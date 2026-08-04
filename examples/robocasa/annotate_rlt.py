"""Stage 1 of the critic pipeline: precompute RL tokens + base-policy action candidates.

Runs a trained Pi0RLT checkpoint over every frame of a RoboCasa task and writes, per frame:

    rl_token     (2048,)          float32   the compact state representation the critic consumes
    action_chunk (H, 12)          float32   the chunk actually executed in the demo, RAW space
                                            (this is the `a` in Q(s, a) that the critic regresses)
    base_action  (N, H, 12)       float32   N action chunks sampled from the VLA, in RAW action space
                                            (the `a'` candidates the bootstrap maximises over)
    reward       ()               float32   sparse success reward, from the dataset
    mc_return    ()               float32   discounted return-to-go (the critic's MC target)
    progress     ()               float32   1 - time-to-success / horizon (see training/progress.py)
    episode/frame index, done

The whole point is that critic training then never touches the VLA or any video: it reads these
flat arrays straight into GPU memory.

Two things make this pass as cheap as it can be:

  * ONE backbone forward per batch. ``extract_token_and_base_actions`` runs the 3B prefix once and
    reuses its KV cache for the flow-matching sampler, instead of paying for it twice.
  * SEQUENTIAL frame order. The frames live in mp4s; decoding them in order lets the decoder stream,
    while random access pays a keyframe seek per frame. Video decode — not the GPU — is the likely
    bottleneck, so this matters more than it looks.

Output is written incrementally into np.memmap files, so an interrupted run can be resumed with
--resume instead of starting over.

Usage (single task):

    srun --gres=gpu:1 uv run examples/robocasa/annotate_rlt.py \
        --config pi05_robocasa_PrepareCoffee_rlt \
        --checkpoint checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt/100000 \
        --out data/rlt_critic/PrepareCoffee
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.progress as _progress
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def _build_action_decoder(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path):
    """Model-space action chunks -> raw action space, mirroring the serving output chain."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = data_config.norm_stats
    if norm_stats is None and data_config.asset_id is not None:
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    if norm_stats is None:
        raise ValueError("No norm stats: cannot decode actions back to raw space.")
    chain = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]

    state_dim = train_config.model.action_dim  # states are padded to the model action dim

    def decode(actions: np.ndarray) -> np.ndarray:
        # `Unnormalize` is strict about the keys in the norm stats, so `state` has to be present even
        # though only the actions are wanted here — this mirrors what Policy.infer feeds it.
        out = {"state": np.zeros((actions.shape[0], state_dim), np.float32), "actions": actions}
        for t in chain:
            out = t(out)
        return np.asarray(out["actions"], dtype=np.float32)

    return decode, data_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Registered Pi0RLT train config name.")
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path, help="Checkpoint dir (…/<step>).")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="Output directory for the arrays.")
    ap.add_argument("--num-samples", type=int, default=32, help="Action chunks sampled per frame (N).")
    ap.add_argument(
        "--num-heldout",
        type=int,
        default=8,
        help="Extra candidates per frame, written to `base_action_heldout` and never used by the "
        "bootstrap. Scoring these measures whether the critic's ranking generalises to chunks the "
        "policy could have drawn but did not - the max over a finite candidate set is biased upward, "
        "and this is how we see it. They cost one extra sampler pass, not an extra backbone forward.",
    )
    # The registered config carries the DEFAULT model variant; a checkpoint trained with a different
    # objective or decoder has a different parameter structure and will not load without these.
    ap.add_argument("--objective", default=None, help="Override rlt_objective to match the checkpoint.")
    ap.add_argument("--decoder-mode", default=None, help="Override rlt_decoder_mode (autoregressive|parallel).")
    ap.add_argument("--no-proprio", action="store_true", help="Checkpoint was trained with --no-proprio.")
    ap.add_argument(
        "--proprio-key",
        default="observation.state",
        help="Per-frame vector column written alongside the tokens as proprio.dat. The critic always "
        "reads it, because the RL token deliberately does not carry it.",
    )
    ap.add_argument("--token-dim", type=int, default=None, help="Override rlt_token_dim.")
    ap.add_argument("--num-tokens", type=int, default=None, help="Override rlt_num_tokens.")
    ap.add_argument("--num-flow-steps", type=int, default=10, help="Flow-matching denoising steps.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8, help="Frame-decoding workers.")
    ap.add_argument("--stride", type=int, default=1, help="Keep every k-th frame (1 = all frames).")
    ap.add_argument("--discount", type=float, default=0.99, help="Discount for mc_return.")
    ap.add_argument(
        "--no-terminal-success",
        action="store_true",
        help="Keep the raw reward column instead of cutting each episode at its first success. The "
        "raw column pays 1 for every frame success is held (16 of them on RoboCasa) and only then "
        "ends the episode, which makes the return measure how long the environment lingered as well "
        "as how fast the policy got there, and leaves the value ceiling depending on which episodes "
        "were annotated. Only for reproducing older runs.",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="Debug: only annotate this many frames.")
    ap.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First kept-frame index this process is responsible for. The arrays are always sized for "
        "the whole dataset, so several processes can annotate disjoint ranges into the same output "
        "directory on different GPUs and no merge step is needed - the sampler dominates the cost, so "
        "this scales close to linearly.",
    )
    ap.add_argument("--frame-end", type=int, default=0, help="One past the last index (0 = to the end).")
    ap.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Storage dtype for the token/action arrays. float32 is the safe default; the 16-bit "
        "options halve the files (6.7 -> 3.4 GB per task) and matter mainly at many-task scale. "
        "float16 beats bfloat16 here: the values sit around O(1)-O(10), so the extra mantissa is "
        "worth more than bfloat16's wider exponent.",
    )
    ap.add_argument("--resume", action="store_true", help="Continue a partially written output dir.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_config = _config.get_config(args.config)
    overrides = {}
    if args.objective:
        overrides["rlt_objective"] = args.objective
    if args.decoder_mode:
        overrides["rlt_decoder_mode"] = args.decoder_mode
    if args.no_proprio:
        overrides["rlt_include_proprio"] = False
    if args.token_dim:
        overrides["rlt_token_dim"] = args.token_dim
    if args.num_tokens is not None:
        overrides["rlt_num_tokens"] = args.num_tokens
    if overrides:
        train_config = dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, **overrides))
        logger.info(f"model overrides: {overrides}")
    model_config = train_config.model
    if not hasattr(model_config, "rlt_token_dim"):
        raise ValueError(f"{args.config} is not a Pi0RLT config (no rlt_token_dim).")

    decode_actions, data_config = _build_action_decoder(train_config, args.checkpoint)

    # Sequential, untransformed-order dataset: shuffling here would cost a keyframe seek per frame.
    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    n_total = len(dataset)
    keep = np.arange(0, n_total, args.stride)
    if args.max_frames:
        keep = keep[: args.max_frames]
    n_keep = len(keep)  # arrays are always this size, whatever range this process handles
    lo = max(0, args.frame_start)
    hi = min(n_keep, args.frame_end or n_keep)
    if lo >= hi:
        raise ValueError(f"empty range [{lo}, {hi})")
    logger.info(f"{args.config}: {n_total} frames, keeping {n_keep} (stride {args.stride})")

    # --- reward / mc_return / progress, straight from the dataset's sparse success reward ---
    labels = _progress.compute_progress_labels(data_config.repo_id)
    from lerobot.datasets import lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    reward_all = _progress.read_reward_column(pathlib.Path(meta.root), meta.total_frames)
    # Proprioception is written HERE, not joined back later. The `noprop` bottleneck leaves it out of
    # the RL token by design, so every critic needs it, and the dataset is already open on the exact
    # frame indices being annotated - taking it now is one column read, whereas recovering it
    # afterwards means re-opening the parquet and matching on (episode_index, frame_index).
    proprio_all = _progress.read_vector_column(pathlib.Path(meta.root), meta.total_frames, args.proprio_key)
    if proprio_all is None:
        raise ValueError(
            f"{args.proprio_key!r} is not a column of {data_config.repo_id}. The critic cannot judge an "
            f"action chunk without knowing where the arm is; pass --proprio-key to name the right column."
        )
    logger.info(f"proprio: {args.proprio_key} is {proprio_all.shape[1]}-d")
    if reward_all is None:  # no success column: the sparse reward is all zeros
        reward_all = np.zeros(meta.total_frames, dtype=np.float32)
    # NOT lo/hi: those name the frame range this process owns, a few lines up.
    ep_lo = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    ep_hi = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)

    if not args.no_terminal_success:
        # Success is terminal: the first frame the reward fires pays 1 and nothing after it pays
        # anything. The raw column instead pays 1 on every frame success is held - 16 of them here -
        # and only then ends the episode, so the return conflates "how fast did the policy succeed"
        # with "how long did the simulator keep the flag up", which is no part of the policy's doing.
        # It also leaves the reachable value depending on which episodes were annotated (14.85 under
        # the usual 16-frame hold, 20.01 for one episode where success flickered and re-fired). With
        # success terminal, V*(s) = gamma^(steps to success), bounded by exactly 1.
        cut = 0
        for a, b in zip(ep_lo, ep_hi, strict=True):
            fired = np.flatnonzero(reward_all[a:b])
            if len(fired) == 0:
                continue
            first = a + int(fired[0])
            reward_all[first + 1 : b] = 0.0
            cut += b - 1 - first
        logger.info(f"success is terminal: {cut} post-success frames pay nothing; value ceiling is 1.0")
    ep_of = np.zeros(meta.total_frames, dtype=np.int32)
    frame_of = np.zeros(meta.total_frames, dtype=np.int32)
    mc_all = np.zeros(meta.total_frames, dtype=np.float32)
    done_all = np.zeros(meta.total_frames, dtype=np.int8)
    for e, (a, b) in enumerate(zip(ep_lo, ep_hi, strict=True)):
        ep_of[a:b] = e
        frame_of[a:b] = np.arange(b - a)
        # Done at the episode's last frame, and at a terminal success so nothing bootstraps past it.
        done_all[b - 1] = 1
        fired = np.flatnonzero(reward_all[a:b])
        if len(fired):
            done_all[a + int(fired[0])] = 1
        # Discounted return-to-go of the sparse reward, computed backwards within the episode.
        g = 0.0
        seg = reward_all[a:b]
        for t in range(b - a - 1, -1, -1):
            g = float(seg[t]) + args.discount * g
            mc_all[a + t] = g
    progress_all = labels.progress(ep_of, frame_of)

    # --- output arrays (memmap, written incrementally so --resume works) ---
    args.out.mkdir(parents=True, exist_ok=True)
    # The encoder emits rlt_num_tokens vectors, concatenated; that product is what lands on disk.
    D = model_config.rlt_token_dim * getattr(model_config, "rlt_num_tokens", 1)
    H = model_config.action_horizon
    probe = decode_actions(np.zeros((1, H, model_config.action_dim), np.float32))
    raw_dim = int(probe.shape[-1])
    logger.info(f"token dim {D}, chunk {H}x{raw_dim} raw, N={args.num_samples}")

    spec = {
        "num_frames": int(n_keep),
        "token_dim": D,
        "horizon": H,
        "action_dim": raw_dim,
        "num_samples": args.num_samples,
        "num_heldout": args.num_heldout,
        "stride": args.stride,
        "discount": args.discount,
        "dtype": args.dtype,
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "repo_id": data_config.repo_id,
        "proprio_dim": int(proprio_all.shape[1]),
        "proprio_key": args.proprio_key,
    }
    (args.out / "meta.json").write_text(json.dumps(spec, indent=2))

    import ml_dtypes  # bfloat16 as a numpy dtype (ships with jax)

    store_dtype = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[args.dtype]

    def _mm(name, shape, dtype=None):
        return np.memmap(
            args.out / f"{name}.dat",
            dtype=store_dtype if dtype is None else dtype,
            # Existing files are opened in place so shards do not clobber each other.
            mode="r+" if (args.resume or lo or (args.out / f"{name}.dat").exists()) else "w+",
            shape=shape,
        )

    tok_mm = _mm("rl_token", (n_keep, D))
    proprio_mm = _mm("proprio", (n_keep, proprio_all.shape[1]), np.float32)
    chunk_mm = _mm("action_chunk", (n_keep, H, raw_dim))
    act_mm = _mm("base_action", (n_keep, args.num_samples, H, raw_dim))
    held_mm = _mm("base_action_heldout", (n_keep, args.num_heldout, H, raw_dim)) if args.num_heldout else None
    scalars = {
        k: _mm(k, (n_keep,), dt)
        for k, dt in [
            ("reward", np.float32),
            ("mc_return", np.float32),
            ("progress", np.float32),
            ("episode_index", np.int32),
            ("frame_index", np.int32),
            ("done", np.int8),
        ]
    }
    done_path = args.out / (f"_progress_{lo}.json" if (lo or hi != n_keep) else "_progress.json")
    start = json.loads(done_path.read_text())["done"] if (args.resume and done_path.exists()) else lo
    if start:
        logger.info(f"resuming at frame {start}/{n_keep}")

    # --- model ---
    model = model_config.load(_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()

    # Draw train and held-out candidates in ONE call: the expensive part is the 3B prefix forward,
    # which is shared, so the extra chunks cost only additional flow-matching passes.
    n_cand = args.num_samples + args.num_heldout

    @jax.jit
    def extract(rng, obs):
        return model.extract_token_and_base_actions(rng, obs, num_samples=n_cand, num_steps=args.num_flow_steps)

    # Decode ahead of the GPU. Frame decoding and the backbone forward are both slow; run serially
    # they simply add, and the accelerator idles through every decode. A worker pool filling a bounded
    # queue keeps the next batch ready before the current one finishes, so the wall clock is whichever
    # of the two is slower rather than their sum.
    rng = jax.random.key(args.seed)
    t0 = time.perf_counter()

    def fetch(ix):
        items = [dataset[int(i)] for i in ix]
        return jax.tree.map(lambda *xs: np.stack(xs), *items)

    # Overlap decoding with the GPU WITHOUT threads. The video decoders underneath the dataset are
    # not thread-safe (decoding one from several threads segfaults), but jax dispatch is already
    # asynchronous: `extract` returns as soon as the work is queued. So decode the next batch while
    # the accelerator is still busy with the previous one, and only block when writing results out.
    pending = None
    n_written = 0

    def _write_scalars(s0, ix, nb):
        nonlocal n_written
        scalars["reward"][s0 : s0 + nb] = reward_all[ix]
        scalars["mc_return"][s0 : s0 + nb] = mc_all[ix]
        scalars["progress"][s0 : s0 + nb] = progress_all[ix]
        scalars["episode_index"][s0 : s0 + nb] = ep_of[ix]
        scalars["frame_index"][s0 : s0 + nb] = frame_of[ix]
        scalars["done"][s0 : s0 + nb] = done_all[ix]
        proprio_mm[s0 : s0 + nb] = proprio_all[ix]
        n_written = s0 + nb
        if (s0 // args.batch_size) % 20 == 0:
            el = time.perf_counter() - t0
            rate = (n_written - start) / max(el, 1e-6)
            logger.info(f"{n_written}/{hi}  {rate:.1f} frames/s  eta {(hi - n_written) / max(rate, 1e-6) / 60:.1f} min")
            done_path.write_text(json.dumps({"done": int(n_written)}))

    def flush(p):
        s0, ix, bat, z_dev, base_dev = p
        z_rl = np.asarray(z_dev, np.float32)  # blocks here, once, on work queued an iteration ago
        base = np.asarray(base_dev, np.float32)  # [b, N, H, model_dim]
        nb = len(ix)
        raw = decode_actions(base.reshape(nb * n_cand, H, -1)).reshape(nb, n_cand, H, raw_dim)
        tok_mm[s0 : s0 + nb] = z_rl
        # The demo's own chunk, decoded through the same chain so it lives in the same space as the
        # candidates -- without it there is no `a` to evaluate Q(s, a) on.
        chunk_mm[s0 : s0 + nb] = decode_actions(np.asarray(bat["actions"], np.float32))
        act_mm[s0 : s0 + nb] = raw[:, : args.num_samples]
        if held_mm is not None:
            held_mm[s0 : s0 + nb] = raw[:, args.num_samples :]
        return s0, ix, nb

    for s in range(start, hi, args.batch_size):
        idx = keep[s : min(s + args.batch_size, hi)]
        batch = fetch(idx)  # CPU decode, concurrent with the previous batch's GPU work
        z_dev, base_dev = extract(jax.random.fold_in(rng, s), _model.Observation.from_dict(batch))
        if pending is not None:
            ps, pidx, pb = flush(pending)
            _write_scalars(ps, pidx, pb)
        pending = (s, idx, batch, z_dev, base_dev)

    if pending is not None:
        _write_scalars(*flush(pending))

    for m in (tok_mm, chunk_mm, act_mm, proprio_mm, *scalars.values()):
        m.flush()
    done_path.write_text(json.dumps({"done": int(hi)}))
    logger.info(f"wrote frames [{lo}, {hi}) to {args.out} in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
