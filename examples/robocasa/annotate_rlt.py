"""Stage 1 of the critic pipeline: precompute RL tokens + base-policy action candidates.

Runs a trained Pi0RLT checkpoint over every frame of a RoboCasa task and writes, per frame:

    rl_token     (2048,)          float32   the compact state representation the critic consumes
    action_chunk (H, 12)          float32   the chunk actually executed in the demo, RAW space
                                            (this is the `a` in Q(s, a) that the critic regresses)
    base_action  (N, H, 12)       float32   N action chunks sampled from the VLA, in RAW action space
                                            (the `a'` candidates the bootstrap maximises over)
    reward       ()               float32   sparse success reward, from the dataset
    mc_return    ()               float32   discounted return-to-go (the critic's MC target)
    progress     ()               float32   1 - time-to-success / horizon (see robocasa_progress.py)
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
import openpi.training.robocasa_progress as _progress
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
    ap.add_argument("--num-flow-steps", type=int, default=10, help="Flow-matching denoising steps.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8, help="Frame-decoding workers.")
    ap.add_argument("--stride", type=int, default=1, help="Keep every k-th frame (1 = all frames).")
    ap.add_argument("--discount", type=float, default=0.99, help="Discount for mc_return.")
    ap.add_argument("--max-frames", type=int, default=0, help="Debug: stop after this many frames.")
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
    n_keep = len(keep)
    logger.info(f"{args.config}: {n_total} frames, keeping {n_keep} (stride {args.stride})")

    # --- reward / mc_return / progress, straight from the dataset's sparse success reward ---
    labels = _progress.compute_progress_labels(data_config.repo_id)
    from lerobot.datasets import lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    reward_all = _progress._read_reward_column(pathlib.Path(meta.root), meta.total_frames)
    lo = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    hi = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)
    ep_of = np.zeros(meta.total_frames, dtype=np.int32)
    frame_of = np.zeros(meta.total_frames, dtype=np.int32)
    mc_all = np.zeros(meta.total_frames, dtype=np.float32)
    done_all = np.zeros(meta.total_frames, dtype=np.int8)
    for e, (a, b) in enumerate(zip(lo, hi)):
        ep_of[a:b] = e
        frame_of[a:b] = np.arange(b - a)
        done_all[b - 1] = 1
        # Discounted return-to-go of the sparse reward, computed backwards within the episode.
        g = 0.0
        seg = reward_all[a:b]
        for t in range(b - a - 1, -1, -1):
            g = float(seg[t]) + args.discount * g
            mc_all[a + t] = g
    progress_all = labels.progress(ep_of, frame_of)

    # --- output arrays (memmap, written incrementally so --resume works) ---
    args.out.mkdir(parents=True, exist_ok=True)
    D = model_config.rlt_token_dim
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
        "stride": args.stride,
        "discount": args.discount,
        "dtype": args.dtype,
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "repo_id": data_config.repo_id,
    }
    (args.out / "meta.json").write_text(json.dumps(spec, indent=2))

    import ml_dtypes  # bfloat16 as a numpy dtype (ships with jax)

    store_dtype = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[args.dtype]

    def _mm(name, shape, dtype=None):
        return np.memmap(
            args.out / f"{name}.dat",
            dtype=store_dtype if dtype is None else dtype,
            mode="r+" if args.resume else "w+",
            shape=shape,
        )

    tok_mm = _mm("rl_token", (n_keep, D))
    chunk_mm = _mm("action_chunk", (n_keep, H, raw_dim))
    act_mm = _mm("base_action", (n_keep, args.num_samples, H, raw_dim))
    scalars = {
        k: _mm(k, (n_keep,), dt)
        for k, dt in [("reward", np.float32), ("mc_return", np.float32), ("progress", np.float32),
                      ("episode_index", np.int32), ("frame_index", np.int32), ("done", np.int8)]
    }
    done_path = args.out / "_progress.json"
    start = json.loads(done_path.read_text())["done"] if (args.resume and done_path.exists()) else 0
    if start:
        logger.info(f"resuming at frame {start}/{n_keep}")

    # --- model ---
    model = model_config.load(_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()

    @jax.jit
    def extract(rng, obs):
        return model.extract_token_and_base_actions(
            rng, obs, num_samples=args.num_samples, num_steps=args.num_flow_steps
        )

    rng = jax.random.key(args.seed)
    t0 = time.perf_counter()
    for s in range(start, n_keep, args.batch_size):
        idx = keep[s : s + args.batch_size]
        items = [dataset[int(i)] for i in idx]
        batch = jax.tree.map(lambda *xs: np.stack(xs), *items)
        obs = _model.Observation.from_dict(batch)

        z_rl, base = extract(jax.random.fold_in(rng, s), obs)
        z_rl = np.asarray(z_rl, np.float32)
        base = np.asarray(base, np.float32)  # [b, N, H, model_dim]
        b = len(idx)
        raw = decode_actions(base.reshape(b * args.num_samples, H, -1)).reshape(b, args.num_samples, H, raw_dim)

        tok_mm[s : s + b] = z_rl
        # The demo's own chunk, decoded through the same chain so it lives in the same space as the
        # candidates -- without it there is no `a` to evaluate Q(s, a) on.
        chunk_mm[s : s + b] = decode_actions(np.asarray(batch["actions"], np.float32))
        act_mm[s : s + b] = raw
        scalars["reward"][s : s + b] = reward_all[idx]
        scalars["mc_return"][s : s + b] = mc_all[idx]
        scalars["progress"][s : s + b] = progress_all[idx]
        scalars["episode_index"][s : s + b] = ep_of[idx]
        scalars["frame_index"][s : s + b] = frame_of[idx]
        scalars["done"][s : s + b] = done_all[idx]

        if (s // args.batch_size) % 20 == 0:
            el = time.perf_counter() - t0
            rate = (s - start + b) / max(el, 1e-6)
            eta = (n_keep - s - b) / max(rate, 1e-6)
            logger.info(f"{s + b}/{n_keep}  {rate:.1f} frames/s  eta {eta / 60:.1f} min")
            done_path.write_text(json.dumps({"done": int(s + b)}))

    for m in (tok_mm, chunk_mm, act_mm, *scalars.values()):
        m.flush()
    done_path.write_text(json.dumps({"done": int(n_keep)}))
    logger.info(f"wrote {n_keep} frames to {args.out} in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
