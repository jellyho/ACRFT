"""Roll out the VLA under critic-guided best-of-N with adaptive chunking, and record what it chose.

At every replan the RL token is extracted once, N chunks are drawn from the flow policy off that
same backbone forward, and the critic scores every (candidate, prefix) pair. The arg-max picks both
WHICH chunk to run and HOW FAR to commit to it, so the executed length varies per replan - that is
the whole method, and it is why this cannot reuse a fixed ``replan_steps``.

Three baselines share the loop so the comparison is like-for-like on the same scenes:

    critic    arg-max over candidates x prefixes            (the method)
    bon       arg-max over candidates, always the full chunk (best-of-N without adaptive chunking)
    vla       the first sample, full chunk                   (the policy on its own)

Videos are optional and carry a HUD: the value of every candidate, which one won, how far it
committed, and the value trace so far. What the critic believed is not recoverable from the frames
otherwise, and a rollout that fails while the value climbs is a different bug from one that fails
while it drops.

Usage:
    uv run examples/robocasa/eval_critic.py --config pi05_robocasa_PrepareCoffee_rlt \
        --checkpoint checkpoints/.../30000 --critic .scratch/critic3_ref/params.msgpack \
        --task PrepareCoffee --num-trials 20 --video-dir .scratch/critic_videos
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import rollout as _ro

import openpi.models.model as _model
import openpi.rlt_critic.critic as _critic
import openpi.shared.download as _download
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Replan:
    """What the policy decided at one replan, kept for the HUD and the per-trial trace."""

    q: np.ndarray  # [N, P] value of every candidate at every prefix length
    cand: np.ndarray  # [N, H, A] the candidate chunks themselves, for the path overlay
    best_cand: int
    best_prefix: int  # 0-indexed macro prefix
    n_exec: int  # real steps committed to
    value: float  # the winning value


def build_policy(config_name, checkpoint, critic_path, *, mode, num_samples, flow_steps, seed):
    """A `policy_fn(element) -> (chunk, n_exec, Replan)` for run_trials."""
    train_config = _config.get_config(config_name)
    checkpoint = pathlib.Path(_download.maybe_download(str(checkpoint)))
    model_config = train_config.model
    data_config = train_config.data.create(train_config.assets_dirs, model_config)

    norm_stats = data_config.norm_stats
    if norm_stats is None and data_config.asset_id is not None:
        import openpi.training.checkpoints as _checkpoints

        norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    input_tf = _transforms.compose(
        [
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_tf = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )

    model = model_config.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()
    H = model_config.action_horizon

    @jax.jit
    def extract(rng, obs):
        return model.extract_token_and_base_actions(rng, obs, num_samples=num_samples, num_steps=flow_steps)

    score = macro = None
    if mode != "vla":
        # The critic was fitted on decoded (raw) chunks, so candidates are decoded before scoring -
        # feeding it model-space actions would be a silent unit mismatch, not an error.
        probe = output_tf(
            {
                "state": np.zeros((1, model_config.action_dim), np.float32),
                "actions": np.zeros((1, H, model_config.action_dim), np.float32),
            }
        )
        raw_dim = int(np.asarray(probe["actions"]).shape[-1])
        score, _, macro = _critic.load_trained(critic_path, action_dim=raw_dim, horizon=H)
        score = jax.jit(score)

    rng_holder = [jax.random.key(seed)]

    def decode(actions):
        out = output_tf(
            {"state": np.zeros((actions.shape[0], model_config.action_dim), np.float32), "actions": actions}
        )
        return np.asarray(out["actions"], np.float32)

    def fn(element):
        inp = input_tf(element)
        inp = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inp)
        obs = _model.Observation.from_dict(inp)
        rng_holder[0], k = jax.random.split(rng_holder[0])
        z, base = extract(k, obs)  # [1, D], [1, N, H, adim]
        cand = decode(np.asarray(base[0], np.float32))  # [N, H, raw]
        if mode == "vla":
            return cand[0], H, None

        q = np.asarray(score(jnp.asarray(np.asarray(z, np.float32)), jnp.asarray(cand)[None]))
        q = np.min(q, axis=0)[0]  # ensemble-min -> [N, P]
        if mode == "bon":  # choose the chunk, but always run all of it
            i, p = int(np.argmax(q[:, -1])), q.shape[1] - 1
        else:
            i, p = np.unravel_index(int(np.argmax(q)), q.shape)
        n_exec = int((p + 1) * macro)
        return cand[i], n_exec, Replan(q, cand, int(i), int(p), n_exec, float(q[i, p]))

    return fn, H, macro


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--critic", type=pathlib.Path, default=None, help="Trained critic params.msgpack.")
    ap.add_argument("--task", default="PrepareCoffee")
    ap.add_argument("--modes", nargs="+", default=["critic", "bon", "vla"], choices=["critic", "bon", "vla"])
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=16, help="Candidates per replan.")
    ap.add_argument("--num-flow-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="Scene seed; identical across modes and runs.")
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--video-dir", type=pathlib.Path, default=None)
    ap.add_argument("--num-videos", type=int, default=4)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--path-scale",
        type=float,
        default=0.12,
        help="Metres the drawn candidate paths should span. The raw chunk deltas are in normalised "
        "control units, so a fixed scale would make the fan invisible on some replans and fill the "
        "frame on others; this fixes the executed chunk's span and scales the rest to match.",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Write the per-trial traces here.")
    args = ap.parse_args()

    if "vla" not in args.modes and args.critic is None:
        ap.error("--critic is required unless --modes vla")

    env = _ro.make_env(args.task, camera_size=args.camera_size, seed=args.seed)
    results = {}
    for mode in args.modes:
        policy, H, macro = build_policy(
            args.config,
            args.checkpoint,
            args.critic,
            mode=mode,
            num_samples=args.num_samples,
            flow_steps=args.num_flow_steps,
            seed=args.seed,
        )
        trace, frames, box = [], [], {"trial": 0, "dash": None}
        record = args.video_dir is not None
        projector = None
        if record:
            import action_overlay as _ov

            projector = _ov.CameraProjector(env.sim, "robot0_agentview_left", args.camera_size, args.camera_size)

        def on_step(
            obs, info, step, *, _trace=trace, _frames=frames, _box=box, _mode=mode, _rec=record, _proj=projector, _hz=H
        ):
            if info is not None:
                _trace.append({"step": step, "value": info.value, "n_exec": info.n_exec, "prefix": info.best_prefix})
            if not (_rec and _box["trial"] < args.num_videos):
                return
            import action_overlay as _ov
            import hud as _hud

            if _box["dash"] is None:
                _box["dash"] = _hud.Dashboard(mode=_mode, horizon=_hz, camera_size=args.camera_size)
            paths = None
            if info is not None:
                # Anchor every candidate at the LIVE end-effector so the fan stays attached to the
                # gripper as it moves, rather than to wherever the replan happened.
                ee, bq = np.asarray(obs["robot0_eef_pos"]), np.asarray(obs["robot0_base_quat"])
                sc = _ov._adaptive_scale(info.cand[info.best_cand], args.path_scale)
                paths = [_proj.project(_ov.predict_path(ee, bq, c, sc)) for c in info.cand]
            _frames.append(
                _box["dash"].frame(
                    _ro.image_from_obs(obs, _ro.CAMERAS["observation/image"]),
                    _ro.image_from_obs(obs, _ro.CAMERAS["observation/wrist_image"]),
                    info,
                    step,
                    paths=paths,
                    chosen=(info.best_cand if info is not None else 0),
                    success=bool(env._check_success()),
                )
            )

        def on_trial(trial, success, steps, *, _frames=frames, _box=box, _mode=mode, _rec=record):
            logger.info(
                f"[{_mode}] trial {trial + 1}/{args.num_trials}: {'SUCCESS' if success else 'failure'} in {steps} steps"
            )
            if _rec and trial < args.num_videos and _frames:
                import imageio

                args.video_dir.mkdir(parents=True, exist_ok=True)
                out = args.video_dir / f"{args.task}_{_mode}_t{trial:02d}_{'succ' if success else 'fail'}.mp4"
                imageio.mimwrite(out, _frames, fps=args.fps, quality=9)
                logger.info(f"  saved {out}  ({len(_frames)} frames)")
            _frames.clear()
            _box["trial"] = trial + 1
            _box["dash"] = None

        np.random.seed(args.seed)  # robosuite placement samplers read the legacy global RNG
        res = _ro.run_trials(
            env,
            policy,
            task=args.task,
            num_trials=args.num_trials,
            seed=args.seed,
            replan_steps=H,
            on_trial=on_trial,
            on_step=on_step,
        )
        res["trace"] = trace
        results[mode] = res
        logger.info(f"[{mode}] success {res['successes']}/{res['num_trials']} = {res['success_rate']:.0%}")
        if trace:
            ns = np.array([t["n_exec"] for t in trace])
            logger.info(
                f"[{mode}] committed steps: mean {ns.mean():.1f} of {H}, histogram {np.bincount(ns, minlength=H + 1)[1:].tolist()}"
            )

    print("\n=== success rate ===")
    for mode, r in results.items():
        print(f"  {mode:8s} {r['successes']:3d}/{r['num_trials']}  {r['success_rate']:.0%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
