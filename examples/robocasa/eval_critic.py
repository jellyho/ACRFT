"""Roll out the VLA under critic-guided best-of-N with adaptive chunking, and record what it chose.

At every replan the RL token is extracted once, N chunks are drawn from the flow policy off that
same backbone forward, and the critic scores every (candidate, prefix) pair. The arg-max picks both
WHICH chunk to run and HOW FAR to commit to it, so the executed length varies per replan - that is
the whole method, and it is why this cannot reuse a fixed ``replan_steps``.

The critic makes two decisions - which chunk, and how far to commit - and they are separable, so
the modes form a 2x2 and any gain can be attributed to one of them rather than to "the method":

                     full chunk        critic picks the prefix
    first sample     vla               prefix
    best of N        bon               critic

`vla` is the policy alone. If `bon` ~ `vla` the candidates are not rankable and best-of-N is a coin
flip; if `prefix` ~ `vla` adaptive chunking is buying nothing. All four share the loop, the scenes
and the seed, so the four numbers are directly comparable.

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


class VLA:
    """The frozen policy side of a critic-guided rollout, loaded once and reused across rollouts.

    Kept separate from the critic so the trainer can hold ONE VLA resident and swap in live critic
    params every evaluation, instead of reloading the 3B model each time.
    """

    def __init__(self, config_name, checkpoint, *, num_samples, flow_steps, seed):
        train_config = _config.get_config(config_name)
        checkpoint = pathlib.Path(_download.maybe_download(str(checkpoint)))
        self.model_config = train_config.model
        data_config = train_config.data.create(train_config.assets_dirs, self.model_config)
        norm_stats = data_config.norm_stats
        if norm_stats is None and data_config.asset_id is not None:
            import openpi.training.checkpoints as _checkpoints

            norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
        self._in = _transforms.compose(
            [
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._out = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )
        model = self.model_config.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        model.eval()
        self.H = self.model_config.action_horizon
        self._rng = [jax.random.key(seed)]

        @jax.jit
        def _extract(rng, obs):
            return model.extract_token_and_base_actions(rng, obs, num_samples=num_samples, num_steps=flow_steps)

        self._extract = _extract
        probe = self._out(
            {
                "state": np.zeros((1, self.model_config.action_dim), np.float32),
                "actions": np.zeros((1, self.H, self.model_config.action_dim), np.float32),
            }
        )
        self.raw_dim = int(np.asarray(probe["actions"]).shape[-1])

    def _decode(self, actions):
        out = self._out(
            {"state": np.zeros((actions.shape[0], self.model_config.action_dim), np.float32), "actions": actions}
        )
        return np.asarray(out["actions"], np.float32)

    def token_and_candidates(self, element):
        """One backbone forward -> (rl token [1, D], decoded candidates [N, H, raw])."""
        inp = self._in(element)
        inp = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inp)
        obs = _model.Observation.from_dict(inp)
        self._rng[0], k = jax.random.split(self._rng[0])
        z, base = self._extract(k, obs)
        return np.asarray(z, np.float32), self._decode(np.asarray(base[0], np.float32))


def make_policy_fn(vla, score, macro, *, mode, query_noise=0.0, softmax_temp=0.0, seed=0):
    """policy_fn(element) -> (chunk, n_exec, Replan). `score(obs, actions)` is a live critic; the vla
    is reused. mode='vla' ignores the critic entirely.

    Two selection knobs, both aimed at the arg-max picking whichever candidate the critic most
    over-values by noise rather than the best one:
      query_noise   perturb each candidate chunk before scoring, so a lone spurious peak that depends
                    on the exact action does not survive; the score is of a neighbourhood, not a point.
      softmax_temp  sample the candidate from softmax(Q / temp) instead of taking the arg-max, so a
                    near-tie is not always resolved toward the noisiest estimate. 0 = hard arg-max.
    """
    rng = np.random.default_rng(seed)

    def fn(element):
        z, cand = vla.token_and_candidates(element)
        if mode == "vla":
            return cand[0], vla.H, None
        scored = cand
        if query_noise > 0:
            # Temporally coherent offset+drift per candidate, scaled by the chunk's own spread, so the
            # perturbed chunk stays a plausible neighbour rather than jittering into noise.
            sd = scored.std(axis=(0, 1), keepdims=True) + 1e-6
            ramp = np.linspace(-1, 1, scored.shape[1])[None, :, None]
            off = rng.standard_normal((scored.shape[0], 1, scored.shape[2]))
            drift = rng.standard_normal((scored.shape[0], 1, scored.shape[2])) * ramp
            scored = scored + query_noise * sd * (off + drift)
        zc = jnp.repeat(jnp.asarray(z)[:, None], cand.shape[0], axis=1)  # [1, N, D]
        q = np.asarray(score(zc, jnp.asarray(scored)[None]))
        q = np.min(q, axis=0)[0]  # ensemble-min -> [N, P]
        flat = q[:, -1] if mode == "bon" else (q[0] if mode == "prefix" else q.reshape(-1))
        if softmax_temp > 0:  # sample instead of arg-max
            w = np.exp((flat - flat.max()) / softmax_temp)
            choice = rng.choice(len(flat), p=w / w.sum())
        else:
            choice = int(np.argmax(flat))
        if mode == "bon":
            i, pp = choice, q.shape[1] - 1
        elif mode == "prefix":
            i, pp = 0, choice
        else:
            i, pp = np.unravel_index(choice, q.shape)
        n_exec = int((pp + 1) * macro)
        return cand[i], n_exec, Replan(q, cand, int(i), int(pp), n_exec, float(q[i, pp]))

    return fn


def build_policy(
    config_name, checkpoint, critic_path, *, mode, num_samples, flow_steps, seed, query_noise=0.0, softmax_temp=0.0
):
    """CLI path: load the VLA and (for critic modes) a critic from disk. Returns (policy_fn, H, macro)."""
    vla = VLA(config_name, checkpoint, num_samples=num_samples, flow_steps=flow_steps, seed=seed)
    kw = {"query_noise": query_noise, "softmax_temp": softmax_temp, "seed": seed}
    if mode == "vla":
        return make_policy_fn(vla, None, None, mode=mode, **kw), vla.H, None
    score, _, macro = _critic.load_trained(critic_path, action_dim=vla.raw_dim, horizon=vla.H)
    return make_policy_fn(vla, jax.jit(score), macro, mode=mode, **kw), vla.H, macro


def main() -> None:
    # force=True: robosuite configures the root logger at import time, which turns a plain
    # basicConfig into a no-op and swallows every progress line - a multi-hour eval then looks
    # identical to a hung one until it prints its final table.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--critic", type=pathlib.Path, default=None, help="Trained critic params.msgpack.")
    ap.add_argument("--task", default="PrepareCoffee")
    ap.add_argument(
        "--modes", nargs="+", default=["critic", "bon", "prefix", "vla"], choices=["critic", "bon", "prefix", "vla"]
    )
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=16, help="Candidates per replan.")
    ap.add_argument(
        "--query-noise", type=float, default=0.0, help="Perturb candidates by this many action-std before scoring."
    )
    ap.add_argument(
        "--softmax-temp", type=float, default=0.0, help="Sample the candidate from softmax(Q/temp); 0 = arg-max."
    )
    ap.add_argument("--num-flow-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="Scene seed; identical across modes and runs.")
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--video-dir", type=pathlib.Path, default=None)
    ap.add_argument("--num-videos", type=int, default=4)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--path-scale",
        type=float,
        default=None,
        help="Metres of end-effector motion per unit of normalised action. Defaults to the value the "
        "evaluation controller actually uses (0.05 for OSC_POSE on PandaOmron), which makes the drawn "
        "paths the predicted trajectory rather than a direction indicator at an arbitrary length.",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Write the per-trial traces here.")
    args = ap.parse_args()

    if "vla" not in args.modes and args.critic is None:
        ap.error("--critic is required unless --modes vla")

    if args.path_scale is None:
        import action_overlay as _ov0

        args.path_scale = _ov0.EE_METRES_PER_UNIT
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
            query_noise=args.query_noise,
            softmax_temp=args.softmax_temp,
        )
        trace, frames, box = [], [], {"trial": 0, "dash": None, "proj": None}
        record = args.video_dir is not None

        def on_step(obs, info, step, *, _trace=trace, _frames=frames, _box=box, _mode=mode, _rec=record, _hz=H):
            if info is not None:
                _trace.append({"step": step, "value": info.value, "n_exec": info.n_exec, "prefix": info.best_prefix})
            if not (_rec and _box["trial"] < args.num_videos):
                return
            import action_overlay as _ov
            import hud as _hud

            if _box["proj"] is None:
                # env.sim is only populated by the first reset, so the projector cannot be built
                # alongside the env - it is built on the first recorded frame instead.
                _box["proj"] = _ov.CameraProjector(env.sim, "robot0_agentview_left", args.camera_size, args.camera_size)
            if _box["dash"] is None:
                _box["dash"] = _hud.Dashboard(mode=_mode, horizon=_hz, camera_size=args.camera_size)
            paths = None
            if info is not None:
                # Anchor every candidate at the LIVE end-effector so the fan stays attached to the
                # gripper as it moves, rather than to wherever the replan happened.
                ee, bq = np.asarray(obs["robot0_eef_pos"]), np.asarray(obs["robot0_base_quat"])
                sc = args.path_scale
                paths = [_box["proj"].project(_ov.predict_path(ee, bq, c, sc)) for c in info.cand]
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
