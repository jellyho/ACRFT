"""Annotate ROLLOUT trajectories (successes AND failures) into the critic's memmap format.

The demo pipeline (annotate_rlt.py) reads a LeRobot dataset; this one reads the npz trajectories
that ``eval_critic.py --dump-traj`` records during rollouts. Output layout is identical, so
critic training mixes rollout data with demo data by simply pointing --data at the merged dir.

Why failures matter: the demos are success-only, so nothing in them distinguishes a good action
chunk from a bad one — the measured result is a critic whose Q is constant across candidates.
Failed rollouts are the missing contrast: states from which the behavior policy did NOT reach
success carry mc_return = 0 and (with the discount) drag V down exactly where it should be.

Reward convention matches the demo annotation: sparse 1.0 at the first success frame, success is
terminal, failures pay nothing anywhere and terminate at their last frame (timeout). mc_return is
the discounted return-to-go of that sparse reward — exactly gamma^(steps to success) on successes,
identically 0 on failures.

    uv run --no-sync python examples/robocasa/annotate_rollouts.py \
        --traj-dir $CACHE_DIR/rollout_traj --out $CACHE_DIR/annot/rollouts \
        --config pi05_robocasa_PrepareCoffee_rlt --checkpoint $VLA_CKPT \
        --vla-override rlt_decoder_mode=parallel --vla-override rlt_include_proprio=false \
        --shard 0/8   # optional: disjoint trial ranges may run in parallel into the same out dir
"""

import argparse
import io
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

logger = logging.getLogger(__name__)


def load_traj(f):
    z = np.load(f, allow_pickle=True)
    return {
        "images": z["images"],
        "wrist": z["wrist"],
        "states": z["states"].astype(np.float32),
        "actions": z["actions"].astype(np.float32),
        "success": bool(z["success"]),
        "prompt": str(z["prompt"]) if "prompt" in z else str(z["task"]),
        "name": pathlib.Path(f).stem,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--vla-override", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--num-flow-steps", type=int, default=10)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/k — this process annotates trials i, i+k, i+2k, …")
    args = ap.parse_args()

    files = sorted(args.traj_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no npz trajectories in {args.traj_dir}")
    trajs_meta = []  # (file, num_frames) for every trial, in global order — all shards agree on this
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            trajs_meta.append((f, int(z["states"].shape[0])))
    offsets = np.concatenate([[0], np.cumsum([n for _, n in trajs_meta])])
    total = int(offsets[-1])

    si, sk = (int(x) for x in args.shard.split("/"))
    own = list(range(si, len(files), sk))
    logger.info(f"{len(files)} trials / {total} frames total; shard {args.shard} owns {len(own)} trials")

    from eval_critic import VLA  # noqa: E402  (single VLA loader shared with rollouts)

    vla = VLA(
        args.config,
        args.checkpoint,
        num_samples=args.num_samples,
        flow_steps=args.num_flow_steps,
        seed=args.seed,
        model_overrides=dict(kv.split("=", 1) for kv in args.vla_override) if args.vla_override else None,
    )
    H, A = vla.H, vla.raw_dim

    # Probe one frame for the token dim, then open/create the shared memmaps.
    first = load_traj(trajs_meta[own[0]][0])

    def element(tr, t):
        from PIL import Image

        return {
            "observation/image": np.asarray(Image.open(io.BytesIO(tr["images"][t]))),
            "observation/wrist_image": np.asarray(Image.open(io.BytesIO(tr["wrist"][t]))),
            "observation/state": tr["states"][t],
            "prompt": tr["prompt"],
        }

    z0, _ = vla.token_and_candidates(element(first, 0))
    token_dim = int(np.asarray(z0).shape[-1])
    proprio_dim = int(first["states"].shape[-1])

    args.out.mkdir(parents=True, exist_ok=True)
    spec = {
        "num_frames": total,
        "horizon": H,
        "action_dim": A,
        "num_samples": args.num_samples,
        "token_dim": token_dim,
        "proprio_dim": proprio_dim,
        "discount": args.discount,
        "config": args.config,
        "source": "rollouts",
        "reward_scheme": "sparse",
        "trials": [{"file": str(f.name), "frames": n} for f, n in trajs_meta],
    }
    meta_f = args.out / "meta.json"
    if not meta_f.exists():
        meta_f.write_text(json.dumps(spec, indent=2))

    def mm(name, shape, dtype=np.float32):
        path = args.out / f"{name}.dat"
        mode = "r+" if path.exists() else "w+"
        return np.memmap(path, dtype, mode, shape=shape)

    rl_token = mm("rl_token", (total, token_dim))
    proprio = mm("proprio", (total, proprio_dim))
    action_chunk = mm("action_chunk", (total, H, A))
    base_action = mm("base_action", (total, args.num_samples, H, A))
    reward = mm("reward", (total,))
    mc_return = mm("mc_return", (total,))
    done = mm("done", (total,), np.int8)

    for k, ti in enumerate(own):
        f, n = trajs_meta[ti]
        lo = int(offsets[ti])
        tr = load_traj(f)
        # Sparse reward: 1.0 at the trial's last recorded frame when it succeeded (run_trials stops
        # the moment _check_success fires, so the last frame IS the success frame); terminal there.
        r = np.zeros(n, np.float32)
        if tr["success"]:
            r[n - 1] = 1.0
        g = 0.0
        mc = np.zeros(n, np.float32)
        for t in range(n - 1, -1, -1):
            g = r[t] + args.discount * g
            mc[t] = g
        acts = tr["actions"]
        pad = np.concatenate([acts, np.repeat(acts[-1:], H, axis=0)], axis=0)
        for t in range(n):
            zt, cand = vla.token_and_candidates(element(tr, t))
            rl_token[lo + t] = np.asarray(zt, np.float32)[0]
            base_action[lo + t] = np.asarray(cand, np.float32)
            action_chunk[lo + t] = pad[t : t + H]
            proprio[lo + t] = tr["states"][t]
        reward[lo : lo + n] = r
        mc_return[lo : lo + n] = mc
        done[lo : lo + n] = 0
        done[lo + n - 1] = 1
        for arr in (rl_token, base_action, action_chunk, proprio, reward, mc_return, done):
            arr.flush()
        logger.info(
            f"[{k + 1}/{len(own)}] {tr['name']}: {n} frames {'SUCCESS' if tr['success'] else 'fail'} "
            f"(mc[0]={mc[0]:.4f})"
        )
    logger.info(f"shard {args.shard} done -> {args.out}")


if __name__ == "__main__":
    main()
