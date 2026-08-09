"""GR1 rollout client — runs in .venv-gr1 (mujoco 3.2.6 stack), queries a policy server.

The server (main venv) exposes the standard openpi websocket infer(obs) -> {"actions": chunk}
interface; BoN/critic logic lives server-side, and the commit length is expressed by the LENGTH of
the returned chunk — this loop just executes what it receives (variable-horizon replan).

    # server (main venv):  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_gr1_rlt ...
    # client (.venv-gr1):  python examples/gr1/rollout_client.py --task PnPCanToDrawerClose \\
    #                        --num-trials 50 --seed 5000 --host localhost --port 8000

Paired protocol (mirrors examples/robocasa/rollout.py): trial i uses env seed (seed + i), so every
server/policy variant evaluated against the same seed sees identical scenes.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "third_party/robocasa-gr1-tabletop-tasks")
sys.path.insert(0, "third_party/robosuite-gr1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCanToDrawerClose")
    ap.add_argument("--robot", default="GR1ArmsAndWaistFourierHands")
    ap.add_argument("--num-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import gymnasium as gym
    from openpi_client import websocket_client_policy  # the lightweight client package
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers gr1_unified/* envs)

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    env = gym.make(f"gr1_unified/{args.task}_{args.robot}_Env", enable_render=True)

    trials = []
    for trial in range(args.num_trials):
        obs, _ = env.reset(seed=args.seed + trial)
        success, step = False, 0
        while step < args.max_steps and not success:
            element = {
                "observation/image": np.asarray(obs["video.ego_view_pad_res256_freq20"])[-1],
                "observation/state": np.asarray(obs["state"], np.float32)
                if "state" in obs
                else np.concatenate(
                    [np.asarray(v, np.float32).reshape(-1) for k, v in sorted(obs.items()) if k.startswith("state.")]
                ),
                "prompt": getattr(env.unwrapped, "language_instruction", args.task),
            }
            chunk = np.asarray(policy.infer(element)["actions"])  # [n_exec, 44]
            for action in chunk:
                obs, reward, term, trunc, info = env.step(action)
                success = success or bool(reward > 0) or bool(info.get("success", False))
                step += 1
                if success or step >= args.max_steps or term or trunc:
                    break
            if term or trunc:
                break
        trials.append({"trial": trial, "success": bool(success), "steps": step})
        print(f"trial {trial + 1}/{args.num_trials}: {'SUCCESS' if success else 'failure'} in {step}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "task": args.task,
                "seed": args.seed,
                "num_trials": args.num_trials,
                "success_rate": float(np.mean([t["success"] for t in trials])),
                "trials": trials,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
