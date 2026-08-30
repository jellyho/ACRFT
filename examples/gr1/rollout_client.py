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

# 44-d GR00T layout (meta/modality.json). The ArmsAndWaist robot exposes neither legs nor neck;
# those channels are exactly 0 throughout the demos, so we zero-fill state and drop them from actions.
STATE_SLICES = [
    ("state.left_arm", 0, 7),
    ("state.left_hand", 7, 13),
    (None, 13, 19),  # left_leg
    (None, 19, 22),  # neck
    ("state.right_arm", 22, 29),
    ("state.right_hand", 29, 35),
    (None, 35, 41),  # right_leg
    ("state.waist", 41, 44),
]
ACTION_SLICES = {
    "action.left_arm": (0, 7),
    "action.left_hand": (7, 13),
    "action.right_arm": (22, 29),
    "action.right_hand": (29, 35),
    "action.waist": (41, 44),
}


def pack_state(obs: dict) -> np.ndarray:
    state = np.zeros(44, np.float32)
    for key, a, b in STATE_SLICES:
        if key is not None:
            state[a:b] = np.asarray(obs[key], np.float32).reshape(-1)
    return state


def split_action(flat: np.ndarray) -> dict:
    return {k: np.asarray(flat[a:b], np.float32) for k, (a, b) in ACTION_SLICES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCanToDrawerClose")
    ap.add_argument("--robot", default="GR1ArmsAndWaistFourierHands")
    ap.add_argument("--prompt", default="PnPCanToDrawerClose", help="must match the training task string")
    ap.add_argument("--num-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--video-dir", type=pathlib.Path, default=None, help="save per-trial ego_view mp4s here")
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
        writer = None
        if args.video_dir is not None:
            import cv2

            args.video_dir.mkdir(parents=True, exist_ok=True)
            vpath = args.video_dir / f"{args.task}_seed{args.seed + trial}.mp4"
            writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 20, (256, 256))
        while step < args.max_steps and not success:
            element = {
                "observation/image": np.asarray(obs["video.ego_view_pad_res256_freq20"]),  # [256,256,3] uint8
                "observation/state": pack_state(obs),
                "prompt": args.prompt,
            }
            chunk = np.asarray(policy.infer(element)["actions"])  # [n_exec, 44]
            for action in chunk:
                obs, reward, term, trunc, info = env.step(split_action(action))
                success = success or bool(reward > 0) or bool(info.get("success", False))
                if writer is not None:
                    frame = np.asarray(obs["video.ego_view_pad_res256_freq20"])
                    writer.write(frame[:, :, ::-1])  # RGB → BGR
                step += 1
                if success or step >= args.max_steps or term or trunc:
                    break
            if term or trunc:
                break
        if writer is not None:
            writer.release()
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
