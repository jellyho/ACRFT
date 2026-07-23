"""RoboCasa 365 evaluation client for openpi policies.

Runs a RoboCasa (robosuite / MuJoCo) simulation, queries an openpi policy server over a
websocket, and rolls out the returned action chunks — the RoboCasa analogue of
``examples/libero/main.py``.

Two processes, two environments:

  1. Policy server (openpi training venv, LeRobot v3.0 / numpy>=2):
        uv run scripts/serve_policy.py policy:checkpoint \
            --policy.config pi05_robocasa --policy.dir /path/to/checkpoint

  2. This client (a RoboCasa venv with robosuite + robocasa + openpi-client; robosuite pins
     numpy<2, so keep it separate from the training venv):
        python examples/robocasa/main.py --task PickPlaceCounterToCabinet --host <server-host>

The server applies the RoboCasa data transforms (see robocasa_policy.py): it receives three
256x256 images, a 16-d state, and a prompt, and returns a chunk of 12-d actions.
"""

import argparse
import pathlib

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.env_utils import create_env

# obs keys that make up the 16-d ``observation.state`` (in LeRobot order), from
# robocasa's LEROBOT_STATE_TO_HDF5_STATE mapping.
_STATE_KEYS = (
    "robot0_base_pos",  # 3  base position
    "robot0_base_quat",  # 4  base rotation
    "robot0_base_to_eef_pos",  # 3  EE position (relative)
    "robot0_base_to_eef_quat",  # 4  EE rotation (relative)
    "robot0_gripper_qpos",  # 2  gripper qpos
)

_CAMERAS = {
    "observation/image": "robot0_agentview_left",
    "observation/wrist_image": "robot0_eye_in_hand",
    "observation/image_right": "robot0_agentview_right",
}


def _state(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in _STATE_KEYS])


def _image(obs: dict, camera: str) -> np.ndarray:
    # Raw robosuite camera obs are stored bottom-origin; RoboCasa flips them vertically to get
    # the upright frames the datasets (and thus the policy) were trained on. Match that here.
    return np.ascontiguousarray(obs[f"{camera}_image"][::-1])


def _lerobot_action_to_env(action: np.ndarray) -> np.ndarray:
    """Reorder a 12-d model action (LeRobot order) into the env's expected flat order.

    LeRobot order : [base_motion(0:4), control_mode(4:5), ee_pos(5:8), ee_rot(8:11), gripper(11:12)]
    Env/HDF5 order: [ee_pos(0:3), ee_rot(3:6), gripper(6:7), base_motion(7:11), control_mode(11:12)]
    """
    return np.concatenate([action[5:8], action[8:11], action[11:12], action[0:4], action[4:5]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="RoboCasa env/task name, e.g. PickPlaceCounterToCabinet.")
    ap.add_argument("--host", default="localhost", help="Policy server host.")
    ap.add_argument("--port", type=int, default=8000, help="Policy server port.")
    ap.add_argument("--num-trials", type=int, default=10, help="Number of rollouts.")
    ap.add_argument(
        "--max-steps", type=int, default=None, help="Max env steps per rollout. Defaults to the task horizon."
    )
    ap.add_argument(
        "--replan-steps",
        type=int,
        default=10,
        help="How many actions of each returned chunk to execute before re-querying.",
    )
    ap.add_argument("--camera-size", type=int, default=256, help="Camera height/width for the env.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--video-dir", type=pathlib.Path, default=None, help="If set, save mp4 rollouts here (needs imageio)."
    )
    ap.add_argument(
        "--num-videos",
        type=int,
        default=5,
        help="Max number of rollout videos to save when --video-dir is set (0 = none).",
    )
    ap.add_argument(
        "--action-dist-samples",
        type=int,
        default=0,
        help="On saved videos, overlay the policy's action distribution: at each replan, "
        "sample this many extra action chunks and draw their predicted EE paths "
        "(translucent) plus the executed one (bright). 0 = off. Adds N inferences "
        "per replan on video trials only.",
    )
    ap.add_argument(
        "--action-dist-scale",
        type=float,
        default=0.25,
        help="Target world length (metres) of the executed chunk's drawn path in the "
        "action-distribution overlay; the draw scale is derived from it and shared "
        "by all candidates, so their relative spread stays faithful. Tune for size.",
    )
    ap.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=None,
        help="If set, write a JSON summary (success rate + per-trial results) here.",
    )
    args = ap.parse_args()

    env = create_env(
        env_name=args.task,
        robots="PandaOmron",
        split="target",  # heldout target layouts/objects, matching the target/human demos
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=args.camera_size,
        camera_heights=args.camera_size,
        seed=args.seed,
    )
    max_steps = args.max_steps or int(getattr(env, "horizon", 500))
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    trials = []
    for trial in range(args.num_trials):
        # Re-seed the env's RNG before every reset with a deterministic per-trial seed. RoboCasa
        # samples the layout/style/objects/textures from env.rng on reset, so this makes trial i
        # the exact same scene for every checkpoint (fair comparison) regardless of how the
        # rollout consumed the RNG.
        env.rng = np.random.default_rng(args.seed + trial)
        obs = env.reset()
        prompt = env.get_ep_meta().get("lang", args.task)
        record = args.video_dir is not None and trial < args.num_videos
        print(f"[trial {trial + 1}/{args.num_trials}] prompt: {prompt!r}")

        overlay_on = record and args.action_dist_samples > 0
        projector = None
        if overlay_on:
            import action_overlay as _ov

            projector = _ov.CameraProjector(env.sim, "robot0_agentview_left", args.camera_size, args.camera_size)

        frames = []
        success = False
        step = 0
        while step < max_steps and not success:
            element = {
                "observation/state": _state(obs),
                "observation/image": _image(obs, _CAMERAS["observation/image"]),
                "observation/wrist_image": _image(obs, _CAMERAS["observation/wrist_image"]),
                "observation/image_right": _image(obs, _CAMERAS["observation/image_right"]),
                "prompt": prompt,
            }
            action_chunk = np.asarray(client.infer(element)["actions"])

            # Action-distribution overlay: sample extra chunks (the distribution at THIS replan) and,
            # on every frame of the chunk, draw their predicted EE paths anchored at the LIVE EE so
            # the spray stays attached to the moving gripper.
            candidates = None
            if overlay_on:
                extras = [np.asarray(client.infer(element)["actions"]) for _ in range(args.action_dist_samples)]
                candidates = [action_chunk, *extras]  # executed chunk is index 0

            for action in action_chunk[: args.replan_steps]:
                obs, _, _, _ = env.step(_lerobot_action_to_env(np.asarray(action)))
                step += 1
                if record:
                    frame = _image(obs, _CAMERAS["observation/image"])
                    if overlay_on:
                        frame = _ov.draw_overlay(
                            frame,
                            projector,
                            np.asarray(obs["robot0_eef_pos"]),
                            np.asarray(obs["robot0_base_quat"]),
                            candidates,
                            executed_idx=0,
                            target_len=args.action_dist_scale,
                        )
                    frames.append(frame)
                if env._check_success():
                    success = True
                    break
                if step >= max_steps:
                    break

        successes += success
        trials.append({"trial": trial, "success": bool(success), "steps": step})
        print(f"[trial {trial + 1}] {'SUCCESS' if success else 'failure'} in {step} steps")

        if record and frames:
            import imageio

            out = args.video_dir / f"{args.task}_trial{trial:02d}_{'succ' if success else 'fail'}.mp4"
            imageio.mimwrite(out, frames, fps=20)
            print(f"  saved {out}")

    rate = successes / args.num_trials
    print(f"\n{args.task}: success rate {successes}/{args.num_trials} = {rate:.1%}")

    if args.output_json is not None:
        import json

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "task": args.task,
                    "num_trials": args.num_trials,
                    "successes": successes,
                    "success_rate": rate,
                    "trials": trials,
                },
                indent=2,
            )
        )
        print(f"  wrote {args.output_json}")


if __name__ == "__main__":
    main()
