"""Collect pi05 rollouts in the RoboCasa sim for training the standalone patch-critic.

Deploys the (official) pi05 checkpoint through the SAME serving path that scored 44% on OpenDrawer
(``create_trained_policy`` -> RoboCasa transforms, mean/std norm, HDF5-order action output), and
logs one record PER ENV STEP -- the canonical (s, a, r, s') transition dataset:

    images   (N, 3, 224, 224, 3) uint8   3 cameras (agentview_left, eye_in_hand, agentview_right)
    state    (N, 16)             float32  proprio (LeRobot order)
    action   (N, 12)             float32  the SINGLE action executed at this step (env/HDF5 order)
    reward   (N,)                float32  sparse: 1.0 on the success step, else 0
    done     (N,)                int8     1 on the last step of an episode (success or horizon)
    episode_index (N,)           int32

The critic's action CHUNK is derived at train time as action[t : t+H] within the same episode; the
per-prefix (adaptive-K) and MC/TD targets come from the dense per-step reward/done. Nothing chunk-
level is stored (that was redundant). Candidates are NOT stored -- they are resampled from the pi05
actor at deploy/eval time, which is exactly how BoN / adaptive-K selection works.

Uses env-action-order (the official checkpoint already outputs HDF5 order, so no LeRobot->env remap).

    uv run --group eval examples/robocasa/collect_rollouts_patchcritic.py \
        --task PickPlaceCounterToMicrowave --num-episodes 60 --max-steps 500 \
        --out /data5/jellyho/pc_rollouts/PickPlaceCounterToMicrowave
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="pi05_robocasa_pretrained")
    ap.add_argument(
        "--ckpt",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pi05_robocasa_pretrained/human300_pretrain/75000"),
    )
    ap.add_argument("--num-episodes", type=int, default=60)
    ap.add_argument(
        "--max-steps", type=int, default=500, help="cap per episode (env horizon is 1000; failures truncate)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224, help="stored image size (DINOv2 native 224)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    import sys

    from PIL import Image

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import rollout as R  # noqa: N812

    cfg = _config.get_config(a.config)
    policy = _policy_config.create_trained_policy(cfg, a.ckpt)
    env = R.make_env(a.task, camera_size=256, seed=a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    def resize(img):
        return np.asarray(Image.fromarray(img).resize((a.img_size, a.img_size), Image.BILINEAR))

    def imgs_of(obs):
        return np.stack(
            [
                resize(R.image_from_obs(obs, R.CAMERAS["observation/image"])),
                resize(R.image_from_obs(obs, R.CAMERAS["observation/wrist_image"])),
                resize(R.image_from_obs(obs, R.CAMERAS["observation/image_right"])),
            ]
        ).astype(np.uint8)  # [3, S, S, 3]

    IMG, STATE, ACT, REW, DONE, EPI = [], [], [], [], [], []
    n_success = 0
    for ep in range(a.num_episodes):
        env.rng = np.random.default_rng(a.seed + ep)
        np.random.seed(a.seed + ep)
        obs = env.reset()
        prompt = env.get_ep_meta().get("lang", a.task)
        step, success = 0, False
        while step < a.max_steps and not success:
            chunk = np.asarray(policy.infer(R.obs_to_element(obs, prompt))["actions"], np.float32)  # [H,12]
            for act in chunk:  # execute the full chunk, one env step per action (env-action-order)
                a12 = np.asarray(act, np.float32)[:12]
                IMG.append(imgs_of(obs))  # state BEFORE the action
                STATE.append(R.state_from_obs(obs).astype(np.float32))
                ACT.append(a12)
                EPI.append(ep)
                obs, _, _, _ = env.step(a12)
                step += 1
                success = bool(env._check_success())
                REW.append(1.0 if success else 0.0)
                DONE.append(1 if (success or step >= a.max_steps) else 0)
                if success or step >= a.max_steps:
                    break
        n_success += int(success)
        print(f"[ep {ep + 1}/{a.num_episodes}] {'SUCCESS' if success else 'fail'} ({step} steps)", flush=True)

    n = len(REW)
    save = {
        "images": (np.stack(IMG), np.uint8),
        "state": (np.stack(STATE), np.float32),
        "action": (np.stack(ACT), np.float32),
        "reward": (np.asarray(REW, np.float32), np.float32),
        "done": (np.asarray(DONE, np.int8), np.int8),
        "episode_index": (np.asarray(EPI, np.int32), np.int32),
    }
    for name, (arr, dt) in save.items():
        mm = np.memmap(a.out / f"{name}.dat", dtype=dt, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
    meta = {
        "num_steps": n,
        "num_episodes": a.num_episodes,
        "num_success": n_success,
        "action_dim": 12,
        "state_dim": 16,
        "img_size": a.img_size,
        "max_steps": a.max_steps,
        "task": a.task,
        "shapes": {k: list(v[0].shape) for k, v in save.items()},
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {n} steps ({n_success}/{a.num_episodes} success) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
