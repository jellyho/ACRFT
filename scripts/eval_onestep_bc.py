"""Offline BC-quality evaluation: alpha-Flow few-step sampling vs the 10-step BC baseline (YAM).

The go/no-go gate before RL: did turning pi05 into a mean-velocity model cost action quality?
Real-robot rollouts are not available offline, so this measures the standard proxies on held-out
frames, in UNNORMALIZED robot space so numbers are comparable across checkpoints with different
norm stats (each policy unnormalizes with its own stats via create_trained_policy):

  mse_gt      ||a_pred - a_demo||^2 over the 30-step chunk (robot space, 14-d)   lower = closer to demo
  self_gap    ||a_1step - a_10step||^2 for the SAME alpha-Flow model, same noise  how much 1-step loses

Same per-frame noise across variants (variance reduction). Caveats printed with the results: the BC
baseline was trained on s300 success-only while alpha-Flow trains on s347 all-episodes, so this is a
reference comparison, not a method-only diff; and demo-MSE is a proxy, not success rate.

    uv run python scripts/eval_onestep_bc.py \
      --af-ckpt /data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/160000 \
      --bc-ckpt /data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/70000
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af-ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--bc-ckpt", type=pathlib.Path, default=None)
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--episodes", type=int, nargs="+", default=[320, 79, 23, 214, 5, 141])
    ap.add_argument("--frames-per-ep", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/eval_onestep"))
    a = ap.parse_args()

    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    import openpi.policies.policy_config as policy_config
    import openpi.training.config as train_config

    H = a.horizon

    # dataset: images at t + the raw 30-step action window in ONE item (training-style delta_timestamps)
    meta = lerobot_dataset.LeRobotDatasetMetadata(a.repo_id, root=a.root)
    fps = meta.fps

    def make_ds(episode):
        return lerobot_dataset.LeRobotDataset(
            a.repo_id,
            root=a.root,
            episodes=[episode],
            video_backend="pyav",
            delta_timestamps={"action": [t / fps for t in range(H)]},
        )

    frames = []  # list of (episode, idx_in_ep, obs_dict, gt_chunk[H,14])
    for e in a.episodes:
        ds = make_ds(e)
        n = len(ds)
        picks = np.linspace(0.05, 0.85, a.frames_per_ep) * n
        for t in picks.astype(int):
            item = ds[int(t)]
            obs = {
                "observation/image": (np.asarray(item["observation.images.agentview"]).transpose(1, 2, 0) * 255).astype(
                    np.uint8
                ),
                "observation/wrist_image": (
                    np.asarray(item["observation.images.wrist_left"]).transpose(1, 2, 0) * 255
                ).astype(np.uint8),
                "observation/image_right": (
                    np.asarray(item["observation.images.wrist_right"]).transpose(1, 2, 0) * 255
                ).astype(np.uint8),
                "observation/state": np.asarray(item["observation.state"], np.float32),
                "prompt": item.get("task", "pick up the lego and place it in the taxi"),
            }
            gt = np.asarray(item["action"], np.float32)  # [H,14]
            frames.append((e, int(t), obs, gt))
        del ds
    print(f"{len(frames)} eval frames from {len(a.episodes)} episodes", flush=True)

    rng = np.random.default_rng(0)
    noises = rng.standard_normal((len(frames), H, 32)).astype(np.float32)

    def run_policy(cfg_name, ckpt, num_steps, tag):
        cfg = train_config.get_config(cfg_name)
        pol = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={"num_steps": num_steps})
        outs = []
        for i, (_e, _t, obs, _gt) in enumerate(frames):
            res = pol.infer(dict(obs), noise=noises[i])
            outs.append(np.asarray(res["actions"], np.float32)[:H, :14])
        del pol
        import gc

        import jax

        jax.clear_caches()
        gc.collect()
        print(f"[{tag}] done", flush=True)
        return np.stack(outs)  # [N,H,14]

    results = {}
    preds = {}
    preds["af_1step"] = run_policy("pi05_yam_lego_taxi_alphaflow", a.af_ckpt, 1, "alphaflow 1-step")
    preds["af_2step"] = run_policy("pi05_yam_lego_taxi_alphaflow", a.af_ckpt, 2, "alphaflow 2-step")
    preds["af_10step"] = run_policy("pi05_yam_lego_taxi_alphaflow", a.af_ckpt, 10, "alphaflow 10-step")
    if a.bc_ckpt is not None:
        preds["bc_10step"] = run_policy("pi05_yam_lego_taxi", a.bc_ckpt, 10, "BC baseline 10-step")

    gts = np.stack([gt for (_, _, _, gt) in frames])  # [N,H,14]
    for k, p in preds.items():
        results[f"mse_gt/{k}"] = float(np.mean((p - gts) ** 2))
        # intra-chunk temporal roughness (jerk): mean ||second difference along the 30-step axis||^2.
        # demo-MSE cannot see this (per-step errors average out); the robot feels it directly.
        results[f"jerk/{k}"] = float(np.mean(np.diff(p, n=2, axis=-2) ** 2))
    results["self_gap/af_1_vs_10"] = float(np.mean((preds["af_1step"] - preds["af_10step"]) ** 2))
    results["self_gap/af_2_vs_10"] = float(np.mean((preds["af_2step"] - preds["af_10step"]) ** 2))
    # per-dimension scale reference so MSEs are interpretable
    results["gt_action_var"] = float(np.var(gts))
    results["jerk/demo"] = float(np.mean(np.diff(gts, n=2, axis=-2) ** 2))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "af_ckpt": str(a.af_ckpt),
                "bc_ckpt": str(a.bc_ckpt),
                "episodes": a.episodes,
                "frames_per_ep": a.frames_per_ep,
                "metrics": results,
            },
            indent=1,
        )
    )
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
