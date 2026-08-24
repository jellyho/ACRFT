"""Turn recorded rollouts into a LeRobot v3.0 dataset for the fine-tuning ladder.

The rollout client (`examples/robocasa/main.py --traj-dir`) records EVERY episode, successes and
failures alike, as one directory holding three camera mp4s and a `traj.npz` of states, actions,
decision boundaries and the outcome. The filter belongs here, not at collection time, so one GPU
campaign feeds every rung of the ladder:

    B1  --filter success            imitate only the episodes that succeeded
    B2  --filter weighted           keep all, attach a per-episode weight (written to the dataset)
    B0  --filter all                the unfiltered control, to separate "more data" from "better data"

The schema matches the converted demonstrations (`convert_pretrain_tars.py` output) so a run can
train on demos alone, rollouts alone, or their concatenation without a second code path.

    uv run examples/robocasa/convert_rollouts.py \
        --collect-root /scratch/jellyho/acrft/collect --task PickPlaceSinkToCounter \
        --filter success --repo-id jellyho/robocasa_b1_PickPlaceSinkToCounter \
        --root /scratch/jellyho/acrft/rollout_v3
"""

import argparse
import json
import pathlib

import numpy as np

# rollout camera key -> demonstration feature name (the demos are the schema of record)
CAMERA_MAP = {
    "image": "observation.images.robot0_agentview_left",
    "wrist_image": "observation.images.robot0_eye_in_hand",
    "image_right": "observation.images.robot0_agentview_right",
}


def features_like_demos(state_dim: int, action_dim: int) -> dict:
    feats = {
        name: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
        for name in CAMERA_MAP.values()
    }
    # float64 matches the converted demonstrations, so a run can concatenate the two datasets and
    # share one set of normalization statistics
    feats["observation.state"] = {"dtype": "float64", "shape": (state_dim,), "names": None}
    feats["action"] = {"dtype": "float64", "shape": (action_dim,), "names": None}
    feats["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    feats["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
    # provenance carried per frame: which decision this step belongs to, and the episode's outcome
    # and weight. Downstream filtering/weighting reads these instead of re-deriving them.
    feats["rollout.decision_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    feats["rollout.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    feats["rollout.weight"] = {"dtype": "float32", "shape": (1,), "names": None}
    return feats


def episode_dirs(root: pathlib.Path, task: str):
    d = root / task / "traj"
    return sorted(p for p in d.glob("*") if p.is_dir() and (p / "traj.npz").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-root", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/collect"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--filter", choices=["success", "all", "weighted"], default="success")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=pathlib.Path, required=True, help="local dataset root (no hub push)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-episodes", type=int, default=None)
    a = ap.parse_args()

    import imageio.v3 as iio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = episode_dirs(a.collect_root, a.task)
    if not eps:
        raise SystemExit(f"no recorded episodes under {a.collect_root / a.task / 'traj'}")

    meta = []
    for p in eps:
        z = np.load(p / "traj.npz", allow_pickle=True)
        meta.append({"dir": p, "success": bool(z["success"]), "steps": int(z["steps"]), "prompt": str(z["prompt"])})
    n_succ = sum(m["success"] for m in meta)
    print(f"{a.task}: {len(meta)} episodes recorded, {n_succ} successful ({n_succ / len(meta):.2f})", flush=True)

    keep = [m for m in meta if m["success"]] if a.filter == "success" else meta
    if a.max_episodes:
        keep = keep[: a.max_episodes]
    if not keep:
        raise SystemExit("filter kept no episodes")

    z0 = np.load(keep[0]["dir"] / "traj.npz", allow_pickle=True)
    ds = LeRobotDataset.create(
        repo_id=a.repo_id,
        fps=a.fps,
        features=features_like_demos(z0["state"].shape[1], z0["action"].shape[1]),
        root=a.root / a.repo_id,
        robot_type="panda_mobile",
        use_videos=True,
    )

    written = 0
    for m in keep:
        z = np.load(m["dir"] / "traj.npz", allow_pickle=True)
        state, action = z["state"], z["action"]
        decisions = {int(x) for x in z["decision_steps"]}
        vids = {k: iio.imread(m["dir"] / f"{k}.mp4") for k in CAMERA_MAP}
        n = min(len(state), *(len(v) for v in vids.values()))
        # a rollout carries one outcome; the per-episode weight is 1 for successes and, under
        # --filter weighted, 0 for failures. B2 replaces this with an advantage at training time.
        w = 1.0 if m["success"] else (0.0 if a.filter == "weighted" else 1.0)
        dec_idx = -1
        for t in range(n):
            if t in decisions:
                dec_idx += 1
            frame = {name: vids[k][t] for k, name in CAMERA_MAP.items()}
            frame["observation.state"] = state[t].astype(np.float64)
            frame["action"] = action[t].astype(np.float64)
            # sparse outcome reward on the final frame of a successful episode
            last = t == n - 1
            frame["next.reward"] = np.array([1.0 if (last and m["success"]) else 0.0], np.float32)
            frame["next.done"] = np.array([last], bool)
            frame["rollout.decision_index"] = np.array([dec_idx], np.int64)
            frame["rollout.success"] = np.array([m["success"]], bool)
            frame["rollout.weight"] = np.array([w], np.float32)
            frame["task"] = m["prompt"]
            ds.add_frame(frame)
        ds.save_episode()
        written += 1
        if written % 10 == 0:
            print(f"  {written}/{len(keep)} episodes written", flush=True)

    ds.finalize()
    summary = {
        "task": a.task,
        "filter": a.filter,
        "episodes_recorded": len(meta),
        "episodes_successful": n_succ,
        "episodes_written": written,
        "frames_written": ds.num_frames,
        "root": str(a.root / a.repo_id),
    }
    (a.root / a.repo_id / "conversion_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
