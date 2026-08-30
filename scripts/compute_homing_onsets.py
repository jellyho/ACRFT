"""Per-episode homing onset for the yam dataset, from the dataset's own control_mode field.

Every teleop episode is [teleop task frames][homing return-to-home frames]. The dataset marks this
directly: observation.control_mode == 0.0 during teleop, == 4.0 during the homing reset. Those trailing
homing frames are not task behaviour; for FAILURE episodes we drop them (a failure's homing = "arms back
near home, task not done", which visually collides with SUCCESS starts and would mislabel the critic).

homing_onset(e) = 1 + (last teleop frame index) = start of the trailing homing run. Writes
{episode: {"len": L, "homing_onset": k, "task_frac": k/L}} to --out (JSON). No video is decoded.

    uv run python scripts/compute_homing_onsets.py --out .scratch/yam_homing_onsets.json
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--teleop-value", type=float, default=0.0, help="control_mode value during teleop (task)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/yam_homing_onsets.json"))
    a = ap.parse_args()

    import lerobot.datasets.lerobot_dataset as lrd

    ds = lrd.LeRobotDataset(a.repo_id, root=a.root, tolerance_s=0.05)
    # low-dim fields straight from the parquet (no video decode)
    cm = np.asarray(ds.hf_dataset["observation.control_mode"], np.float32).reshape(-1)  # [N]
    epds = ds.meta.episodes
    starts = {int(e): int(f) for e, f in zip(epds["episode_index"], epds["dataset_from_index"], strict=True)}
    ends = {int(e): int(t) for e, t in zip(epds["episode_index"], epds["dataset_to_index"], strict=True)}

    out = {}
    fracs = []
    for e in sorted(starts):
        s, t = starts[e], ends[e]
        seq = cm[s:t]
        L = t - s
        # onset = 1 + last teleop frame (start of the trailing homing run); no teleop -> keep all
        teleop = np.flatnonzero(seq == a.teleop_value)
        onset = int(teleop[-1]) + 1 if len(teleop) else L
        out[str(e)] = {"len": int(L), "homing_onset": int(onset), "task_frac": round(onset / L, 3)}
        fracs.append(onset / L)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    fracs = np.array(fracs)
    print(f"wrote {len(out)} episodes -> {a.out}", flush=True)
    print(
        f"task_frac: mean {fracs.mean():.3f}  min {fracs.min():.3f}  max {fracs.max():.3f}  (1.0 = no homing detected)",
        flush=True,
    )
    print(f"episodes with >20% homing: {int((fracs < 0.8).sum())}", flush=True)


if __name__ == "__main__":
    main()
