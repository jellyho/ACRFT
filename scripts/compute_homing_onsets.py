"""Per-episode homing onset for the yam dataset (where the arms start returning to home).

Every teleop episode starts near a common home pose, goes out to do the task (proprio far from home),
then RETURNS to home at the end (the operator's reset motion). Those trailing "homing" frames are not
task behaviour. For FAILURE episodes we want to drop them (a failure's homing = "arms back near home,
lego not placed", which visually collides with SUCCESS starts and would mislabel the critic).

homing_onset(e) = 1 + (last frame index, within the episode, whose proprio is still FAR from home,
i.e. dist_to_home > tau). Frames at/after that index are the homing return and get truncated. Writes
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
    ap.add_argument(
        "--tau", type=float, default=3.0, help="dist-to-home threshold separating task (~20) from home (~0.7)"
    )
    ap.add_argument("--margin", type=int, default=0, help="keep this many frames past the last task frame")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/yam_homing_onsets.json"))
    a = ap.parse_args()

    import lerobot.datasets.lerobot_dataset as lrd

    ds = lrd.LeRobotDataset(a.repo_id, root=a.root, tolerance_s=0.05)
    # low-dim state + episode boundaries straight from the parquet (no video decode)
    state = np.asarray(ds.hf_dataset["observation.state"], np.float32)  # [N, state_dim]
    epds = ds.meta.episodes
    starts = {int(e): int(f) for e, f in zip(epds["episode_index"], epds["dataset_from_index"], strict=True)}
    ends = {int(e): int(t) for e, t in zip(epds["episode_index"], epds["dataset_to_index"], strict=True)}

    # home pose ~= mean of every episode's FIRST frame
    home = np.mean([state[starts[e]] for e in starts], 0)

    out = {}
    fracs = []
    for e in sorted(starts):
        s, t = starts[e], ends[e]
        seq = state[s:t]
        L = t - s
        dist = np.linalg.norm(seq - home[None], axis=1)  # [L]
        far = np.flatnonzero(dist > a.tau)  # task frames (away from home)
        # never leaves home (degenerate) -> keep all; else cut just past the last task frame
        onset = L if len(far) == 0 else min(L, int(far[-1]) + 1 + a.margin)
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
