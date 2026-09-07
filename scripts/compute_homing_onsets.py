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


def _homing_onset(seq: np.ndarray, teleop_value: float, tol: int) -> int:
    """Start of the trailing homing run, or ``len(seq)`` when the episode has no homing tail.

    The obvious reading -- 1 + the last teleop frame -- assumes the homing run is the strict suffix
    of the episode. Every jellyho/yam_cable_tie episode breaks that assumption by exactly one frame:
    each ends [... 4. 4. 4. 4. 0.], so the last teleop frame is the LAST frame, the onset comes out
    as the episode length, and all 14,324 homing frames (5.7% of the set) stay labelled as task
    progress. Since homing_onset is the denominator of the whole cost_to_goal target, that is not a
    cosmetic error.

    So find the last run of homing frames instead and accept it as the tail when it reaches within
    `tol` frames of the end. A stray teleop frame after the arms have already gone home does not
    make the episode task behaviour again.
    """
    homing = seq != teleop_value
    L = len(seq)
    if not homing.any():
        return L
    idx = np.flatnonzero(homing)
    if idx[-1] < L - 1 - tol:  # the last homing run is interior, not a tail
        return L
    # walk back over the contiguous run that ends at idx[-1]
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return int(idx[breaks[-1] + 1] if len(breaks) else idx[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    # None = wherever LeRobot resolves --repo-id (HF_LEROBOT_HOME). A hard-coded default is how a
    # cable-tie invocation ends up reading lego frames.
    ap.add_argument("--root", default=None)
    ap.add_argument("--teleop-value", type=float, default=0.0, help="control_mode value during teleop (task)")
    ap.add_argument(
        "--tol",
        type=int,
        default=5,
        help="a homing run counts as the trailing tail if it reaches within this many frames of the end",
    )
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
        onset = _homing_onset(seq, a.teleop_value, a.tol)
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
