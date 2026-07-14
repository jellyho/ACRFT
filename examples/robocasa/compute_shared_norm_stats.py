"""Compute a SHARED normalization across all RoboCasa 365 target tasks.

Why: per-task normalization is ill-conditioned for near-stationary tasks. E.g. in WashLettuce the
base barely moves and the control-mode is ~constant, so those action dims have a quantile range
(q99-q01) of ~0; the few frames that do vary then get amplified ~1e6x by normalization, which
blows up the training loss. RoboCasa is one robot with one action space across all tasks, so the
right fix is a single normalization computed over all tasks — mobile/gripper/control variation
from other tasks gives every dim a well-conditioned range.

This streams the raw ``action`` (12-d) and ``observation.state`` (16-d) of every task through
openpi's RunningStats (mean/std/q01/q99) and writes the same ``norm_stats.json`` into each per-task
config's asset dir, so no config change is needed — ``pi05_robocasa`` / ``pi05_robocasa_<Task>``
just load it as usual.

    uv run examples/robocasa/compute_shared_norm_stats.py --output-dir /data5/jellyho/robocasa365
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import openpi.shared.normalize as normalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Base config (pi05_robocasa) points at this task; per-task configs use their own <Task>.
_BASE_CONFIG_TASK = "PrepareCoffee"


def _target_asset_dirs(tasks: list[str], hf_user: str) -> list[Path]:
    """Asset dirs (assets/<config>/<repo_id>/) that should receive the shared norm_stats.json."""
    dirs = [_REPO_ROOT / "assets" / f"pi05_robocasa_{t}" / hf_user / f"robocasa365-{t}" for t in tasks]
    if _BASE_CONFIG_TASK in tasks:
        dirs.append(_REPO_ROOT / "assets" / "pi05_robocasa" / hf_user / f"robocasa365-{_BASE_CONFIG_TASK}")
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("/data5/jellyho/robocasa365"),
                    help="Dir with the converted per-task datasets.")
    ap.add_argument("--hf-user", type=str, default="jellyho", help="Owner used in the repo id / asset path.")
    ap.add_argument("--max-frames-per-task", type=int, default=None,
                    help="Optionally subsample this many frames per task (default: use all).")
    args = ap.parse_args()

    tasks = sorted(p.name for p in args.output_dir.iterdir() if (p / "meta").is_dir())
    if not tasks:
        raise SystemExit(f"No task datasets found under {args.output_dir}")
    print(f"Computing shared norm stats over {len(tasks)} tasks ...")

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    total = 0
    for i, task in enumerate(tasks, 1):
        files = sorted(glob.glob(str(args.output_dir / task / "data" / "chunk-*" / "*.parquet")))
        seen = 0
        for f in files:
            t = pq.read_table(f, columns=["action", "observation.state"]).to_pydict()
            action = np.asarray(t["action"], dtype=np.float32)
            state = np.asarray(t["observation.state"], dtype=np.float32)
            if args.max_frames_per_task is not None and seen + len(action) > args.max_frames_per_task:
                keep = args.max_frames_per_task - seen
                action, state = action[:keep], state[:keep]
            action_stats.update(action)
            state_stats.update(state)
            seen += len(action)
            if args.max_frames_per_task is not None and seen >= args.max_frames_per_task:
                break
        total += seen
        print(f"  [{i}/{len(tasks)}] {task}: {seen} frames")

    norm_stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(f"\nDone ({total} frames total).")
    print("action std   :", norm_stats["actions"].std)
    print("action q99-q01:", norm_stats["actions"].q99 - norm_stats["actions"].q01)

    targets = _target_asset_dirs(tasks, args.hf_user)
    for d in targets:
        d.mkdir(parents=True, exist_ok=True)
        normalize.save(d, norm_stats)
    print(f"\nWrote shared norm_stats.json to {len(targets)} asset dirs (per-task configs + base).")


if __name__ == "__main__":
    main()
