"""Repair GR1 quantile norm stats whose q-span collapsed on near-constant dims.

Root cause (2026-08-10): the GR1 hands' last joints (flat-44 dims 12/34) sit parked at 3.0 for
>99% of demo frames, so q01=q99=2.9994 while the joint actually swings 0..3 when grasping. The
quantile normalizer then maps a grasp to (x-q01)/1e-6 ~ 1e6 — those dims dominate the flow loss
and the finetuned policy never learns the real action dims (measured: 20k pilot worse than
hold-still open-loop, 0/25 rollouts).

Rule: for any dim with (q99-q01) < --min-span, widen [q01, q99] to the dataset's true [min, max]
from the original GR00T stats.json; if min==max (legs/neck, constant 0) the dim is left alone —
a constant maps to a constant and is harmless. Mean/std are left untouched (quantile path only).

Usage:
    uv run python slurm/repair_gr1_norm_stats.py \
        --norm-stats assets/pi05_gr1_rlt/gr1_unified.PnPCanToDrawerClose/norm_stats.json \
        --raw-stats /scratch/.../LeRobot/gr1_unified.PnPCanToDrawerClose/meta/stats.json
"""

import argparse
import json
import pathlib

import numpy as np


def repair(norm_stats_path: pathlib.Path, raw_stats_path: pathlib.Path, min_span: float) -> list[str]:
    doc = json.loads(norm_stats_path.read_text())
    raw = json.loads(raw_stats_path.read_text())
    raw_key = {"actions": "action", "state": "observation.state"}
    log = []
    for key, stats in doc["norm_stats"].items():
        rk = raw_key.get(key)
        if rk is None or rk not in raw:
            continue
        q01, q99 = np.asarray(stats["q01"], np.float64), np.asarray(stats["q99"], np.float64)
        lo, hi = np.asarray(raw[rk]["min"], np.float64), np.asarray(raw[rk]["max"], np.float64)
        narrow = (q99 - q01 < min_span) & (hi - lo >= min_span)
        for d in np.where(narrow)[0]:
            log.append(f"{key}[{d}]: q[{q01[d]:.4f},{q99[d]:.4f}] -> [{lo[d]:.4f},{hi[d]:.4f}]")
            q01[d], q99[d] = lo[d], hi[d]
        stats["q01"], stats["q99"] = q01.tolist(), q99.tolist()
    backup = norm_stats_path.with_suffix(".json.pre_repair")
    if not backup.exists():
        backup.write_text(norm_stats_path.read_text())
    norm_stats_path.write_text(json.dumps(doc, indent=2))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-stats", type=pathlib.Path, required=True)
    ap.add_argument("--raw-stats", type=pathlib.Path, required=True)
    ap.add_argument("--min-span", type=float, default=0.1)
    args = ap.parse_args()
    log = repair(args.norm_stats, args.raw_stats, args.min_span)
    print("\n".join(log) if log else "no dims repaired")
    print(f"repaired {len(log)} dims -> {args.norm_stats}")


if __name__ == "__main__":
    main()
