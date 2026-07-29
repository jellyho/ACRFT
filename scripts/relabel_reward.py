"""Re-label an annotation's reward/return/done columns. Success becomes terminal either way.

RoboCasa's raw `next.reward` pays 1 on every frame success is *held* - 16 of them - and only then
ends the episode. So the return measures how long the simulator lingered as well as how fast the
policy arrived, and the largest reachable value moves with the slice that happened to be annotated
(14.85 under the usual hold, 20.01 for one episode where success flickered and re-fired). Cutting
each episode at its first success removes both problems. These columns come from the dataset rather
than the VLA, so re-labelling costs seconds and leaves the tokens and candidate chunks untouched.

    sparse (default)   r = 1 at the terminal when the task was achieved, 0 everywhere else.
                       V*(s) = gamma^(steps to success), so the support is exactly [0, 1].

    v3                 the reference scheme (adaptive_q_chunking/data_annoation/reward_annotate.py,
                       docs/reward_value_design.html): living -1 per step, 0 at a successful
                       terminal, -C_fail = -0.4*T_max at a failed one, gamma 0.9999, then divided by
                       Z = |min return| over the dataset so the support is exactly [-1, 0]. Denser -
                       arriving two steps sooner is worth two living units rather than a factor of
                       1.02 - at the cost of a reward the environment did not hand out.

Usage:
    uv run scripts/relabel_reward.py --data .scratch/annot_pilot [--scheme v3] [--dry-run]
"""

import argparse
import json
import pathlib

import numpy as np

import openpi.training.progress as _progress

DISCOUNT = 0.9999
C_FAIL_FRAC = 0.4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    ap.add_argument("--scheme", choices=["sparse", "v3"], default="sparse")
    ap.add_argument("--discount", type=float, default=None, help="Defaults to 0.99 sparse / 0.9999 v3.")
    ap.add_argument("--c-fail-frac", type=float, default=C_FAIL_FRAC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    T, stride = meta["num_frames"], meta.get("stride", 1)
    if stride != 1:
        raise ValueError(f"stride={stride}: the return has to be accumulated over consecutive frames.")

    from lerobot.datasets import lerobot_dataset

    dsm = lerobot_dataset.LeRobotDatasetMetadata(meta["repo_id"])
    raw = _progress.read_reward_column(pathlib.Path(dsm.root), dsm.total_frames)
    if raw is None:
        raise ValueError(f"{meta['repo_id']} carries no reward column; nothing to re-label.")
    lo = np.asarray(dsm.episodes["dataset_from_index"], dtype=np.int64)
    hi = np.asarray(dsm.episodes["dataset_to_index"], dtype=np.int64)
    g = args.discount if args.discount is not None else (0.99 if args.scheme == "sparse" else DISCOUNT)

    # The terminal is where the task was achieved, not where the simulator stopped: RoboCasa keeps
    # paying 1 while success is held, and those frames are past the decision problem.
    term = np.empty(len(lo), np.int64)
    succ = np.zeros(len(lo), bool)
    for e, (a, b) in enumerate(zip(lo, hi, strict=True)):
        fired = np.flatnonzero(raw[a:b])
        succ[e] = len(fired) > 0
        term[e] = a + int(fired[0]) if succ[e] else b - 1
    steps = term - lo + 1  # decision steps per episode, terminal included
    c_fail = args.c_fail_frac * float(steps.max())

    r_raw = np.zeros(dsm.total_frames, np.float64)
    mc_raw = np.zeros(dsm.total_frames, np.float64)
    done = np.zeros(dsm.total_frames, np.int8)
    for e, (a, t) in enumerate(zip(lo, term, strict=True)):
        done[t] = 1
        k = np.arange(t - a, -1, -1, dtype=np.float64)  # steps remaining, terminal = 0
        if args.scheme == "sparse":
            r_raw[t] = 1.0 if succ[e] else 0.0
            mc_raw[a : t + 1] = g**k if succ[e] else 0.0
        else:
            r_raw[a:t] = -1.0
            r_raw[t] = 0.0 if succ[e] else -c_fail
            mc_raw[a : t + 1] = -(1.0 - g**k) / (1.0 - g) + (0.0 if succ[e] else -(g**k) * c_fail)
        # Post-terminal frames are outside the MDP.
        r_raw[t + 1 : hi[e]] = 0.0
        mc_raw[t + 1 : hi[e]] = 0.0

    z = abs(float(mc_raw.min())) or 1.0 if args.scheme == "v3" else 1.0
    keep = np.arange(0, dsm.total_frames, stride)[:T]
    out = {
        "reward": (r_raw / z)[keep].astype(np.float32),
        "mc_return": (mc_raw / z)[keep].astype(np.float32),
        "done": done[keep].astype(bool),
    }
    alive = np.zeros(dsm.total_frames, bool)
    for a, t in zip(lo, term, strict=True):
        alive[a : t + 1] = True

    support = [-1.0, 0.0] if args.scheme == "v3" else [0.0, 1.0]
    print(f"scheme {args.scheme}   gamma {g}   value support {support}")
    print(f"{len(lo)} episodes, {int(succ.sum())} successful, {int((~succ).sum())} failed")
    print(f"steps to terminal: min {steps.min()}  p50 {int(np.median(steps))}  max {steps.max()}")
    if args.scheme == "v3":
        print(f"T_max {steps.max()} -> C_fail {c_fail:.1f}   Z {z:.1f}")
        print(f"living (normalised) {-1 / z:.3e}   failure terminal {-c_fail / z:.3f}")
    print(f"mc_return over the kept {T} frames: min {out['mc_return'].min():.4f}  max {out['mc_return'].max():.4f}")
    print(f"frames at or before their terminal: {int(alive[keep].sum())} of {T} ({100 * alive[keep].mean():.1f}%)")
    if args.dry_run:
        print("dry run: nothing written")
        return

    for name, arr in out.items():
        mm = np.memmap(args.data / f"{name}.dat", dtype=arr.dtype, mode="r+", shape=(T,))
        mm[:] = arr
        mm.flush()
    meta |= {"reward_scheme": args.scheme, "discount": g, "terminal_success": True, "value_support": support}
    if args.scheme == "v3":
        meta |= {"c_fail": c_fail, "z": z}
    (args.data / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"rewrote reward/mc_return/done in {args.data}; value support is {support}")


if __name__ == "__main__":
    main()
