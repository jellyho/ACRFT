"""Collect the M4 fixed-k sweep into one table + figure (regenerable; no hand-copied numbers).

    python slurm/probes/ksweep_collect.py --root /scratch/jellyho/acrft/gr1_eval/ksweep \
        --fig /scratch/jellyho/acrft/hub_figs/ksweep.png --json slurm/probes/ksweep_results.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "slurm")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PALETTE as COLORS
from plot_style import apply as apply_style

KS = [1, 2, 4, 8, 12, 16]
TASKS = ["OpenDrawer", "CoffeeServeMug", "TurnOnStove", "PickPlaceSinkToCounter", "PickPlaceCounterToMicrowave"]


def load(root):
    out = {}
    for t in TASKS:
        for k in KS:
            f = root / f"k{k}" / t / "results.json"
            if f.exists():
                r = json.loads(f.read_text())
                out[(t, k)] = (r["successes"], r["num_trials"], r["success_rate"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/gr1_eval/ksweep"))
    ap.add_argument("--fig", type=pathlib.Path, default=None)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()
    d = load(a.root)

    rows = {}
    for t in TASKS:
        sr = {k: d[(t, k)][2] for k in KS if (t, k) in d}
        if not sr:
            continue
        best = max(sr, key=lambda k: sr[k])
        # non-monotone if the best is an interior point of the observed grid
        ks_seen = sorted(sr)
        interior = best not in (ks_seen[0], ks_seen[-1])
        rows[t] = {"sr": sr, "best_k": best, "best_sr": sr[best], "interior_peak": interior, "n": d[(t, ks_seen[0])][1]}
    # how much a merely BETTER CONSTANT buys over the default full chunk (k=H): the baseline that
    # any adaptive method must beat, not k=16 (worker C's lesson, quantified here)
    gains = {t: r["best_sr"] - r["sr"][16] for t, r in rows.items() if 16 in r["sr"]}
    interior = sum(r["interior_peak"] for r in rows.values())
    k1_worst = sum(1 for r in rows.values() if 1 in r["sr"] and r["sr"][1] == min(r["sr"].values()))
    payload = {
        "per_task": rows,
        "ks": KS,
        "n_trials": 20,
        "seed": 3000,
        "best_minus_full_chunk": gains,
        "mean_best_minus_full_chunk": (sum(gains.values()) / len(gains)) if gains else None,
        "interior_peaks": [interior, len(rows)],
        "k1_is_worst": [k1_worst, len(rows)],
        "distinct_best_k": sorted({r["best_k"] for r in rows.values()}),
    }
    if a.json:
        a.json.write_text(json.dumps(payload, indent=1))
    for t, r in rows.items():
        print(
            f"{t:<30} best k={r['best_k']:>2} at {r['best_sr']:.2f}  interior={r['interior_peak']}  "
            + " ".join(f"k{k}={v:.2f}" for k, v in sorted(r["sr"].items()))
        )

    if a.fig:
        apply_style()
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for i, (t, r) in enumerate(rows.items()):
            ks = sorted(r["sr"])
            sr = [r["sr"][k] for k in ks]
            # binomial standard error at n trials (selection-grade, single seed)
            se = [np.sqrt(max(s * (1 - s), 1e-9) / r["n"]) for s in sr]
            ax.errorbar(ks, sr, yerr=se, marker="o", ms=4, lw=1.3, color=COLORS[i % len(COLORS)], label=t, capsize=2)
            ax.plot([r["best_k"]], [r["best_sr"]], marker="*", ms=13, color=COLORS[i % len(COLORS)])
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("fixed execution length k (chunk H = 16)")
        ax.set_ylabel("success rate")
        ax.set_title("fixed-k sweep")
        ax.legend(fontsize=6.5, title="★ = best k for that task", title_fontsize=6.5)
        fig.tight_layout()
        a.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=180, bbox_inches="tight")
        print("wrote", a.fig)


if __name__ == "__main__":
    main()
