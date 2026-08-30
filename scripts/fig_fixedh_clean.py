"""Minimal figure for wc-r-0822-fixedh: adaptive vs every fixed execution length, per task.

Numbers are the published means from the hub entry wc-r-0822-fixedh (worker C, cube-double,
aqc_iql best cell, 3 seeds x 0.8/0.9/1.0M average, 50 eval episodes; fixed arms share the
adaptive code path with a single-element prefix_candidates). Worker C's raw run JSONs are not
in this repo, so the entry table is the source of record here -- do not edit these constants
without re-reading that entry.
"""

# ruff: noqa: ICN001  (matplotlib.use must precede pyplot; probe-local imports intentional)

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))
import plot_style

plot_style.apply()
C = plot_style.PALETTE if hasattr(plot_style, "PALETTE") else ["#4C72B0", "#DD8452", "#55A868"]

FIX_H = [1, 2, 3, 5]
DATA = {
    "task2": {"fixed": [0.456, 0.356, 0.416, 0.227], "adapt": (2.43, 0.689)},
    "task5": {"fixed": [0.780, 0.818, 0.842, 0.847], "adapt": (2.57, 0.882)},
}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for ax, (task, d) in zip(axes, DATA.items(), strict=True):
    ax.plot(FIX_H, d["fixed"], "o-", color=C[0], ms=6, lw=1.8, label="fixed $h$")
    x, y = d["adapt"]
    ax.scatter([x], [y], marker="*", s=260, color=C[1], zorder=5, label="adaptive")
    ax.axhline(y, color=C[1], lw=0.9, ls=":", alpha=0.55)
    ax.set_title(task)
    ax.set_xlabel("executed length")
    ax.set_xticks(FIX_H)
    lo = min(d["fixed"]) - 0.06
    hi = max(y, *d["fixed"]) + 0.06
    ax.set_ylim(lo, hi)
axes[0].set_ylabel("success rate")
axes[0].legend(loc="lower left")
fig.tight_layout()
out = pathlib.Path(__file__).resolve().parents[1] / ".scratch/fig_fixedh_clean"
fig.savefig(f"{out}.png", dpi=220)
fig.savefig(f"{out}.svg")
print(f"wrote {out}.png/.svg")
