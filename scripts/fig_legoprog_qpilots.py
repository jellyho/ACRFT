"""QPILOTS-U steering strength on the real robot (LEGOPROG block 4).

One critic, one base policy; only alpha changes. alpha = 0 is the base sampler exactly (the
steering term vanishes), so it is the control rather than a separate condition -- which is what
makes this a dose-response curve instead of a set of arms.

Note the range: the paper sweeps {0.1, 0.2, 0.3} on pi0.5-LIBERO, so every point here except the
largest sits BELOW its smallest tested value.
"""

# ruff: noqa: E402, ICN001

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "slurm"))
import plot_style

plot_style.apply()
PAL = plot_style.PALETTE

XLSX = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(pathlib.Path.home() / ".claude/uploads/9b8fe2d1-78b2-49b5-88f6-b730fd35fe92/c0b2be4f-LEGOPROG.xlsx")
)
d = pd.read_excel(XLSX, header=None)

COLS = {0: 0.1, 3: 0.05, 6: 0.025, 9: 0.01, 12: 0.005, 15: 0.0}
runs = {}
for col, alpha in COLS.items():
    vals = [d.iloc[r, col + 2] for r in range(37, 47)]
    runs[alpha] = np.array([v for v in vals if pd.notna(v)], float)

alphas = sorted(runs)
means = np.array([runs[a].mean() for a in alphas])
cis = np.array([1.96 * runs[a].std(ddof=1) / np.sqrt(len(runs[a])) for a in alphas])

fig, ax = plt.subplots(figsize=(6.0, 3.8))

# alpha = 0 belongs ON the curve -- it is the same sampler with the steering term at zero, so the
# dose-response starts there. Linear axis at the true spacing: alpha IS a continuous quantity and
# the crowding below 0.05 is a fact about where the sweep sampled, not something to even out.
base_m, base_c = means[0], cis[0]
ax.axhspan(base_m - base_c, base_m + base_c, color=PAL[7], alpha=0.15, zorder=0)
ax.axhline(base_m, color=PAL[7], lw=1.2, ls="--", zorder=1)

ax.errorbar(
    alphas,
    means,
    yerr=cis,
    marker="o",
    ms=6,
    lw=1.6,
    color=PAL[3],
    capsize=4,
    zorder=4,
    label="QPILOTS-U",
)
ax.scatter(
    [0.0], [base_m], s=90, facecolor="white", edgecolor=PAL[7], linewidth=2.0, zorder=5, label="α = 0 (no steering)"
)
for a in alphas:
    ax.scatter(np.full(len(runs[a]), a), runs[a], s=14, color="0.35", alpha=0.45, zorder=3, linewidths=0)
ax.set_xlabel("steering strength α")
ax.set_ylabel("mean progress (0-4)")
ax.set_title("steering hurts at every strength tried")
ax.set_ylim(-0.15, 4.2)
ax.legend(loc="upper right")

fig.tight_layout()
out = R / ".scratch/extraction/fig_legoprog_qpilots.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print("wrote", out)
for a in alphas:
    v = runs[a]
    print(
        f"alpha={a:<6g} n={len(v)} mean={v.mean():.2f} ± {1.96 * v.std(ddof=1) / np.sqrt(len(v)):.2f}  {v.astype(int).tolist()}"
    )
