"""Real-robot progress for the three SELECTION-RULE runs (LEGOPROG block 3).

All three share one base policy and one critic (`fixed`); only the rule that picks which of the
N sampled chunks to execute differs. bon1 is the control: with a single candidate there is
nothing to select between, so it is the base policy's own sampling.

Progress is an integer stage count per episode (0-4), 10 episodes per run, so the run-level
statistic is a mean over 10 paired-by-nothing episodes -- the CI is wide by construction and is
drawn as such rather than hidden.
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
    else str(pathlib.Path.home() / ".claude/uploads/9b8fe2d1-78b2-49b5-88f6-b730fd35fe92/325539e2-LEGOPROG.xlsx")
)
d = pd.read_excel(XLSX, header=None)

# block 3 starts at row 25; the first episode shares the header row, so values run rows 25..34
RUNS = [
    (3, "implicit\nN=1", "no selection (control)"),
    (6, "argmax\nN=8", "BoN"),
    (0, "implicit\nN=8", "expectile lottery"),
]
data = {}
for col, label, _ in RUNS:
    vals = [d.iloc[r, col + 2] for r in range(25, 35)]
    data[label] = np.array([v for v in vals if pd.notna(v)], float)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})

labels = [lab for _, lab, _ in RUNS]
means = [data[k].mean() for k in labels]
cis = [1.96 * data[k].std(ddof=1) / np.sqrt(len(data[k])) for k in labels]
x = np.arange(len(labels))
ax.bar(x, means, yerr=cis, color=[PAL[7], PAL[0], PAL[2]], width=0.6, error_kw={"lw": 1.2, "capsize": 4})
for i, k in enumerate(labels):
    ax.scatter(
        np.full(len(data[k]), i) + np.random.default_rng(0).uniform(-0.13, 0.13, len(data[k])),
        data[k],
        s=16,
        color="0.25",
        alpha=0.55,
        zorder=3,
        linewidths=0,
    )
ax.set_xticks(x, labels)
ax.set_ylabel("mean progress (0–4)")
ax.set_title("selection rule, one critic, one base policy")
ax.set_ylim(0, 4.3)

# distribution of per-episode stages: what the mean is hiding
stages = np.arange(5)
w = 0.26
for i, (k, c) in enumerate(zip(labels, [PAL[7], PAL[0], PAL[2]], strict=True)):
    counts = [(data[k] == s).sum() for s in stages]
    ax2.bar(stages + (i - 1) * w, counts, width=w, color=c, label=k.replace("\n", " "))
ax2.set_xticks(stages)
ax2.set_xlabel("progress stage reached")
ax2.set_ylabel("episodes")
ax2.set_title("per-episode outcomes")
ax2.legend()

fig.tight_layout()
out = R / ".scratch/extraction/fig_legoprog_selection.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print("wrote", out)
for k in labels:
    v = data[k]
    print(
        f"{k.replace(chr(10), ' '):18s} n={len(v)} mean={v.mean():.2f} ± {1.96 * v.std(ddof=1) / np.sqrt(len(v)):.2f}  vals={v.astype(int).tolist()}"
    )
