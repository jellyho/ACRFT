# ruff: noqa
"""Run-level (6-seed) closed-loop comparison: vla / rand / td-max / DEAS, HL-Gauss, PrepareCoffee.

The honest high-power picture: run-level mean +/- 95% t-CI bars with the per-seed points overlaid, so
the spread that made n=25 single-seed conclusions unreliable is visible. All three critic arms sit at
VLA parity (Δ̄ ~ 0, CI includes 0); only rand dips. Regenerates from compare_tdmax_deas.json.

Outputs: plots/31_runlevel_cmp.png
"""

import json, os, sys
import matplotlib, numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "slurm")
from plot_style import PALETTE, apply

apply()
C = os.environ["CACHE_DIR"]
r = json.load(open(f"{C}/gr1_eval/compare_tdmax_deas.json"))
order = ["vla", "rand", "tdmax", "deas"]
labs = {"vla": "VLA\n(baseline)", "rand": "rand\n(null)", "tdmax": "td-max\n(ours)", "deas": "DEAS\n(expectile-V)"}
cols = {"vla": "#1a1a1a", "rand": PALETTE[7], "tdmax": PALETTE[0], "deas": PALETTE[2]}
xs = np.arange(len(order))
means = [r[m]["run_level_mean"] for m in order]
cis = [r[m]["run_level_t95"] for m in order]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.bar(
    xs,
    means,
    color=[cols[m] for m in order],
    width=0.62,
    yerr=cis,
    error_kw={"ecolor": "#333", "elinewidth": 1.4, "capsize": 5},
    zorder=2,
)
ax.axhline(r["vla"]["run_level_mean"], color="#1a1a1a", lw=1.0, ls="--", zorder=1)
rng = np.random.default_rng(0)
for x, m in zip(xs, order):
    pts = r[m]["per_seed"]
    jit = (rng.random(len(pts)) - 0.5) * 0.22
    ax.scatter(x + jit, pts, s=22, color="#222", alpha=0.5, zorder=3, edgecolor="none")
    ax.text(
        x, means[order.index(m)] + cis[order.index(m)] + 0.03, f"{means[order.index(m)]:.2f}", ha="center", fontsize=10
    )
ax.set_xticks(xs)
ax.set_xticklabels([labs[m] for m in order])
ax.set_ylim(0, 1.0)
ax.set_ylabel("closed-loop success rate")
sd = r["meta"]["seeds"]
tps = r["meta"]["trials_per_seed"]
ax.set_title(
    f"run-level ({len(sd)} seeds x {tps} = {len(sd)*tps}/arm, 95% t-CI)  PrepareCoffee, HL-Gauss, N={r['meta']['N']}"
)
fig.tight_layout()
os.makedirs(f"{C}/plots", exist_ok=True)
fig.savefig(f"{C}/plots/31_runlevel_cmp.png", dpi=150)
print("CMP_PLOT_DONE")
