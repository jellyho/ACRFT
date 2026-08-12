# ruff: noqa
"""Closed-loop success bars for the per-prefix TD-max critic comparison (joint candidate x prefix BoN).

Six modes on PrepareCoffee (N=8, n=25), Wilson 95% CIs: vla baseline, rand null, randh joint-null, and
scalar / HL-Gauss / floq BoN. The story: with the bootstrap fixed to TD-max over candidates and
selection to a joint (candidate, prefix) arg-max, STILL no critic beats the VLA — rand (pure resample)
tops the critics, the winner's curse persists (coverage unsolved). Regenerates from bon_pfx_compare.json.

Outputs: plots/30_pfx_bon.png
"""

import json, os, sys
import matplotlib, numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "slurm")
from plot_style import PALETTE, apply

apply()
C = os.environ["CACHE_DIR"]
r = json.load(open(f"{C}/gr1_eval/bon_pfx_compare.json"))
order = ["vla", "rand", "randh", "scalar", "hlgauss", "floq"]
labs = {
    "vla": "VLA\n(baseline)",
    "rand": "rand\n(null)",
    "randh": "randh\n(joint null)",
    "scalar": "scalar\nBoN",
    "hlgauss": "HL-Gauss\nBoN",
    "floq": "floq\nBoN",
}
cols = {
    "vla": "#1a1a1a",
    "rand": PALETTE[7],
    "randh": "#b0b0b0",
    "scalar": PALETTE[7],
    "hlgauss": PALETTE[2],
    "floq": PALETTE[3],
}
sr = [r[m]["success_rate"] for m in order]
lo = [r[m]["success_rate"] - r[m]["wilson95"][0] for m in order]
hi = [r[m]["wilson95"][1] - r[m]["success_rate"] for m in order]
xs = np.arange(len(order))
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.bar(
    xs,
    sr,
    color=[cols[m] for m in order],
    width=0.66,
    yerr=[lo, hi],
    error_kw={"ecolor": "#333", "elinewidth": 1.3, "capsize": 4},
)
ax.axhline(r["vla"]["success_rate"], color="#1a1a1a", lw=1.0, ls="--", zorder=0)
for x, m in zip(xs, order):
    ax.text(
        x,
        r[m]["success_rate"] + hi[order.index(m)] + 0.03,
        f"{r[m]['successes']}/{r[m]['num_trials']}",
        ha="center",
        fontsize=9,
    )
ax.set_xticks(xs)
ax.set_xticklabels([labs[m] for m in order])
ax.set_ylim(0, 1.0)
ax.set_ylabel("closed-loop success rate")
ax.set_title(
    f"per-prefix TD-max, joint (candidate x prefix) BoN  (PrepareCoffee, N={r['meta']['N']}, n={r['vla']['num_trials']})"
)
fig.tight_layout()
os.makedirs(f"{C}/plots", exist_ok=True)
fig.savefig(f"{C}/plots/30_pfx_bon.png", dpi=150)
print("PFX_PLOT_DONE", f"{C}/plots/30_pfx_bon.png")
