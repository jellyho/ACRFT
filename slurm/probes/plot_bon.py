# ruff: noqa
"""Two-panel figure for the 3-way critic-head comparison: offline metrics vs closed-loop BoN success.

Left  : offline demo-winrate (does the critic rank the demo action above random candidates) for the
        three heads on the SAME AQC trunk — scalar / HL-Gauss / floq.
Right : closed-loop PrepareCoffee success for BoN with each critic, against the VLA baseline (execute
        candidate 0) and the `rand` null (execute a random candidate), with Wilson 95% CIs.

The story is the disconnect: HL-Gauss/floq win the offline ranking (0.85-0.91) yet NO BoN mode beats
the VLA, and scalar/floq fall below even the random null — the winner's curse of argmax-ing a critic
over the policy's own candidates. Regenerates from the two probe JSONs (no GPU).

Outputs: plots/29_bon_compare.png
Usage: plot_bon.py
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "slurm")
from plot_style import PALETTE
from plot_style import apply

apply()
C = os.environ["CACHE_DIR"]
off = json.load(open(f"{C}/probes/floq_critic.json"))
bon = json.load(open(f"{C}/gr1_eval/bon_critic_compare.json"))

heads = [("scalar", "scalar\n(regression)"), ("hlgauss", "HL-Gauss\n(classification)"), ("floq", "floq\n(flow)")]
hcol = {"scalar": PALETTE[7], "hlgauss": PALETTE[2], "floq": PALETTE[3]}  # gray / green / red

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})

# ---- left: offline demo-winrate ----
wr = [off[f"{k}_arq"]["demo_winrate"] for k, _ in heads]
xs = np.arange(len(heads))
axL.bar(xs, wr, color=[hcol[k] for k, _ in heads], width=0.62)
for x, v in zip(xs, wr):
    axL.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
axL.set_xticks(xs)
axL.set_xticklabels([lab for _, lab in heads])
axL.set_ylim(0, 1.0)
axL.set_ylabel("offline demo-winrate")
axL.set_title(f"offline: does the head rank actions?  (γ={off['gamma']})")

# ---- right: closed-loop BoN success with Wilson CI ----
order = ["vla", "rand", "scalar", "hlgauss", "floq"]
labs = {
    "vla": "VLA\n(baseline)",
    "rand": "rand\n(null)",
    "scalar": "scalar\nBoN",
    "hlgauss": "HL-Gauss\nBoN",
    "floq": "floq\nBoN",
}
cols = {
    "vla": "#1a1a1a",
    "rand": PALETTE[7],
    "scalar": hcol["scalar"],
    "hlgauss": hcol["hlgauss"],
    "floq": hcol["floq"],
}
sr = [bon[m]["success_rate"] for m in order]
lo = [bon[m]["success_rate"] - bon[m]["wilson95"][0] for m in order]
hi = [bon[m]["wilson95"][1] - bon[m]["success_rate"] for m in order]
xs2 = np.arange(len(order))
axR.bar(
    xs2,
    sr,
    color=[cols[m] for m in order],
    width=0.66,
    yerr=[lo, hi],
    error_kw={"ecolor": "#333", "elinewidth": 1.3, "capsize": 4},
)
axR.axhline(bon["vla"]["success_rate"], color="#1a1a1a", lw=1.0, ls="--", zorder=0)
for x, m in zip(xs2, order):
    n, k = bon[m]["num_trials"], bon[m]["successes"]
    axR.text(x, bon[m]["success_rate"] + hi[list(order).index(m)] + 0.03, f"{k}/{n}", ha="center", fontsize=9)
axR.set_xticks(xs2)
axR.set_xticklabels([labs[m] for m in order])
axR.set_ylim(0, 1.0)
axR.set_ylabel("closed-loop success rate")
axR.set_title(
    f"closed-loop: does BoN beat the VLA?  (PrepareCoffee, N={bon['meta']['N']}, n={bon['vla']['num_trials']})"
)

fig.tight_layout()
os.makedirs(f"{C}/plots", exist_ok=True)
fig.savefig(f"{C}/plots/29_bon_compare.png", dpi=150)
print("BON_PLOT_DONE", f"{C}/plots/29_bon_compare.png")
