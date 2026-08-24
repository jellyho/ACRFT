"""Figure for the CFAC toy (reads results.json written by toy_cfac.py; regenerable anytime).

    python slurm/probes/toy_cfac_fig.py --res /scratch/jellyho/acrft/probes/toy_cfac/results.json \
        --out /scratch/jellyho/acrft/hub_figs/toy_cfac.png
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

ARMS = ["A0_naive", "A1_hist", "A2_polboot", "A3_cfac"]
ARM_LABELS = ["naive\n(chunk-reg + data-V)", "+history", "+policy\nbootstrap", "CFAC\n(+composed)"]
FIXED = ["k1", "k2", "k3", "k4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    s = json.loads(a.res.read_text())["summary"]
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    # (1) deployed SR: critic arms as bars, fixed-k sweep as a line, oracle as hline
    ax = axes[0]
    m = [s[x]["sr"][0] for x in ARMS]
    e = [s[x]["sr"][1] for x in ARMS]
    ax.bar(range(4), m, yerr=e, capsize=3, color=[COLORS[0]] * 3 + [COLORS[2]], width=0.62)
    ax.axhline(s["oracle"]["sr"][0], ls="--", lw=1.2, color=COLORS[3], label="hand-crafted oracle")
    kx = [s[f]["sr"][0] for f in FIXED]
    ke = [s[f]["sr"][1] for f in FIXED]
    ax.errorbar(np.arange(4) - 0.0, kx, yerr=ke, color=COLORS[1], marker="o", ms=4, lw=1.2, label="fixed k=1..4")
    ax.set_xticks(range(4), ARM_LABELS, fontsize=7)
    ax.set_ylabel("deployed success rate")
    ax.set_title("deployment")
    ax.legend(fontsize=7)

    # (2) chosen commitment at the two decision types
    ax = axes[1]
    w = 0.36
    kc = [s[x]["mean_k_C"][0] for x in ARMS]
    kce = [s[x]["mean_k_C"][1] for x in ARMS]
    kj = [s[x]["mean_k_J"][0] for x in ARMS]
    kje = [s[x]["mean_k_J"][1] for x in ARMS]
    ax.bar(np.arange(4) - w / 2, kc, w, yerr=kce, capsize=3, color=COLORS[0], label="corridor entry (commit is right)")
    ax.bar(np.arange(4) + w / 2, kj, w, yerr=kje, capsize=3, color=COLORS[1], label="junction entry (react is right)")
    ax.axhline(4, ls=":", lw=1, color="gray")
    ax.axhline(1, ls=":", lw=1, color="gray")
    ax.set_xticks(range(4), ARM_LABELS, fontsize=7)
    ax.set_ylabel("mean selected k")
    ax.set_title("commitment by state type")
    ax.legend(fontsize=7)

    # (3) belief minus realized (calibration of the critic's own forecast)
    ax = axes[2]
    g = [s[x]["belief_gap"][0] for x in ARMS]
    ge = [s[x]["belief_gap"][1] for x in ARMS]
    ax.bar(range(4), g, yerr=ge, capsize=3, color=[COLORS[0]] * 3 + [COLORS[2]], width=0.62)
    ax.axhline(0, lw=1, color="gray")
    ax.set_xticks(range(4), ARM_LABELS, fontsize=7)
    ax.set_ylabel("believed − realized (discounted)")
    ax.set_title("self-deception")

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
