"""Figures for the neural CFAC toy (regenerable from results.json).

    python slurm/probes/toy_cfac_nn_fig.py --res /scratch/jellyho/acrft/probes/toy_cfac_nn/results.json \
        --out /scratch/jellyho/acrft/hub_figs/toy_cfac_nn.png [--curric <curriculum-run>/results.json]
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

CELLS = ["cfac_neither_sel", "cfac_nohist_sel", "cfac_nointerv_sel", "cfac_sel"]
CELL_LABELS = ["neither", "+interventional\ncomposition", "+history", "both\n(CFAC)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=pathlib.Path, required=True)
    ap.add_argument("--curric", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.res.read_text())
    s = d["summary"]
    apply_style()

    ncol = 3 if a.curric else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.3))

    # (1) deployed return: baselines, the 2x2 critic cells, and joint
    ax = axes[0]
    order = ["bc_k1", "bc_k2", "bc_k4", "naive_sel", *CELLS, "cfac_joint", "bc_oracle"]
    lab = [
        "k=1",
        "k=2",
        "k=4",
        "naive\ncritic",
        "neither",
        "+interv",
        "+hist",
        "CFAC\n(sel)",
        "CFAC\n(joint)",
        "oracle",
    ]
    col = [COLORS[7]] * 3 + [COLORS[3]] + [COLORS[0]] * 3 + [COLORS[2], COLORS[2], COLORS[1]]
    m = [s[k]["ret"][0] for k in order]
    e = [s[k]["ret"][1] for k in order]
    ax.bar(range(len(order)), m, yerr=e, capsize=3, color=col, width=0.68)
    ax.axhline(s["bc_oracle"]["ret"][0], ls="--", lw=1, color=COLORS[1])
    ax.set_xticks(range(len(order)), lab, fontsize=6.5)
    ax.set_ylabel("deployed discounted return")
    ax.set_title("deployment")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=COLORS[7], label="fixed execution length (BC policy)"),
            Patch(color=COLORS[3], label="naive chunk-outcome critic"),
            Patch(color=COLORS[0], label="ablated CFAC critic"),
            Patch(color=COLORS[2], label="CFAC"),
            Patch(color=COLORS[1], label="hand-crafted oracle"),
        ],
        fontsize=6.2,
        loc="lower right",
    )

    # (2) the 2x2: what each ingredient buys, in reaction rate at the junction reveal
    ax = axes[1]
    rr = [s[c]["react_rate"][0] for c in CELLS]
    re = [s[c]["react_rate"][1] for c in CELLS]
    ax.bar(range(4), rr, yerr=re, capsize=3, color=[COLORS[0]] * 3 + [COLORS[2]], width=0.62)
    ax.set_xticks(range(4), CELL_LABELS, fontsize=7)
    ax.set_ylabel("fraction reacting after the reveal")
    ax.set_ylim(0, 1.05)
    ax.set_title("junction: does it re-query?")

    # (3) curriculum, if a second run was supplied
    if a.curric:
        c = json.loads(a.curric.read_text())
        cps = c["per_seed"]
        rounds = ["cfac_sel"] + [f"cfac_joint_r{i}" for i in (1, 2, 3)]
        x = np.arange(len(rounds))
        kc = np.array([[p[r]["k_corridor"] for r in rounds] for p in cps])
        rt = np.array([[p[r]["ret"] for r in rounds] for p in cps])
        ax = axes[2]
        ax.errorbar(x, kc.mean(0), yerr=kc.std(0), marker="o", color=COLORS[0], label="mean commitment (corridor)")
        ax.set_xticks(x, ["before", "round 1", "round 2", "round 3"], fontsize=7)
        ax.set_ylabel("mean selected k at corridor entry", color=COLORS[0])
        ax2 = ax.twinx()
        ax2.errorbar(x, rt.mean(0), yerr=rt.std(0), marker="s", ls="--", color=COLORS[2], label="return")
        ax2.set_ylabel("deployed return", color=COLORS[2])
        ax2.grid(visible=False)
        ax.set_title("curriculum under improvement")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
