"""The background slide for CFAC: what the environment is, and why the two segments are different.

This figure carries no results. It exists so that a result figure can be explained: it draws the
information timeline of one episode (when the latent is visible, when it is hidden, when the event
arrives), what the demonstrator knows at each step, and what a Markov policy can know, which is the
gap the commitment is asked to bridge.

    python slurm/probes/toy_cfac_setup_fig.py --out /scratch/jellyho/acrft/hub_figs/toy_cfac_setup.png
"""

import argparse
import pathlib
import sys

sys.path.insert(0, "slurm")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from plot_style import PALETTE as COLORS
from plot_style import apply as apply_style

H = 4
SEGS = [("corridor", "C"), ("junction", "J"), ("corridor", "C")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/hub_figs/toy_cfac_setup.png")
    )
    a = ap.parse_args()
    apply_style()

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.6), height_ratios=[1.25, 1.0])

    # ---- top: information timeline of one episode
    ax = axes[0]
    T = len(SEGS) * H
    for i, (name, kind) in enumerate(SEGS):
        x0 = i * H
        ax.axvspan(x0, x0 + H, color=COLORS[0] if kind == "C" else COLORS[1], alpha=0.10, lw=0)
        ax.text(x0 + H / 2, 3.42, name, ha="center", fontsize=9, color="0.35")
    rows = [
        ("the target direction g", 2.6),
        ("what the demonstrator knows", 1.7),
        ("what a Markov policy sees", 0.8),
    ]
    for label, y in rows:
        ax.text(-0.35, y, label, ha="right", va="center", fontsize=8)

    for i, (_name, kind) in enumerate(SEGS):
        x0 = i * H
        for step in range(H):
            t = x0 + step
            visible = (kind == "C" and step == 0) or (kind == "J" and step >= 1)
            # row 1: is g present in the world's observation at this step
            ax.add_patch(mpatches.Rectangle((t + 0.1, 2.35), 0.8, 0.5, color=COLORS[2] if visible else "0.87", lw=0))
            # row 2: the demonstrator remembers within a segment, so it knows g from the moment it appears
            knows = (kind == "C") or (kind == "J" and step >= 1)
            ax.add_patch(mpatches.Rectangle((t + 0.1, 1.45), 0.8, 0.5, color=COLORS[2] if knows else "0.87", lw=0))
            # row 3: a Markov policy only ever sees the current observation
            ax.add_patch(mpatches.Rectangle((t + 0.1, 0.55), 0.8, 0.5, color=COLORS[2] if visible else "0.87", lw=0))
    ax.annotate(
        "plan visible only here,\nthen hidden",
        xy=(0.5, 2.9),
        xytext=(1.6, 3.05),
        fontsize=7.5,
        color="0.3",
        arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 1},
    )
    ax.annotate(
        "event arrives here",
        xy=(5.5, 2.9),
        xytext=(6.4, 3.05),
        fontsize=7.5,
        color="0.3",
        arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 1},
    )
    ax.set_xlim(-3.4, T + 0.3)
    ax.set_ylim(0.2, 3.7)
    ax.set_yticks([])
    ax.set_xticks([t + 0.5 for t in range(T)], [str(t) for t in range(T)], fontsize=7)
    ax.set_xlabel("environment step")
    ax.set_title("what is known, and when")
    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS[2], label="knows the target direction"),
            mpatches.Patch(color="0.87", label="does not"),
        ],
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.02),
        ncol=2,
    )

    # ---- bottom: the two situations, and what each choice costs
    ax = axes[1]
    ax.axis("off")
    box = {"boxstyle": "round,pad=0.5", "linewidth": 1.2}
    ax.text(
        0.02,
        0.92,
        "corridor  ·  the latent is in the PAST",
        fontsize=10,
        color=COLORS[0],
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.74,
        "The plan was shown at the entry and then hidden. The actions that follow\n"
        "carry it; the observation no longer does. Committing keeps the plan.\n"
        "Re-querying discards it, and a Markov policy has to guess.\n\n"
        "→ a critic that sees only the observation cannot represent the difference,\n"
        "   which is why CFAC conditions on the executed history.",
        fontsize=8,
        va="top",
        transform=ax.transAxes,
        bbox={**box, "facecolor": "#f4f7fb", "edgecolor": COLORS[0]},
    )
    ax.text(
        0.53,
        0.92,
        "junction  ·  the latent is in the FUTURE",
        fontsize=10,
        color=COLORS[1],
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.53,
        0.74,
        "The event is revealed one step after the segment starts. Committing across\n"
        "that moment acts before the information exists. Reacting waits for it.\n\n"
        "→ in the demonstrations the person already knew the event when choosing\n"
        "   those actions, so regressing outcomes on the chunk credits the executor\n"
        "   with knowledge it will not have. CFAC composes the value through a\n"
        "   resampled successor instead, which is the do(·) intervention.",
        fontsize=8,
        va="top",
        transform=ax.transAxes,
        bbox={**box, "facecolor": "#fdf6f0", "edgecolor": COLORS[1]},
    )

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
