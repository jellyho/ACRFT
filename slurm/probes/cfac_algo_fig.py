"""What CFAC actually computes: the critic, its targets, the update loop, and deployment.

The task figures explain the problem; this one explains the algorithm. Four panels:

  (1) the critic. One forward pass over the chunk emits a value for every prefix, so scoring all
      commitment lengths costs one query, not H.
  (2) how a prefix target is built, and the one change that matters. The naive target bootstraps
      from the demonstration's own successor, which already carries the event the demonstrator saw
      when choosing that tail. CFAC resamples the successor among other episodes at the same
      decision point while holding the tail fixed, which is the do(.) intervention.
  (3) the offline loop: critic regression and full-chunk advantage-weighted policy improvement,
      alternating, with the backbone frozen.
  (4) deployment: one policy query, one critic pass, execute the longest prefix within epsilon.

    python slurm/probes/cfac_algo_fig.py --out /scratch/jellyho/acrft/hub_figs/cfac_algo.png
"""

import argparse
import pathlib
import sys

sys.path.insert(0, "slurm")

import matplotlib.patches as mp
import matplotlib.pyplot as plt
from plot_style import PALETTE as COLORS
from plot_style import apply as apply_style

BLUE, ORANGE, GREEN, RED, PURPLE = COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4]
INK, SOFT = "0.25", "0.55"


def panel(ax, title, w=10.0, h=6.4):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.85")
    ax.set_title(title, fontsize=10, pad=7)


def box(ax, x, y, w, h, text, color, *, fs=8, tc="white", alpha=1.0, radius=0.12):
    ax.add_patch(
        mp.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def outline(ax, x, y, w, h, text, color, *, fs=8, radius=0.12, ls="-"):
    ax.add_patch(
        mp.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor="none",
            edgecolor=color,
            lw=1.3,
            ls=ls,
            zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=SOFT, lw=1.4, ls="-", style="-|>"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": style, "color": color, "lw": lw, "ls": ls, "shrinkA": 2, "shrinkB": 2},
        zorder=2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/hub_figs/cfac_algo.png"))
    a = ap.parse_args()
    apply_style()

    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.16)
    fig.suptitle("CFAC, as an algorithm", fontsize=13.5, y=0.975)

    # ---------------------------------------------------------------- (1) the critic
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "① one pass over the chunk, a value for every prefix")
    box(ax, 0.4, 4.6, 2.4, 0.85, "observation  $o_t$", BLUE, fs=8.5)
    box(ax, 0.4, 3.45, 2.4, 0.85, "history  $a_{t-m:t-1}$", BLUE, fs=8.5, alpha=0.75)
    ax.text(1.6, 2.95, "frozen backbone features", fontsize=7, color=SOFT, ha="center")
    outline(ax, 3.35, 3.6, 1.5, 1.7, "key\nencoder", SOFT, fs=8)
    arr(ax, 2.8, 5.0, 3.35, 4.7)
    arr(ax, 2.8, 3.9, 3.35, 4.3)

    xs = [5.3, 6.5, 7.7, 8.9]
    for i, x in enumerate(xs):
        box(ax, x, 4.05, 0.95, 0.8, f"$a_{i + 1}$", "0.35", fs=8.5)
        outline(ax, x, 2.55, 0.95, 1.05, "", PURPLE, radius=0.1)
        arr(ax, x + 0.47, 4.05, x + 0.47, 3.6)
        if i:
            arr(ax, xs[i - 1] + 0.95, 3.07, x, 3.07, PURPLE, lw=1.2)
        # prefix value read out under each step
        box(ax, x, 1.15, 0.95, 0.75, f"$Q_{i + 1}$", GREEN, fs=8.5)
        arr(ax, x + 0.47, 2.55, x + 0.47, 1.9, GREEN)
    arr(ax, 4.85, 4.45, 5.3, 4.45)
    arr(ax, 4.85, 3.1, 5.3, 3.1, PURPLE, lw=1.2)
    ax.text(7.1, 5.55, "the chunk the policy proposed", fontsize=7.5, color=SOFT, ha="center")
    ax.text(7.1, 0.55, "the value of committing 1, 2, 3, 4 actions", fontsize=7.5, color=GREEN, ha="center")
    ax.text(1.6, 1.5, "causal: prefix $k$\nsees only $a_{1:k}$", fontsize=7.5, color=PURPLE, ha="center")

    # ---------------------------------------------------------------- (2) the target
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "② how a prefix target is built, and the one change that matters")

    # naive row
    ax.text(0.35, 5.75, "naive", fontsize=8.5, color=RED, ha="left", weight="bold")
    box(ax, 0.4, 4.5, 1.5, 0.9, "$h_t$", "0.55", fs=8.5)
    box(ax, 2.3, 4.5, 1.9, 0.9, "tail $a_{2:k}$", "0.55", fs=8, alpha=0.85)
    arr(ax, 4.25, 4.95, 5.1, 4.95, RED)
    box(ax, 5.1, 4.5, 3.5, 0.9, "its OWN successor\n(event already known)", RED, fs=7.5)
    ax.text(8.9, 4.95, "looks\nhigh", fontsize=7.5, color=RED, ha="left", va="center")
    ax.text(
        5.0,
        4.12,
        "the demonstrator chose that tail knowing the event, so the pairing keeps the confound",
        fontsize=7.3,
        color=RED,
        ha="center",
    )

    # CFAC row
    ax.text(0.35, 3.25, "CFAC", fontsize=8.5, color=GREEN, ha="left", weight="bold")
    box(ax, 0.4, 1.95, 1.5, 0.9, "$h_t$", "0.55", fs=8.5)
    box(ax, 2.3, 1.95, 1.9, 0.9, "same tail", GREEN, fs=8)
    for i, dy in enumerate((1.0, 0.0, -1.0)):
        box(
            ax,
            5.1,
            2.6 + dy - 0.34,
            3.5,
            0.68,
            ["another episode, same point", "…another event", "…another again"][i],
            GREEN,
            fs=7.2,
            alpha=0.85 - i * 0.12,
        )
        arr(ax, 4.25, 2.4, 5.1, 2.6 + dy, GREEN, lw=1.1)
    ax.text(6.85, 0.95, "averaged: what open-loop execution actually faces", fontsize=7.4, color=GREEN, ha="center")
    ax.text(
        5.0,
        0.32,
        "resample the successor, hold the tail fixed  =  $Q(h,\\,\\mathrm{do}(a_{1:k}))$",
        fontsize=8,
        color=INK,
        ha="center",
    )

    # ---------------------------------------------------------------- (3) the loop
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "③ the offline loop (no environment interaction)")
    box(ax, 0.7, 3.9, 3.6, 1.5, "critic\nfit prefix targets", GREEN, fs=8.5)
    box(ax, 5.7, 3.9, 3.6, 1.5, "policy\nweight the recorded chunk\nby its advantage", BLUE, fs=8)
    arr(ax, 4.3, 4.9, 5.7, 4.9, SOFT, lw=1.6)
    arr(ax, 5.7, 4.25, 4.3, 4.25, SOFT, lw=1.6)
    ax.text(5.0, 5.55, "the critic scores the current policy", fontsize=7.3, color=SOFT, ha="center")
    ax.text(5.0, 3.5, "the policy moves toward what the critic prefers", fontsize=7.3, color=SOFT, ha="center")
    ax.text(0.75, 2.75, "improvement targets the FULL chunk", fontsize=8, color=INK, ha="left")
    ax.text(
        0.75,
        2.2,
        "improving only the short prefix leaves the deployed\nlower bound unmoved, so commitments never lengthen",
        fontsize=7.4,
        color=SOFT,
        ha="left",
    )
    box(
        ax,
        0.7,
        0.45,
        8.6,
        1.05,
        "backbone frozen · demonstrations only\nnothing is sampled and ranked at deployment",
        "0.88",
        fs=7.6,
        tc=INK,
    )

    # ---------------------------------------------------------------- (4) deployment
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "④ deployment: one query, one critic pass")
    box(ax, 0.5, 4.4, 2.3, 1.0, "policy\nproposes a chunk", BLUE, fs=8)
    arr(ax, 2.8, 4.9, 3.6, 4.9)
    box(ax, 3.6, 4.4, 2.3, 1.0, "critic scores\nevery prefix", GREEN, fs=8)
    arr(ax, 5.9, 4.9, 6.7, 4.9)
    outline(ax, 6.7, 4.4, 2.8, 1.0, "take the longest $k$\nwithin $\\epsilon$ of the best", ORANGE, fs=8)

    # a small profile illustrating the rule
    base = [0.55, 0.95, 0.92, 0.4]
    for i, v in enumerate(base):
        x = 1.4 + i * 1.5
        ax.add_patch(mp.Rectangle((x, 1.2), 0.9, 1.9 * v, color=GREEN, alpha=0.85, zorder=3))
        ax.text(x + 0.45, 1.0, f"k={i + 1}", ha="center", fontsize=7.5, color=SOFT)
    ax.plot([1.3, 7.4], [1.2 + 1.9 * 0.95, 1.2 + 1.9 * 0.95], ls="--", lw=1, color=ORANGE, zorder=4)
    ax.text(7.5, 1.2 + 1.9 * 0.95, "best", fontsize=7.5, color=ORANGE, va="center")
    ax.plot([1.3, 7.4], [1.2 + 1.9 * 0.87, 1.2 + 1.9 * 0.87], ls=":", lw=1, color=ORANGE, zorder=4)
    ax.text(7.5, 1.2 + 1.9 * 0.85, "best − ε", fontsize=7.5, color=ORANGE, va="center")
    ax.annotate(
        "chosen: the longer\nof the two that tie",
        xy=(5.3, 1.2 + 1.9 * 0.92),
        xytext=(8.3, 1.55),
        fontsize=7.5,
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1},
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        3.55,
        "ties break toward the longer commitment: breaking them short re-queries constantly,\n"
        "which re-injects policy error and discards the plan the chunk was carrying",
        fontsize=7.4,
        color=SOFT,
        ha="left",
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
