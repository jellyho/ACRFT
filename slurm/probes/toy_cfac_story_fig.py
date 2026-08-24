"""The friendly explainer for the toy task: a storyboard of what actually happens.

The information-timeline figure is precise but abstract. This one is the version you show first:
what the agent sees, what it has to decide, and why the same decision rule cannot be right in both
halves of the episode.

Top row is one episode as four moments. Bottom row is what each strategy does with it.

    python slurm/probes/toy_cfac_story_fig.py --out /scratch/jellyho/acrft/hub_figs/toy_cfac_story.png
"""

import argparse
import pathlib
import sys

sys.path.insert(0, "slurm")

import matplotlib.patches as mp
import matplotlib.pyplot as plt
from plot_style import PALETTE as COLORS
from plot_style import apply as apply_style

BLUE, ORANGE, GREEN, RED, GREY = COLORS[0], COLORS[1], COLORS[2], COLORS[3], "0.72"


def panel(ax, title, tint=None):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.85")
    if tint:
        ax.add_patch(mp.Rectangle((0, 0), 10, 6.4, color=tint, alpha=0.07, lw=0, zorder=0))
    ax.set_title(title, fontsize=9.5, pad=6)


def agent(ax, x, y, color="0.25", label=None):
    ax.plot([x], [y], "o", ms=15, color=color, zorder=5, clip_on=False)
    if label:
        ax.text(x, y - 1.0, label, ha="center", fontsize=7.5, color="0.35")


def sign(ax, x, y, text, color, *, faded=False):
    """A signpost: the environment telling the agent where to go."""
    a = 0.28 if faded else 1.0
    ax.add_patch(
        mp.FancyBboxPatch(
            (x - 0.75, y - 0.42),
            1.5,
            0.84,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="none",
            alpha=a,
            zorder=4,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=12, color="white", alpha=a, zorder=5, fontweight="bold")


def arrow(ax, x0, y0, dx, dy, color, ls="-", lw=2.2, alpha=1.0, head=0.3):
    ax.annotate(
        "",
        xy=(x0 + dx, y0 + dy),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": f"-|>,head_width={head},head_length={head * 1.6}",
            "color": color,
            "lw": lw,
            "ls": ls,
            "alpha": alpha,
        },
        zorder=3,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/hub_figs/toy_cfac_story.png")
    )
    a = ap.parse_args()
    apply_style()

    fig = plt.figure(figsize=(12.2, 6.9))
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1], hspace=0.45, wspace=1.4)
    fig.suptitle("the toy task, one episode at a time", fontsize=13, y=0.98)

    # ---------------- row 1: the four moments of an episode
    ax = fig.add_subplot(gs[0, 0:3])
    panel(ax, "① corridor starts: the sign is up", BLUE)
    sign(ax, 2.2, 4.6, "↗", BLUE)
    agent(ax, 2.2, 2.3, label="agent")
    arrow(ax, 2.9, 2.6, 3.4, 2.0, BLUE)
    ax.text(7.3, 2.0, "go this way\nfor the next 4 steps", fontsize=8, color="0.35", ha="center")
    ax.text(0.35, 5.9, "t = 0", fontsize=7.5, color="0.55", ha="left")

    ax = fig.add_subplot(gs[0, 3:6])
    panel(ax, "② the sign disappears", BLUE)
    sign(ax, 2.2, 4.6, "↗", BLUE, faded=True)
    ax.text(2.2, 3.85, "hidden", fontsize=7.5, color="0.5", ha="center")
    agent(ax, 4.2, 3.0)
    arrow(ax, 4.9, 3.3, 2.1, 1.3, BLUE, alpha=0.4)
    ax.text(7.6, 5.0, "kept plan", fontsize=7.5, color=BLUE, ha="center")
    for dx, dy in [(2.2, -0.2), (1.5, -1.5), (0.3, -2.1)]:
        arrow(ax, 4.9, 2.8, dx, dy, RED, ls=":", lw=1.4, alpha=0.55, head=0.22)
    ax.text(2.5, 1.5, "if it re-asks here,\nthe agent no longer knows\nand guesses", fontsize=8, color=RED, ha="center")
    ax.text(0.35, 5.9, "t = 1..3", fontsize=7.5, color="0.55", ha="left")

    ax = fig.add_subplot(gs[0, 6:9])
    panel(ax, "③ junction: nothing to see yet", ORANGE)
    agent(ax, 2.2, 3.2)
    ax.add_patch(mp.FancyBboxPatch((3.9, 4.3), 1.6, 0.9, boxstyle="round,pad=0.08", facecolor=GREY, edgecolor="none"))
    ax.add_patch(mp.FancyBboxPatch((3.9, 1.3), 1.6, 0.9, boxstyle="round,pad=0.08", facecolor=GREY, edgecolor="none"))
    ax.text(4.7, 4.75, "?", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax.text(4.7, 1.75, "?", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    arrow(ax, 2.9, 3.4, 1.0, 1.2, GREY, lw=1.6)
    arrow(ax, 2.9, 3.0, 1.0, -1.2, GREY, lw=1.6)
    ax.text(4.9, 0.7, "which way is right is not decided yet", fontsize=8, color="0.35", ha="center")
    ax.text(0.35, 5.9, "t = 4", fontsize=7.5, color="0.55", ha="left")

    ax = fig.add_subplot(gs[0, 9:12])
    panel(ax, "④ one step later: the signal lights up", ORANGE)
    agent(ax, 2.2, 3.2)
    ax.add_patch(mp.FancyBboxPatch((3.9, 4.3), 1.6, 0.9, boxstyle="round,pad=0.08", facecolor=GREEN, edgecolor="none"))
    ax.add_patch(
        mp.FancyBboxPatch((3.9, 1.3), 1.6, 0.9, boxstyle="round,pad=0.08", facecolor=GREY, edgecolor="none", alpha=0.5)
    )
    ax.text(4.7, 4.75, "↗", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    arrow(ax, 2.9, 3.4, 1.0, 1.2, GREEN)
    ax.text(
        4.9,
        0.7,
        "an agent that waited turns correctly;\none that committed already guessed",
        fontsize=8,
        color="0.35",
        ha="center",
    )
    ax.text(0.35, 5.9, "t = 5", fontsize=7.5, color="0.55", ha="left")

    # ---------------- row 2: the three strategies
    strategies = [
        (
            "always commit (k = 4)",
            [("corridor", GREEN, "keeps the plan"), ("junction", RED, "guesses, wrong half the time")],
            "good in the corridor, blind at the junction",
        ),
        (
            "always re-ask (k = 1)",
            [("corridor", RED, "forgets the plan"), ("junction", GREEN, "sees the signal")],
            "reactive at the junction, lost in the corridor",
        ),
        (
            "decide per state (what we want)",
            [("corridor", GREEN, "commits"), ("junction", GREEN, "waits one step")],
            "needs a critic that prices both correctly",
        ),
    ]
    for i, (title, rows, foot) in enumerate(strategies):
        ax = fig.add_subplot(gs[1, i * 4 : i * 4 + 4])
        panel(ax, title)
        for j, (where, col, what) in enumerate(rows):
            y = 4.7 - j * 1.9
            ax.add_patch(
                mp.Rectangle((0.6, y - 0.55), 2.5, 1.1, color=BLUE if where == "corridor" else ORANGE, alpha=0.16, lw=0)
            )
            ax.text(1.85, y, where, ha="center", va="center", fontsize=8.5, color="0.3")
            ax.add_patch(mp.Circle((3.75, y), 0.3, color=col, zorder=4))
            ax.text(
                3.75,
                y,
                "✓" if col == GREEN else "✗",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                zorder=5,
                fontweight="bold",
            )
            ax.text(4.4, y, what, ha="left", va="center", fontsize=8.2, color="0.35")
        ax.text(5.0, 1.0, foot, ha="center", fontsize=8, color="0.5", style="italic")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
