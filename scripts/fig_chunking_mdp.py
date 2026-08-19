"""Schematic of the three MDPs the decoupled-horizon scheme lives in.

(A) the base MDP M            -- a decision every step
(B) the full-commitment MDP M_H -- one decision per chunk; the H-step open-loop rollout is ONE transition,
                                   reward = discounted H-step sum, discount gamma^H.  Improvement lives here.
(C) adaptive execution        -- variable-length transitions chosen per state. Deployment lives here.

Schematic (no data), drawn by script so it is regenerable.
"""

import itertools
import pathlib

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

OUT = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/figs_chunking_easy")
C = plt.get_cmap("Dark2").colors
GREY = "#8a8a84"


def state(ax, x, y, label, r=0.19, face="white", edge="#555", lw=1.4, fs=9):
    ax.add_patch(plt.Circle((x, y), r, facecolor=face, edgecolor=edge, lw=lw, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


def arrow(ax, x0, x1, y, color="#555", lw=1.4, rad=0.0, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle=style,
            mutation_scale=11,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )


def decision_mark(ax, x, y, color):
    """A filled marker meaning 'the policy is queried here'."""
    ax.plot([x], [y + 0.42], marker="v", ms=7, color=color, zorder=5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 6.6))
    xs = [0.6 + 1.05 * i for i in range(7)]
    y = 0.5

    # ---------------- (A) base MDP
    ax = axes[0]
    for i, x in enumerate(xs):
        state(ax, x, y, f"$s_{i}$")
        decision_mark(ax, x, y, C[7 % len(C)] if False else "#444")
    for i in range(6):
        arrow(ax, xs[i] + 0.19, xs[i + 1] - 0.19, y)
        ax.text((xs[i] + xs[i + 1]) / 2, y + 0.17, f"$a_{i}$", ha="center", fontsize=8, color="#444")
    ax.set_title(
        "(A)  the base MDP $M$ — a decision at every step (discount γ)", fontsize=10.5, fontweight="regular", loc="left"
    )

    # ---------------- (B) full-commitment MDP M_H
    ax = axes[1]
    state(ax, xs[0], y, "$s_0$", face="#eaf3ee", edge=C[0], lw=2)
    state(ax, xs[6], y, "$s_H$", face="#eaf3ee", edge=C[0], lw=2)
    for i in range(1, 6):
        state(ax, xs[i], y, "", r=0.075, face="#dddddd", edge="#bbbbbb", lw=1)
    decision_mark(ax, xs[0], y, C[0])
    decision_mark(ax, xs[6], y, C[0])
    ax.add_patch(
        FancyArrowPatch(
            (xs[0] + 0.19, y + 0.16),
            (xs[6] - 0.19, y + 0.16),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2.4,
            color=C[0],
            connectionstyle="arc3,rad=-0.28",
            zorder=2,
        )
    )
    ax.text(
        (xs[0] + xs[6]) / 2,
        y + 0.92,
        r"one action $=$ the whole chunk $a\in\mathcal{A}^H$;  reward $r_H=\sum_{j<H}\gamma^j r$;  discount $\gamma^H$",
        ha="center",
        fontsize=9,
        color=C[0],
    )
    ax.text(
        (xs[0] + xs[6]) / 2,
        y - 0.42,
        "executed open-loop — no decision in between",
        ha="center",
        fontsize=8.5,
        color=GREY,
    )
    ax.set_title(
        "(B)  the full-commitment MDP $M_H$ — ONE transition.   ← our IMPROVEMENT loop lives here",
        fontsize=10.5,
        fontweight="regular",
        loc="left",
    )

    # ---------------- (C) adaptive execution
    ax = axes[2]
    stops = [0, 2, 3, 6]
    for i, x in enumerate(xs):
        if i in stops:
            state(ax, x, y, f"$s_{i}$", face="#fdf0e7", edge=C[1], lw=2)
            decision_mark(ax, x, y, C[1])
        else:
            state(ax, x, y, "", r=0.075, face="#dddddd", edge="#bbbbbb", lw=1)
    for a, b in itertools.pairwise(stops):
        ax.add_patch(
            FancyArrowPatch(
                (xs[a] + 0.19, y + 0.16),
                (xs[b] - 0.19, y + 0.16),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=2.2,
                color=C[1],
                connectionstyle="arc3,rad=-0.35",
                zorder=2,
            )
        )
        ax.text((xs[a] + xs[b]) / 2, y + 0.72, f"$k={b - a}$", ha="center", fontsize=9, color=C[1])
    ax.set_title(
        "(C)  deployment — commit $\\kappa(s)$ steps, then re-query.   ← the SELECTOR lives here",
        fontsize=10.5,
        fontweight="regular",
        loc="left",
    )

    for ax in axes:
        ax.set_xlim(0, 7.4)
        ax.set_ylim(-0.55, 1.55)
        ax.axis("off")

    fig.text(0.5, 0.005, "▼ = the policy is queried here", ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = OUT / "fig_mdp.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
