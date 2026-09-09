"""Figures for the plain-language action-chunking entry. Regenerable: pure computation, no data files.

Panel A: DQC Theorem 1's bias bound as a function of commitment length k, for several discounts, with
epsilon_k = 3(1-(1-eps)^(k-1)) (DQC Prop. 4). Shows the bound grows monotonically in k but saturates
against the value-range ceiling -- i.e. it fixes the direction but is vacuous at large k.

Panel B: the same bound as a FRACTION of the value ceiling 1/(1-gamma), which is the quantity that
matters for a selector comparing across k: even a few percent of the ceiling dwarfs real advantages.
"""

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bias_bound(k, gamma, eps):
    """DQC Thm 1 bound with Prop 4's eps_k for eps-deterministic dynamics."""
    ek = np.minimum(3.0 * (1.0 - (1.0 - eps) ** (k - 1)), 1.0)  # TV distance caps at 1
    return gamma * ek / ((1.0 - gamma) * (1.0 - (1.0 - ek) * gamma**k))


def main():
    out = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/figs_chunking_easy")
    out.mkdir(parents=True, exist_ok=True)

    ks = np.arange(1, 51)
    gammas = [0.99, 0.999, 0.99964]
    eps = 0.001
    colors = plt.get_cmap("Dark2").colors

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    ax = axes[0]
    for i, g in enumerate(gammas):
        ax.plot(ks, bias_bound(ks, g, eps), color=colors[i], lw=2, label=f"γ = {g}")
    ax.set_xlabel("commitment length k")
    ax.set_ylabel("bias bound")
    ax.set_yscale("log")
    ax.set_title("bias bound grows with k", fontsize=11, fontweight="regular")

    ax = axes[1]
    for i, g in enumerate(gammas):
        ceil = 1.0 / (1.0 - g)
        ax.plot(ks, 100 * bias_bound(ks, g, eps) / ceil, color=colors[i], lw=2, label=f"γ = {g}")
    ax.set_xlabel("commitment length k")
    ax.set_ylabel("bound as % of value range")
    ax.set_ylim(0, 100)
    ax.set_title("and saturates against the ceiling", fontsize=11, fontweight="regular")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"ε = {eps} per-step stochasticity", fontsize=10, y=1.02, color="#555")
    fig.tight_layout()
    p = out / "bias_vs_k.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p)

    # print the numbers that the entry's table quotes, so text and figure cannot drift
    for g in gammas:
        row = [f"{bias_bound(k, g, eps):.1f}" for k in (2, 5, 10, 30, 50)]
        mono = all(bias_bound(k + 1, g, eps) >= bias_bound(k, g, eps) - 1e-12 for k in range(1, 60))
        print(f"gamma={g} eps={eps} ceiling={1 / (1 - g):.0f}  k=2,5,10,30,50 -> {row}  monotone={mono}")


if __name__ == "__main__":
    main()
