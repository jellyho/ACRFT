"""Explanatory figures for the plain-language action-chunking entry.

All panels are computed, not drawn by hand, except the clearly-labelled schematic in fig_curriculum
(which illustrates the SHAPE our theory predicts, not measured data).

  fig_lucky.png      DQC Proposition 1's 6-state counterexample, evaluated exactly: what the chunked
                     critic believes about each 2-step chunk vs. what open-loop execution actually gets.
  fig_epsilon.png    The nominal-vs-actual gap (DQC Thm 1 + Prop 4) as a function of how unpredictable
                     the world is, for several commitment lengths -- zero at eps=0 for every length.
  fig_curriculum.png Schematic of the decomposition (aleatoric floor + shrinking epistemic term) and the
                     mean-commitment-length curve it predicts. LABELLED AS SCHEMATIC.
"""

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/figs_chunking_easy")
C = plt.get_cmap("Dark2").colors


def style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_lucky(gamma=0.99, c=0.4):
    """DQC Prop 1's 6-state MDP, evaluated exactly.

    Construction (paper): from A, action 0 goes to B w.p. delta and C w.p. 1-delta; action 1 goes to C.
    The BEHAVIOUR policy is closed-loop: it plays 0 at B and 1 at C. So in the data the chunk (0,0) only
    ever appears when the coin gave B -- hence the critic believes it always reaches the reward state D.
    Executed open-loop it reaches D only with probability delta.
    """
    delta, ctil = 0.5, c + 0.5
    scale = gamma / (1 - gamma)
    chunks = ["(0,0)\n'lucky' chunk", "(0,1)", "(1,1)\nactually best"]
    nominal = [scale, ctil * scale, ctil * scale]  # what the chunked critic believes
    actual = [delta * scale, ctil * scale, ctil * scale]  # what open-loop execution gets

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, nominal, w, color=C[2], label="what the critic believes (nominal)")
    ax.bar(x + w / 2, actual, w, color=C[1], label="what actually happens (open-loop)")
    ax.set_xticks(x)
    ax.set_xticklabels(chunks, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("the critic picks the chunk it is most deluded about", fontsize=11, fontweight="regular", pad=14)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.set_ylim(0, max(nominal) * 1.30)
    ax.annotate(
        "critic picks THIS\n(highest believed value)",
        xy=(0 - w / 2, nominal[0]),
        xytext=(0.28, nominal[0] * 1.16),
        fontsize=8.5,
        color="#444",
        arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1},
    )
    ax.annotate(
        f"but gets {actual[0]:.0f}, not {nominal[0]:.0f}",
        xy=(0 + w / 2, actual[0]),
        xytext=(0.6, actual[0] * 0.62),
        fontsize=8.5,
        color="#444",
        arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1},
    )
    fig.suptitle(
        f"DQC Proposition 1's counterexample, evaluated exactly (γ={gamma}, δ={delta})",
        fontsize=9.5,
        y=1.0,
        color="#555",
    )
    fig.tight_layout()
    p = OUT / "fig_lucky.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p, "| nominal", [f"{v:.1f}" for v in nominal], "actual", [f"{v:.1f}" for v in actual])


def gap(k, gamma, eps):
    ek = np.minimum(3.0 * (1.0 - (1.0 - eps) ** (k - 1)), 1.0)
    return gamma * ek / ((1.0 - gamma) * (1.0 - (1.0 - ek) * gamma**k))


def fig_epsilon(gamma=0.99):
    """The price of committing blind, as a function of world unpredictability. Zero at eps=0."""
    eps = np.linspace(0, 0.02, 200)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, k in enumerate([2, 5, 10, 25]):
        ax.plot(100 * eps, gap(k, gamma, eps), color=C[i], lw=2, label=f"commit k = {k}")
    ax.set_xlabel("how unpredictable the world is  (ε, % chance of a surprise per step)")
    ax.set_ylabel("price of committing blind")
    ax.set_title("a predictable world makes commitment free", fontsize=11, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.axvline(0, color="#999", lw=1, ls=":")
    ax.annotate(
        "ε = 0 → price is exactly 0\nfor EVERY commitment length\n(our Theorem 2)",
        xy=(0.01, 1.0),
        xytext=(0.95, gap(25, gamma, 0.02) * 0.10),
        fontsize=8.5,
        color="#444",
        arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1},
    )
    fig.suptitle(f"DQC Theorem 1 bound with Prop. 4's ε_k  (γ={gamma})", fontsize=9.5, y=1.0, color="#555")
    fig.tight_layout()
    p = OUT / "fig_epsilon.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p)


def fig_curriculum(gamma=0.99, H=10, eps_dyn=1e-4, d_epis0=20.0, tol=4.0, n_max=60):  # noqa: N803
    """DERIVED curriculum curve (no free-hand shapes).

    Ingredients, all from the theorems:
      * Theorem A: improvement is policy iteration in M_H, which converges at least as fast as value
        iteration there; VI in M_H contracts by gamma^H per sweep, so
              Delta_epis(n) <= gamma^(H n) * Delta_epis(0).
      * Theorem 3 (III.5): the value still obtainable by breaking the chunk is at most
              B(n) = Delta_react + Delta_epis(n),      Delta_react <= Delta_alea.
      * DQC Cor. 1 + Prop. 4: Delta_alea <= eps_H * H_eff * Hbar with eps_H = 3(1-(1-eps)^(H-1)).
      * Lexicographic rule with tolerance tol: once B(n) <= tol, full commitment is provably selected.
    """
    Heff = 1.0 / (1.0 - gamma)
    Hbar = 1.0 / (1.0 - gamma**H)
    eps_H = 3.0 * (1.0 - (1.0 - eps_dyn) ** (H - 1))
    floor = eps_H * Heff * Hbar  # Delta_alea bound
    n = np.arange(0, n_max + 1)
    d_epis = d_epis0 * (gamma**H) ** n  # Theorem A contraction
    B = floor + d_epis  # what breaking the chunk can still buy

    below = np.where(tol >= B)[0]
    nstar = int(below[0]) if len(below) else None

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    ax = axes[0]
    ax.fill_between(n, 0, floor, color=C[1], alpha=0.75, label="aleatoric  Δ_alea  (bound, policy-independent)")
    ax.fill_between(n, floor, B, color=C[0], alpha=0.75, label="epistemic  Δ_epis(n) = γ^(Hn)·Δ_epis(0)")
    ax.set_xlabel("improvement iteration n  (policy iteration in $M_H$)")
    ax.set_ylabel("gap to the closed-loop optimum")
    ax.set_title("Theorem A contracts the epistemic term geometrically", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.plot(n, B, color=C[3], lw=2.2, label="B(n) = Δ_react + Δ_epis(n)  — what breaking can still buy")
    ax.axhline(floor, color="#999", lw=1.2, ls="--")
    ax.axhline(tol, color=C[5], lw=1.4, ls="-.", label=f"lexicographic tolerance ε = {tol:g}")
    ax.text(n_max * 0.42, floor * 1.10, "aleatoric floor", fontsize=8.5, color="#555")
    if nstar is not None:
        ax.axvline(nstar, color="#444", lw=1, ls=":")
        ax.annotate(
            f"n* = {nstar}\nfrom here the rule\nprovably commits fully",
            xy=(nstar, tol),
            xytext=(nstar + 3, tol * 2.4),
            fontsize=8.5,
            color="#444",
            arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1},
        )
    ax.set_xlabel("improvement iteration n")
    ax.set_ylabel("value still available from breaking")
    ax.set_title("so the selector runs out of reasons to break", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        f"derived from Theorem A + Theorem 3 + DQC Cor.1/Prop.4   (γ={gamma}, H={H}, ε={eps_dyn:g} → floor ≤ {floor:.2f})",
        fontsize=9,
        y=1.02,
        color="#555",
    )
    fig.tight_layout()
    p = OUT / "fig_curriculum.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {p} | eps_H={eps_H:.5f} floor={floor:.3f} rate=gamma^H={gamma**H:.4f} n*={nstar}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_lucky()
    fig_epsilon()
    fig_curriculum()
