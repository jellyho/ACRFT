"""The local model that gives the optimal commitment length, and its ceiling.

Two forces decide how long to commit.

  Exposure loss (worse when long).  While committed you cannot react. Per step, something can go wrong
  from two sources: environment stochasticity eps, and the policy's tail inaccuracy (1-q). A mishap early
  in a long chunk ruins the WHOLE remainder, so with a uniform mishap time the expected loss is
  quadratic in k:           (rho(s)*eps + (1-q)*c) * k^2 / 2
  where rho(s) is the local cost of being unable to react (large near contact).

  Decision loss (worse when short).  Every re-query is a decision made by an imperfect critic/actor and
  costs delta.  One decision per cycle.

  Loss per unit time:  L(k) = (rho*eps + (1-q)*c) * k / 2  +  delta / k
  Minimising:          k*(s) = min(H, sqrt( 2*delta / (rho(s)*eps + (1-q)*c) ))
  Ceiling at q=1:      k_ceil(s) = min(H, sqrt( 2*delta / (rho(s)*eps) ))

Consequences the figure shows: k* grows as the policy's tail becomes accurate (q -> 1) even in a NOISY
environment (no exact ties, no tie-break rule needed); the growth stops at a ceiling set by eps and
rho(s); the ceiling follows a 1/sqrt(eps) law and is capped by H; and as delta -> 0 (a perfect decision
maker) the ceiling collapses -- with a perfect critic one should re-query every step.

This is a LOCAL LINEARISED MODEL, not a theorem: it quantifies the tendency the theorems establish.
"""

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/figs_chunking_easy")
C = plt.get_cmap("Dark2").colors


def loss(k, q, eps, rho=1.0, delta=0.5, c=1.0):
    return (rho * eps + (1 - q) * c) * k / 2 + delta / k


def kstar(q, eps, rho=1.0, delta=0.5, c=1.0, cap=30):
    return np.minimum(cap, np.sqrt(2 * delta / (rho * eps + (1 - q) * c)))


def style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(cap=30, delta=0.5, eps0=1e-3):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    # (A) the U-shaped trade-off and how its minimum moves as the policy improves
    ax = axes[0]
    ks = np.linspace(1, cap, 400)
    for i, q in enumerate([0.5, 0.9, 0.99, 1.0]):
        ax.plot(ks, loss(ks, q, eps0, delta=delta), color=C[i], lw=2, label=f"tail accuracy q = {q:g}")
        km = kstar(q, eps0, delta=delta, cap=cap)
        ax.plot([km], [loss(km, q, eps0, delta=delta)], marker="o", ms=6, color=C[i])
    ax.set_xlabel("commitment length k")
    ax.set_ylabel("loss per unit time")
    ax.set_yscale("log")
    ax.set_title("two forces, one optimum", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=8.3)
    ax.text(
        0.30,
        0.06,
        "● = optimum $k^*$ — it moves RIGHT as the tail becomes accurate",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#444",
    )

    # (B) Q1: k* grows as the tail becomes accurate, in a NOISY environment, and stops at a ceiling.
    # Plotted against (1-q) on a log axis: that is the scale on which the approach to q=1 is visible
    # and on which the different ceilings separate.
    ax = axes[1]
    gaps = np.logspace(0, -5, 400)  # 1-q from 1 down to 1e-5
    for i, eps in enumerate([3e-3, 1e-3, 1e-4, 0.0]):
        lbl = "ε = 0  (deterministic)" if eps == 0 else f"ε = {eps:g}"
        ax.plot(gaps, kstar(1 - gaps, eps, delta=delta, cap=cap), color=C[i], lw=2.1, label=lbl)
        if eps > 0:
            ceil = kstar(1.0, eps, delta=delta, cap=cap)
            ax.axhline(ceil, color=C[i], lw=1, ls="--", alpha=0.55)
    ax.axhline(cap, color="#bbb", lw=1, ls=":")
    ax.set_xscale("log")
    ax.invert_xaxis()  # policy improves to the right
    ax.set_xlabel("policy tail error  $1-q$   (improves →)")
    ax.set_ylabel("optimal commitment $k^*$")
    ax.set_ylim(0, cap * 1.12)
    ax.set_title("Q1: it grows as the tail improves — even with noise", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")
    ax.text(0.97, 0.90, f"H = {cap}", transform=ax.transAxes, ha="right", fontsize=8.2, color="#888")
    ax.text(
        0.97,
        0.06,
        "dashed = the ceiling each ε imposes",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.2,
        color="#444",
    )

    # (C) Q2: the ceiling set by environment noise and local reactivity cost
    ax = axes[2]
    epss = np.logspace(-5, -1.3, 300)
    for i, rho in enumerate([0.3, 1.0, 3.0]):
        lbl = {0.3: "free space (ρ small)", 1.0: "typical (ρ = 1)", 3.0: "contact-critical (ρ large)"}[rho]
        ax.plot(epss, kstar(1.0, epss, rho=rho, delta=delta, cap=cap), color=C[i], lw=2.1, label=lbl)
    ax.axhline(cap, color="#bbb", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("environment stochasticity ε")
    ax.set_ylabel("ceiling $k_{\\rm ceil}$ at q = 1")
    ax.set_title("Q2: the ceiling falls as $1/\\sqrt{ε}$", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.grid(which="both", axis="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")

    fig.suptitle(
        "local model:  $L(k)=\\frac{1}{2}(\\rho\\,\\varepsilon+(1-q)c)\\,k+\\delta/k$   "
        "$\\Rightarrow$   $k^*=\\min(H,\\ \\sqrt{2\\delta/(\\rho\\varepsilon+(1-q)c)})$   "
        f"(δ={delta:g}, c=1)   —   a linearised model, not a theorem",
        fontsize=9.2,
        y=1.045,
        color="#555",
    )
    fig.tight_layout()
    p = OUT / "fig_kstar.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p)
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  q={q:<6g} eps={eps0:g}  k* = {kstar(q, eps0, delta=delta, cap=cap):.2f}")
    for eps in (1e-2, 1e-3, 1e-4):
        print(
            f"  q=1 eps={eps:<7g} ceiling: rho=0.3 -> {kstar(1.0, eps, rho=0.3, delta=delta, cap=cap):5.1f}"
            f"   rho=3 -> {kstar(1.0, eps, rho=3.0, delta=delta, cap=cap):5.1f}"
        )


if __name__ == "__main__":
    main()
