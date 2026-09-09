"""The optimal commitment length under all FOUR forces, and where it stops.

Updates fig_chunking_kstar.py. That version had a "decision loss delta" -- a cost charged per re-query --
standing in for whatever makes short commitments bad. Two things are now known and change the model:

  * Re-planning has no intrinsic cost. What re-planning loses is INFORMATION: it re-conditions on the
    current observation and discards what the plan still carried from when the scene was informative.
    So delta is replaced by nu(s), the value of that discarded information -- a per-cycle loss that is
    real, is a property of the state's observability, and needs no invented cost term.
  * That information loss is what the concurrent literature calls the non-Markovianity of the
    demonstrations, and it is the only force that makes a LONGER commitment strictly better.

Forces on the commitment length k, per unit time:

  push SHORT, exposure while committed (cannot react):   (rho(s)*eps + (1-q)*c) * k / 2
      rho(s)*eps  environment stochasticity, weighted by how costly being unreactive is here
                  -> ALEATORIC: survives any amount of learning. This sets the floor.
      (1-q)*c     the policy's tail inaccuracy
                  -> EPISTEMIC: absorbed as the policy improves (q -> 1).

  push LONG, information carried forward:                nu(s) / k
      nu(s)       value of what a re-query would discard (occlusion, hidden intent)
                  -> NON-MARKOV: shrinks only if the policy's context grows, which a short-context
                     VLA cannot do; on the critic side it is what commitment is FOR.

  L(k) = (rho*eps + (1-q)*c)*k/2 + nu/k      ->   k*(s) = min(H, sqrt(2*nu / (rho*eps + (1-q)*c)))
  and at a perfect policy (q=1):                  k_floor(s) = min(H, sqrt(2*nu / (rho*eps)))

The three panels: (A) the two-sided trade-off and how its optimum moves as the policy improves;
(B) k* rises with tail accuracy and stops at a floor set by the environment, not by the policy;
(C) the floor as a function of the two things that set it -- how occluded the state is (nu) and how
stochastic it is (rho*eps). A LOCAL LINEARISED MODEL, not a theorem.
"""

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/figs_chunking_easy")
C = plt.get_cmap("Dark2").colors
LONG, SHORT = C[3], C[2]


def loss(k, q, eps, nu, rho=1.0, c=1.0):
    return (rho * eps + (1 - q) * c) * k / 2 + nu / k


def kstar(q, eps, nu, rho=1.0, c=1.0, cap=30):
    return np.minimum(cap, np.sqrt(2 * nu / (rho * eps + (1 - q) * c)))


def style(ax):
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(cap=30, nu=0.5, eps0=1e-3):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))

    # (A) the two-sided trade-off; the optimum slides right as the policy's tail becomes accurate
    ax = axes[0]
    ks = np.linspace(1, cap, 400)
    for i, q in enumerate([0.5, 0.9, 0.99, 1.0]):
        lab = f"tail accuracy q = {q:g}" + ("  (perfect)" if q == 1.0 else "")
        ax.plot(ks, loss(ks, q, eps0, nu), color=C[i], lw=2.2, label=lab)
        km = kstar(q, eps0, nu, cap=cap)
        ax.plot([km], [loss(km, q, eps0, nu)], marker="o", ms=7, color=C[i])
    ax.set_yscale("log")
    ax.set_xlabel("commitment length  $k$")
    ax.set_ylabel("loss per unit time")
    ax.set_title("(A) two sides, one optimum", fontsize=12.5)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.text(0.30, 0.055, "long: exposure while unreactive", transform=ax.transAxes, fontsize=9, color=SHORT)
    ax.text(0.30, 0.005, "short: information a re-query discards", transform=ax.transAxes, fontsize=9, color=LONG)

    # (B) k* vs tail accuracy: rises, then stops at a floor the policy cannot move
    ax = axes[1]
    gaps = np.logspace(0, -5, 400)  # 1-q from 1 down to 1e-5
    for i, eps in enumerate([3e-3, 1e-3, 1e-4]):
        ax.plot(gaps, kstar(1 - gaps, eps, nu, cap=cap), color=C[i], lw=2.4, label=f"env noise ε = {eps:g}")
        ax.axhline(kstar(1.0, eps, nu, cap=cap), color=C[i], lw=1.1, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("policy tail error  $1-q$      (improvement →)")
    ax.set_ylabel("optimal commitment  $k^*$")
    ax.set_ylim(0, cap * 1.15)
    ax.set_title("(B) it grows, then the environment stops it", fontsize=12.5)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(
        0.97,
        0.30,
        "dashed = aleatoric floor\n(no policy can pass it)",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=SHORT,
    )
    ax.text(
        0.03, 0.06, "epistemic force absorbed\nas the policy improves", transform=ax.transAxes, fontsize=9, color=LONG
    )

    # (C) the floor itself: set by how occluded (nu) against how stochastic (rho*eps) the state is
    ax = axes[2]
    nus = np.logspace(-1.4, 0.9, 300)
    for i, re_ in enumerate([3e-3, 1e-3, 2e-4]):
        lab = {3e-3: "contact / high noise", 1e-3: "typical", 2e-4: "free space / low noise"}[re_]
        ax.plot(nus, kstar(1.0, re_, nus, cap=cap), color=C[i], lw=2.4, label=lab)
    ax.axhline(cap, color="#bbb", lw=1.1, ls=":")
    ax.text(nus[-1], cap * 1.03, "H", fontsize=9, color="#888", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"information a re-query discards   $\nu(s)$   (occlusion →)")
    ax.set_ylabel(r"floor  $k_{\rm floor}$")
    ax.set_title("(C) the floor is a property of the state", fontsize=12.5)
    style(ax)
    ax.grid(which="both", axis="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        r"$k^*(s)=\min(H,\ \sqrt{2\nu(s)/(\rho(s)\varepsilon+(1-q)c)})$      "
        "long ← information the plan carries   |   short ← exposure while unreactive",
        fontsize=11.5,
        y=1.045,
    )
    fig.tight_layout()
    p = OUT / "fig_four_forces_kstar.png"
    fig.savefig(p, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print("wrote", p)
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  q={q:<5g} eps={eps0:g} nu={nu:g}  ->  k* = {kstar(q, eps0, nu, cap=cap):.2f}")
    for nu_ in (0.1, 0.5, 2.0):
        print(
            f"  q=1 nu={nu_:<4g}  floor: contact {kstar(1.0, 3e-3, nu_, cap=cap):5.1f}   "
            f"free space {kstar(1.0, 2e-4, nu_, cap=cap):5.1f}"
        )


if __name__ == "__main__":
    main()
