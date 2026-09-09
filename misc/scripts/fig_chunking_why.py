"""Why the commitment length MUST grow — an exactly solvable model.

Model (everything below is exact, no simulation):
  A corridor. At each step exactly one action is correct: correct -> reward 1, wrong -> reward 0.
  A chunk policy proposes H actions. Its FIRST slot is always correct; each later slot is correct with
  probability q. (This is the well-documented shape of chunk policies: early timesteps are more accurate
  than late ones.)  Committing k steps means executing slots 1..k and then re-querying -- so re-querying
  is a way of using only the accurate early slots.

  Value of committing k:      V(k) = [1 + q*gamma*(1-gamma^(k-1))/(1-gamma)] / (1 - gamma^k)
  Value of the shortest:      V(1) = 1/(1-gamma)                     <-- does NOT depend on q at all
  Advantage of breaking:      V(1) - V(k) = gamma (1-q) (1-gamma^(k-1)) / [(1-gamma)(1-gamma^k)]

Three consequences, all visible in the figure:
  1. the ONLY thing that makes breaking attractive is the tail inaccuracy (1-q) -- the advantage is
     exactly proportional to it, and is exactly zero at q=1 (all k tie);
  2. improving at short k cannot fix it, because V(1) does not contain q;
  3. improving at k=H raises q, which monotonically destroys the advantage of breaking. Hence the
     commitment length must grow, and it stops when the advantage hits the floor (here: zero).
"""

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/figs_chunking_easy")
C = plt.get_cmap("Dark2").colors


def V(k, q, g):  # noqa: N802  (V is the standard value-function symbol)
    per = 1.0 + q * g * (1 - g ** (k - 1)) / (1 - g)
    return per / (1 - g**k)


def adv(k, q, g):
    """closed form: V(1) - V(k)"""
    return g * (1 - q) * (1 - g ** (k - 1)) / ((1 - g) * (1 - g**k))


def style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(g=0.9, H=10):  # noqa: N803
    OUT.mkdir(parents=True, exist_ok=True)
    ks = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    ax = axes[0]
    for i, q in enumerate([0.0, 0.5, 0.8, 0.95, 1.0]):
        vals = [V(k, q, g) for k in ks]
        ax.plot(
            ks,
            vals,
            marker="o",
            ms=3.5,
            color=C[i],
            lw=1.9,
            label=f"tail accuracy q = {q:g}" + ("  ← after improvement" if q == 1.0 else ""),
        )
    ax.set_xlabel("commitment length k")
    ax.set_ylabel("value  V(k)")
    ax.set_title("as the tail gets accurate, the curve flattens", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")
    ax.annotate(
        "q = 1: every k has the SAME value\n(exact tie — nothing to gain by breaking)",
        xy=(H * 0.62, V(int(H * 0.62), 1.0, g)),
        xytext=(2.5, V(1, 1.0, g) * 0.33),
        fontsize=8.3,
        color="#444",
        arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1},
    )

    ax = axes[1]
    qs = np.linspace(0, 1, 200)
    for i, k in enumerate([2, 5, 10]):
        ax.plot(qs, adv(k, qs, g), color=C[i], lw=2.1, label=f"commit k = {k} instead of 1")
    ax.set_xlabel("tail accuracy q   (this is what improving at k = H raises)")
    ax.set_ylabel("advantage of breaking\nV(1) − V(k)")
    ax.set_title("the reason to break is exactly the tail's inaccuracy", fontsize=10.5, fontweight="regular")
    style(ax)
    ax.legend(frameon=False, fontsize=8.5)
    ax.axvline(1.0, color="#999", lw=1, ls=":")
    ax.annotate(
        "→ improvement at k = H pushes q this way\n    and the reason to break is consumed",
        xy=(0.985, adv(10, 0.985, g)),
        xytext=(0.12, adv(10, 0.0, g) * 0.55),
        fontsize=8.3,
        color="#444",
        arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1},
    )

    fig.suptitle(
        f"exactly solvable model (γ={g}):   V(1) − V(k) = γ(1−q)(1−γ^(k−1)) / [(1−γ)(1−γ^k)]   —   "
        "note V(1) does not contain q at all",
        fontsize=9,
        y=1.03,
        color="#555",
    )
    fig.tight_layout()
    p = OUT / "fig_why.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p)
    for q in (0.0, 0.5, 0.8, 0.95, 0.99, 1.0):
        vals = [V(k, q, g) for k in ks]
        print(f"  q={q:4.2f}  argmax_k = {int(np.argmax(vals)) + 1:2d}   V(1)-V(H) = {adv(H, q, g):.4f}")


if __name__ == "__main__":
    main()
