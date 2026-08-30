"""Figures for the flow/diffusion-RL policy-extraction survey note (house style).

1. taxonomy — the 6 families, organized by HOW they get ∇Q into an iterative generator.
2. routes   — our setting (frozen pi05 flow decoder + patch critic): three ways to run DDPG on a
              one-step latent actor WITHOUT backprop through the decoder ODE.

  uv run python scripts/flowrl_survey_figs.py --out space_v2/figs
"""

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import report_style as rs


def _panel(ax, x, y, w, h, title, body, meths, ec):
    """A titled panel: bold title (top), body (middle), italic methods (bottom) — with padding."""
    ax.add_patch(
        mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc="#f9f9f7", ec=ec, lw=1.3)
    )
    pad = 0.26
    ax.text(x + pad, y + h - pad, title, fontsize=9.5, fontweight="bold", color=ec, va="top", ha="left")
    ax.text(x + pad, y + h - pad - 0.62, body, fontsize=7.8, color=rs.INK, va="top", ha="left", linespacing=1.5)
    ax.text(x + pad, y + pad, meths, fontsize=7.2, style="italic", color=ec, va="bottom", ha="left", linespacing=1.45)


def _box(ax, x, y, w, h, lines, *, fc="#f2f1ec", ec=rs.INK, fs=9):
    ax.add_patch(
        mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=1.3)
    )
    ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center", fontsize=fs, color=rs.INK, linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1, *, c=rs.GRAY, lw=1.7, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": lw, "ls": ls})


def fig_taxonomy(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    W, H, GX = 4.9, 3.55, 0.55
    xs = [0.3, 0.3 + W + GX, 0.3 + 2 * (W + GX)]
    yt, yb = 5.15, 0.7
    P = [
        (
            xs[0],
            yt,
            "1 · backprop through chain",
            "differentiate Q back through\nall T denoising steps — honest\nbut O(T) memory, unstable\n→ keep T tiny",
            "Diffusion-QL",
            rs.RED,
        ),
        (
            xs[1],
            yt,
            "2 · distill to 1-step, then RL",
            "collapse the sampler to ONE\ndifferentiable map → cheap\nreparam ∇Q, no BPTT, no\niterative test-time sampling",
            "FQL · Consistency-AC\nCPQL · OFQL",
            rs.GREEN,
        ),
        (
            xs[2],
            yt,
            "3 · score ↔ ∇Q matching",
            "never sample; set the\ngenerator's score / velocity\nfield to (a function of) ∇_a Q",
            "QSM · SRPO · DAC",
            rs.ORANGE,
        ),
        (
            xs[0],
            yb,
            "4 · denoising-as-MDP (PG)",
            "each denoising step = MDP\naction (Gaussian); policy-\ngradient over steps —\nno ∇ through Q at all",
            "DPPO · ReinFlow\nFlow-GRPO",
            rs.PURPLE,
        ),
        (
            xs[1],
            yb,
            "5 · weighted resample / regress",
            "the critic only reweights or\nselects samples (BoN-flavored)\n— zero generator ∇Q",
            "IDQL · QVPO",
            rs.TEAL,
        ),
        (
            xs[2],
            yb,
            "6 · latent-noise RL / adjoint",
            "RL in the FROZEN generator's\nnoise space (critic on the\nnoise) → ∇ never enters the\ndenoiser",
            "DSRL ★ · Adjoint Matching",
            rs.BLUE,
        ),
    ]
    for x, y, t, b, m, c in P:
        _panel(ax, x, y, W, H, t, b, m, c)
    ax.text(
        0.3,
        9.75,
        "Getting the DDPG-style ∇Q into an ITERATIVE flow / diffusion generator — six families",
        fontsize=11.5,
        color=rs.INK,
    )
    ax.text(xs[2], 0.42, "★ DSRL = the precedent for our latent-actor idea", fontsize=7.8, color=rs.BLUE, va="top")
    rs.save(fig, path)
    plt.close(fig)


def fig_routes(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    # top pipeline
    _box(ax, 0.3, 4.5, 2.7, 1.4, "state\ns", fc="#eef4fb", fs=9.5)
    _box(ax, 3.7, 4.4, 3.5, 1.55, "one-step\nlatent actor h_φ\n z = h_φ(s, ε)", fc="#fdeee6", ec=rs.ORANGE, fs=8.6)
    _box(ax, 8.0, 4.4, 3.5, 1.55, "frozen decoder D\n a = D(z)\n (flow, iterative)", fc="#eafaea", ec=rs.GREEN, fs=8.6)
    _box(ax, 12.3, 4.5, 3.3, 1.4, "patch critic\n Q^π(s, a)", fc="#f9f9f7", ec=rs.PURPLE, fs=8.8)
    _arrow(ax, 3.0, 5.2, 3.7, 5.2)
    _arrow(ax, 7.2, 5.2, 8.0, 5.2)
    _arrow(ax, 11.5, 5.2, 12.3, 5.2)
    ax.text(8.6, 6.35, "∂a/∂z through D's ODE  =  the wall", fontsize=8.5, color=rs.RED)
    # three routes (bottom), well separated
    W, GX = 4.9, 0.6
    xs = [0.3, 0.3 + W + GX, 0.3 + 2 * (W + GX)]
    _panel(
        ax,
        xs[0],
        0.5,
        W,
        2.7,
        "A · latent critic  (DSRL)",
        "learn  Q̃(s,z) ≈ Q^π(s, D(z));\nDDPG:  max Q̃(s, h_φ) + BC\n→ ∇_z Q̃ is direct, no D",
        "most direct",
        rs.BLUE,
    )
    _panel(
        ax,
        xs[1],
        0.5,
        W,
        2.7,
        "B · 1-step decoder  (FQL)",
        "distill  D → D̂  (one step);\n∇_z Q(D̂(z)) = ∇_a Q · ∂D̂/∂z\n→ one cheap Jacobian",
        "exact decoder, distill cost",
        rs.GREEN,
    )
    _panel(
        ax,
        xs[2],
        0.5,
        W,
        2.7,
        "C · latent Q-score matching",
        "regress  h_φ → Π(z + η ∇_z Q̃)\n= cos / tangential (sphere-loss)\n→ stable, no ascent",
        "recommended",
        rs.ORANGE,
    )
    _arrow(ax, xs[0] + W / 2, 3.2, 4.9, 4.4, c=rs.BLUE, ls="--")
    _arrow(ax, xs[1] + W / 2, 3.2, 6.0, 4.4, c=rs.GREEN, ls="--")
    _arrow(ax, xs[2] + W / 2, 3.2, 5.4, 4.4, c=rs.ORANGE, ls="--")
    ax.text(
        0.3, 7.25, "Three ways to run DDPG on the latent actor without ∂D/∂z through the ODE", fontsize=11, color=rs.INK
    )
    rs.save(fig, path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2/figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_taxonomy(a.out / "fr_taxonomy.png")
    fig_routes(a.out / "fr_routes.png")
    print("wrote 2 figures ->", a.out)


if __name__ == "__main__":
    main()
