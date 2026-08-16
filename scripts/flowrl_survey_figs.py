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


def _box(ax, x, y, w, h, title, body, *, fc="#f2f1ec", ec=rs.INK, tfs=9, bfs=7.6):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(x + 0.12, y + h - 0.16, title, fontsize=tfs, fontweight="bold", color=ec, va="top")
    ax.text(x + 0.12, y + h - 0.5, body, fontsize=bfs, color=rs.INK, va="top")


def _arrow(ax, x0, y0, x1, y1, *, c=rs.GRAY, lw=1.6, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": lw, "ls": ls})


def fig_taxonomy(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    groups = [
        (0.2, 4.9, "1 · backprop through chain", "differentiate Q back through all T\ndenoising steps — honest but O(T),\nunstable → T tiny", "Diffusion-QL", rs.RED),
        (4.15, 4.9, "2 · distill to 1-step, then RL", "collapse the sampler to ONE\ndifferentiable map → cheap reparam ∇Q,\nno BPTT, no iterative test-time", "FQL · Consistency-AC · CPQL · OFQL", rs.GREEN),
        (8.1, 4.9, "3 · score ↔ ∇Q matching", "never sample; set the generator's\nscore/velocity field to (a function of) ∇_a Q", "QSM · SRPO · DAC", rs.ORANGE),
        (0.2, 1.6, "4 · denoising-as-MDP (PG)", "each step = MDP action (Gaussian);\npolicy-gradient over steps, no ∇ through Q", "DPPO · ReinFlow · Flow-GRPO", rs.PURPLE),
        (4.15, 1.6, "5 · weighted resample / regress", "critic only reweights/selects samples\n(BoN-flavored) — zero generator ∇Q", "IDQL · QVPO", rs.TEAL),
        (8.1, 1.6, "6 · latent-noise RL / adjoint", "RL in the FROZEN generator's noise\nspace (critic on noise) — ∇ never enters\nthe denoiser", "DSRL ★ · Adjoint Matching", rs.BLUE),
    ]
    for x, y, t, b, meths, c in groups:
        _box(ax, x, y, 3.7, 2.0, t, b, ec=c)
        ax.text(x + 0.12, y + 0.22, meths, fontsize=7.2, style="italic", color=c, va="bottom")
    ax.text(0.2, 6.95, "How to get the DDPG-style ∇Q into an ITERATIVE flow/diffusion generator — 6 families",
            fontsize=10.5, color=rs.INK)
    ax.text(8.1, 1.35, "★ DSRL = our idea's precedent", fontsize=7.5, color=rs.BLUE, va="top")
    rs.save(fig, path)
    plt.close(fig)


def fig_routes(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    _box(ax, 0.2, 3.1, 2.4, 1.1, "state s", "", ec=rs.INK)
    _box(ax, 3.0, 3.0, 2.7, 1.25, "one-step\nlatent actor h_φ", "z = h_φ(s, ε)", ec=rs.ORANGE)
    _box(ax, 6.1, 3.0, 2.7, 1.25, "frozen decoder D", "action a = D(z)\n(flow, iterative)", ec=rs.GREEN)
    _box(ax, 9.2, 3.05, 2.5, 1.15, "patch critic", "Q^π(s, a)", ec=rs.PURPLE)
    _arrow(ax, 2.6, 3.65, 3.0, 3.65)
    _arrow(ax, 5.7, 3.65, 6.1, 3.65)
    _arrow(ax, 8.8, 3.65, 9.2, 3.6)
    ax.text(6.4, 4.55, "∂a/∂z through D's ODE = the wall", fontsize=8, color=rs.RED)
    # three routes into the actor, none through D's ODE
    _box(ax, 3.0, 0.4, 3.0, 1.35, "A · latent critic (DSRL)", "learn Q̃(s,z) ≈ Q^π(s,D(z));\nDDPG: max Q̃(s,h_φ) + BC\n→ ∇_z Q̃ direct, no D", ec=rs.BLUE)
    _box(ax, 6.35, 0.4, 3.0, 1.35, "B · 1-step decoder (FQL)", "distill D → D̂ (1-step);\n∇_z Q(D̂(z)) = ∇_a Q·∂D̂/∂z\n→ one cheap Jacobian", ec=rs.GREEN)
    _box(ax, 9.7, 0.4, 3.0, 1.35, "C · latent Q-score match", "regress h_φ → z+η∇_z Q̃\n(= cos/tangential, our sphere-loss)\n→ stable, no ascent", ec=rs.ORANGE)
    _arrow(ax, 4.5, 1.75, 4.4, 3.0, c=rs.BLUE, ls="--")
    _arrow(ax, 7.85, 1.75, 7.4, 2.99, c=rs.GREEN, ls="--")
    _arrow(ax, 10.9, 1.75, 4.7, 3.0, c=rs.ORANGE, ls="--")
    ax.text(0.2, 5.25, "Three ways to run DDPG on the latent actor without ∂D/∂z through the ODE", fontsize=10,
            color=rs.INK)
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
