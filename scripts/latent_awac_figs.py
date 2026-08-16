"""Figures for the latent-space AWAC note (house style, report_style).

  1. shift  — the improvement moves from deploy-time (BoN) into train-time (AWR baked into the actor).
  2. beta   — effective sample size of the exp(A/beta) weights vs beta: the usable band between
              argmax-collapse (EMaQ) and mean/BC (no improvement). Raw vs standardized advantages.

    uv run python scripts/latent_awac_figs.py --out space_v2/figs
"""

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import report_style as rs


def _box(ax, x, y, w, h, text, *, fc="#f2f1ec", ec=rs.INK, tc=rs.INK, fs=8.5, lw=1.1):
    ax.add_patch(
        mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02", fc=fc, ec=ec, lw=lw)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, *, c=rs.GRAY, lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": lw})


def fig_shift(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    # BoN (deploy-time)
    ax.text(0.2, 4.95, "Best-of-N  —  improvement at DEPLOY (every step, N× cost, actor unchanged)",
            fontsize=9.5, color=rs.RED)
    _box(ax, 0.3, 3.4, 2.3, 1.0, "actor\nsample N latents", fc="#fdeee6", ec=rs.ORANGE)
    _box(ax, 3.1, 3.4, 2.2, 1.0, "frozen D\n→ N chunks", fc="#eafaea", ec=rs.GREEN)
    _box(ax, 5.8, 3.4, 2.2, 1.0, "critic Q\nscore", fc="#f9f9f7", ec=rs.PURPLE)
    _box(ax, 8.5, 3.4, 2.0, 1.0, "argmax\n(BoN)", fc="#e8f1fc", ec=rs.BLUE)
    _box(ax, 11.0, 3.4, 2.0, 1.0, "execute", fc="#f2f1ec")
    for x0, x1 in [(2.6, 3.1), (5.3, 5.8), (8.0, 8.5), (10.5, 11.0)]:
        _arrow(ax, x0, 3.9, x1, 3.9)
    # AWR (train-time)
    ax.text(0.2, 2.35, "AWR  —  improvement at TRAIN (baked into the actor; deploy = 1 sample)",
            fontsize=9.5, color=rs.GREEN)
    _box(ax, 0.3, 0.7, 2.3, 1.0, "sampled / logged\n(s, z, a)", fc="#eef4fb")
    _box(ax, 3.1, 0.7, 2.6, 1.0, "advantage\nA = Q^π(D(z)) − V", fc="#f9f9f7", ec=rs.PURPLE)
    _box(ax, 6.2, 0.7, 3.0, 1.0, "weighted flow-matching\ne^{A/β} · ‖v_φ − (z−ε)‖²", fc="#fdeee6", ec=rs.ORANGE)
    _box(ax, 9.8, 0.7, 2.2, 1.0, "improved\nactor q_φ", fc="#fdeee6", ec=rs.ORANGE, lw=1.6)
    _box(ax, 12.5, 0.7, 2.2, 1.0, "deploy:\n1 sample", fc="#eafaea", ec=rs.GREEN)
    for x0, x1 in [(2.6, 3.1), (5.7, 6.2), (9.2, 9.8), (12.0, 12.5)]:
        _arrow(ax, x0, 1.2, x1, 1.2)
    ax.set_title("Move the improvement from deploy-time BoN into the actor (AWR)", loc="left")
    rs.save(fig, path)
    plt.close(fig)


def fig_beta(path):
    rs.use()
    # a fixed set of standardized advantages (N=16), sweep beta -> effective sample size of exp(A/beta)
    rng = np.random.default_rng(0)
    A = np.sort(rng.standard_normal(16))
    A = (A - A.mean()) / A.std()
    betas = np.logspace(-1.2, 1.4, 200)
    ess = []
    for b in betas:
        w = np.exp(A / b)
        w /= w.sum()
        ess.append(1.0 / np.sum(w**2))  # effective sample size in [1, N]
    ess = np.array(ess) / len(A)  # normalize to [1/N, 1]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhspan(0.0, 0.18, color=rs.RED, alpha=0.06)
    ax.axhspan(0.82, 1.0, color=rs.GRAY, alpha=0.08)
    ax.plot(betas, ess, color=rs.BLUE, lw=2.2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("advantage temperature  β  (log)")
    ax.set_ylabel("effective sample fraction  ESS / N")
    ax.text(0.09, 0.10, "β→0: collapse to argmax\n= greedy / EMaQ (high variance)", fontsize=8.5, color=rs.RED)
    ax.text(3.0, 0.90, "β→∞: flatten to mean\n= BC (no improvement)", fontsize=8.5, color=rs.GRAY, ha="left")
    ax.text(0.62, 0.5, "usable\nAWR band", fontsize=9, color=rs.GREEN, ha="center")
    ax.set_title("The AWR temperature: usable band between argmax-collapse and BC-flatten", loc="left", fontsize=10)
    rs.save(fig, path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2/figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_shift(a.out / "aw_shift.png")
    fig_beta(a.out / "aw_beta.png")
    print("wrote 2 figures ->", a.out)


if __name__ == "__main__":
    main()
