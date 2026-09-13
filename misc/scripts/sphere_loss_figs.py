"""Figures for the spherical-latent loss note (house style, report_style).

1. tangent   — on the unit circle: cosine loss's gradient is purely tangential (Riemannian);
               MSE-on-raw adds a wasted radial component.
2. es        — best-of-N / softmax over latent samples is a zeroth-order (ES) estimate of the
               latent DPG direction; beta->0 collapses to argmax (cosine-to-z*).

  uv run python scripts/sphere_loss_figs.py --out space_v2/figs
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import report_style as rs


def _unit(a):
    return np.array([np.cos(a), np.sin(a)])


def fig_tangent(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=rs.GRAY, lw=1.2)
    au, az = np.deg2rad(38), np.deg2rad(78)
    u, z = _unit(au), _unit(az)
    # vectors from origin
    ax.annotate("", xy=u, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": rs.INK, "lw": 1.6})
    ax.annotate("", xy=z, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": rs.GREEN, "lw": 1.6})
    ax.text(u[0] * 1.08, u[1] * 1.08, "û", fontsize=13, color=rs.INK)
    ax.text(z[0] * 1.08, z[1] * 1.08, "z*", fontsize=13, color=rs.GREEN)
    # tangent direction at û (unit), toward z*
    tan = z - (u @ z) * u
    tan = tan / np.linalg.norm(tan)
    scl = 0.5
    # cosine / Riemannian gradient: purely tangential
    ax.annotate("", xy=u + scl * tan, xytext=u, arrowprops={"arrowstyle": "-|>", "color": rs.ORANGE, "lw": 2.6})
    # MSE extra: radial component along û (wasted)
    ax.annotate("", xy=u + 0.32 * u, xytext=u, arrowprops={"arrowstyle": "-|>", "color": rs.RED, "lw": 2.2, "ls": "--"})
    # tangent line (faint)
    ax.plot(
        [u[0] - 0.7 * tan[0], u[0] + 0.7 * tan[0]],
        [u[1] - 0.7 * tan[1], u[1] + 0.7 * tan[1]],
        color=rs.GRID,
        lw=1,
        zorder=0,
    )
    ax.text(
        u[0] + scl * tan[0] + 0.02,
        u[1] + scl * tan[1] + 0.03,
        "cosine grad\n= tangential (Riemannian)",
        fontsize=8.5,
        color=rs.ORANGE,
    )
    ax.text(u[0] + 0.34 * u[0] + 0.03, u[1] + 0.34 * u[1], "MSE-on-raw:\nradial (wasted)", fontsize=8.5, color=rs.RED)
    ax.set_xlim(-1.25, 1.55)
    ax.set_ylim(-0.35, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Sphere geometry: cosine = pure tangential, MSE wastes the radial", loc="left", fontsize=10.5)
    rs.save(fig, path)
    plt.close(fig)


def fig_es(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    au = np.deg2rad(30)
    u = _unit(au)
    # N samples along the arc, Q rising toward z* (62 deg)
    angs = np.deg2rad(np.array([24, 32, 40, 47, 53, 58, 62, 66]))
    zstar = np.deg2rad(62)
    Q = np.exp(-((angs - zstar) ** 2) / (2 * np.deg2rad(12) ** 2))
    th = np.linspace(np.deg2rad(18), np.deg2rad(74), 200)
    ax.plot(np.cos(th), np.sin(th), color=rs.GRAY, lw=1.3)
    pts = np.stack([np.cos(angs), np.sin(angs)], 1)
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=50 + 300 * Q, c=Q, cmap="viridis", zorder=4, edgecolor="white", lw=0.9)
    beta = 0.25
    w = np.exp(Q / beta)
    w /= w.sum()
    zbar = (w[:, None] * pts).sum(0)
    zbar /= np.linalg.norm(zbar)
    istar = int(np.argmax(Q))
    # current point û (black), the estimated update arrow to z̄ (green)
    ax.scatter([u[0]], [u[1]], s=90, color=rs.INK, zorder=6)
    ax.annotate("", xy=zbar * 1.005, xytext=u, arrowprops={"arrowstyle": "-|>", "color": rs.GREEN, "lw": 2.8})
    ax.scatter([pts[istar, 0]], [pts[istar, 1]], s=170, facecolor="none", edgecolor=rs.ORANGE, lw=2.4, zorder=5)
    ax.text(u[0] - 0.02, u[1] - 0.11, "û  (current)", fontsize=9, color=rs.INK, ha="center")
    ax.text(0.37, 1.17, "softmax-weighted z̄  ≈  DPG direction", fontsize=9, color=rs.GREEN)
    ax.text(pts[istar, 0] + 0.03, pts[istar, 1] + 0.02, "z* = argmax\n(β→0 limit)", fontsize=8.8, color=rs.ORANGE)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("critic value Q", fontsize=8.5)
    ax.set_xlim(0.35, 1.15)
    ax.set_ylim(0.35, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Best-of-N in latent space = zeroth-order (ES) estimate of the DPG gradient", loc="left", fontsize=10)
    rs.save(fig, path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2/figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_tangent(a.out / "sl_tangent.png")
    fig_es(a.out / "sl_es.png")
    print("wrote 2 figures ->", a.out)


if __name__ == "__main__":
    main()
