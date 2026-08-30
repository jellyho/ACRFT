"""Figures for the latent-actor-expert + patch-critic design note (house style, report_style).

Three scripted figures (English text — the report prose is bilingual, the figures are the shared asset):
  1. arch      — pi05 (VLM + frozen action expert) + a trainable latent-actor expert (MoT) + patch critic
  2. quadrant  — the risk x ceiling map: IQL / TD-BoN(EMaQ) / latent-actor-AC (the target quadrant)
  3. loop      — the meanflow-free improvement loop (forward-sample flow -> critic select -> distill)

    uv run python scripts/latent_actor_figs.py --out space_v2/figs
"""

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import report_style as rs


def _box(ax, x, y, w, h, text, *, fc="#f2f1ec", ec=rs.INK, tc=rs.INK, fs=9, lw=1.1):
    ax.add_patch(
        mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02", fc=fc, ec=ec, lw=lw)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, *, c=rs.GRAY, style="-|>", lw=1.5, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": style, "color": c, "lw": lw, "ls": ls})


def fig_arch(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    _box(ax, 0.15, 1.9, 2.0, 1.15, "obs\n3-cam + state", fc="#eef4fb", fs=8.5)
    _box(ax, 2.45, 1.8, 2.1, 1.35, "pi05 VLM\nbackbone", fc="#e8f1fc", ec=rs.BLUE, fs=9)
    _box(ax, 4.95, 1.8, 2.5, 1.35, "latent-actor expert\n★ trainable (MoT)", fc="#fdeee6", ec=rs.ORANGE, fs=8.3)
    _box(ax, 7.9, 1.8, 2.6, 1.35, "action expert (flow)\n❄ frozen = safe decoder", fc="#eafaea", ec=rs.GREEN, fs=8.3)
    _box(ax, 10.95, 1.85, 2.0, 1.25, "action chunk\n(H steps)", fc="#f2f1ec", fs=8.5)
    _arrow(ax, 2.2, 2.45, 2.45, 2.45)
    _arrow(ax, 4.6, 2.45, 4.95, 2.45)
    _arrow(ax, 7.5, 2.45, 7.9, 2.45)
    _arrow(ax, 10.55, 2.45, 10.95, 2.45)
    ax.text(7.7, 3.28, "z  (spherical, ‖z‖=1)", fontsize=8, color=rs.ORANGE, ha="center")
    ax.add_patch(mpatches.Circle((6.75, 3.3), 0.12, fill=False, ec=rs.ORANGE, lw=1.0))
    ax.plot([6.75], [3.42], marker="o", ms=3, color=rs.ORANGE)
    # patch critic below, reads state+chunk, sends improve signal to the latent actor
    _box(ax, 8.0, 0.15, 3.1, 1.15, "patch critic\nDINOv2 · ARQ · cost-to-goal  (Q^π)", fc="#f9f9f7", ec=rs.PURPLE, fs=8)
    _arrow(ax, 11.7, 1.85, 10.6, 1.32)  # chunk -> critic
    _arrow(ax, 8.0, 0.72, 6.2, 1.78, c=rs.RED, lw=1.7, ls="--")  # critic -> latent actor (improve)
    ax.text(6.7, 0.95, "improve (∇)", fontsize=8, color=rs.RED, ha="center")
    ax.text(
        0.15,
        3.9,
        "leash is architectural: improvement moves only the latent; the frozen decoder keeps every"
        " action in-support",
        fontsize=8.5,
        color=rs.GRAY,
    )
    ax.set_title("Latent-actor expert on pi05 + patch critic", loc="left")
    rs.save(fig, path)
    plt.close(fig)


def fig_quadrant(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axvspan(0.5, 1, ymin=0, ymax=0.5, color=rs.GREEN, alpha=0.07)
    ax.axhline(0.5, color=rs.GRID, lw=1)
    ax.axvline(0.5, color=rs.GRID, lw=1)
    pts = [
        ("IQL (expectile-V)", 0.30, 0.24, rs.BLUE, (8, 8)),
        ("on-policy V (SARSA)", 0.40, 0.14, rs.TEAL, (-14, -16)),
        ("TD-BoN / EMaQ (max)", 0.80, 0.83, rs.RED, (-6, 8)),
        ("latent-actor AC (target)", 0.82, 0.22, rs.GREEN, (-30, 12)),
    ]
    for name, x, y, c, off in pts:
        ax.scatter([x], [y], s=95, color=c, zorder=5, edgecolor="white", lw=1.3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off, fontsize=8.6, color=c)
    _arrow(ax, 0.34, 0.24, 0.78, 0.235, c=rs.GRAY, lw=1.2, ls=":")
    _arrow(ax, 0.80, 0.79, 0.815, 0.28, c=rs.GRAY, lw=1.2, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("expressiveness / ceiling  →")
    ax.set_ylabel("offline instability (risk)  →")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.97, 0.05, "safe + high-ceiling\n(target quadrant)", fontsize=9, color=rs.GREEN, ha="right")
    ax.set_title("The two-axis tension, and the target", loc="left")
    rs.save(fig, path)
    plt.close(fig)


def fig_loop(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    _box(
        ax,
        0.25,
        1.9,
        2.9,
        1.3,
        "frozen flow\nN noises → N chunks\n(forward-sample only)",
        fc="#eafaea",
        ec=rs.GREEN,
        fs=8,
    )
    _box(ax, 3.55, 1.9, 2.9, 1.3, "patch critic\nensemble-min Q^π\ncost-to-goal", fc="#f9f9f7", ec=rs.PURPLE, fs=8)
    _box(ax, 6.85, 1.9, 2.7, 1.3, "select\nbest-of-N /\nadvantage-weight", fc="#e8f1fc", ec=rs.BLUE, fs=8)
    _box(ax, 9.85, 1.9, 2.6, 1.3, "distill latent actor\n(Flow-DAgger)", fc="#fdeee6", ec=rs.ORANGE, fs=8)
    _arrow(ax, 3.15, 2.55, 3.55, 2.55)
    _arrow(ax, 6.45, 2.55, 6.85, 2.55)
    _arrow(ax, 9.55, 2.55, 9.85, 2.55)
    _arrow(ax, 11.15, 1.9, 11.15, 1.0, c=rs.GRAY)
    _arrow(ax, 11.15, 1.0, 1.7, 1.0, c=rs.GRAY)
    _arrow(ax, 1.7, 1.0, 1.7, 1.85, c=rs.GRAY)
    ax.text(
        6.4,
        0.72,
        "iterate: the latent actor proposes better noise → resample",
        fontsize=8.5,
        color=rs.GRAY,
        ha="center",
    )
    ax.text(
        6.4,
        4.05,
        "no meanflow · no ODE backprop  (the flow is only ever forward-sampled)",
        fontsize=9,
        color=rs.RED,
        ha="center",
    )
    ax.set_title("meanflow-free improvement loop (flow-matching pi05)", loc="left")
    rs.save(fig, path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2/figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_arch(a.out / "la_arch.png")
    fig_quadrant(a.out / "la_quadrant.png")
    fig_loop(a.out / "la_loop.png")
    print("wrote 3 figures ->", a.out)


if __name__ == "__main__":
    main()
