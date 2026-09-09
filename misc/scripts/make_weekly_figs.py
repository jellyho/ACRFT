"""Background + result figures for the weekly presentation decks (house style).

The weekly rule: every result plot needs the background visualization that makes it presentable --
method schematics, architecture diagrams, pipeline maps -- all script-regenerable. This module owns
the deck-specific figures; report figures (30/31/32) live in slurm/make_figures.py and are reused.

    CACHE_DIR=/data5/jellyho/acrft_cache uv run python scripts/make_weekly_figs.py
"""

import json
import os
import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import report_style as rs

P = pathlib.Path(os.environ.get("CACHE_DIR", "/data5/jellyho/acrft_cache")) / "plots"
REPO = pathlib.Path(__file__).parent.parent


def _box(ax, x, y, w, h, text, fc="#f2f1ec", ec=rs.INK, fs=9):
    ax.add_patch(
        mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.3)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=rs.INK, linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1, c=rs.GRAY, lw=1.6, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": lw, "ls": ls})


def fig_40_af_concept():
    """BACKGROUND: what alpha-Flow changes -- instantaneous velocity (10 Euler steps) vs mean
    velocity (one jump), and why that matters for the RL actor update."""
    rs.use()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))

    for ax, title, n in [
        (axes[0], "pi0.5 as-is: instantaneous v(z,t), 10-step ODE", 10),
        (axes[1], "alpha-Flow: mean velocity u(z,r,t), one jump", 1),
    ]:
        # a wiggly flow path from noise (t=1) to action (t=0)
        t = np.linspace(0, 1, 200)
        path = 1.1 * (1 - t) * 0 + np.sin(2.2 * np.pi * t) * 0.25 * (1 - t) + t * 1.0
        ax.plot(t, path, color="#cccccc", lw=2.2, zorder=1)
        pts = np.linspace(0, 1, n + 1)
        py = np.interp(pts, t, path)
        if n > 1:
            for i in range(n):
                _arrow(ax, pts[i], py[i], pts[i + 1], py[i + 1], c=rs.BLUE, lw=1.6)
            ax.scatter(pts, py, s=18, color=rs.BLUE, zorder=5)
        else:
            _arrow(ax, 0, py[0], 1, py[-1], c=rs.ORANGE, lw=2.6)
            ax.scatter([0, 1], [py[0], py[-1]], s=26, color=rs.ORANGE, zorder=5)
        ax.scatter([0], [py[0]], s=60, color="k", zorder=6, marker="o")
        ax.annotate("noise z (t=1)", (0.0, py[0]), textcoords="offset points", xytext=(4, 10), fontsize=8)
        ax.annotate("action chunk (t=0)", (1.0, py[-1]), textcoords="offset points", xytext=(-92, -14), fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].text(
        0.5,
        -0.08,
        "every actor update pays 10 expert forwards",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=8.5,
        color=rs.RED,
    )
    axes[1].text(
        0.5,
        -0.08,
        "actor update = ONE forward (the offline-RL bottleneck removed)",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8.5,
        color=rs.GREEN,
    )
    fig.tight_layout()
    fig.savefig(P / "40_af_concept.png", dpi=160)
    plt.close(fig)


def fig_41_af_200k_curves():
    """RESULT: the 200k run's alpha curriculum and delta2, pulled from wandb (c4vy84yy)."""
    import wandb

    api = wandb.Api()
    r = api.run("RSS-PFT_RLLAB/yam-rlt/c4vy84yy")
    steps, alpha, d2 = [], [], []
    for k in ["alpha", "delta2"]:
        h = r.history(keys=[k], samples=800, pandas=False)
        if k == "alpha":
            steps = [row["_step"] for row in h]
            alpha = [row[k] for row in h]
        else:
            d2 = [(row["_step"], row[k]) for row in h]
    rs.use()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    ax = axes[0]
    ax.plot(steps, alpha, color=rs.BLUE, lw=1.8)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 1.3)
    for x, lab in [(57600, "leaves clamp (theory 57.6k)"), (142400, "hits floor")]:
        ax.axvline(x, color="0.6", ls=":", lw=1.1)
        ax.annotate(lab, (x, 0.5), rotation=90, fontsize=7.5, va="center", ha="right")
    ax.set_xlabel("step")
    ax.set_ylabel("α")
    ax.set_title("the curriculum, as measured (no incident)")
    ax = axes[1]
    s2, v2 = zip(*d2, strict=False)
    ax.plot(s2, v2, color=rs.GREEN, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("delta² (raw error)")
    ax.set_title("delta2: 0.052 -> 0.0026 (watch this, not the loss)")
    fig.tight_layout()
    fig.savefig(P / "41_af_200k_curves.png", dpi=160)
    plt.close(fig)


def fig_42_gate_bars():
    """RESULT: the one-step gate -- demo-MSE bars from the two results.json files (recomputed)."""
    rs.use()
    data = {}
    for tag, d in (("160k", "eval_onestep_160k"), ("200k", "eval_onestep_200k")):
        f = REPO / ".scratch" / d / "results.json"
        if f.exists():
            data[tag] = json.loads(f.read_text())["metrics"]
    names = [
        ("af_1step", "a-Flow\n1-step"),
        ("af_2step", "a-Flow\n2-step"),
        ("af_10step", "a-Flow\n10-step"),
        ("bc_10step", "BC baseline\n10-step"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(names))
    w = 0.36
    for j, (tag, m) in enumerate(data.items()):
        vals = [m[f"mse_gt/{k}"] for k, _ in names]
        cols = [rs.ORANGE if k == "af_1step" else rs.BLUE for k, _ in names]
        ax.bar(x + (j - 0.5) * w, vals, w, color=cols, alpha=0.55 + 0.45 * j, label=f"@{tag}")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in names], fontsize=8.5)
    ax.set_ylabel("demo-MSE (robot space)")
    ax.set_title("1-step gate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(P / "42_gate_bars.png", dpi=160)
    plt.close(fig)


def fig_43_fql_arch():
    """BACKGROUND: the FQL 4-expert MoT over one shared VLM prefix, with the staged recipe."""
    rs.use()
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    _box(ax, 0.3, 5.6, 3.4, 1.7, "VLM prefix\npaligemma 2b (FROZEN)\nimage+prompt -> KV", fc="#eef4fb")
    experts = [
        (4.6, 6.3, "flow expert u_th 300m\nFROZEN - distill target", "#eafaea", rs.GREEN),
        (4.6, 4.3, "one-step actor u_om 300m\nTRAIN - z->chunk, 1 forward", "#fdeee6", rs.ORANGE),
        (4.6, 2.3, "critic Q_phi 150m\nTRAIN - distributional HL-Gauss Q", "#f3eefc", rs.PURPLE),
    ]
    for x, y, txt, fc, ec in experts:
        _box(ax, x, y, 4.2, 1.6, txt, fc=fc, ec=ec, fs=8.6)
        _arrow(ax, 3.7, 6.4, x, y + 0.8)
    ax.text(3.6, 7.6, "prefix KV computed once, shared by all experts (the OOM lesson)", fontsize=8, color=rs.GRAY)
    _box(
        ax,
        9.6,
        5.9,
        4.1,
        1.5,
        "stage 1 - critic warmup\ny = MC return (no bootstrap)\nactor: distill only",
        fc="#f9f9f7",
        fs=8.2,
    )
    _box(
        ax,
        9.6,
        4.0,
        4.1,
        1.5,
        "stage 2 - actor-critic\ncritic: TD + MC floor\nactor: distill + Q-max",
        fc="#f9f9f7",
        fs=8.2,
    )
    _box(ax, 9.6, 2.1, 4.1, 1.5, "knob - critic->backbone grad\n{never, warmup, always}", fc="#f9f9f7", fs=8.2)
    _arrow(ax, 11.65, 5.9, 11.65, 5.5, c=rs.GRAY)
    _arrow(ax, 11.65, 4.0, 11.65, 3.6, c=rs.GRAY)
    ax.text(9.6, 7.6, "staged QC-FQL recipe (train-step verified, CPU + real backbone)", fontsize=9, color=rs.INK)
    ax.text(
        0.3,
        0.9,
        "alpha-Flow junction: replace the distill target (10-step ODE) with the 1-step forward -> teacher cost vanishes",
        fontsize=9,
        color=rs.ORANGE,
    )
    fig.tight_layout()
    fig.savefig(P / "43_fql_arch.png", dpi=160)
    plt.close(fig)


def main():
    P.mkdir(parents=True, exist_ok=True)
    fig_40_af_concept()
    fig_41_af_200k_curves()
    fig_42_gate_bars()
    fig_43_fql_arch()
    print("wrote 40-43 →", P)


if __name__ == "__main__":
    main()
