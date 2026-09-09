"""Generate the patch-critic Method figures — scripted, house style (report_style.py).

Four figures, all produced from this script so they can never drift from the design/data:
  1. observation    — frozen DINOv2 dense-patch grid pipeline (schematic)
  2. mask           — the block-causal attention mask (DATA: the actual mask matrix)
  3. hlgauss        — HL-Gauss projection of a scalar target onto 51 atoms (DATA)
  4. results        — RoboCasa 365 success rate, actor vs BoN vs adaptive-K (DATA, honest)

    uv run python scripts/patch_critic_figs.py --out <dir>
"""

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import report_style as rs

# ---- the honest RoboCasa 365 numbers (25 rollouts / task) -------------------
TASKS = ["OpenDrawer", "CoffeeSetupMug", "PnP→Microwave", "PnP→Counter"]
SR = {  # success rate (%)
    "actor (no selection)": [44, 12, 24, 24],
    "best-of-N (dist-V)": [32, 4, 4, 12],
    "adaptive-K (dist-V)": [16, 0, 8, 24],
}


def _box(ax, x, y, w, h, text, *, fc="#f2f1ec", ec=rs.INK, tc=rs.INK, fs=9, lw=1.1):
    ax.add_patch(
        mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02", fc=fc, ec=ec, lw=lw)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, *, c=rs.GRAY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.4})


def fig_observation(path):
    rs.use()
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for k, (yy, cam) in enumerate([(2.75, "left"), (1.55, "wrist"), (0.35, "right")]):
        _box(ax, 0.1, yy, 1.5, 0.9, f"cam: {cam}\n224×224×3", fc="#eef4fb", fs=8)
        _arrow(ax, 1.65, yy + 0.45, 2.35, 1.9 + (k - 1) * 0.05)
    _box(ax, 2.4, 1.35, 2.0, 1.15, "frozen\nDINOv2 ViT-S/14\n❄  stop-grad", fc="#e8f1fc", ec=rs.BLUE, fs=8.5)
    _arrow(ax, 4.45, 1.9, 5.05, 1.9)
    _box(ax, 5.1, 1.35, 1.7, 1.15, "16×16 patches\nper camera", fc="#f2f1ec", fs=8.5)
    _arrow(ax, 6.85, 1.9, 7.45, 1.9)
    _box(ax, 7.5, 1.35, 1.6, 1.15, "2×2 avg-pool\n→ 8×8 / cam", fc="#f2f1ec", fs=8.5)
    ax.text(9.15, 1.9, "→", ha="center", va="center", fontsize=15, color=rs.GRAY)
    _box(ax, 8.35, 0.05, 1.6, 1.0, "ϕ(s):\n192 tokens\n× 384-d", fc="#eafaea", ec=rs.GREEN, fs=8.5)
    ax.set_title("Observation — frozen DINOv2 dense-patch grid", loc="left")
    rs.save(fig, path)
    plt.close(fig)


def fig_mask(path):
    """The actual block-causal attention mask (context + M=8 action macro-groups)."""
    rs.use()
    M = 8
    labels = ["ctx"] + [f"a{j + 1}" for j in range(M)]  # 1 context block + 8 action macro-groups
    n = len(labels)
    mask = np.zeros((n, n))
    mask[:, 0] = 1  # everyone attends to context
    for q in range(1, n):
        for k in range(1, n):
            if k <= q:  # action token q sees action groups 1..q (block-causal)
                mask[q, k] = 1
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    ax.imshow(mask, cmap="Blues", vmin=0, vmax=1.6, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("key (attended-to)")
    ax.set_ylabel("query")
    for q in range(n):
        for k in range(n):
            if mask[q, k]:
                ax.text(k, q, "•", ha="center", va="center", color=rs.BLUE, fontsize=9)
    # value read-out annotation: each action token j yields the horizon-h value
    ax.text(
        n - 0.5,
        0.2,
        "value read at each\naction token → Q₂,Q₄,…,Q₁₆",
        ha="right",
        va="center",
        color=rs.ORANGE,
        fontsize=8,
    )
    ax.grid(visible=False)
    ax.set_title("Block-causal mask → one value per horizon", loc="left")
    rs.save(fig, path)
    plt.close(fig)


def fig_hlgauss(path):
    """HL-Gauss: a scalar target y softened onto 51 atoms over [0,1]."""
    rs.use()
    m, y, sigma = 51, 0.62, 0.045
    z = np.linspace(0, 1, m)
    delta = z[1] - z[0]
    from math import erf
    from math import sqrt

    def cdf(v):
        return 0.5 * (1 + erf((v - y) / (sqrt(2) * sigma)))

    p = np.array([cdf(zi + delta / 2) - cdf(zi - delta / 2) for zi in z])
    p /= p.sum()
    fig, ax = plt.subplots(figsize=rs.FIGSIZE)
    ax.bar(z, p, width=delta * 0.9, color=rs.BLUE, alpha=0.85, label="projected atoms  Φ(y)")
    xf = np.linspace(0, 1, 400)
    g = np.exp(-((xf - y) ** 2) / (2 * sigma**2))
    g = g / g.max() * p.max()
    ax.plot(xf, g, color=rs.ORANGE, lw=2, label="target  N(y, σ²)")
    rs.baseline(ax, 0, "")
    ax.axvline(y, color=rs.GRAY, ls=":", lw=1.3)
    ax.text(y + 0.01, p.max() * 0.98, "  y", color=rs.GRAY, fontsize=9, va="top")
    ax.set_xlabel("value support  z ∈ [0, 1]  (51 atoms)")
    ax.set_ylabel("probability")
    ax.set_title("HL-Gauss soft target")
    ax.legend()
    rs.save(fig, path)
    plt.close(fig)


def fig_results(path):
    rs.use()
    x = np.arange(len(TASKS))
    w = 0.26
    colors = [rs.GRAY, rs.BLUE, rs.ORANGE]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (name, vals) in enumerate(SR.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[i], alpha=0.9 if i else 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=9)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 50)
    ax.set_title("RoboCasa 365 — value selection does not yet beat the actor", loc="left")
    ax.legend(ncol=3, loc="upper right")
    rs.save(fig, path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2/figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_observation(a.out / "pcm_observation.png")
    fig_mask(a.out / "pcm_mask.png")
    fig_hlgauss(a.out / "pcm_hlgauss.png")
    fig_results(a.out / "pcm_results.png")
    print("wrote 4 figures ->", a.out)


if __name__ == "__main__":
    main()
