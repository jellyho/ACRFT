"""Plot the checkpoint-sweep success rates written by run_eval.sh (summary.csv).

uv run examples/robocasa/plot_eval.py --task PrepareCoffee
uv run examples/robocasa/plot_eval.py --summary eval/.../summary.csv --out /tmp/eval.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=str, default=None, help="Task name (derives the summary.csv path).")
    ap.add_argument("--exp-suffix", type=str, default="run")
    ap.add_argument("--summary", type=Path, default=None, help="Path to summary.csv (overrides --task).")
    ap.add_argument("--out", type=Path, default=None, help="Output image path (default: next to summary.csv).")
    args = ap.parse_args()

    if args.summary is not None:
        summary = args.summary
    elif args.task is not None:
        cfg = f"pi05_robocasa_{args.task}"
        summary = _REPO_ROOT / "eval" / cfg / f"{args.task}_{args.exp_suffix}" / "summary.csv"
    else:
        raise SystemExit("Pass --task or --summary")
    if not summary.exists():
        raise SystemExit(f"No summary.csv at {summary}")

    steps, rates = [], []
    with open(summary) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            rates.append(float(row["success_rate"]) * 100.0)
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    rates = [rates[i] for i in order]

    task = args.task or summary.parent.parent.name.replace("pi05_robocasa_", "")
    best = max(range(len(rates)), key=lambda i: rates[i])

    accent, ink, muted, grid = "#2f6df0", "#1a1a1a", "#8a8f98", "#e6e8eb"
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.plot(steps, rates, "-o", color=accent, lw=2, ms=6, mfc="white", mec=accent, zorder=3)
    # highlight the best checkpoint
    ax.scatter([steps[best]], [rates[best]], s=140, facecolor=accent, edgecolor="white", zorder=4, lw=1.5)
    ax.annotate(
        f"best: {rates[best]:.0f}% @ {steps[best]:,}",
        (steps[best], rates[best]),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
        color=ink,
        weight="bold",
    )
    for x, y in zip(steps, rates, strict=True):
        ax.annotate(
            f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5, color=muted
        )

    ax.set_title(f"RoboCasa · {task} — success rate vs checkpoint", fontsize=12, weight="bold", color=ink, pad=12)
    ax.set_xlabel("training step", fontsize=10, color=muted)
    ax.set_ylabel("success rate (%)", fontsize=10, color=muted)
    ax.set_ylim(0, max(100, max(rates) + 12))
    ax.grid(visible=True, color=grid, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    n = len(Path(summary).read_text().splitlines()) - 1
    ax.text(
        0.99,
        -0.16,
        f"{n} checkpoints · 50 trials each · fixed seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=muted,
    )
    fig.tight_layout()

    out = args.out or summary.with_suffix(".png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
