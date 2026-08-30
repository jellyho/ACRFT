"""Training-progress figure for the cable-tie BC finetune.

Reads the live wandb console capture (no checkpoint is loaded, no policy is rolled
out) and plots the two scalars that say whether the run is healthy. Writes the raw
arrays next to the figure so it can be restyled without re-parsing the log.

    python plot_progress.py [path/to/output.log]
"""

import os
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(os.path.expanduser("~/dotfiles/claude/paper.mplstyle"))
sys.path.insert(0, os.path.expanduser("~/dotfiles/claude"))
from plot_palette import BLUE  # noqa: E402
from plot_palette import ORANGE_TEXT  # noqa: E402

HERE = pathlib.Path(__file__).parent
DEFAULT_LOG = pathlib.Path(
    "/NHNHOME/WORKSPACE/gwanwoo/gwanwoo/ACRFT/wandb/run-20260817_185512-l1gyxtbg/files/output.log"
)
LINE = re.compile(r"Step (\d+): bc_loss=([\d.]+), grad_norm=([\d.]+), loss=[\d.]+, param_norm=([\d.]+)")


def main() -> None:
    log = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    rows = [m.groups() for m in LINE.finditer(log.read_text(errors="replace").replace("\r", "\n"))]
    if not rows:
        raise SystemExit(f"no 'Step N: bc_loss=...' lines in {log}")
    step = np.array([int(r[0]) for r in rows])
    bc = np.array([float(r[1]) for r in rows])
    gn = np.array([float(r[2]) for r in rows])
    np.savez(HERE / "data" / "progress.npz", step=step, bc_loss=bc, grad_norm=gn)

    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    ax.plot(step, bc, color=BLUE, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("BC loss")
    # Range from the data, not from the 100k target - the target is stated in the caption, and
    # padding the axis out to it would leave most of the panel empty.
    ax.set_xlim(0, step.max())
    ax.set_yticks([0.08, 0.04, 0.02, 0.01, 0.007])
    ax.set_yticklabels(["0.08", "0.04", "0.02", "0.01", "0.007"])

    ax2 = ax.twinx()
    ax2.plot(step, gn, color=ORANGE_TEXT, lw=1.2, alpha=0.85)
    ax2.set_ylabel("gradient norm")
    ax2.set_ylim(0, max(gn) * 1.05)

    # Warmup ends at 1k; the LR is constant at 5e-5 from there on. Labelled in the caption, not here.
    ax.axvline(1_000, color="#999999", lw=0.9, ls=":")
    ax.legend(
        handles=[
            plt.Line2D([], [], color=BLUE, lw=1.8, label="BC loss (left)"),
            plt.Line2D([], [], color=ORANGE_TEXT, lw=1.2, label="grad norm (right)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    fig.savefig(HERE / "figs" / "progress.png", dpi=200)
    print(f"steps {step.min()}..{step.max()}  bc_loss {bc[0]:.4f} -> {bc[-1]:.4f}")


if __name__ == "__main__":
    main()
