"""QPILOTS-U steering sweep on the real robot -- the re-run.

Ten episodes per steering strength, episode indices shared across conditions, 30-step execution.
alpha=0 is the no-steering control collected inside this same run, which is the only fair anchor for
the rest of the sweep. Error bars are 95% t-CI over the 10 episodes.

Regenerates from .scratch/legoprog_qpilots.json, which holds the operator's per-episode values.
"""

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plot_style import GRAY
from plot_style import PALETTE
from plot_style import apply

ALPHAS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1]
RUN = 1  # the re-run (rows 62-71 of the operator's sheet)
SRC = pathlib.Path(".scratch/legoprog_qpilots.json")
OUT = pathlib.Path("hub_figs/fig_qpilots_rerun.png")


def _ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def main():
    d = json.loads(SRC.read_text())["conditions"]
    arm = lambda a: np.array(d["fixed_qpilot_0.000" if a == 0 else f"fixed_qpilot_{a:g}"][RUN], float)  # noqa: E731

    apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    x = np.arange(len(ALPHAS))
    m, e = zip(*[_ci(arm(a)) for a in ALPHAS], strict=True)

    ctl = m[0]
    ax.axhline(ctl, color=GRAY, lw=1.2, ls=":")
    ax.text(len(ALPHAS) - 0.62, ctl + 0.07, "no steering ($\\alpha{=}0$)", color=GRAY, ha="right", fontsize=9.5)
    ax.errorbar(x, m, yerr=e, fmt="o-", color=PALETTE[0], lw=1.9, ms=6)

    ax.set_xticks(x, [f"{a:g}" for a in ALPHAS])
    ax.set_xlim(-0.4, len(ALPHAS) - 0.6)
    ax.set_xlabel("steering strength  $\\alpha$")
    ax.set_ylabel("progress stage reached  (0-4)")
    ax.set_title("QPILOTS-U sweep, re-run")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}  (n=10 per point)")
    for a, mm, ee in zip(ALPHAS, m, e, strict=True):
        d_ = arm(a) - arm(0.0)
        md, ed = _ci(d_) if a else (0.0, 0.0)
        print(f"  alpha={a:<6g} {mm:.2f} +- {ee:.2f}   paired vs alpha=0: {md:+.2f} +- {ed:.2f}")


if __name__ == "__main__":
    main()
