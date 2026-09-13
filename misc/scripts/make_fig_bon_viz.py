"""The same timeline visualization, for a BoN-selection rollout.

BoN commits the full chunk every replan, so commit length is constant — the thing that varies
is WHICH candidate the critic picked and HOW confident it was (the winning Q). Each trial is a
timeline; every segment is one replan (length = full chunk), colored by the chosen candidate's
Q in a single-hue gradient (dark = high Q = critic confident). Reads the per-step trace
eval_critic writes (step / value / n_exec / prefix). Regenerates from a rollout JSON.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "scripts")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import report_style

report_style.use()

ap = argparse.ArgumentParser()
ap.add_argument("--rollout", type=pathlib.Path, default=pathlib.Path(".scratch/rollout_rltphi.json"))
ap.add_argument("--arm", default="bon")
ap.add_argument("--max-trials", type=int, default=8)
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/fig_bon_viz.jpg"))
a = ap.parse_args()

tr = json.loads(a.rollout.read_text())[a.arm]["trace"]
trials, cur = [], []
for r in tr:
    if r["step"] == 1 and cur:
        trials.append(cur)
        cur = []
    cur.append(r)
if cur:
    trials.append(cur)

# collapse each trial to its replan segments (a new replan = value changes)
segs_per_trial = []
for tt in trials:
    segs, last = [], None
    for r in tt:
        sig = (r["n_exec"], round(r["value"], 5))
        if sig != last:
            segs.append((r["n_exec"], r["value"]))
            last = sig
    segs_per_trial.append(segs)

allq = np.array([q for segs in segs_per_trial for _, q in segs])
lo, hi = np.percentile(allq, 5), np.percentile(allq, 95)
norm = Normalize(vmin=lo, vmax=hi)
base = np.array([76, 114, 176]) / 255


def col(q):
    t = 0.2 + 0.8 * float(np.clip(norm(q), 0, 1))
    return tuple(1 - (1 - base) * t)


show = segs_per_trial[: a.max_trials]
fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(show) + 1.2))
qmed = float(np.median(allq))
for ti, segs in enumerate(show):
    y = len(show) - 1 - ti
    x = 0
    for ne, q in segs:
        ax.barh(y, ne, left=x, height=0.62, color=col(q), edgecolor="white", linewidth=0.6)
        if q < qmed:  # replans where the best candidate was still below-median (critic unsure)
            ax.plot(x + ne / 2, y + 0.42, "v", color="#c44e52", ms=3.5, zorder=3)
        x += ne
    ax.text(-6, y, f"trial {ti}", ha="right", va="center", fontsize=8.5, color="#555")

ax.set_ylim(-0.6, len(show) - 0.4)
ax.set_xlim(-40, max(300, ax.get_xlim()[1]))
ax.set_yticks([])
ax.set_xlabel("environment step")
ax.set_title("BoN selection timeline")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", [col(lo), col(hi)])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02)
cb.set_label("chosen candidate Q", fontsize=9)
ax.plot([], [], "v", color="#c44e52", ms=5, label="best cand below median Q")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(a.out, dpi=150, pil_kwargs={"quality": 86})
print(f"saved {a.out}  (replans total {sum(len(s) for s in segs_per_trial)}, Q range {lo:.3f}..{hi:.3f})")
