"""Visualize adaptive chunking from a rollout trace: when did the policy re-plan, and how far
did it commit each time?

Each trial is a horizontal timeline of env steps; every replan is a segment whose length = the
committed steps (n_exec) and whose color = that commit length in a single-hue gradient (short =
looked again soon = more adaptive; long = committed the full chunk). Reads the per-step trace
that eval_critic writes (step / value / n_exec / prefix). Regenerates from a rollout JSON.
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "scripts")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import report_style

report_style.use()

ap = argparse.ArgumentParser()
ap.add_argument("--rollout", type=pathlib.Path, default=pathlib.Path(".scratch/rollout_mbacv_tau13.json"))
ap.add_argument("--arm", default="mbacv")
ap.add_argument("--max-trials", type=int, default=8)
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/fig_adaptiveness.jpg"))
a = ap.parse_args()

tr = json.loads(a.rollout.read_text())[a.arm]["trace"]
# Split the flat trace into trials wherever step resets to 1, then into replan segments.
trials = []
cur = []
for r in tr:
    if r["step"] == 1 and cur:
        trials.append(cur)
        cur = []
    cur.append(r)
if cur:
    trials.append(cur)

H = 16  # macro horizon (full chunk)
base = np.array([76, 114, 176]) / 255


def shade(nexec):  # 4..16 -> light..dark (short commit = light = more adaptive)
    t = 0.25 + 0.75 * (nexec / H)
    return tuple(1 - (1 - base) * t)


show = trials[: a.max_trials]
fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(show) + 1.2))
all_commits = []
for ti, tt in enumerate(show):
    y = len(show) - 1 - ti
    x = 0
    last = None
    for r in tt:
        sig = (r["n_exec"], round(r["value"], 4))
        if sig != last:  # a new replan
            ne = r["n_exec"]
            all_commits.append(ne)
            ax.barh(y, ne, left=x, height=0.62, color=shade(ne), edgecolor="white", linewidth=0.6)
            if ne <= 8:  # mark the adaptive early cuts
                ax.plot(x + ne, y, "v", color="#c44e52", ms=4, zorder=3)
            x += ne
            last = sig
    ax.text(-6, y, f"trial {ti}", ha="right", va="center", fontsize=8.5, color="#555")

ax.set_ylim(-0.6, len(show) - 0.4)
ax.set_xlim(-40, max(300, ax.get_xlim()[1]))
ax.set_yticks([])
ax.set_xlabel("environment step")
ax.set_title("Adaptive commit timeline")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
hist = dict(sorted(collections.Counter(all_commits).items()))
tot = sum(hist.values())
handles = [Patch(color=shade(k), label=f"commit {k}  ({100 * v / tot:.0f}%)") for k, v in hist.items()]
handles.append(plt.Line2D([], [], marker="v", color="#c44e52", ls="", ms=5, label="early cut (≤8)"))
ax.legend(handles=handles, fontsize=8, loc="lower right", ncols=2)
fig.tight_layout()
fig.savefig(a.out, dpi=150, pil_kwargs={"quality": 86})
print(f"saved {a.out}  (commit histogram {hist})")
