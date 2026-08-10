"""Nested milestone towers for the YAM pi05 real-robot evaluation.

One vertical slot per run; the four milestone bars overlap in one slot, bottom-aligned,
single hue light -> dark for milestone 1 -> 4. The visible band between level m and m+1
is the fraction of trials that stopped at exactly milestone m; the full height of the
lightest bar is P(>=1). Regenerates from docs/reports/yam_pi05_progress_2026-08-10.json.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "scripts")
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import report_style

report_style.use()

d = json.loads(pathlib.Path("docs/reports/yam_pi05_progress_2026-08-10.json").read_text())
ORDER = ["rel_s200_50k", "rel_s200_100k", "rel_s200_150k", "rel_s200_200k", "rel_s200_100k_h60"]
LABEL = {
    "rel_s200_50k": "50k",
    "rel_s200_100k": "100k",
    "rel_s200_150k": "150k",
    "rel_s200_200k": "200k",
    "rel_s200_100k_h60": "100k\nH60",
}
XPOS = {"rel_s200_50k": 0, "rel_s200_100k": 1, "rel_s200_150k": 2, "rel_s200_200k": 3, "rel_s200_100k_h60": 4.4}

base = np.array([76, 114, 176]) / 255  # seaborn deep blue


def shade(mi):  # milestone index 0..3: light -> full hue
    return tuple(1 - (1 - base) * (0.30 + 0.70 * mi / 3))


W = 0.62
fig, ax = plt.subplots(figsize=(6.4, 3.5))
for run in ORDER:
    p = d["runs"][run]["prog"]
    n = len(p)
    fr = [sum(x >= m for x in p) / n for m in (1, 2, 3, 4)]
    x = XPOS[run]
    for mi, f in enumerate(fr):
        if f > 0:
            ax.bar(x, f, width=W, color=shade(mi), zorder=2 + mi)
    # per-band numbers: value of level m at the top of its visible band
    lv = [*fr, 0.0]
    for mi in range(4):
        f, nxt = lv[mi], lv[mi + 1]
        if f == 0:
            if mi == 0 or lv[mi - 1] > 0:  # topmost zero level only
                ax.text(x, 0.035, "0", ha="center", va="bottom", fontsize=8, color="#999999", zorder=6)
            break
        if f - nxt >= 0.07:
            ax.text(
                x,
                (f + nxt) / 2,
                f"{f:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )
        elif mi == 3 or lv[mi + 1] == 0:
            ax.text(
                x,
                f + 0.025,
                f"{f:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

ax.set_xticks([XPOS[r] for r in ORDER], [LABEL[r] for r in ORDER])
ax.axvline(3.75, color="#dddddd", lw=0.8, ls=":")
ax.set_ylim(0, 1.06)
ax.set_ylabel("fraction of trials")
ax.set_title("Milestone attainment")
ax.legend(
    handles=[Patch(color=shade(mi), label=rf"$\geq$ milestone {mi + 1}") for mi in range(4)],
    fontsize=8.5,
    loc="upper right",
    ncols=1,
)
fig.tight_layout()
out = pathlib.Path(".scratch/fig_yam_pi05_funnel.jpg")
fig.savefig(out, dpi=150, pil_kwargs={"quality": 86})
print("saved", out)
