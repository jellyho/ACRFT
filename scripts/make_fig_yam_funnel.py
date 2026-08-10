"""YAM pi05 real-robot milestone figure: stacked funnel towers + mean-progress line.

Left: one stacked bar per run. Segments stack from the bottom, milestone 1 (lightest) up
to milestone 4 (darkest); each segment's height = fraction of trials that stopped at exactly
that milestone, so the total tower height = P(reach >=1) and the top edge of each segment
sits at the cumulative P(>=m). Single blue hue, light->dark. Right: mean progress (0-4) with
SEM across checkpoints, the H60 variant as an open diamond. Regenerates from
docs/reports/yam_pi05_progress_2026-08-10.json.
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


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})

# --- left: stacked funnel towers ---
W = 0.62
for run in ORDER:
    p = d["runs"][run]["prog"]
    n = len(p)
    ge = [sum(x >= m for x in p) / n for m in (1, 2, 3, 4)]  # P(>=m)
    bands = [ge[0] - ge[1], ge[1] - ge[2], ge[2] - ge[3], ge[3]]  # exactly-at-m, bottom->top
    x = XPOS[run]
    bottom = 0.0
    for mi, h in enumerate(bands):
        if h > 0:
            ax_l.bar(x, h, width=W, bottom=bottom, color=shade(mi), zorder=2)
        # label cumulative P(>=m+1) at the TOP edge of this segment, if that milestone was reached
        top = bottom + h
        if ge[mi] > 0 and h >= 0.06:
            ax_l.text(
                x,
                top - h / 2,
                f"{ge[mi]:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )
        bottom = top
    # annotate the highest milestone reached, at the tower top
    hi = max((m for m in range(4) if ge[m] > 0), default=-1)
    if hi >= 0 and bands[hi] < 0.06:
        ax_l.text(
            x,
            ge[0] + 0.02,
            f"{ge[hi]:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

ax_l.set_xticks([XPOS[r] for r in ORDER], [LABEL[r] for r in ORDER])
ax_l.axvline(3.75, color="#dddddd", lw=0.8, ls=":")
ax_l.set_ylim(0, 1.06)
ax_l.set_ylabel("fraction of trials")
ax_l.set_title("Milestone funnel")
ax_l.legend(
    handles=[Patch(color=shade(mi), label=f"milestone {mi + 1}") for mi in range(4)],
    fontsize=8,
    loc="upper right",
    title="stops at",
    title_fontsize=8,
)

# --- right: mean progress line ---
main = {d["runs"][r]["ckpt"]: d["runs"][r]["prog"] for r in ORDER if d["runs"][r]["h"] == 16}
cks = sorted(main)
xs = [c / 1000 for c in cks]
mean = [np.mean(main[c]) for c in cks]
sem = [np.std(main[c], ddof=1) / np.sqrt(len(main[c])) for c in cks]
h60 = d["runs"]["rel_s200_100k_h60"]["prog"]
ax_r.errorbar(xs, mean, yerr=sem, fmt="o-", color=shade(2), ms=5, capsize=3, label="H16")
ax_r.errorbar(
    [100],
    [np.mean(h60)],
    yerr=[np.std(h60, ddof=1) / np.sqrt(len(h60))],
    fmt="D",
    color=shade(2),
    markerfacecolor="white",
    ms=6,
    capsize=3,
    label="100k H60",
)
ax_r.set_xlabel("checkpoint (k steps)")
ax_r.set_ylabel("mean progress (0 to 4)")
ax_r.set_xticks(xs)
ax_r.set_ylim(0, 4)
ax_r.axhline(4, ls=":", color="#bbb", lw=1)
ax_r.set_title("Mean progress")
ax_r.legend(fontsize=8.5, loc="upper right")

fig.tight_layout()
out = pathlib.Path(".scratch/fig_yam_pi05_funnel.jpg")
fig.savefig(out, dpi=150, pil_kwargs={"quality": 86})
print("saved", out)
