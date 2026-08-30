"""YAM pi05 real-robot milestone figure: equal-interval milestone grid + mean-progress line.

Left: each milestone occupies an EQUAL vertical row (milestone 1 at the bottom -> 4 at top),
so runs are compared on identical footing. Within each row, a fixed-length light track marks
the 100% reference and a filled bar of length = P(reach >=m) shows the rate, in a single blue
hue that darkens with milestone (gradient). Numbers give the exact rate. Right: mean progress
(0-4) with SEM across checkpoints, H60 as an open diamond. Regenerates from
docs/reports/yam_pi05_progress_2026-08-10.json.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "scripts")
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
    "rel_s200_100k_h60": "100k H60",
}
base = np.array([76, 114, 176]) / 255  # seaborn deep blue


def shade(mi):  # milestone index 0..3: light -> full hue
    return tuple(1 - (1 - base) * (0.32 + 0.68 * mi / 3))


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})

# --- left: equal-interval milestone grid ---
n_runs = len(ORDER)
row_h = 0.66  # bar thickness within each unit-height milestone row
col_w = 1.0
track_w = 0.86  # full-rate reference length within a column
for ci, run in enumerate(ORDER):
    p = d["runs"][run]["prog"]
    n = len(p)
    ge = [sum(x >= m for x in p) / n for m in (1, 2, 3, 4)]
    x0 = ci * col_w
    for mi in range(4):
        y = mi + 0.5
        ax_l.add_patch(
            plt.Rectangle(
                (x0 - track_w / 2, y - row_h / 2),
                track_w,
                row_h,
                facecolor="#f0f0f0",
                edgecolor="#e2e2e2",
                lw=0.5,
                zorder=1,
            )
        )
        w = ge[mi] * track_w
        if w > 0:
            ax_l.add_patch(
                plt.Rectangle(
                    (x0 - track_w / 2, y - row_h / 2), w, row_h, facecolor=shade(mi), edgecolor="none", zorder=2
                )
            )
        inside = ge[mi] >= 0.34
        tx = x0 - track_w / 2 + (w - 0.08 if inside else w + 0.06)
        ax_l.text(
            tx,
            y,
            f"{ge[mi]:.1f}",
            ha="right" if inside else "left",
            va="center",
            fontsize=8.5,
            color="white" if inside else "#333333",
            zorder=3,
            path_effects=None if inside else [pe.withStroke(linewidth=1.8, foreground="white")],
        )

ax_l.axvline(3.5, color="#cccccc", lw=0.8, ls=":")
ax_l.set_xlim(-0.6, n_runs - 0.4)
ax_l.set_ylim(0, 4)
ax_l.set_xticks(range(n_runs), [LABEL[r] for r in ORDER], fontsize=9)
ax_l.set_yticks([mi + 0.5 for mi in range(4)], [f"milestone {mi + 1}" for mi in range(4)])
ax_l.set_title("Milestone attainment (equal rows)")
for s in ("top", "right", "left", "bottom"):
    ax_l.spines[s].set_visible(False)
ax_l.tick_params(length=0)

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
ax_r.legend(fontsize=8.5, loc="lower left")

fig.tight_layout()
out = pathlib.Path(".scratch/fig_yam_pi05_funnel.jpg")
fig.savefig(out, dpi=150, pil_kwargs={"quality": 86})
print("saved", out)
