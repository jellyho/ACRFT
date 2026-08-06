"""Failure-taxonomy figures from autopsy rollout JSONs (stage_at / end_state logs).

Three panels, one story: where trials die, per policy mode.
  1. Stage survival funnel  - fraction of trials reaching each task stage (partial progress)
  2. Outcome composition    - stacked failure categories per mode, pressed_no_success split into
                              sub-modes (mug displaced vs retreat failure) where end_state exists
  3. Stage timing           - distribution of the step at which each stage was first reached

Reads every autopsy-style JSON it can find and groups trials by mode; rerun after any new batch.

    uv run --no-sync python slurm/make_autopsy_plots.py     # -> $CACHE_DIR/plots/15_autopsy.png
"""

import glob
import json
import os
import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
# Cohorts keep comparisons honest: modes are drawn together ONLY when they share the scene set
# (same seeds -> identical kitchens, paired per trial) and, for critic-driven modes, the same
# critic checkpoint. Cross-cohort comparison is the reader's job, clearly labelled.
COHORTS = [
    (
        "scenes 2000-2300 · critic = iql_e70 (v6, 200k)",
        [
            ("vla", C / "critic_runs/v6_iql/iql_e70/rollout/autopsy_s*.json"),
            ("vla", C / "critic_runs/v6_iql/iql_e70/rollout/vla_end_s*.json"),
            ("critic", C / "critic_runs/v6_iql/iql_e70/rollout/cr_autopsy_s*.json"),
        ],
    ),
    (
        "scenes 2400-2700 · critic = v10 e70_aqc (100k)",
        [
            ("vla", C / "critic_runs/v10_aqc/e70_aqc/rollout/tri_s*.json"),
            ("critic", C / "critic_runs/v10_aqc/e70_aqc/rollout/tri_s*.json"),
            ("aqc", C / "critic_runs/v10_aqc/e70_aqc/rollout/tri_s*.json"),
        ],
    ),
]
STAGE_KEYS = ["grasped", "placed", "machine_on"]
STAGE_LABELS = ["grasped\n(mug in hand)", "placed\n(under dispenser)", "machine_on\n(button pressed)", "success"]
CAT_ORDER = [
    "no_grasp",
    "grasp_only",
    "placed_no_press",
    "pressed_noS: displaced",
    "pressed_noS: retreat",
    "pressed_noS: (unlogged)",
    "success",
]
CAT_COLORS = {
    "no_grasp": "#9ca3af",
    "grasp_only": "#f59e0b",
    "placed_no_press": "#d97706",
    "pressed_noS: displaced": "#dc2626",
    "pressed_noS: retreat": "#7c3aed",
    "pressed_noS: (unlogged)": "#f3a5a5",
    "success": "#16a34a",
}


def classify(t):
    sa = t.get("stage_at", {}) or {}
    if t["success"]:
        return "success"
    if "grasped" not in sa:
        return "no_grasp"
    if "placed" not in sa:
        return "grasp_only"
    if "machine_on" not in sa:
        return "placed_no_press"
    e = t.get("end_state", {}) or {}
    if not e:
        return "pressed_noS: (unlogged)"
    if e.get("placed") is False:
        return "pressed_noS: displaced"
    if e.get("grip_obj_dist", 9) < 0.25 or e.get("grip_button_far") is False:
        return "pressed_noS: retreat"
    return "pressed_noS: (unlogged)"


def load_cohort(patterns):
    modes = {}
    for mode, pat in patterns:
        seen = set()
        for f in sorted(glob.glob(str(pat))):
            j = json.loads(pathlib.Path(f).read_text())
            trials = j.get(mode, {}).get("trials")
            if not trials or any("stage_at" not in t for t in trials):
                continue
            if (mode, f) in seen:
                continue
            seen.add((mode, f))
            modes.setdefault(mode, []).extend(trials)
    return modes


def main():
    cohorts = [(name, load_cohort(p)) for name, p in COHORTS]
    cohorts = [(n, m) for n, m in cohorts if m]
    if not cohorts:
        raise SystemExit("no stage-logged autopsy JSONs found yet")
    fig, rows = plt.subplots(len(cohorts), 3, figsize=(17, 5.4 * len(cohorts)), constrained_layout=True, dpi=200)
    rows = np.atleast_2d(rows)
    for ci, (cname, modes) in enumerate(cohorts):
        order = [m for m in ("vla", "critic", "aqc") if m in modes]
        print(cname, {m: len(t) for m, t in modes.items()})
        draw_cohort(rows[ci], cname, modes, order)
    fig.suptitle(
        "PrepareCoffee failure taxonomy — env-predicate stage logs; each row is one paired cohort "
        "(same scenes, same critic checkpoint)",
        fontsize=13,
    )
    out = C / "plots/15_autopsy.png"
    fig.savefig(out)
    print(f"saved {out}")


def draw_cohort(axes, cname, modes, order):
    # ---- 1. survival funnel
    ax = axes[0]
    colors = {"vla": "#6b7280", "critic": "#dc2626", "aqc": "#2563eb"}
    for m in order:
        tr = modes[m]
        surv = [np.mean([1.0 if k in (t.get("stage_at") or {}) else 0.0 for t in tr]) for k in STAGE_KEYS] + [
            np.mean([t["success"] for t in tr])
        ]
        ax.plot(range(4), surv, "o-", lw=2.2, ms=7, color=colors[m], label=f"{m} (n={len(tr)})")
        for x, y in enumerate(surv):
            ax.text(x, y + 0.012 + 0.02 * order.index(m), f"{y:.2f}", ha="center", fontsize=8, color=colors[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.set_ylabel("fraction of trials reaching stage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage survival funnel (higher = further progress)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # ---- 2. outcome composition
    ax = axes[1]
    for i, m in enumerate(order):
        c = Counter(classify(t) for t in modes[m])
        n = len(modes[m])
        bottom = 0.0
        for cat in CAT_ORDER:
            v = c.get(cat, 0) / n
            if v == 0:
                continue
            ax.bar(
                i,
                v,
                bottom=bottom,
                color=CAT_COLORS[cat],
                width=0.6,
                label=cat if i == 0 or cat not in [classify(t) for t in modes[order[0]]] else None,
            )
            if v > 0.03:
                ax.text(
                    i,
                    bottom + v / 2,
                    f"{c.get(cat, 0)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cat != "success" else "black",
                )
            bottom += v
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c]) for c in CAT_ORDER]
    ax.legend(
        handles, CAT_ORDER, fontsize=7.5, frameon=True, framealpha=0.9, loc="center left", bbox_to_anchor=(1.0, 0.5)
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("fraction of trials")
    ax.set_title("Outcome composition (counts printed in segments)")

    # ---- 3. stage timing
    ax = axes[2]
    width = 0.25
    for i, m in enumerate(order):
        for k, key in enumerate(STAGE_KEYS):
            vals = [(t.get("stage_at") or {}).get(key) for t in modes[m]]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            pos = k + (i - (len(order) - 1) / 2) * width
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=width * 0.85,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black"),
            )
            bp["boxes"][0].set_facecolor(colors[m])
            bp["boxes"][0].set_alpha(0.65)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["grasped", "placed", "machine_on"], fontsize=10)
    ax.set_ylabel("env step when stage first reached")
    ax.set_title("Stage timing (box color = mode, as in panel 1)")
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.65) for m in order]
    ax.legend(handles, order, frameon=False, fontsize=8)

    axes[0].annotate(cname, xy=(0, 1.10), xycoords="axes fraction", fontsize=11, fontweight="bold")


if __name__ == "__main__":
    main()
