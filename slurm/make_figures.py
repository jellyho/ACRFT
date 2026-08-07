"""Regenerate the report's aggregate figures from raw eval JSONs, in the paper style
(plot_style — Seohong-Park-convention: white bg, y-grid, no top/right spines, deep palette).

Figures whose inputs are plain JSON are rebuilt here on every run, so the report's images
can never drift from the raw data:
  16_run_level.png  v11 demo-only forest (4 methods x 16 seeds, run-level delta + 95% t-CI)
  18_band_open.png  candidate-band opening across critic generations (band_width.json)
  15_autopsy.png    failure-stage decomposition, programmatic end_state rules
  20_final_forest.png  FINAL campaign forest — auto-updates as evals arrive

    uv run --no-sync python slurm/make_figures.py
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plot_style import PALETTE, apply  # noqa: E402

C = pathlib.Path("/scratch/jellyho/acrft")
P = C / "plots"
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 8: 2.365, 16: 2.131}


def _runs(pattern, mode):
    """[(delta, critic_sr, vla_sr)] per seed — one JSON per (run, seed); dedupe by seed suffix."""
    out = {}
    for f in sorted(C.glob(pattern)):
        d = json.loads(f.read_text())
        if mode not in d or "vla" not in d:
            continue
        seed = f.stem.split("_s")[-1]
        cs = np.mean([t["success"] for t in d[mode]["trials"]])
        vs = np.mean([t["success"] for t in d["vla"]["trials"]])
        out[seed] = (cs - vs, cs, vs)  # later batches overwrite earlier reruns of the same seed
    return list(out.values())


def forest(ax, rows, title):
    """rows: [(label, deltas array)] bottom-up; point cloud + mean +/- 95% t-CI."""
    ys = np.arange(len(rows))
    for y, (label, ds) in zip(ys, rows):
        ds = np.asarray(ds, float)
        color = PALETTE[y % len(PALETTE)]
        ax.scatter(ds, np.full_like(ds, y) + np.random.default_rng(0).uniform(-0.13, 0.13, ds.size),
                   s=22, alpha=0.45, color=color, linewidths=0)
        if len(ds) >= 2:
            m, se = ds.mean(), ds.std(ddof=1) / np.sqrt(len(ds))
            t = TCRIT.get(len(ds), 2.1)
            ax.errorbar([m], [y], xerr=[[t * se], [t * se]], fmt="D", color=color,
                        markersize=8, elinewidth=2.6, capsize=5, zorder=5)
    ax.axvline(0, color="#444444", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("run-level Δ success rate (method − in-job VLA)")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def fig_16_v11():
    rows = []
    for m in ["td", "qc", "iql", "aqc"]:
        mode = {"td": "critic", "qc": "critic", "iql": "critic", "aqc": "aqc"}[m]
        # 16 seeds per method, split across three submission batches (std/old/nseed prefixes)
        r = _runs(f"critic_runs/v11_std/{m}/rollout/*_s*.json", mode)
        if r:
            rows.append((f"{m.upper()}  (n={len(r)})", [x[0] for x in r]))
    if not rows:
        return
    apply()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    forest(ax, rows, "v11 demo-only")
    fig.savefig(P / "16_run_level.png")
    plt.close(fig)
    print("16_run_level.png")


def fig_20_final():
    arms = ["td_max", "td_soft", "td_aqcmax", "iql", "qc", "td_max_a101", "td_max_a201",
            "iql_a101", "iql_a201", "td_max_online", "iql_online", "td_max_demo", "iql_demo", "qc_demo"]
    rows = []
    for a in arms:
        r = _runs(f"critic_runs/final/{a}/rollout/f_s*.json", "critic")
        rows.append((f"{a}  (n={len(r)})", [x[0] for x in r]))
    apply()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    forest(ax, rows[::-1], "FINAL campaign")
    fig.savefig(P / "20_final_forest.png")
    plt.close(fig)
    print("20_final_forest.png")


def fig_18_band():
    src = C / "probes/band_width.json"
    if not src.exists():
        return
    data = json.loads(src.read_text())
    agg = {}
    for row in data:
        agg.setdefault(row["label"], []).append(row["band"])
    labels = list(agg)
    meds = [float(np.median(v)) for v in agg.values()]
    apply()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    xs = np.arange(len(labels))
    colors = [PALETTE[0] if v < 0.03 else PALETTE[2] for v in meds]
    ax.bar(xs, meds, color=colors, width=0.62)
    ax.set_xticks(xs, labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("candidate band  q99 − q01")
    ax.set_title("Candidate-band opening")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[0], label="closed (< 0.03)"),
                       Patch(color=PALETTE[2], label="opened (mixed-data critics)")], loc="upper left")
    fig.savefig(P / "18_band_open.png")
    plt.close(fig)
    print("18_band_open.png")


def _stage(t):
    """Programmatic failure-stage rules (same predicates as rollout.end_state)."""
    if t["success"]:
        return "success"
    e = t.get("end_state") or {}
    if e.get("machine_on"):
        return "machine_on_not_success"
    if e.get("placed"):
        return "placed_no_press"
    if e.get("grasped"):
        return "grasped_not_placed"
    return "no_grasp"


def fig_15_autopsy():
    cats = ["no_grasp", "grasped_not_placed", "placed_no_press", "machine_on_not_success"]
    groups = {
        "v11 TD critic": ("critic_runs/v11_std/td/rollout/nseed_s*.json", "critic"),
        "v11 IQL critic": ("critic_runs/v11_std/iql/rollout/nseed_s*.json", "critic"),
        "VLA (v11 in-job)": ("critic_runs/v11_std/td/rollout/nseed_s*.json", "vla"),
    }
    fracs = {}
    for g, (pat, mode) in groups.items():
        trials = []
        pat = pat.replace("nseed_s*", "*_s*")
        for f in sorted(C.glob(pat)):
            d = json.loads(f.read_text())
            if mode in d:
                trials += d[mode]["trials"]
        fails = [t for t in trials if not t["success"]]
        if not fails:
            continue
        fracs[g] = [np.mean([_stage(t) == c for t in fails]) for c in cats]
    if not fracs:
        return
    apply()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = np.arange(len(cats))
    w = 0.8 / len(fracs)
    for i, (g, v) in enumerate(fracs.items()):
        ax.bar(xs + i * w - 0.4 + w / 2, v, width=w * 0.92, label=g, color=PALETTE[i])
    ax.set_xticks(xs, ["no grasp", "grasped,\nnot placed", "placed,\nno press", "machine on,\nnot success"], fontsize=9.5)
    ax.set_ylabel("share of failures")
    ax.set_title("Failure stages")
    ax.legend()
    fig.savefig(P / "15_autopsy.png")
    plt.close(fig)
    print("15_autopsy.png")


def main():
    P.mkdir(exist_ok=True)
    fig_16_v11()
    fig_18_band()
    fig_15_autopsy()
    fig_20_final()


if __name__ == "__main__":
    main()
