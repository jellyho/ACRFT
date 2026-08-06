"""Run-level Δ(method − in-job vla) chart for the v11 fair-comparison campaign.

One dot per evaluation run (one seed, 30 paired scenes); the black bar is the mean over runs with
its 95% t-CI whiskers. A method "beats vla" when the CI clears zero. Dot color names the scene
pool, so pool-specific behaviour is visible instead of hiding in the pool.

Panel 2 shows the generational flip that motivated the campaign: the SAME selection rule (IQL
joint argmax) on the SAME old scene pool, v6 checkpoint (200k, raw actions) vs v11 (100k,
z-scored actions, current trainer).

    uv run --no-sync python slurm/make_run_level_plot.py   # -> $CACHE_DIR/plots/16_run_level.png
"""

import glob
import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
METHODS = [("td", "critic"), ("iql", "critic"), ("qc", "critic"), ("aqc", "aqc")]
POOLS = {
    "std": ("scenes 3000-3300", "#2563eb"),
    "old": ("scenes 0-300", "#dc2626"),
    "nseed": ("scenes 4000-4700", "#16a34a"),
}
TCRIT = {
    k: v
    for k, v in zip(
        range(2, 40),
        [
            12.7,
            4.30,
            3.18,
            2.78,
            2.57,
            2.45,
            2.36,
            2.31,
            2.26,
            2.23,
            2.20,
            2.18,
            2.16,
            2.14,
            2.13,
            2.12,
            2.11,
            2.10,
            2.09,
            2.09,
            2.08,
            2.07,
            2.07,
            2.06,
            2.06,
            2.06,
            2.05,
            2.05,
            2.05,
            2.04,
            2.04,
            2.04,
            2.03,
            2.03,
            2.03,
            2.03,
            2.03,
            2.02,
        ],
        strict=False,
    )
}


def run_deltas(method, mode):
    out = []
    for prefix, (pool_label, color) in POOLS.items():
        for f in sorted(glob.glob(str(C / f"critic_runs/v11_std/{method}/rollout/{prefix}_s*.json"))):
            j = json.loads(pathlib.Path(f).read_text())
            if mode not in j or "vla" not in j:
                continue
            a = np.mean([t["success"] for t in j[mode]["trials"]])
            v = np.mean([t["success"] for t in j["vla"]["trials"]])
            out.append((a - v, pool_label, color))
    return out


def main():
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True, dpi=200, gridspec_kw={"width_ratios": [2.2, 1]}
    )

    ax = axes[0]
    xticks, xlabels = [], []
    rng = np.random.default_rng(0)
    for i, (m, mode) in enumerate(METHODS):
        ds = run_deltas(m, mode)
        if not ds:
            xticks.append(i)
            xlabels.append(f"{m}\n(no data yet)")
            continue
        vals = np.array([d[0] for d in ds])
        for d, pool_label, color in ds:
            ax.scatter(
                i + rng.uniform(-0.13, 0.13),
                d,
                s=42,
                color=color,
                alpha=0.75,
                zorder=3,
                label=pool_label if (pool_label, "dot") not in getattr(ax, "_seen", set()) else None,
            )
            seen = getattr(ax, "_seen", set())
            seen.add((pool_label, "dot"))
            ax._seen = seen
        n = len(vals)
        mmean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        ci = TCRIT.get(n, 2.0) * se
        ax.errorbar(i, mmean, yerr=ci, color="black", capsize=6, lw=2.4, zorder=4)
        ax.scatter([i], [mmean], marker="_", s=600, color="black", zorder=5)
        xticks.append(i)
        xlabels.append(f"{m}\nΔ̄={mmean:+.3f}\nn={n} runs")
    ax.axhline(0, color="#888", lw=1.2, ls="--")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Δ success rate  (method − in-job vla, per 30-scene run)")
    ax.set_title(
        "v11 fair checkpoints: run-level Δ vs paired vla\n(dot = one seed-run · black = mean ± 95% t-CI · beat-vla criterion: CI above 0)"
    )
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, title="scene pool", title_fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # ---- panel 2: generational flip, old pool, IQL joint argmax
    ax = axes[1]
    gens = []
    v6 = []
    for f in sorted(glob.glob(str(C / "critic_runs/v6_iql/iql_e70/rollout/replay_s*.json"))):
        j = json.loads(pathlib.Path(f).read_text())
        v6.append(
            np.mean([t["success"] for t in j["critic"]["trials"]]) - np.mean([t["success"] for t in j["vla"]["trials"]])
        )
    v11 = [d for d, lbl, _ in run_deltas("iql", "critic") if lbl == "scenes 0-300"]
    gens = [
        ("v6 iql_e70\n200k · raw actions\n(yesterday's replay)", v6, "#9ca3af"),
        ("v11 iql\n100k · z-scored actions\n(fair set)", v11, "#2563eb"),
    ]
    for i, (name, vals, color) in enumerate(gens):
        vals = np.array(vals)
        if len(vals) == 0:
            continue
        ax.scatter(
            np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=48, color=color, alpha=0.85, zorder=3
        )
        n = len(vals)
        se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        ax.errorbar(i, vals.mean(), yerr=TCRIT.get(n, 2.8) * se, color="black", capsize=6, lw=2.2, zorder=4)
        ax.text(i, vals.mean() + 0.012, f"{vals.mean():+.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="#888", lw=1.2, ls="--")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([g[0] for g in gens], fontsize=8)
    ax.set_ylabel("Δ success rate vs in-job vla")
    ax.set_title("Same rule, same scenes (pool 0-300):\ncheckpoint generation flips the sign")
    ax.grid(axis="y", alpha=0.25)

    out = C / "plots/16_run_level.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
