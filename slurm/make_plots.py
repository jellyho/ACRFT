"""Render every headline result as a PNG figure.

Reads only what the experiments wrote (rollout/*.json, diag.json, vbias*.json, pfx_curve.json)
and writes figures to $CACHE_DIR/plots/. Layout rules: constrained_layout everywhere, legends
outside the axes or direct-labelled, and nothing rotated past 0 degrees - if labels would collide,
the figure gets taller instead.

    uv run slurm/make_plots.py
"""

import json
import math
import os
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
OUT = ROOT / "plots"
MODES = ["vla", "rand", "bon", "prefix", "critic"]
BANDS = ["5-15", "15-30", "30-60", "60-120", "120-250", "250-600"]
C = {"td": "#c0563f", "iql": "#2f855a", "g999": "#3b78ae", "duel": "#8168b3", "vla": "#666a5e", "warn": "#b9892e"}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
    }
)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rollouts(sweep, prefer=None):
    out = {}
    for d in sorted((ROOT / "critic_runs" / sweep).glob("*")):
        files = sorted((d / "rollout").glob("*.json"))
        if not files:
            continue
        pick = None
        if prefer:
            for f in files:
                if prefer in f.name:
                    pick = f
        pick = pick or files[-1]
        out[d.name] = json.loads(pick.read_text())
    return out


def pool(runs):
    tot = {m: [0, 0] for m in MODES}
    for d in runs.values():
        for m in MODES:
            if m in d:
                tot[m][0] += d[m]["successes"]
                tot[m][1] += d[m]["num_trials"]
    return tot


def mcnemar(d):
    cv = {t["trial"]: t["success"] for t in d["critic"]["trials"]}
    vv = {t["trial"]: t["success"] for t in d["vla"]["trials"]}
    b = sum(1 for t in cv if cv[t] and not vv.get(t))
    c = sum(1 for t in cv if not cv[t] and vv.get(t))
    n = b + c
    return 1.0 if n == 0 else min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2 / 2**n)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    v3 = rollouts("v3_fixedmask")
    v6 = rollouts("v6_iql")
    v8 = rollouts("v8_iql2", prefer="params_150000")
    vb3 = json.loads((ROOT / "vbias.json").read_text())
    vb6 = json.loads((ROOT / "vbias_v6_iql.json").read_text())
    vb8 = json.loads((ROOT / "vbias_v8_iql2.json").read_text())
    pfx = json.loads((ROOT / "pfx_curve.json").read_text())

    # ---------------------------------------------------------------- 1. success by mode, by family
    groups = [
        ("TD (v3, 14 runs)", C["td"], pool(v3)),
        ("IQL γ=.99 (v6, 4 runs)", C["iql"], pool(v6)),
        ("IQL γ=.999+ (v8, 3 runs @150k)", C["g999"], pool({k: v for k, v in v8.items() if k.startswith("g")})),
        ("IQL dueling (duel_e70 @150k)", C["duel"], pool({k: v for k, v in v8.items() if k.startswith("duel")})),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.6), constrained_layout=True)
    xs = np.arange(len(MODES))
    for gi, (name, col, tot) in enumerate(groups):
        off = (gi - (len(groups) - 1) / 2) * 0.17
        for i, m in enumerate(MODES):
            k, n = tot[m]
            if not n:
                continue
            p = k / n
            lo, hi = wilson(k, n)
            ax.errorbar(
                xs[i] + off,
                p,
                yerr=[[p - lo], [hi - p]],
                fmt="o",
                ms=6,
                lw=1.6,
                capsize=3,
                color=col,
                label=name if i == 0 else None,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["vla\n(no critic)", "rand\n(random cand)", "bon\n(pick cand)", "prefix\n(pick commit)", "critic\n(pick both)"]
    )
    ax.set_ylabel("success rate")
    ax.set_ylim(0.30, 0.95)
    ax.set_title("Rollout success by evaluation mode — dots with 95% Wilson intervals")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.savefig(OUT / "1_success_by_mode.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 2. value bias b(d)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True, sharex=True)
    x = np.arange(len(BANDS))
    a = axs[0]
    a.plot(x, [r["b"] for r in vb3["base"]["rows"]], "o-", color=C["td"], label="TD base")
    for r, ls in (("iql_e50", "-"), ("iql_e70", "--"), ("iql_e90", "-."), ("iql_e95", ":")):
        a.plot(
            x,
            [q["b"] for q in vb6[r]["rows"]],
            marker="o",
            ls=ls,
            color=C["iql"],
            alpha=0.55 + 0.15 * ["-", "--", "-.", ":"].index(ls),
            label=r.replace("iql_", "τ="),
        )
    a.axhline(0, color="k", lw=0.8, ls=(0, (2, 2)))
    a.set_title("γ = 0.99:  TD inflates, IQL collapses,\nhigher τ re-inflates (dose-response)")
    a.set_ylabel("value bias  b(d) = V̂ − γ^d")
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    b = axs[1]
    for r, col, ls in (
        ("g999_e50", C["g999"], "-"),
        ("g999_e70", C["g999"], "--"),
        ("g9995_e70", C["warn"], "-"),
        ("duel_e50", C["duel"], "-"),
        ("duel_e70", C["duel"], "--"),
    ):
        b.plot(x, [q["b"] for q in vb8[r]["rows"]], marker="o", ls=ls, color=col, label=r)
    b.axhline(0, color="k", lw=0.8, ls=(0, (2, 2)))
    b.set_ylim(a.get_ylim())  # same scale so flatness is legible
    b.set_title("γ = 0.999 / 0.9995 and dueling:\nessentially exact at every distance")
    b.legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax_ in axs:
        ax_.set_xticks(x)
        ax_.set_xticklabels(BANDS)
        ax_.set_xlabel("steps to goal")
    fig.savefig(OUT / "2_value_bias.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 3. expectile dose-response
    runs6 = [r for r in ("iql_e50", "iql_e70", "iql_e90", "iql_e95") if r in v6]
    taus = [{"iql_e50": 0.5, "iql_e70": 0.7, "iql_e90": 0.9, "iql_e95": 0.95}[r] for r in runs6]
    meanb = [float(np.mean([q["b"] for q in vb6[r]["rows"]])) for r in runs6]  # offline uses all four
    diffs, dlo, dhi = [], [], []
    for r in runs6:
        d = v6[r]
        pc, pv = d["critic"]["success_rate"], d["vla"]["success_rate"]
        diffs.append(pc - pv)
        se = math.sqrt(pc * (1 - pc) / 30 + pv * (1 - pv) / 30)
        dlo.append(1.96 * se)
        dhi.append(1.96 * se)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    axs[0].plot(taus, meanb, "o-", color=C["iql"])
    axs[0].set_xlabel("expectile τ")
    axs[0].set_ylabel("mean value bias  b")
    axs[0].set_title("Offline: bias returns as τ → max")
    axs[1].errorbar(taus, diffs, yerr=[dlo, dhi], fmt="o-", color=C["iql"], capsize=3)
    axs[1].axhline(0, color="k", lw=0.8, ls=(0, (2, 2)))
    axs[1].set_xlabel("expectile τ")
    axs[1].set_ylabel("critic − vla  (success rate)")
    axs[1].set_title("Rollout: harm returns as τ → max")
    fig.suptitle("The expectile dose-response that pins the inflation on the arg-max", fontsize=11)
    fig.savefig(OUT / "3_dose_response.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 4. per-prefix target ratio
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    hs = pfx["pfx"]
    cmap = plt.get_cmap("plasma")
    for i, bkt in enumerate(pfx["buckets"]):
        col = cmap(0.12 + 0.75 * i / max(len(pfx["buckets"]) - 1, 1))
        ax.plot(hs, bkt["ratio"], "o-", color=col, lw=1.8)
        ax.annotate(
            f"{bkt['lo']}–{bkt['hi']}",
            (hs[-1], bkt["ratio"][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=col,
            fontsize=8.5,
            va="center",
        )
    ax.axhline(1.0, color="k", lw=0.9, ls=(0, (2, 2)))
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5])
    ax.set_yticklabels(["1.0", "1.5", "2", "3", "5"])
    ax.set_xticks(hs)
    ax.set_xlabel("prefix length h (steps committed)")
    ax.set_ylabel("target / truth   y_h ÷ γ^d")
    ax.set_xlim(hs[0] - 0.5, hs[-1] + 3.2)
    ax.set_title(
        "TD per-prefix targets vs truth, by distance to goal\n"
        "(each line slopes down in h and far bands float above 1 → arg-max prefers short commits)"
    )
    fig.savefig(OUT / "4_prefix_targets.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 5. per-run critic - vla
    rows = []
    for _fam, col, runs in (("TD", C["td"], v3), ("IQL", C["iql"], v6), ("v8", C["g999"], v8)):
        for r, d in runs.items():
            if "critic" not in d or "vla" not in d:
                continue
            col_ = C["duel"] if r.startswith("duel") else col
            rows.append((f"{r}", col_, d["critic"]["success_rate"] - d["vla"]["success_rate"], mcnemar(d)))
    rows.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(rows) + 1.6), constrained_layout=True)
    ys = np.arange(len(rows))
    for y, (_name, col, diff, p) in zip(ys, rows, strict=True):
        ax.plot([0, diff], [y, y], color=col, lw=2, alpha=0.35)
        ax.plot(diff, y, "o", ms=6.5, color=col, mfc=col if p < 0.05 else "white", mew=1.6)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("critic − vla success rate  (30 paired trials; filled dot = McNemar p < 0.05)")
    ax.set_title("Per-run harm of the full critic, every rollout so far")
    fig.savefig(OUT / "5_per_run_harm.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 6. success vs training steps (v8)
    ckpts = ["params_50000", "params_100000", "params_150000"]
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 6.6), constrained_layout=True, sharey=True)
    for ax_, run in zip(axs.ravel(), ["g999_e50", "g999_e70", "g9995_e70", "duel_e70"], strict=True):
        pts = {m: [] for m in ("vla", "prefix", "critic")}
        steps = []
        for ck in ckpts:
            f = ROOT / "critic_runs/v8_iql2" / run / "rollout" / f"{ck}_seed0.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            steps.append(int(ck.split("_")[1]) // 1000)
            for m, series in pts.items():
                series.append(d[m]["success_rate"])
        ax_.plot(steps, pts["vla"], "s--", color=C["vla"], label="vla")
        ax_.plot(steps, pts["prefix"], "o-", color=C["g999"], label="prefix")
        ax_.plot(steps, pts["critic"], "o-", color=C["td"], label="critic")
        ax_.set_title(run, fontsize=10)
        ax_.set_xticks(steps)
        ax_.set_xticklabels([f"{s}k" for s in steps])
        ax_.set_ylim(0.3, 1.0)
    axs[0][0].legend(frameon=False, fontsize=9)
    axs[1][0].set_xlabel("training steps")
    axs[1][1].set_xlabel("training steps")
    axs[0][0].set_ylabel("success rate")
    axs[1][0].set_ylabel("success rate")
    fig.suptitle("v8: success vs critic training steps (30 trials per point — noisy, read trends only)")
    fig.savefig(OUT / "6_success_vs_steps.png")
    plt.close(fig)

    for f in sorted(OUT.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
