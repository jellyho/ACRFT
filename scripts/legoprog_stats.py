"""Run-level statistics for every LEGOPROG robot condition, against the CORRECT control.

Source of record: .scratch/legoprog_v3.xlsx (block 1 has EIGHT columns -- the two bc30 columns are the
ones that matter and the ones the earlier analyses left out). Every critic condition serves the h30 BC
checkpoint (src/openpi/extraction/serving.py BC_CKPT = yam_bc_s300_h30_successonly) and executes the
whole 30-step chunk (patch_critic_policy.py: n_exec = min(policy_horizon, critic_horizon) = 30; the
`fixed` critics have macro_group_size=30 so even adaptive mode has exactly one prefix). So the
method-only-diff control is bc30_ex_30: same checkpoint, same 30 executed steps, no critic.

Comparing against bc50_ex_10 instead -- which the earlier analyses did -- changes TWO things at once
(checkpoint h50->h30 AND execution length 10->30) and picks the peak of a six-point sweep as the
baseline. The decomposition below shows which of the two carries the difference.

Everything is recomputed from the sheet on every run; nothing here is typed in by hand.
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

R = pathlib.Path(__file__).resolve().parents[1]
BLOCKS = {
    "bc": (1, [0, 3, 6, 9, 12, 15, 18, 21]),
    "adaptive": (13, [0, 3, 6, 9, 12, 15]),
    "select": (25, [0, 3, 6]),
    "steer": (37, [0, 3, 6, 9, 12, 15]),
}
CONTROL = "bc30_ex_30"


def welch(v, c):
    d = v.mean() - c.mean()
    se = np.sqrt(v.var(ddof=1) / len(v) + c.var(ddof=1) / len(c))
    num = (v.var(ddof=1) / len(v) + c.var(ddof=1) / len(c)) ** 2
    den = (v.var(ddof=1) / len(v)) ** 2 / (len(v) - 1) + (c.var(ddof=1) / len(c)) ** 2 / (len(c) - 1)
    tc = stats.t.ppf(0.975, num / den)
    return d, d - tc * se, d + tc * se, float(stats.ttest_ind(v, c, equal_var=False).pvalue)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, ps[i] * (len(ps) - rank))
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=pathlib.Path, default=R / ".scratch/legoprog_v3.xlsx")
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/legoprog_stats.json")
    ap.add_argument("--fig", type=pathlib.Path, default=R / ".scratch/extraction/fig_legoprog_vs_control.png")
    a = ap.parse_args()
    d = pd.read_excel(a.xlsx, header=None)

    runs = {}
    for bname, (r0, cols) in BLOCKS.items():
        for c in cols:
            name = d.iloc[r0, c]
            if isinstance(name, str):
                v = [d.iloc[r, c + 2] for r in range(r0, r0 + 10)]
                runs[name] = (bname, np.array([x for x in v if pd.notna(x)], float))
    C = runs[CONTROL][1]

    print(
        f"control {CONTROL}: {C.mean():.2f} ± {1.96 * C.std(ddof=1) / np.sqrt(len(C)):.2f}  {C.astype(int).tolist()}\n"
    )
    print(f"{'condition':22s} {'block':9s} {'mean':>5} {'Δ vs ctrl':>10} {'95% CI':>17} {'p':>7} {'Holm':>7}")
    crit = [(n, b, v) for n, (b, v) in runs.items() if b != "bc"]
    res = [(n, b, v, *welch(v, C)) for n, b, v in crit]
    hp = holm(np.array([r[-1] for r in res]))
    table = []
    for (n, b, v, dd, lo, hi, p), ph in sorted(zip(res, hp, strict=True), key=lambda x: -x[0][2].mean()):
        flag = "**" if ph < 0.05 else ("*" if p < 0.05 else "")
        print(f"{n:22s} {b:9s} {v.mean():5.2f} {dd:+10.2f} [{lo:+6.2f},{hi:+6.2f}] {p:7.3f} {ph:7.3f} {flag}")
        table.append(
            {
                "name": n,
                "block": b,
                "mean": float(v.mean()),
                "delta": float(dd),
                "ci": [float(lo), float(hi)],
                "p": p,
                "holm": float(ph),
                "episodes": v.astype(int).tolist(),
            }
        )
    print(f"\nraw p<.05: {sum(r[-1] < 0.05 for r in res)}/{len(res)}   Holm p<.05: {int((hp < 0.05).sum())}/{len(res)}")

    # ---- decomposition of the WRONG comparison ---------------------------------------------------
    print("\n### why 'vs bc50_ex_10' looked like 15/15 worse: two changes at once ###")
    dec = {}
    for lab, x, y in [
        ("bc50_ex_10 -> bc30_ex_10   (checkpoint h50->h30, same 10 steps)", "bc50_ex_10", "bc30_ex_10"),
        ("bc30_ex_10 -> bc30_ex_30   (execution 10->30, same h30 checkpoint)", "bc30_ex_10", "bc30_ex_30"),
        ("bc50_ex_10 -> bc50_ex_30   (execution 10->30, h50 checkpoint)", "bc50_ex_10", "bc50_ex_30"),
        ("bc50_ex_10 -> bc30_ex_30   (both at once = the comparison that was made)", "bc50_ex_10", "bc30_ex_30"),
    ]:
        dd, lo, hi, p = welch(runs[y][1], runs[x][1])
        print(f"  {lab:70s} Δ={dd:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
        dec[lab] = {"delta": float(dd), "ci": [float(lo), float(hi)], "p": p}
    others = np.concatenate([runs[k][1] for k in runs if k.startswith("bc50") and k != "bc50_ex_10"])
    dd, lo, hi, p = welch(runs["bc50_ex_10"][1], others)
    print(f"  bc50_ex_10 vs the other five bc50 points pooled (peak-picking check)     Δ={dd:+.2f} p={p:.4f}")
    dec["peak_pick_p"] = p

    # ---- steering: graded, or flat-then-collapse? ------------------------------------------------
    print("\n### QPILOTS: is the dose-response graded? ###")
    st = {k: v for k, (b, v) in runs.items() if b == "steer"}
    alpha = {k: float(k.rsplit("_", 1)[1]) for k in st}
    ks = sorted(st, key=lambda k: alpha[k])
    xs = np.concatenate([[alpha[k]] * len(st[k]) for k in ks])
    ys = np.concatenate([st[k] for k in ks])
    rho_all = stats.spearmanr(xs, ys)
    keep = [k for k in ks if alpha[k] < 0.1]
    rho_no01 = stats.spearmanr(
        np.concatenate([[alpha[k]] * len(st[k]) for k in keep]), np.concatenate([st[k] for k in keep])
    )
    mid = [k for k in ks if 0 < alpha[k] < 0.1]
    kw_mid = stats.kruskal(*[st[k] for k in mid])
    pooled_on = np.concatenate([st[k] for k in ks if alpha[k] > 0])
    dd, lo, hi, p = welch(pooled_on, st[ks[0]])
    print("  means by alpha: " + "  ".join(f"{alpha[k]:g}:{st[k].mean():.2f}" for k in ks))
    print(f"  Spearman all six levels : rho={rho_all.statistic:+.3f} p={rho_all.pvalue:.1e}")
    print(f"  Spearman without a=0.1  : rho={rho_no01.statistic:+.3f} p={rho_no01.pvalue:.3f}   <- trend gone")
    print(f"  Kruskal across 0.005-0.05: p={kw_mid.pvalue:.3f}   <- the four intermediate levels are flat")
    print(f"  pooled a>0 (n={len(pooled_on)}) vs a=0: Δ={dd:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.4f}")
    steer = {
        "spearman_all_p": float(rho_all.pvalue),
        "spearman_no01_p": float(rho_no01.pvalue),
        "kruskal_mid_p": float(kw_mid.pvalue),
        "pooled_on_delta": float(dd),
        "pooled_on_ci": [float(lo), float(hi)],
        "pooled_on_p": p,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "control": CONTROL,
                "control_mean": float(C.mean()),
                "table": table,
                "decomposition": dec,
                "steering": steer,
                "xlsx": str(a.xlsx),
            },
            indent=1,
        )
    )
    print(f"\nwrote {a.out}")

    import sys

    sys.path.insert(0, str(R / "slurm"))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    PAL = plot_style.PALETTE
    col = {"adaptive": PAL[0], "select": PAL[2], "steer": PAL[3]}
    order = sorted(table, key=lambda r: ({"select": 0, "adaptive": 1, "steer": 2}[r["block"]], -r["mean"]))
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    cc = 1.96 * C.std(ddof=1) / np.sqrt(len(C))
    ax.axhspan(C.mean() - cc, C.mean() + cc, color=PAL[7], alpha=0.18, zorder=0)
    ax.axhline(
        C.mean(), color=PAL[7], lw=1.3, ls="--", zorder=1, label=f"{CONTROL}: same checkpoint, same 30 steps, no critic"
    )
    x = np.arange(len(order))
    for i, r in enumerate(order):
        v = np.array(r["episodes"], float)
        ax.bar(
            i,
            r["mean"],
            yerr=1.96 * v.std(ddof=1) / np.sqrt(len(v)),
            color=col[r["block"]],
            width=0.68,
            error_kw={"lw": 1.1, "capsize": 3},
            zorder=2,
        )
        ax.scatter(
            np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)),
            v,
            s=9,
            color="0.25",
            alpha=0.5,
            zorder=3,
            linewidths=0,
        )
        if r["p"] < 0.05:
            ax.text(
                i,
                r["mean"] + 1.96 * v.std(ddof=1) / np.sqrt(len(v)) + 0.12,
                "**" if r["holm"] < 0.05 else "*",
                ha="center",
                fontsize=11,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["name"].replace("fixed_", "").replace("qpilot_", "α=") for r in order], rotation=40, ha="right", fontsize=8.5
    )
    for b, lab in (("select", "selection"), ("adaptive", "adaptive commitment"), ("steer", "QPILOTS steering")):
        ax.bar([0], [0], color=col[b], label=lab)
    ax.set_ylabel("mean progress (0–4), n=10")
    ax.set_ylim(0, 4.2)
    ax.set_title("at the 30-step operating point, only steering moves the needle — down")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=170)
    print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
