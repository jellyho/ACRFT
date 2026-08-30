# ruff: noqa
"""Apple-to-apple ablation figure from worker C's acrft_ogbench runs. Recomputes success (last-3-eval
mean, seed-averaged) and plots the cleanest single-component ladders: objective, alpha, expectile.

Outputs: plots/32_aqc_ablation.png
"""

import csv, glob, os, re, sys
from collections import defaultdict
import matplotlib, numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "slurm")
from plot_style import PALETTE, apply

apply()

ROOT = "/scratch/gwanwoo13/aqc/exp/aqc-ogbench"
OBJ = {"iql", "iqlnt", "notgt", "calql", "sarsa"}
AGG = {"mean", "max"}
C = os.environ["CACHE_DIR"]


def parse(rn):
    m = re.match(r"^(.*?)-(\d{4}_\d{4})$", rn)
    if not m:
        return None
    toks = m.group(1).split("-")
    if not toks or toks[0] != "aqc":
        return None
    toks = toks[1:]
    d = {"objective": "plain", "agg": "-", "expectile": "-", "alpha": "-"}
    while toks:
        t = toks[0]
        if t in OBJ:
            d["objective"] = t
        elif t in AGG:
            d["agg"] = t
        elif re.fullmatch(r"t\d+", t):
            d["expectile"] = t
        elif re.fullmatch(r"a\d+", t):
            d["alpha"] = t
        else:
            break
        toks.pop(0)
    d["task"] = toks.pop() if (toks and re.fullmatch(r"task\d+", toks[-1])) else "-"
    d["env"] = "-".join(toks) if toks else "?"
    return d


def succ(ev):
    try:
        s = [float(r["success"]) for r in csv.DictReader(open(ev)) if r.get("success")]
        return sum(s[-3:]) / len(s[-3:]) if s else None
    except:
        return None


runs = defaultdict(list)
for run in os.listdir(ROOT):
    p = parse(run)
    if not p:
        continue
    for ev in glob.glob(os.path.join(ROOT, run, "*", "*", "eval.csv")):
        s = succ(ev)
        if s is not None:
            runs[(p["objective"], p["agg"], p["expectile"], p["alpha"], p["env"], p["task"])].append(s)
agg = {k: (np.mean(v), len(v)) for k, v in runs.items()}


def get(obj, ag, ex, al, env, task="-"):
    return agg.get((obj, ag, ex, al, env, task))


def rows(specs):
    out = []
    for lab, r in specs:
        v = get(*r)
        if v:
            out.append((lab, v[0], v[1]))
    return out


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
# Panel A: objective ladder (cube-double, a300)
ax = axes[0]
b = rows(
    [
        ("plain", ("plain", "-", "-", "a300", "cube-double")),
        ("iql", ("iql", "mean", "t09", "a300", "cube-double")),
        ("iqlnt", ("iqlnt", "mean", "t09", "a300", "cube-double")),
        ("notgt", ("notgt", "-", "-", "a300", "cube-double")),
    ]
)
cols = [PALETTE[2], PALETTE[0], PALETTE[9], PALETTE[3]]
ax.bar(range(len(b)), [x[1] for x in b], color=cols[: len(b)], width=0.62)
for i, x in enumerate(b):
    ax.text(i, x[1] + 0.02, f"{x[1]:.2f}\nn{x[2]}", ha="center", fontsize=9)
ax.set_xticks(range(len(b)))
ax.set_xticklabels([x[0] for x in b])
ax.set_ylim(0, 1)
ax.set_ylabel("success (last-3 eval, seed avg)")
ax.set_title("objective  (cube-double, a300)")
# Panel B: alpha U-curve (cube-double, iqlnt/t09)
ax = axes[1]
b = rows([(a, ("iqlnt", "mean", "t09", a, "cube-double")) for a in ["a100", "a170", "a300", "a900", "a2700"]])
ax.plot(range(len(b)), [x[1] for x in b], "o-", color=PALETTE[0], lw=2)
for i, x in enumerate(b):
    ax.text(i, x[1] + 0.03, f"{x[1]:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(b)))
ax.set_xticklabels([x[0] for x in b])
ax.set_ylim(0, 1)
ax.set_title("alpha  (cube-double, iqlnt/t09)")
# Panel C: expectile (cube-double, iql/mean/a900)
ax = axes[2]
b = rows([(e, ("iql", "mean", e, "a900", "cube-double")) for e in ["t08", "t09", "t095"]])
ax.bar(range(len(b)), [x[1] for x in b], color=PALETTE[5], width=0.55)
for i, x in enumerate(b):
    ax.text(i, x[1] + 0.02, f"{x[1]:.2f}\nn{x[2]}", ha="center", fontsize=9)
ax.set_xticks(range(len(b)))
ax.set_xticklabels([x[0] for x in b])
ax.set_ylim(0, 1)
ax.set_title("expectile  (cube-double, iql/a900)")
fig.suptitle("acrft_ogbench apple-to-apple ablation (worker C runs · worker B analysis)", fontsize=12)
fig.tight_layout()
os.makedirs(f"{C}/plots", exist_ok=True)
fig.savefig(f"{C}/plots/32_aqc_ablation.png", dpi=150)
print("AQC_ABLATION_PLOT_DONE")
