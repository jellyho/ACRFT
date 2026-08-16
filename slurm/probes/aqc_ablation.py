# ruff: noqa
"""Apple-to-apple ablation over worker C's acrft_ogbench runs (/scratch/gwanwoo13/aqc/exp/aqc-ogbench).

Run-dir names encode components: aqc-[objective-][agg-][tXX-][aXXX-]ENV[-taskN]-DATE
  objective: plain / iql / iqlnt / notgt      agg: mean / max     expectile: t09 / t095
  alpha:     a300 / a900 / a8100              env: cube-double / scene / puzzle-4x4 / ...
Success = worker C's standard: mean of the LAST 3 eval rows' `success`, per seed, then mean over seeds.
Groups runs so that exactly ONE component differs (holding env+task+the rest fixed) and prints the effect.

Usage: aqc_ablation.py
"""

import csv
import glob
import os
import re
from collections import defaultdict

ROOT = "/scratch/gwanwoo13/aqc/exp/aqc-ogbench"
OBJ = {"iql", "iqlnt", "notgt", "calql", "sarsa"}
AGG = {"mean", "max"}


def parse(runname):
    """run dir name -> dict(components) or None."""
    m = re.match(r"^(.*?)-(\d{4}_\d{4})$", runname)  # strip trailing DATE
    if not m:
        return None
    core, date = m.group(1), m.group(2)
    toks = core.split("-")
    if not toks or toks[0] != "aqc":
        return None
    toks = toks[1:]
    d = {"objective": "plain", "agg": "-", "expectile": "-", "alpha": "-", "date": date}
    # peel known tokens from front
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
    # peel taskN from back
    task = "-"
    if toks and re.fullmatch(r"task\d+", toks[-1]):
        task = toks.pop()
    d["env"] = "-".join(toks) if toks else "?"
    d["task"] = task
    return d


def success_of(evalcsv):
    """mean of last-3 eval rows' success."""
    try:
        rows = list(csv.DictReader(open(evalcsv)))
        s = [float(r["success"]) for r in rows if r.get("success") not in (None, "")]
        if not s:
            return None
        return sum(s[-3:]) / len(s[-3:])
    except Exception:
        return None


# gather: config_key -> list of per-seed success
runs = defaultdict(list)  # key (objective,agg,expectile,alpha,env,task) -> [success,...]
n_eval = 0
for run in os.listdir(ROOT):
    p = parse(run)
    if not p:
        continue
    for ev in glob.glob(os.path.join(ROOT, run, "*", "*", "eval.csv")):
        sr = success_of(ev)
        if sr is None:
            continue
        n_eval += 1
        key = (p["objective"], p["agg"], p["expectile"], p["alpha"], p["env"], p["task"])
        runs[key].append(sr)

agg = {k: (sum(v) / len(v), len(v)) for k, v in runs.items()}  # key -> (mean success, n seeds)
print(f"parsed {len(runs)} configs from {n_eval} eval.csv\n")

FIELDS = ["objective", "agg", "expectile", "alpha", "env", "task"]


def ablate(axis):
    """print single-component comparisons: hold all-but-`axis` fixed, vary axis."""
    ai = FIELDS.index(axis)
    groups = defaultdict(dict)  # fixed-tuple -> {axis_val: (succ,n)}
    for k, (s, n) in agg.items():
        fixed = tuple(x for i, x in enumerate(k) if i != ai)
        groups[fixed][k[ai]] = (s, n)
    print(f"===== ablate {axis} (한 컴포넌트만 변화) =====")
    shown = 0
    for fixed, variants in sorted(groups.items()):
        if len(variants) < 2:
            continue
        env = fixed[FIELDS.index("env") - (1 if FIELDS.index("env") > ai else 0)]
        task = fixed[-1]
        lab = ", ".join(f"{v}={s:.2f}(n{n})" for v, (s, n) in sorted(variants.items(), key=lambda x: -x[1][0]))
        others = "/".join(str(x) for x in fixed)
        print(f"  [{others}]  {lab}")
        shown += 1
        if shown >= 30:
            print("  … (더 있음)")
            break
    print()


for ax in ["objective", "expectile", "alpha", "agg"]:
    ablate(ax)
