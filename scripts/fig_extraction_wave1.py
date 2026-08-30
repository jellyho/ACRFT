"""Wave-1 offline extraction comparison — paired uplifts of the training-free arms vs BC.

Reads .scratch/extraction/eval/*.json (eval_extraction.py outputs; per-state arrays are paired
across arms by shared noise keys). Left: paired critic-Q uplift with 95% CI. Right: the price —
demo-MSE and chunk-jerk deltas. Regenerable at any time from the JSONs.
"""

# ruff: noqa: E402, ICN001  (matplotlib.use must precede pyplot)

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "slurm"))
import plot_style

plot_style.apply()
PAL = plot_style.PALETTE

D = R / ".scratch/extraction/eval"
base = json.loads((D / "bc_base.json").read_text())["per_state"]
ORDER = ["flowdagger", "bon_n8", "idql_n64", "qpilots_a01", "qpilots_a02", "qpilots_a03"]
NAMES = {
    "flowdagger": "FlowDAgger",
    "bon_n8": "BoN N=8",
    "idql_n64": "IDQL N=64",
    "qpilots_a01": "QPILOTS α=.1",
    "qpilots_a02": "QPILOTS α=.2",
    "qpilots_a03": "QPILOTS α=.3",
}


def paired(label, key):
    ps = json.loads((D / f"{label}.json").read_text())["per_state"]
    d = np.array(ps[key]) - np.array(base[key])
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(d.size)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
y = np.arange(len(ORDER))

dq = [paired(a, "q_mean") for a in ORDER]
ax1.barh(y, [v for v, _ in dq], xerr=[c for _, c in dq], color=PAL[0], height=0.62, error_kw={"lw": 1.1})
ax1.axvline(0, color="0.35", lw=0.9)
ax1.set_yticks(y, [NAMES[a] for a in ORDER])
ax1.set_xlabel("paired ΔQ vs BC (95% CI)")
ax1.set_title("critic-Q uplift")

dm = [paired(a, "demo_mse")[0] for a in ORDER]
dj = [paired(a, "jerk")[0] for a in ORDER]
ax2.barh(y + 0.17, dm, color=PAL[1], height=0.32, label="Δ demo-MSE")
ax2.barh(y - 0.17, dj, color=PAL[2], height=0.32, label="Δ jerk")
ax2.axvline(0, color="0.35", lw=0.9)
ax2.set_yticks(y, ["" for _ in ORDER])
ax2.set_xlabel("paired Δ vs BC")
ax2.set_title("the price")
ax2.legend()

fig.tight_layout()
out = R / ".scratch/extraction/fig_extraction_wave1.png"
fig.savefig(out, dpi=160)
print("wrote", out)
