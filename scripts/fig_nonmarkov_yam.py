"""Figure for the YAM non-Markovianity measurement (reads .scratch/nonmarkov_yam/results.json)."""

# ruff: noqa: E402, ICN001  (matplotlib.use must precede pyplot; probe-local imports intentional)

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "slurm"))
import plot_style

plot_style.apply()
PAL = plot_style.PALETTE

res = json.loads((R / ".scratch/nonmarkov_yam/results.json").read_text())
arms = res["arms"]
m0 = arms["0"]["val_mse"]
ks = sorted(int(k) for k in arms if k != "0")
fps = 30.0

fig, ax = plt.subplots(figsize=(4.4, 3.0))
xs = [k / fps for k in ks]
ys = [100 * (m0 - arms[str(k)]["val_mse"]) / m0 for k in ks]
ax.plot(xs, ys, "o-", color=PAL[0], lw=1.8, ms=6)
ax.axhline(0, color="#555", lw=1.0, ls="--")
ax.text(xs[-1], 0.6, "Markov", ha="right", fontsize=8, color="#555")
ax.set_xlabel("history window (s)")
ax.set_ylabel("held-out action-pred.\nimprovement over Markov (%)")
ax.set_xticks(xs)
ax.set_xticklabels([f"{x:g}" for x in xs])
fig.tight_layout()
out = R / ".scratch/fig_nonmarkov_yam"
fig.savefig(f"{out}.png", dpi=220)
fig.savefig(f"{out}.svg")
print("wrote", out, {k: round(y, 1) for k, y in zip(ks, ys, strict=True)})
