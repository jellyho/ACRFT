"""Single-lag non-Markovianity profile (vision-present, the fair primary condition).

Reads .scratch/nonmarkov_yam_lags/results.json (+ the channel-diagnostic runs for reference
lines). Each lag arm's input is exactly [frame(t-n), frame(t)]; Markov = [t, t] (same dims).
"""

# ruff: noqa: E402, ICN001  (matplotlib.use must precede pyplot)

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

d = json.loads((R / ".scratch/nonmarkov_yam_lags/results.json").read_text())
m0 = d["arms"]["0"]["val_mse"]
lags = sorted(int(k) for k in d["arms"] if k != "0")
xs = [k / 30.0 for k in lags]
ys = [100 * (m0 - d["arms"][str(k)]["val_mse"]) / m0 for k in lags]

vel = json.loads((R / ".scratch/nonmarkov_yam_velonly/results.json").read_text())
vel_gain = 100 * (vel["arms"]["0"]["val_mse"] - vel["arms"]["15"]["val_mse"]) / vel["arms"]["0"]["val_mse"]

fig, ax = plt.subplots(figsize=(4.8, 3.1))
ax.plot(xs, ys, "o-", color=PAL[0], lw=1.8, ms=6, label="one raw frame at lag $n$")
ax.axhline(0, color="#555", lw=1.0, ls="--")
ax.axhline(vel_gain, color=PAL[1], lw=1.2, ls=":")
ax.text(4.8, vel_gain + 1.2, "explicit velocity feature", ha="right", fontsize=8, color=PAL[1])
ax.text(4.8, 1.0, "Markov", ha="right", fontsize=8, color="#555")
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["0.03", "0.1", "0.17", "0.5", "1", "2", "5"], fontsize=8)
ax.minorticks_off()
ax.set_xlabel("lag of the added frame (s)")
ax.set_ylabel("held-out action-pred.\nimprovement over Markov (%)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
out = R / ".scratch/fig_nonmarkov_lags"
fig.savefig(f"{out}.png", dpi=220)
fig.savefig(f"{out}.svg")
print("wrote", out, "| vel line", round(vel_gain, 1))
