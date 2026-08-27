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

gains = {}
for run in ("nonmarkov_yam_lags", "nonmarkov_yam_lags2"):
    f = R / f".scratch/{run}/results.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    m0 = d["arms"]["0"]["val_mse"]  # each run's gain vs its OWN baseline (runs differ by ~3%)
    for k in d["arms"]:
        if k != "0":
            gains[int(k)] = 100 * (m0 - d["arms"][k]["val_mse"]) / m0
lags = sorted(gains)
xs = [k / 30.0 for k in lags]
ys = [gains[k] for k in lags]

fig, ax = plt.subplots(figsize=(4.8, 3.1))
ax.plot(xs, ys, "o-", color=PAL[0], lw=1.8, ms=6)
ax.axhline(0, color="#555", lw=1.0, ls="--")
ax.text(4.8, 1.0, "Markov", ha="right", fontsize=8, color="#555")
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([f"{x:.2g}" for x in xs], fontsize=7, rotation=45)
ax.minorticks_off()
ax.set_xlabel("lag of the added frame (s)")
ax.set_ylabel("held-out action-pred.\nimprovement over Markov (%)")
fig.tight_layout()
out = R / ".scratch/fig_nonmarkov_lags"
fig.savefig(f"{out}.png", dpi=220)
fig.savefig(f"{out}.svg")
print("wrote", out)
