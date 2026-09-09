"""Overview map for one episode: where each short-dominant clip region sits on the timeline.

Mirrors make_nonmarkov_region_clips.py's region rule (SHORT {1,3,5,10} vs LONG {15,20,30},
gap > 15%p, sustained >= --min-sec) and labels regions r0, r1, ... to match clip filenames.
"""

# ruff: noqa: E402, ICN001

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "slurm"))
import plot_style

plot_style.apply()
PAL = plot_style.PALETTE

ap = argparse.ArgumentParser()
ap.add_argument("--episode", type=int, default=81)
ap.add_argument("--min-sec", type=float, default=2.0)
ap.add_argument("--gap-thresh", type=float, default=15.0)
a = ap.parse_args()

D = R / ".scratch/nonmarkov_yam_lags"
idx = np.load(D / "perframe_idx.npy")
ERR = {0: np.load(D / "perframe_err_k0.npy")}
for k in (1, 3, 5, 15, 30, 60, 150):
    ERR[k] = np.load(D / f"perframe_err_k{k}.npy")
for extra, ks in (
    (R / ".scratch/nonmarkov_yam_lags2", (10, 20)),
    (R / ".scratch/nonmarkov_yam_lags3", (2, 4, 6, 8, 12)),
):
    f = extra / "perframe_idx.npy"
    if not f.exists():
        continue
    idx2 = np.load(f)
    pos = np.searchsorted(idx2, idx)
    ok = (pos < len(idx2)) & (idx2[np.minimum(pos, len(idx2) - 1)] == idx)
    for k in ks:
        g = extra / f"perframe_err_k{k}.npy"
        if g.exists() and ok.all():
            ERR[k] = np.load(g)[pos]

meta = json.loads(pathlib.Path("/data1/jellyho/pc_cache/yam_s347/meta.json").read_text())
off, n = (meta["episodes"][str(a.episode)][x] for x in ("offset", "full_len"))
m = (idx >= off) & (idx < off + n)
real = idx[m] - off
FPS = 30.0


def smooth(v, w=31):
    return np.convolve(v, np.ones(w) / w, mode="same")


base = max(float(ERR[0][m].mean()), 1e-6)
SHORT, LONG = (1, 3, 5, 10), (15, 20, 30)
g_s = np.mean([100 * smooth(ERR[0][m] - ERR[k][m]) / base for k in SHORT], axis=0)
g_l = np.mean([100 * smooth(ERR[0][m] - ERR[k][m]) / base for k in LONG], axis=0)
diff = g_s - g_l
cls = diff > a.gap_thresh
min_pts = int(a.min_sec * FPS / 2)
edges = np.flatnonzero(np.diff(np.concatenate([[0], cls.astype(int), [0]])))
regions = [(s, t) for s, t in zip(edges[::2], edges[1::2], strict=True) if t - s >= min_pts]

secs = real / FPS
fig, ax = plt.subplots(figsize=(14, 3.2))
ax.plot(secs, g_s, color=PAL[0], lw=1.5, label="short lags {1,3,5,10}")
ax.plot(secs, g_l, color=PAL[1], lw=1.5, label="long lags {15,20,30}")
ax.axhline(0, color="#555", lw=0.9, ls="--")
for r, (s0, s1) in enumerate(regions):
    x0, x1 = secs[s0], secs[min(s1, len(secs) - 1)]
    ax.axvspan(x0, x1, color=PAL[3], alpha=0.14, lw=0)
    ax.text(
        (x0 + x1) / 2,
        ax.get_ylim()[1] * 0.92 if False else max(g_s.max(), g_l.max()) * 0.98,
        f"r{r}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=PAL[3],
    )
    print(f"r{r}: frames {real[s0]}-{real[min(s1, len(real) - 1)]}  ({x0:.1f}s - {x1:.1f}s, {x1 - x0:.1f}s)")
ax.set_xlabel("episode time (s)")
ax.set_ylabel("improvement over Markov (%)")
ax.set_title(f"ep{a.episode}: short-dominant regions (shaded, labels match clip files)", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = R / f".scratch/fig_nm_ep{a.episode}_map"
fig.savefig(f"{out}.png", dpi=200)
print("wrote", out)
