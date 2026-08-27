"""Suite of per-frame trajectory views from the saved lag-run predictors (no retraining).

Outputs (.scratch/):
  fig_nmtraj_grid.png     3x3 grid of episodes: smoothed Delta err(t) = Markov - lag5
  fig_nmtraj_heat.png     position-bin x lag heatmap of mean Delta err (all lag arms)
  fig_nmtraj_hist.png     per-frame Delta distribution, success vs failure
  fig_nmtraj_raw.png      one episode's raw err curves (Markov vs lag-5) + Delta band
"""

# ruff: noqa: E402, ICN001  (matplotlib.use must precede pyplot)

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

D = R / ".scratch/nonmarkov_yam_lags"
D2 = R / ".scratch/nonmarkov_yam_lags2"
idx = np.load(D / "perframe_idx.npy")
is_succ = np.load(D / "perframe_is_succ.npy")
ERR = {0: np.load(D / "perframe_err_k0.npy")}
for k in (1, 3, 5, 15, 30, 60, 150):
    ERR[k] = np.load(D / f"perframe_err_k{k}.npy")
for extra, extra_ks in ((D2, (10, 20)), (R / ".scratch/nonmarkov_yam_lags3", (2, 4, 6, 8, 12))):
    idx2_f = extra / "perframe_idx.npy"
    if not idx2_f.exists():
        continue  # extra runs cover MORE frames (smaller kmax) -> align onto the main index set
    idx2 = np.load(idx2_f)
    pos = np.searchsorted(idx2, idx)
    ok = (pos < len(idx2)) & (idx2[np.minimum(pos, len(idx2) - 1)] == idx)
    for k in extra_ks:
        f = extra / f"perframe_err_k{k}.npy"
        if f.exists() and ok.all():
            ERR[k] = np.load(f)[pos]
meta = json.loads(pathlib.Path("/data1/jellyho/pc_cache/yam_s347/meta.json").read_text())

offsets = {int(k): (v["offset"], v["full_len"]) for k, v in meta["episodes"].items()}
ep_of = np.zeros(len(idx), np.int64)
for e, (off, ln) in offsets.items():
    m = (idx >= off) & (idx < off + ln)
    ep_of[m] = e
eps = np.unique(ep_of)
succ_eps = [e for e in eps if is_succ[ep_of == e][0] and (ep_of == e).sum() > 200]
fail_eps = [e for e in eps if not is_succ[ep_of == e][0] and (ep_of == e).sum() > 100]


def smooth(v, w=31):
    return np.convolve(v, np.ones(w) / w, mode="same")


# ---- (1) one episode per figure, ALL predictors together -------------------------------
cmap = plt.get_cmap("viridis")
FINE15 = True  # <=15 view; extra fine arms (lags3) join automatically when present
ks_all = sorted(k for k in ERR if 0 < k <= 15) if FINE15 else sorted(k for k in ERR if k != 0)
succ_short = sorted(succ_eps, key=lambda e: (ep_of == e).sum())[:4]  # shortest successes
for e in succ_short:
    m = ep_of == e
    frac = np.linspace(0, 1, m.sum())
    fig, ax = plt.subplots(figsize=(19, 3.6))
    ax.axhline(0, color="black", lw=1.2, ls="--", label="Markov", zorder=5)
    base = max(float(ERR[0][m].mean()), 1e-6)  # % of this episode's Markov error, higher = better
    for j, k in enumerate(ks_all):
        ax.plot(
            frac,
            100 * smooth(ERR[0][m] - ERR[k][m]) / base,
            color=cmap(j / (len(ks_all) - 1)),
            lw=1.1,
            label=f"lag {k}",
        )
    # shade where SHORT lags beat LONG lags (blue) or vice versa (orange); |gap| <= 15%p left blank
    # split WITHIN the 30-frame chunk window (lags > 30 are out of scope for commitment)
    SHORT, LONG = (1, 3, 5, 10), (15, 20, 30)
    g_s = np.mean([100 * smooth(ERR[0][m] - ERR[k][m]) / base for k in SHORT], axis=0)
    g_l = np.mean([100 * smooth(ERR[0][m] - ERR[k][m]) / base for k in LONG], axis=0)
    diff = g_s - g_l
    for cls, color in ((diff > 15, "#4C72B0"), (diff < -15, "#DD8452")):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], cls.astype(int), [0]])))
        for a0, a1 in zip(edges[::2], edges[1::2], strict=True):
            ax.axvspan(frac[a0], frac[min(a1, len(frac) - 1)], color=color, alpha=0.10, lw=0)
    ok = is_succ[m][0]
    ax.set_title(
        f"ep{e} ({'success' if ok else 'failure'})  ·  shading: blue = first half of chunk (lag<15) wins, orange = second half (15–30) wins",
        fontsize=10,
    )
    ax.set_xlabel("episode position (fraction)")
    ax.set_ylabel("improvement over Markov (%)")
    ax.legend(fontsize=8, ncol=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(R / f".scratch/fig_nmtraj_ep{e}.png", dpi=200)
    plt.close(fig)

# ---- (2) position x lag heatmap --------------------------------------------------------
nbins = 25
ks = sorted(k for k in ERR if k != 0)
H = np.zeros((len(ks), nbins))
C = np.zeros((len(ks), nbins))
for e in eps:
    m = ep_of == e
    n = m.sum()
    if n < 10:
        continue
    b = np.minimum((np.linspace(0, 1, n) * nbins).astype(int), nbins - 1)
    for i, k in enumerate(ks):
        np.add.at(H[i], b, ERR[0][m] - ERR[k][m])
        np.add.at(C[i], b, 1)
H = H / np.maximum(C, 1)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
vmax = np.abs(H).max()
im = ax.imshow(
    H, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[0, 1, -0.5, len(ks) - 0.5]
)
ax.set_yticks(range(len(ks)))
ax.set_yticklabels([str(k) for k in ks], fontsize=8)
ax.set_xlabel("episode position (fraction)")
ax.set_ylabel("lag of added frame (steps)")
fig.colorbar(im, ax=ax, label=r"mean $\Delta$ err (red = history helps)")
fig.tight_layout()
fig.savefig(R / ".scratch/fig_nmtraj_heat.png", dpi=200)

# ---- (3) Delta distribution success vs failure ----------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.0))
d = ERR[0] - ERR[5]
bins = np.linspace(-1, 2.5, 80)
for s, color, name in ((True, PAL[0], "success"), (False, PAL[3], "failure")):
    ax.hist(d[is_succ == s], bins=bins, density=True, histtype="step", lw=1.6, color=color, label=name)
ax.axvline(0, color="#555", lw=0.9, ls="--")
ax.set_xlabel(r"per-frame $\Delta$ err (Markov $-$ lag5)")
ax.set_ylabel("density")
ax.set_yscale("log")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / ".scratch/fig_nmtraj_hist.png", dpi=200)

# ---- (4) one episode raw curves --------------------------------------------------------
e = succ_eps[0]
m = ep_of == e
frac = np.linspace(0, 1, m.sum())
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(frac, smooth(ERR[0][m]), color="#888", lw=1.4, label="Markov")
ax.plot(frac, smooth(ERR[5][m]), color=PAL[0], lw=1.4, label="with lag-5 frame")
ax.fill_between(frac, smooth(ERR[5][m]), smooth(ERR[0][m]), color=PAL[0], alpha=0.15)
ax.set_xlabel("episode position (fraction)")
ax.set_ylabel("per-frame val MSE (smoothed)")
ax.set_title(f"ep{e} (success)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / ".scratch/fig_nmtraj_raw.png", dpi=200)
print("wrote 4 figures; frames", len(idx), "| eps", len(eps))
