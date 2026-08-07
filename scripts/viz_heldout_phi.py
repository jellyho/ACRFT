"""Held-out generalization view: phi retrained WITHOUT 77 episodes; do the unseen
trajectories still ride the same funnel?

Background ghost points = TRAIN frames (progress colormap). Lines+dots = HELD-OUT episodes
(never used in the TD loss). If they follow the same road with a clean progress ramp, the
metric generalizes across kitchens rather than memorizing the training demos.
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

annot = pathlib.Path(".scratch/annot_noprop")
meta = json.loads((annot/"meta.json").read_text()); n = meta["num_frames"]
ep = np.array(np.memmap(annot/"episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
prog = np.array(np.memmap(annot/"progress.dat", dtype=np.float32, mode="r", shape=(n,)))
phi = np.load(".scratch/rlt_hilp_readout_ho/z.npy").astype(np.float32)
held = np.load(".scratch/rlt_hilp_readout_ho/held_episodes.npy")
rng = np.random.default_rng(0)

heroes = rng.choice(held, size=10, replace=False)
bg = rng.choice(np.flatnonzero(~np.isin(ep, held)), size=3000, replace=False)
hero_rows = []
for e in heroes:
    r = np.flatnonzero(ep == e)
    hero_rows.append(r[np.linspace(0, len(r)-1, min(70, len(r))).astype(int)])
rows = np.unique(np.concatenate([bg] + hero_rows))
pos = {int(r): i for i, r in enumerate(rows)}

from sklearn.manifold import TSNE
zz = phi[rows]; zz = (zz - zz.mean(0)) / (zz.std(0) + 1e-6)
xy = TSNE(n_components=2, init="pca", perplexity=40, random_state=0).fit_transform(zz)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor="#0f1117")
cmap_ep = plt.get_cmap("tab10")
for panel, mode in enumerate(["episode", "progress"]):
    ax = axes[panel]
    bgi = [pos[int(r)] for r in bg]
    ax.scatter(xy[bgi,0], xy[bgi,1], c=prog[bg], cmap="viridis", s=4, alpha=0.18, linewidths=0)
    for k, hr in enumerate(hero_rows):
        pts = xy[[pos[int(r)] for r in hr]]
        pg = prog[hr]
        if mode == "episode":
            col = cmap_ep(k % 10)
            ax.plot(pts[:,0], pts[:,1], "-", color=col, lw=1.2, alpha=0.7)
            ax.scatter(pts[:,0], pts[:,1], color=col, s=11, alpha=0.95, linewidths=0)
            ax.plot(pts[0,0], pts[0,1], "o", color=col, ms=7, mec="w", mew=0.8)
            ax.plot(pts[-1,0], pts[-1,1], "*", color=col, ms=12, mec="w", mew=0.6)
        else:
            seg = pts.reshape(-1,1,2)
            lc = LineCollection(np.concatenate([seg[:-1], seg[1:]], axis=1), cmap="viridis",
                                norm=plt.Normalize(0,1), linewidths=1.6, alpha=0.95)
            lc.set_array((pg[:-1]+pg[1:])/2); ax.add_collection(lc)
            ax.scatter(pts[:,0], pts[:,1], c=pg, cmap="viridis", vmin=0, vmax=1, s=11, alpha=0.95, linewidths=0)
    ax.set_facecolor("#181c25"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#2a3140")
    ax.set_title("HELD-OUT trajectories colored by " + ("EPISODE" if mode=="episode" else "PROGRESS")
                 + "\n(ghost = TRAIN frames)", color="w", fontsize=12)
fig.suptitle("Generalization: 10 episodes NEVER seen by the TD loss, on the phi trained without them",
             color="w", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(".scratch/viz_heldout_phi.png", dpi=140, facecolor="#0f1117")
print("wrote .scratch/viz_heldout_phi.png")
