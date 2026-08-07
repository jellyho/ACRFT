"""Trajectory-line version of the HILP-space viz: each episode drawn as a connected path.

Row 1: each episode is one colored polyline through the t-SNE embedding (O = start, * = end).
       In a pathological space the lines coil up in separate islands; in a healthy one they all
       travel the same road.
Row 2: the same lines colored by progress along their length (all lines sharing one global
       color ramp) — if every trajectory traverses the same gradient, the space encodes task
       phase, not identity.
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from viz_hilp_space import load_rep, phi_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=[
        "RLT-token(VLA)=.scratch/annot_noprop",
        "cheap-z+HILP(v7a)=.scratch/cheap_z_v7a",
    ])
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--per-ep", type=int, default=80)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/viz_hilp_trajlines.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.annot / "meta.json").read_text())
    n = meta["num_frames"]
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))

    rng = np.random.default_rng(args.seed)
    eps = rng.choice(np.unique(ep), size=args.episodes, replace=False)
    per_ep_rows = []
    for e in eps:
        r = np.flatnonzero(ep == e)
        per_ep_rows.append(r[np.linspace(0, len(r) - 1, min(args.per_ep, len(r))).astype(int)])
    rows = np.concatenate(per_ep_rows)
    bounds = np.cumsum([0] + [len(r) for r in per_ep_rows])
    prog_s = prog[rows]

    reps = []
    for spec in args.reps:
        name, z, p = load_rep(spec, n)
        reps.append((name, z[rows]))
        if (p / "head.pt").exists():
            ph = phi_of(p, z[rows])
            if ph is not None:
                reps.append((f"{name} phi-space", ph))

    from sklearn.manifold import TSNE

    R = len(reps)
    fig, axes = plt.subplots(2, R, figsize=(5.6 * R, 10.5), facecolor="#0f1117")
    if R == 1:
        axes = axes[:, None]
    cmap_ep = plt.get_cmap("tab20")

    for c, (name, z) in enumerate(reps):
        zn = (z - z.mean(0)) / (z.std(0) + 1e-6)
        xy = TSNE(n_components=2, init="pca", perplexity=40, random_state=args.seed).fit_transform(zn)

        # row 1: one polyline per episode, colored by episode
        ax = axes[0, c]
        for k in range(len(eps)):
            seg = xy[bounds[k]:bounds[k + 1]]
            col = cmap_ep(k % 20)
            ax.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=1.0, alpha=0.6)
            ax.scatter(seg[:, 0], seg[:, 1], color=col, s=9, alpha=0.9, linewidths=0)
            ax.plot(seg[0, 0], seg[0, 1], "o", color=col, ms=6, mec="w", mew=0.8)
            ax.plot(seg[-1, 0], seg[-1, 1], "*", color=col, ms=10, mec="w", mew=0.5)
        ax.set_title(f"{name}\none line per episode (o = start, * = end)", color="w", fontsize=11)

        # row 2: same lines, segments colored by progress (shared ramp)
        ax = axes[1, c]
        for k in range(len(eps)):
            seg = xy[bounds[k]:bounds[k + 1]]
            pg = prog_s[bounds[k]:bounds[k + 1]]
            pts = seg.reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                                cmap="viridis", norm=plt.Normalize(0, 1), alpha=0.85, linewidths=1.4)
            lc.set_array((pg[:-1] + pg[1:]) / 2)
            ax.add_collection(lc)
            ax.scatter(seg[:, 0], seg[:, 1], c=pg, cmap="viridis", vmin=0, vmax=1,
                       s=9, alpha=0.9, linewidths=0)
        ax.autoscale()
        ax.set_title("same lines, colored by PROGRESS\n(all sharing one road = phase-organized)", color="w", fontsize=11)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
                     ax=ax, fraction=0.04)

    for ax in axes.ravel():
        ax.set_facecolor("#181c25")
        for s in ax.spines.values():
            s.set_color("#2a3140")
        ax.tick_params(colors="#9aa3b2", labelsize=7)

    fig.suptitle("Trajectories as connected paths through the embedding", color="w", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140, facecolor="#0f1117")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
