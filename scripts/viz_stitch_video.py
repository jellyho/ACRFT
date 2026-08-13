"""Video of a stitched trajectory: walk phi from episode A's start to episode B's goal,
showing the nearest REAL frame at every step — the kitchen visibly switches mid-ride.

Left: the retrieved frame (border colored by which episode it came from; label = ep, progress).
Right: the phi map (ghost points colored by progress), the straight walk path, the moving
waypoint, and START/GOAL markers.

This is the moving version of viz_stitch_path.py's strips: cross-episode stitching as film.
"""

import argparse
import json
import pathlib
import subprocess

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout/z.npy"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--repo-id", default="jellyho/robocasa365-PrepareCoffee")
    ap.add_argument("--steps", type=int, default=110)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--bg-frames", type=int, default=2500)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/stitch_ride.mp4"))
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    meta = json.loads((args.annot / "meta.json").read_text())
    n = meta["num_frames"]
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))
    phi = np.load(args.phi).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    A, B = rng.choice(np.unique(ep), size=2, replace=False)
    s = int(rng.choice(np.flatnonzero((ep == A) & (prog > 0.05) & (prog < 0.2))))
    g = int(rng.choice(np.flatnonzero((ep == B) & (prog > 0.8) & (prog < 0.98))))

    pool = np.arange(0, n, 5)
    # make sure the endpoints are in the pool
    pool = np.unique(np.concatenate([pool, [s, g]]))
    phi_pool = phi[pool]

    # GEODESIC walk: straight-line interpolation cuts through empty (off-manifold) space and the
    # nearest-neighbour snaps to the endpoints (measured: the ride was a binary A->B jump). Build a
    # kNN graph over real frames and take the shortest path instead - every step is a real frame and
    # the route must traverse the funnel through actual intermediate states.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=12).fit(phi_pool)
    dist, idx_nn = nn.kneighbors(phi_pool)
    m = len(pool)
    rowi = np.repeat(np.arange(m), idx_nn.shape[1] - 1)
    colj = idx_nn[:, 1:].reshape(-1)
    w = dist[:, 1:].reshape(-1)
    graph = csr_matrix((w, (rowi, colj)), shape=(m, m))
    src = int(np.flatnonzero(pool == s)[0])
    dst = int(np.flatnonzero(pool == g)[0])
    dmat, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dmat[dst]):
        raise SystemExit("no path in kNN graph - increase n_neighbors")
    path = [dst]
    while path[-1] != src:
        path.append(int(pred[path[-1]]))
    path.reverse()
    node_rows = pool[np.array(path)]
    # resample the path evenly to the video length
    sel = np.linspace(0, len(node_rows) - 1, args.steps).astype(int)
    picks = node_rows[sel]
    print(f"geodesic: {len(node_rows)} nodes, episodes visited: {sorted({int(x) for x in ep[node_rows]})[:12]}")

    # 2-D map: t-SNE over background + walk-relevant frames
    bg = rng.choice(n, size=args.bg_frames, replace=False)
    all_rows = np.unique(np.concatenate([bg, picks, [s, g]]))
    pos_of = {int(r): i for i, r in enumerate(all_rows)}
    from sklearn.manifold import TSNE

    zz = phi[all_rows]
    zz = (zz - zz.mean(0)) / (zz.std(0) + 1e-6)
    xy = TSNE(n_components=2, init="pca", perplexity=40, random_state=args.seed).fit_transform(zz)

    from lerobot.datasets import lerobot_dataset

    ds = lerobot_dataset.LeRobotDataset(args.repo_id)
    cam = ds.meta.video_keys[0]
    cache = {}

    def img(i):
        i = int(i)
        if i not in cache:
            cache[i] = np.transpose(np.asarray(ds[i][cam]), (1, 2, 0))
        return cache[i]

    # per-source-episode border colors (A red, B yellow, third episodes blue)
    def col_of(e):
        return "#f87171" if e == A else ("#fbbf24" if e == B else "#60a5fa")

    tmp = pathlib.Path(".scratch/_stitch_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()

    walk_xy = np.array([xy[pos_of[int(r)]] for r in picks])
    for t in range(args.steps):
        r = picks[t]
        e = int(ep[r])
        fig = plt.figure(figsize=(12.8, 5.2), facecolor="#0f1117")
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.04)
        ax = fig.add_subplot(gs[0])
        ax.imshow(img(r))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col_of(e))
            sp.set_linewidth(5)
        tagg = "START ep" if e == A else ("GOAL ep" if e == B else "via ep")
        ax.set_title(f"{tagg}{e}   progress {prog[r]:.2f}", color=col_of(e), fontsize=13)

        ax = fig.add_subplot(gs[1])
        ax.scatter(xy[:, 0], xy[:, 1], c=prog[all_rows], cmap="viridis", s=3, alpha=0.25, linewidths=0)
        ax.plot(walk_xy[: t + 1, 0], walk_xy[: t + 1, 1], "-", color="w", lw=1.8, alpha=0.8)
        ax.scatter(*xy[pos_of[s]], marker="o", s=130, color="#f87171", edgecolors="w", zorder=5, label=f"start (ep{A})")
        ax.scatter(*xy[pos_of[g]], marker="*", s=260, color="#fbbf24", edgecolors="w", zorder=5, label=f"goal (ep{B})")
        ax.scatter(*walk_xy[t], s=110, color=col_of(e), edgecolors="w", linewidths=1.4, zorder=6)
        ax.set_facecolor("#181c25")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2a3140")
        ax.legend(facecolor="#181c25", labelcolor="w", fontsize=9, loc="lower right")
        ax.set_title("walking phi from A's start to B's goal — nearest real frame at each step", color="w", fontsize=12)
        fig.savefig(tmp / f"f{t:04d}.png", dpi=100, facecolor="#0f1117", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        if t % 20 == 0:
            print(f"frame {t}/{args.steps}", flush=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(tmp / "f%04d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            str(args.out),
        ],
        check=True,
        capture_output=True,
    )
    print(f"wrote {args.out}  (A=ep{A}, B=ep{B})")


if __name__ == "__main__":
    main()
