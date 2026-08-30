"""Stitching, shown with real frames: walk phi-space from demo A's start toward demo B's goal,
retrieving the nearest REAL frame (from any episode) at each waypoint.

If the retrieved strip advances through the task while hopping between episodes/kitchens, the
metric genuinely composes trajectories — cross-episode stitching as pictures, not a correlation.

Layout per query (rows): image strip of retrieved frames labeled (episode, progress), with the
start frame (episode A) and the goal frame (episode B) at the ends; an inset shows the walked
path in a 2-D PCA of phi.
"""

import argparse
import json
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout/z.npy"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--repo-id", default="jellyho/robocasa365-PrepareCoffee")
    ap.add_argument("--num-queries", type=int, default=3)
    ap.add_argument("--waypoints", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/viz_stitch_path.png"))
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    meta = json.loads((args.annot / "meta.json").read_text())
    n = meta["num_frames"]
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))
    phi = np.load(args.phi).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    eps_u = np.unique(ep)

    # subsample the retrieval pool for speed (every 3rd frame keeps ~90k candidates)
    pool = np.arange(0, n, 3)
    phi_pool = phi[pool]

    from lerobot.datasets import lerobot_dataset

    ds = lerobot_dataset.LeRobotDataset(args.repo_id)
    cam = ds.meta.video_keys[0]  # agentview

    def frame_img(i):
        x = ds[int(i)][cam]  # [3,H,W] float 0..1
        return np.transpose(np.asarray(x), (1, 2, 0))

    K = args.waypoints
    fig, axes = plt.subplots(
        args.num_queries, K + 2, figsize=(2.1 * (K + 2), 2.6 * args.num_queries), facecolor="#0f1117"
    )
    if args.num_queries == 1:
        axes = axes[None]

    for q in range(args.num_queries):
        A, B = rng.choice(eps_u, size=2, replace=False)
        a_rows = np.flatnonzero((ep == A) & (prog > 0.05) & (prog < 0.2))
        b_rows = np.flatnonzero((ep == B) & (prog > 0.8) & (prog < 0.98))
        s, g = int(rng.choice(a_rows)), int(rng.choice(b_rows))

        # walk phi-space linearly from phi(s) to phi(g); retrieve nearest real frame per waypoint
        ts = np.linspace(0, 1, K)
        picks = []
        for t in ts:
            target = (1 - t) * phi[s] + t * phi[g]
            d = np.linalg.norm(phi_pool - target[None], axis=1)
            picks.append(int(pool[np.argmin(d)]))

        row = axes[q]
        row[0].imshow(frame_img(s))
        row[0].set_title(f"START\nep{A}  p={prog[s]:.2f}", color="#4ade80", fontsize=9)
        for k, r in enumerate(picks):
            row[k + 1].imshow(frame_img(r))
            col = "#60a5fa" if ep[r] not in (A, B) else "#9aa3b2"
            row[k + 1].set_title(f"ep{ep[r]}  p={prog[r]:.2f}", color=col, fontsize=9)
        row[K + 1].imshow(frame_img(g))
        row[K + 1].set_title(f"GOAL\nep{B}  p={prog[g]:.2f}", color="#fbbf24", fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#2a3140")

    fig.suptitle(
        "Walking phi-space from episode A's start to episode B's goal —\n"
        "each waypoint shows the nearest REAL frame from ANY episode (blue title = third episode)",
        color="w",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=130, facecolor="#0f1117")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
