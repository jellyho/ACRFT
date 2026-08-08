"""Video: three different kitchens travelling the SAME road through the embedding.

Left: the agentview frames of three episodes, progress-synchronized (all at the same task
phase at every video frame). Right: the phi-space map (t-SNE, all sampled frames ghosted and
colored by progress) with the three episodes' positions moving along it, trails behind.

If episode identity were still in the space, the three dots would live in three separate
islands; instead they ride the same funnel together while their kitchens look nothing alike.

CPU-only + ffmpeg. A few minutes (video decode of ~3x120 frames dominates).
"""

import argparse
import json
import pathlib
import subprocess

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EP_COLORS = ["#f87171", "#4ade80", "#60a5fa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout/z.npy"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--repo-id", default="jellyho/robocasa365-PrepareCoffee")
    ap.add_argument("--steps", type=int, default=120, help="video frames (progress 0->1)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--bg-frames", type=int, default=2500, help="ghosted background points")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/embedding_ride.mp4"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    meta = json.loads((args.annot / "meta.json").read_text())
    n = meta["num_frames"]
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))
    phi = np.load(args.phi).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    heroes = rng.choice(np.unique(ep), size=3, replace=False)

    # rows: background sample + ALL frames of the three hero episodes (pre-success part)
    bg = rng.choice(n, size=args.bg_frames, replace=False)
    hero_rows = {int(e): np.flatnonzero((ep == e) & (prog < 0.999)) for e in heroes}
    all_rows = np.unique(np.concatenate([bg, *list(hero_rows.values())]))
    pos_of = {int(r): i for i, r in enumerate(all_rows)}

    from sklearn.manifold import TSNE

    zn = phi[all_rows]
    zn = (zn - zn.mean(0)) / (zn.std(0) + 1e-6)
    xy = TSNE(n_components=2, init="pca", perplexity=40, random_state=args.seed).fit_transform(zn)

    # progress-synchronized indexing: for each video step t (progress p), each hero's frame is the
    # one whose progress is closest to p
    ps = np.linspace(0.0, 0.99, args.steps)
    hero_seq = {}
    for e, rows in hero_rows.items():
        pr = prog[rows]
        hero_seq[e] = rows[np.argmin(np.abs(pr[None, :] - ps[:, None]), axis=1)]

    from lerobot.datasets import lerobot_dataset

    ds = lerobot_dataset.LeRobotDataset(args.repo_id)
    cam = ds.meta.video_keys[0]
    cache = {}

    def img(i):
        i = int(i)
        if i not in cache:
            cache[i] = np.transpose(np.asarray(ds[i][cam]), (1, 2, 0))
        return cache[i]

    tmp = pathlib.Path(".scratch/_ride_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()

    for t in range(args.steps):
        fig = plt.figure(figsize=(12.8, 5.4), facecolor="#0f1117")
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.9], hspace=0.12, wspace=0.05)
        for k, e in enumerate(heroes):
            ax = fig.add_subplot(gs[k, 0])
            ax.imshow(img(hero_seq[int(e)][t]))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(EP_COLORS[k])
                sp.set_linewidth(2.5)
            ax.set_ylabel(f"ep{e}", color=EP_COLORS[k], fontsize=10)
        ax = fig.add_subplot(gs[:, 1])
        ax.scatter(xy[:, 0], xy[:, 1], c=prog[all_rows], cmap="viridis", s=3, alpha=0.25, linewidths=0)
        for k, e in enumerate(heroes):
            seq = hero_seq[int(e)][: t + 1]
            pts = xy[[pos_of[int(r)] for r in seq]]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=EP_COLORS[k], lw=2.0, alpha=0.85)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=EP_COLORS[k], s=90, zorder=5, edgecolors="w", linewidths=1.2)
        ax.set_facecolor("#181c25")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2a3140")
        ax.set_title(f"three kitchens, one road  —  task progress {ps[t]:.0%}", color="w", fontsize=13)
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
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
