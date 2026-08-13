"""HILP-paper-style visualization: does episode identity actually disappear from the space?

For each representation (VLA token, cheap-z, and cheap-z's HILP phi-space) this renders:
  row 1  t-SNE of ~1.8k frames colored by EPISODE — islands = episode clustering (the pathology)
  row 2  the SAME t-SNE colored by PROGRESS — a smooth gradient = task phase is the organizing axis
  row 3  distance vs |dprogress| for frame pairs, within-episode (gray) and cross-episode (color):
         HILP's promise is that the CROSS-episode points fall on the same ridge as within —
         the metric measures "how far through the task", not "which kitchen".

Reads only cached arrays; CPU-only (sklearn t-SNE), a few minutes.
"""

import argparse
import json
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rep(spec: str, n: int):
    name, _, p = spec.partition("=")
    p = pathlib.Path(p)
    if (p / "z.npy").exists():
        z = np.load(p / "z.npy").astype(np.float32)
    else:
        m = json.loads((p / "meta.json").read_text())
        z = np.array(np.memmap(p / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, m["token_dim"])))
    return name, z, p


def phi_of(run_dir: pathlib.Path, z: np.ndarray):
    """Apply the run's HILP head phi to z, reconstructed from head.pt (hilp.* keys)."""
    import torch

    sd = torch.load(run_dir / "head.pt", map_location="cpu")
    keys = [k for k in sd if k.startswith("hilp.")]
    if not keys:
        return None
    w0, b0 = sd["hilp.0.weight"], sd["hilp.0.bias"]
    w2, b2 = sd["hilp.2.weight"], sd["hilp.2.bias"]
    zt = torch.from_numpy(z)
    h = torch.nn.functional.gelu(zt @ w0.T + b0)
    return (h @ w2.T + b2).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reps",
        nargs="+",
        default=[
            "VLA-z=.scratch/annot_noprop",
            "cheap-z(v7a)=.scratch/cheap_z_v7a",
        ],
    )
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--per-ep", type=int, default=60)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/viz_hilp_space.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.annot / "meta.json").read_text())
    n = meta["num_frames"]
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))

    rng = np.random.default_rng(args.seed)
    eps = rng.choice(np.unique(ep), size=args.episodes, replace=False)
    rows = []
    for e in eps:
        r = np.flatnonzero(ep == e)
        rows.append(r[np.linspace(0, len(r) - 1, min(args.per_ep, len(r))).astype(int)])
    rows = np.concatenate(rows)
    ep_s, prog_s = ep[rows], prog[rows]
    ep_color = {e: i for i, e in enumerate(eps)}
    ep_idx = np.array([ep_color[e] for e in ep_s])

    # build the representation list; if a cheap-z run has a HILP head, add its phi-space too
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
    fig, axes = plt.subplots(3, R, figsize=(5.2 * R, 13.5), facecolor="#0f1117")
    if R == 1:
        axes = axes[:, None]

    for c, (name, z) in enumerate(reps):
        zn = (z - z.mean(0)) / (z.std(0) + 1e-6)
        xy = TSNE(n_components=2, init="pca", perplexity=40, random_state=args.seed).fit_transform(zn)

        ax = axes[0, c]
        ax.scatter(xy[:, 0], xy[:, 1], c=ep_idx, cmap="tab20", s=6, alpha=0.85, linewidths=0)
        ax.set_title(f"{name}\ncolor = EPISODE ID  (islands = pathology)", color="w", fontsize=11)

        ax = axes[1, c]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=prog_s, cmap="viridis", s=6, alpha=0.9, linewidths=0)
        ax.set_title("color = PROGRESS  (smooth gradient = healthy)", color="w", fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.04)

        # distance vs |dprogress|: 4k within-ep pairs + 4k cross-ep pairs
        from scipy.stats import spearmanr

        m = len(rows)
        i1 = rng.integers(0, m, 6000)
        j1 = rng.integers(0, m, 6000)
        within = ep_s[i1] == ep_s[j1]
        d = np.linalg.norm(zn[i1] - zn[j1], axis=1)
        dp = np.abs(prog_s[i1] - prog_s[j1])
        ax = axes[2, c]
        ax.scatter(dp[within], d[within], s=4, alpha=0.35, color="#9aa3b2", label="within-episode")
        ax.scatter(dp[~within], d[~within], s=4, alpha=0.35, color="#60a5fa", label="cross-episode")
        rho_w = spearmanr(dp[within], d[within]).statistic
        rho_x = spearmanr(dp[~within], d[~within]).statistic
        # the episode-identity offset: cross-episode distance at dp~0 vs within at dp~0
        near = dp < 0.05
        gap = d[~within & near].mean() - d[within & near].mean() if (~within & near).sum() > 10 else float("nan")
        ax.set_title(
            f"distance vs |d(progress)|   rho within {rho_w:.2f} / cross {rho_x:.2f}\n"
            f"episode-offset @dp<0.05: {gap:+.2f}  (0 = episode identity gone)",
            color="w",
            fontsize=10,
        )
        ax.set_xlabel("|d(progress)|", color="w")
        ax.set_ylabel("d(z_i, z_j)", color="w")
        ax.legend(facecolor="#181c25", labelcolor="w", fontsize=8)

    for ax in axes.ravel():
        ax.set_facecolor("#181c25")
        for s in ax.spines.values():
            s.set_color("#2a3140")
        ax.tick_params(colors="#9aa3b2", labelsize=7)

    fig.suptitle("Episode identity vs task progress — HILP-style space visualization", color="w", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor="#0f1117")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
