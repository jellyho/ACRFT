"""Is the policy's chunk distribution multimodal at a given state, or one continuous cloud?

Flow matching can represent multiple modes, and whether it does matters: with several modes the
right way to store a state's candidates is per-mode statistics, and truncating a chunk mid-way risks
switching between them. With one cloud, a single low-rank Gaussian describes it and interpolation
between candidates is safe.

Averaging a clustering score over states cannot answer this. Multimodality is a property of
particular states - a decision point offers two ways to proceed, a mid-reach state offers one - so
pooling washes the structure out and every dataset looks continuous. Everything here is therefore
per state, reported as a distribution over states, and tested against a null:

    for each state, split its candidates with 2-means and take SSE_2 / SSE_1
    draw the same number of points from a Gaussian with that state's own covariance
    a state counts as multimodal when its split beats what the unimodal null achieves

The null matters because 2-means always reduces SSE, unimodal or not - the question is by how much
more than chance.

Distances are taken after scaling each action dimension by its pooled within-state spread; raw units
would let end-effector translation drown out the gripper, which is where a genuine "grasp now or
later" split would live.

Usage:
    uv run scripts/candidate_modality.py --data data/rlt_critic/PrepareCoffee --out modality.png
"""

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def two_means(x, rng, restarts=5, iters=30):
    """Per-state 2-means. x: [S, N, D] -> (SSE_1, SSE_2, labels of the best split)."""
    s, n, _ = x.shape
    mu = x.mean(1, keepdims=True)
    sse1 = ((x - mu) ** 2).sum(-1).mean(-1)  # [S]
    best = np.full(s, np.inf)
    best_lab = np.zeros((s, n), dtype=int)
    for _ in range(restarts):
        c = x[:, rng.permutation(n)[:2]].copy()  # [S, 2, D]
        lab = np.zeros((s, n), dtype=int)
        for _ in range(iters):
            d = ((x[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1)  # [S, N, 2]
            lab = d.argmin(-1)
            for j in range(2):
                m = (lab == j)[..., None]
                cnt = m.sum(1)
                c[:, j] = np.where(cnt > 0, (x * m).sum(1) / np.maximum(cnt, 1), c[:, j])
        sse2 = np.take_along_axis(((x[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1), lab[..., None], 2)[..., 0]
        sse2 = sse2.mean(-1)
        upd = sse2 < best
        best = np.where(upd, sse2, best)
        best_lab = np.where(upd[:, None], lab, best_lab)
    return sse1, best, best_lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--num-states", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("candidate_modality.png"))
    args = ap.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    T, N, H, A = meta["num_frames"], meta["num_samples"], meta["horizon"], meta["action_dim"]
    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]
    cand = np.memmap(args.data / "base_action.dat", dtype=dt, mode="r", shape=(T, N, H, A))
    pp = args.data / "_progress.json"
    done = json.loads(pp.read_text())["done"] if pp.exists() else T

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(done, size=min(args.num_states, done), replace=False))
    raw = np.asarray(cand[idx], np.float32)  # [S, N, H, A]

    # Scale each action dim by its pooled within-state spread so no single dim sets the geometry.
    scale = raw.std(axis=1).mean(axis=(0, 1))  # [A]
    scale = np.where(scale > 1e-6, scale, 1.0)
    x = (raw / scale).reshape(len(idx), N, -1)

    sse1, sse2, lab = two_means(x, rng)
    ratio = sse2 / np.maximum(sse1, 1e-12)

    # Null: points drawn from each state's OWN covariance, so the comparison controls for the shape
    # and rank of that state's cloud and only asks whether it is split into separate lumps.
    xc = x - x.mean(1, keepdims=True)
    null = []
    for _ in range(4):
        w = rng.standard_normal((len(idx), N, N)) / np.sqrt(N)
        sim = np.einsum("snm,smd->snd", w, xc)  # same second moments, unimodal by construction
        s1, s2, _ = two_means(sim, rng, restarts=3)
        null.append(s2 / np.maximum(s1, 1e-12))
    null = np.concatenate(null)

    thr = np.percentile(null, 5)
    frac = float((ratio < thr).mean())
    print(f"{done:,} frames annotated; {len(idx)} states, N={N} candidates")
    print(f"\nSSE2/SSE1   observed: median {np.median(ratio):.3f}   null(unimodal): median {np.median(null):.3f}")
    print(f"5th percentile of the null = {thr:.3f}")
    print(f"\nstates splitting better than the unimodal null: {frac:.1%}")
    if frac < 0.10:
        print("  -> essentially one continuous cloud per state; a low-rank Gaussian is the right model")
    elif frac < 0.35:
        print("  -> a minority of states are genuinely multimodal (decision points), the rest are not")
    else:
        print("  -> multimodality is the norm; per-mode statistics are needed, not a single Gaussian")

    # Where do the split states sit, and how far apart are the two lumps?
    sep = []
    for i in range(len(idx)):
        a, b = x[i][lab[i] == 0], x[i][lab[i] == 1]
        if len(a) and len(b):
            spread = 0.5 * (a.std(0).mean() + b.std(0).mean()) + 1e-9
            sep.append(np.linalg.norm(a.mean(0) - b.mean(0)) / (spread * np.sqrt(x.shape[-1])))
    sep = np.array(sep)
    print(
        f"mode separation (centre gap / within-mode spread): median {np.median(sep):.2f}, p90 {np.percentile(sep, 90):.2f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=140)
    axes[0].hist(null, bins=40, alpha=0.55, label="unimodal null", color="0.6", density=True)
    axes[0].hist(ratio, bins=40, alpha=0.7, label="observed", color="tab:red", density=True)
    axes[0].axvline(thr, color="k", ls="--", lw=1, label="null 5th pct")
    axes[0].set_xlabel("SSE(2 clusters) / SSE(1 cluster)")
    axes[0].set_title(f"lower = more split\n{frac:.0%} of states beat the null", fontsize=9)
    axes[0].legend(fontsize=8)

    axes[1].hist(sep, bins=40, color="tab:purple")
    axes[1].set_xlabel("mode separation")
    axes[1].set_title("centre gap relative to mode width\n(>1 = lumps actually apart)", fontsize=9)

    # The clearest evidence either way: the most-split states, drawn in their own 2-D principal plane.
    order = np.argsort(ratio)[:6]
    ax = axes[2]
    for i in order[:1]:
        xi = x[i] - x[i].mean(0)
        p = xi @ np.linalg.svd(xi, full_matrices=False)[2][:2].T
        ax.scatter(p[:, 0], p[:, 1], c=lab[i], cmap="coolwarm", s=60, edgecolor="k", linewidth=0.5)
        ax.set_title(f"most-split state (ratio {ratio[i]:.2f})\ncandidates in their own principal plane", fontsize=9)
    for a_ in axes:
        a_.grid(visible=True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
