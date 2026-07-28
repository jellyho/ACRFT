"""How many candidates does it take to cover what the VLA would do?

Faithfully estimating a 192-dimensional action-chunk distribution is hopeless at any sample size,
and it is not what the method needs. What it needs is the point at which drawing more chunks stops
producing better ones - beyond that, N only adds critic queries and inflates the arg-max.

Every measure here is computed from the annotation output alone, with no critic, so it can be run
before (or alongside) critic training:

  effective diversity   participation ratio of the candidate set, (sum L)^2 / sum L^2. Sixteen
                        samples spanning three effective directions means the rest are near-copies.
  distinct modes        candidates within `--mode-eps` of each other counted as one, as a function
                        of how many were drawn. Flow matching is multimodal; this counts the modes.
  nearest-neighbour     mean distance to the closest other candidate, against m. Still falling
                        steeply means the mode structure is under-sampled.
  distance to the demo  min over the first m candidates of the distance to the chunk actually
                        executed. This is the one that matters: once it flattens, more samples are
                        not getting closer to good behaviour.

Usage:
    uv run scripts/candidate_diversity.py --data data/rlt_critic/PrepareCoffee --out diversity.png
"""

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    ap.add_argument("--num-states", type=int, default=1024, help="States sampled for the curves.")
    ap.add_argument(
        "--mode-eps",
        type=float,
        default=0.5,
        help="Two candidates count as the same mode within this many median-pairwise-distances.",
    )
    ap.add_argument("--repeats", type=int, default=16, help="Random candidate subsets averaged per m.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("candidate_diversity.png"))
    args = ap.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    T, N, H, A = meta["num_frames"], meta["num_samples"], meta["horizon"], meta["action_dim"]
    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]
    cand = np.memmap(args.data / "base_action.dat", dtype=dt, mode="r", shape=(T, N, H, A))
    demo = np.memmap(args.data / "action_chunk.dat", dtype=dt, mode="r", shape=(T, H, A))
    done = json.loads((args.data / "_progress.json").read_text())["done"] if (args.data / "_progress.json").exists() else T
    print(f"{done:,}/{T:,} frames annotated, N={N}, chunk {H}x{A}")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(done, size=min(args.num_states, done), replace=False))
    C = np.asarray(cand[idx], np.float32).reshape(len(idx), N, -1)  # [S, N, H*A]
    Dm = np.asarray(demo[idx], np.float32).reshape(len(idx), -1)  # [S, H*A]

    # Per-state pairwise distances, used by every curve below.
    diff = C[:, :, None, :] - C[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # [S, N, N]
    med = np.median(dist[:, np.triu_indices(N, 1)[0], np.triu_indices(N, 1)[1]], axis=-1)  # [S]

    # Effective diversity: participation ratio of the candidate set (bounded by min(N-1, H*A)).
    Cc = C - C.mean(1, keepdims=True)
    gram = np.einsum("snd,smd->snm", Cc, Cc)
    tr = np.trace(gram, axis1=1, axis2=2) / N
    fro = np.sum(gram * gram, axis=(1, 2)) / (N * N)
    pr = tr * tr / (fro + 1e-12)
    print(f"\neffective diversity (participation ratio): {pr.mean():.2f}  of at most {min(N - 1, H * A)}")
    print(f"  -> beyond ~{pr.mean():.0f} directions the extra candidates are largely redundant")

    ms = [m for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if m <= N]
    modes, nn, to_demo = [], [], []
    for m in ms:
        mo, nnd, dd = [], [], []
        for _ in range(args.repeats):
            sel = rng.permutation(N)[:m]
            d = dist[:, sel][:, :, sel]
            # Greedy single-link count: how many candidates are further than eps from all earlier ones.
            thr = args.mode_eps * med
            keep = np.ones((len(idx), m), dtype=bool)
            for i in range(1, m):
                closer = d[:, i, :i] < thr[:, None]
                keep[:, i] = ~np.any(closer & keep[:, :i], axis=-1)
            mo.append(keep.sum(-1).mean())
            if m > 1:
                dm = d + np.eye(m)[None] * 1e9
                nnd.append(dm.min(-1).mean())
            dd.append(np.linalg.norm(C[:, sel] - Dm[:, None], axis=-1).min(-1).mean())
        modes.append(np.mean(mo))
        nn.append(np.mean(nnd) if nnd else np.nan)
        to_demo.append(np.mean(dd))

    print("\n m   distinct modes   nn-dist   dist-to-demo   (demo gain vs previous)")
    for i, m in enumerate(ms):
        gain = "" if i == 0 else f"{to_demo[i - 1] - to_demo[i]:+.4f}"
        print(f"{m:3d}   {modes[i]:12.2f}   {nn[i]:7.3f}   {to_demo[i]:12.4f}   {gain:>10s}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=140)
    axes[0].plot(ms, modes, "-o", color="tab:blue")
    axes[0].set_title(f"distinct modes (eps={args.mode_eps} x median dist)\nsaturation = extra draws are copies", fontsize=9)
    axes[0].set_ylabel("modes")
    axes[1].plot(ms, nn, "-o", color="tab:orange")
    axes[1].set_title("mean nearest-neighbour distance\nstill falling = under-sampled", fontsize=9)
    axes[2].plot(ms, to_demo, "-o", color="tab:green")
    axes[2].set_title("distance to the executed chunk\nflat = more samples find nothing better", fontsize=9)
    for ax in axes:
        ax.set_xlabel("candidates drawn (m)")
        ax.set_xscale("log", base=2)
        ax.grid(visible=True, lw=0.5, alpha=0.4)
    fig.suptitle(f"Candidate coverage — effective diversity {pr.mean():.1f} of {min(N - 1, H * A)}", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
