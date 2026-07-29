"""How long does stage 1 need? Measure whether the RL token is still moving.

The probe reports how much policy survives the bottleneck, but it is itself a head learned from a
moving target, so it lags: the latent can stop improving thousands of steps before the probe stops
climbing, and the probe cannot tell you which of the two is happening. What can is the latent itself.
On a fixed set of frames, extract z_rl at successive checkpoints and ask how much it changed:

    drift(T)  =  1 - mean_x cos( z_T(x), z_{T+d}(x) )

No head, no fitting, no lag - just the representation compared against its own past. When it flattens
the encoder has stopped moving and any further probe gain is the probe catching up, not the token
getting better. Reported alongside two properties of each snapshot on their own:

    participation ratio   (sum L)^2 / sum L^2 of the token covariance: how many directions carry the
                          variance. A collapsing bottleneck loses these before anything downstream
                          notices.
    neighbourhood churn   the fraction of each frame's k nearest neighbours (in token space) that are
                          different from the previous checkpoint's. Cosine drift can sit at a small
                          value while the geometry reorders underneath it; this catches that, and it
                          is the part a critic reading the token would actually feel.

Usage:
    uv run scripts/latent_drift.py --config pi05_robocasa_PrepareCoffee_rlt \
        --checkpoint-root checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_reconprog \
        --out .scratch/drift.png
"""

import argparse
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-root", required=True, type=pathlib.Path, help="Dir holding step subdirs.")
    ap.add_argument("--num-frames", type=int, default=512, help="Fixed probe set, identical for every step.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("latent_drift.png"))
    args = ap.parse_args()

    steps = sorted(int(p.name) for p in args.checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit())
    if len(steps) < 2:
        raise SystemExit(f"need at least two checkpoints in {args.checkpoint_root}, found {steps}")
    logger.info(f"checkpoints: {steps}")

    train_config = _config.get_config(args.config)
    model_config = train_config.model
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    dataset = _data_loader.transform_dataset(dataset, data_config)

    # One fixed, evenly spaced probe set: the comparison is only meaningful if every checkpoint sees
    # exactly the same frames, and spreading them over the dataset keeps one episode from dominating.
    idx = np.linspace(0, len(dataset) - 1, args.num_frames).astype(int)
    batches = [
        jax.tree.map(lambda *xs: np.stack(xs), *[dataset[int(i)] for i in idx[s : s + args.batch_size]])
        for s in range(0, len(idx), args.batch_size)
    ]
    logger.info(f"probe set: {len(idx)} frames in {len(batches)} batches")

    tokens = {}
    for step in steps:
        model = model_config.load(
            _model.restore_params(args.checkpoint_root / str(step) / "params", dtype=jnp.bfloat16)
        )
        model.eval()

        @jax.jit
        def extract(rng, obs, _m=model):
            z, _ = _m.extract_token_and_base_actions(rng, obs, num_samples=1, num_steps=4)
            return z

        rng = jax.random.key(args.seed)  # the token is deterministic given the observation
        zs = [np.asarray(extract(rng, _model.Observation.from_dict(b)), np.float32) for b in batches]
        tokens[step] = np.concatenate(zs, 0)
        logger.info(f"  {step}: {tokens[step].shape}")
        del model

    def unit(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

    def knn_sets(a, k):
        u = unit(a)
        sim = u @ u.T
        np.fill_diagonal(sim, -np.inf)
        return np.argsort(-sim, axis=1)[:, :k]

    rows = []
    prev_knn = None
    for i, step in enumerate(steps):
        z = tokens[step]
        zc = z - z.mean(0, keepdims=True)
        ev = np.linalg.svd(zc.astype(np.float64), compute_uv=False) ** 2
        pr = float(ev.sum() ** 2 / (ev**2).sum())
        nn = knn_sets(z, args.knn)
        churn = np.nan
        drift = np.nan
        if i:
            drift = float(1.0 - np.mean(np.sum(unit(z) * unit(tokens[steps[i - 1]]), axis=-1)))
            churn = float(np.mean([1.0 - len(set(a) & set(b)) / args.knn for a, b in zip(nn, prev_knn, strict=True)]))
        prev_knn = nn
        rows.append({"step": step, "drift": drift, "knn_churn": churn, "participation_ratio": pr})
        logger.info(f"{step:>7}  drift {drift:.4f}  knn churn {churn:.3f}  participation {pr:.1f}")

    print("\n   step     drift   knn churn   participation ratio")
    for r in rows:
        print(f"{r['step']:>7}  {r['drift']:8.4f}  {r['knn_churn']:9.3f}  {r['participation_ratio']:19.1f}")
    tail = [r["drift"] for r in rows[1:]]
    if len(tail) >= 3 and tail[-1] > 0.5 * max(tail):
        print("\n  -> drift has NOT flattened: the encoder is still moving, more steps are still buying something")
    elif len(tail) >= 3:
        print("\n  -> drift has flattened: further probe gain is the probe catching up, not a better token")

    xs = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=150)
    axes[0].plot(xs[1:], [r["drift"] for r in rows[1:]], "-o", color="tab:blue")
    axes[0].set_title("token drift between consecutive checkpoints\nflat = stage 1 has converged", fontsize=9)
    axes[0].set_ylabel("1 - cosine")
    axes[1].plot(xs[1:], [r["knn_churn"] for r in rows[1:]], "-o", color="tab:orange")
    axes[1].set_title(f"{args.knn}-NN churn\nhow much the geometry reorders", fontsize=9)
    axes[2].plot(xs, [r["participation_ratio"] for r in rows], "-o", color="tab:green")
    axes[2].set_title("participation ratio\ndirections carrying the variance", fontsize=9)
    for a in axes:
        a.set_xlabel("training step")
        a.grid(visible=True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
