"""Does z_rl encode task phase, or which video it came from?

The bottleneck is trained to reconstruct SigLIP image embeddings, so it is optimised to keep whatever
makes a frame identifiable - and in RoboCasa the thing that makes a frame identifiable is the scene:
counter texture, lighting, object placement, camera pose. Two frames at the same point of the same
task in two different demos then land far apart, while two frames from the *same* demo land together
regardless of what has been accomplished. A critic reading that token can only score trajectories it
has already seen; it cannot stitch a good prefix from one demo onto a good suffix from another,
because in latent space they are not near each other to begin with.

That story is either true or it is not, and it is measurable without training anything:

    knn episode purity   fraction of a frame's k nearest neighbours drawn from its OWN episode.
                         1.0 = the token is an episode fingerprint. Chance is reported alongside;
                         the gap is what matters, not the raw number.
    cross-episode phase  restricted to neighbours from OTHER episodes, mean |Δprogress|. This is the
                         stitching quantity: if frames that are close in latent space but far in
                         video are also at the same point of the task, the geometry supports
                         stitching. Compared against the same quantity over random cross-episode
                         pairs (chance).
    episode-ID accuracy  linear probe predicting which episode a frame came from (held-out frames).
    progress R2          linear probe predicting task progress, held out BY EPISODE - so it measures
                         phase structure that transfers across demos, not within one.
    kendall tau          TCC's label-free alignment score: take two frames of demo A, look up their
                         nearest frames in demo B, and ask whether the order survived. This is the
                         stitching question in its most literal form - can a point in one trajectory
                         be matched to the right point in another. 1 = perfectly aligned, 0 = random.
    episode eta^2        share of latent variance explained by episode identity alone.
    participation ratio  how many directions carry the variance, for context on all of the above.

Every measure is also computed on the mean-pooled prefix image embeddings, i.e. the reconstruction
TARGET. That control decides where the problem lives: if the target is already an episode
fingerprint, the bottleneck is faithfully copying what it was asked to copy and the fix belongs in
the objective, not in the encoder.

Usage:
    uv run scripts/latent_semantics.py --config pi05_robocasa_PrepareCoffee_rlt \
        --checkpoint-root checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt7_pardec_noprop_mae0.5 \
        --out .scratch/semantics_mae05.png
"""

import argparse
import dataclasses
import gc
import json
import logging
import pathlib
import re

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.progress as _progress

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def infer_overrides(name: str) -> dict:
    """Recover the model flags from the run directory name.

    The launcher builds the directory name out of the same switches it passes to train.py
    (run_train_rlt.sh), and several of them change the parameter pytree - loading a `_noprop`
    checkpoint into a default (proprio) config fails on a structure mismatch. Reading the name back
    keeps the diagnostic from needing the launch command to be remembered.
    """
    over = {}
    if "_pardec" in name:
        over["rlt_decoder_mode"] = "parallel"
    if "_noprop" in name:
        over["rlt_include_proprio"] = False
    # Longest tag first: "_reconprog" is a prefix of "_reconprogact", so testing in any other order
    # would load a checkpoint that has an action head into a config that does not build one.
    for tag, obj in (
        ("_reconprogact", "reconstruction+progress+action"),
        ("_reconactbeh", "reconstruction+action+behsim"),
        ("_reconactepadv", "reconstruction+action+epadv"),
        ("_reconbehepadv", "reconstruction+behsim+epadv"),
        ("_reconprog", "reconstruction+progress"),
        ("_reconact", "reconstruction+action"),
        ("_reconbeh", "reconstruction+behsim"),
        ("_reconepadv", "reconstruction+epadv"),
    ):
        if tag in name:
            over["rlt_objective"] = obj
            break
    if "_bbgrad" in name:
        over["rlt_backbone_gradient"] = True
    if "_scalar" in name:
        over["rlt_progress_head"] = "regression"
    if m := re.search(r"_mae([\d.]+)", name):
        over["rlt_mask_ratio"] = float(m.group(1))
    if m := re.search(r"_k(\d+)", name):
        over["rlt_num_tokens"] = int(m.group(1))
    if m := re.search(r"_d(\d+)", name):
        over["rlt_token_dim"] = int(m.group(1))
    return over


def sample_frames(repo_id: str, num_episodes: int, per_episode: int, seed: int):
    """Evenly spaced frames from evenly spaced episodes → (global idx, episode id, progress).

    Both a spread over episodes and a spread within each are required: episode purity needs several
    frames per episode to have anything to be pure about, and cross-episode phase alignment needs
    several episodes covering the same phases.
    """
    from lerobot.datasets import lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    labels = _progress.compute_progress_labels(repo_id)
    lo = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    hi = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)

    rng = np.random.default_rng(seed)
    eps = rng.choice(len(lo), size=min(num_episodes, len(lo)), replace=False)
    eps.sort()

    idx, ep_id, prog = [], [], []
    for e in eps:
        # Stop at the success onset: everything after it is progress==1 by construction, so including
        # the tail would fill the probe set with duplicate-labelled frames and inflate every measure.
        end = min(int(hi[e]), int(lo[e]) + int(labels.onsets[e]) + 1)
        n = max(end - int(lo[e]), 2)
        within = np.linspace(0, n - 1, per_episode).astype(np.int64)
        idx.append(int(lo[e]) + within)
        ep_id.append(np.full(per_episode, e, dtype=np.int64))
        prog.append(labels.progress(np.full(per_episode, e), within))
    return np.concatenate(idx), np.concatenate(ep_id), np.concatenate(prog)


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def neighbour_metrics(z, ep, prog, k):
    """k-NN episode purity and cross-episode phase agreement, each next to its chance level."""
    sim = unit(z.astype(np.float64)) @ unit(z.astype(np.float64)).T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    purity = float(np.mean(ep[nn] == ep[:, None]))
    # Chance: if neighbours were drawn uniformly, the share from one's own episode.
    counts = np.bincount(ep, minlength=ep.max() + 1)[ep]
    purity_chance = float(np.mean((counts - 1) / (len(ep) - 1)))

    same = ep[None, :] == ep[:, None]
    cross = np.where(same, -np.inf, sim)
    nnc = np.argsort(-cross, axis=1)[:, :k]
    phase_err = float(np.mean(np.abs(prog[nnc] - prog[:, None])))
    # Chance: |Δprogress| over ALL cross-episode pairs, i.e. what a latent carrying no phase
    # information would give.
    dp = np.abs(prog[None, :] - prog[:, None])
    phase_chance = float(dp[~same].mean())
    return purity, purity_chance, phase_err, phase_chance


def linear_probes(z, ep, prog, seed):
    """Episode-ID accuracy (held-out frames) and progress R2 (held out BY EPISODE).

    The two splits differ on purpose. Episode identity can only be tested on unseen frames of seen
    episodes - there is no other way to ask the question. Progress is held out by episode, because
    predicting phase on a demo the probe has never seen is precisely the transfer that stitching
    needs; a within-episode split would score high on a latent that has merely memorised each demo's
    own timeline.
    """
    x = StandardScaler().fit_transform(z.astype(np.float64))
    alphas = np.logspace(0, 5, 11)

    # Episode ID: alternate frames, so train and test are interleaved in time rather than split into
    # early/late halves (which would confound episode identity with task phase).
    tr = np.arange(len(ep)) % 2 == 0
    clf = RidgeClassifier(alpha=100.0).fit(x[tr], ep[tr])
    acc = float(clf.score(x[~tr], ep[~tr]))
    acc_chance = float(np.max(np.bincount(ep[~tr])) / (~tr).sum())

    # Progress: leave-episodes-out, averaged over folds.
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uniq), 4)
    r2s = []
    for held in folds:
        te = np.isin(ep, held)
        reg = RidgeCV(alphas=alphas).fit(x[~te], prog[~te])
        pred = reg.predict(x[te])
        # R2 against the variance of the held-out episodes' own progress.
        ss_res = float(np.sum((prog[te] - pred) ** 2))
        ss_tot = float(np.sum((prog[te] - prog[te].mean()) ** 2))
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-9))
    return acc, acc_chance, float(np.mean(r2s))


def kendall_tau(z, ep, prog):
    """TCC's cross-video alignment score, averaged over ordered pairs of episodes.

    For episodes A and B: map every frame of A to its nearest frame of B, then measure how much of
    A's temporal order survives the mapping. Label-free - it uses only the frames' own ordering - and
    it is the one number that asks directly whether a point in one demo can be located in another.
    """
    from scipy.stats import kendalltau

    u = unit(z.astype(np.float64))
    taus = []
    for a in np.unique(ep):
        ia = np.flatnonzero(ep == a)
        # Frames were sampled in increasing time order within each episode, so the row order IS the
        # temporal order and progress is monotone along it.
        order_a = np.argsort(prog[ia])
        ia = ia[order_a]
        for b in np.unique(ep):
            if a == b:
                continue
            ib = np.flatnonzero(ep == b)
            ib = ib[np.argsort(prog[ib])]
            nn = np.argmax(u[ia] @ u[ib].T, axis=1)
            t = kendalltau(np.arange(len(ia)), nn).statistic
            if np.isfinite(t):
                taus.append(float(t))
    return float(np.mean(taus)) if taus else float("nan")


def episode_eta2(z, ep):
    """Fraction of latent variance explained by episode identity: 1 - within/total.

    The blunt version of the whole question. Near 1 means the token is mostly a label for which video
    this frame came from, and whatever phase structure exists lives in the leftover.
    """
    z = z.astype(np.float64)
    total = float(np.sum((z - z.mean(0, keepdims=True)) ** 2))
    within = 0.0
    for e in np.unique(ep):
        m = ep == e
        within += float(np.sum((z[m] - z[m].mean(0, keepdims=True)) ** 2))
    return 1.0 - within / max(total, 1e-9)


def participation_ratio(z):
    zc = z.astype(np.float64) - z.astype(np.float64).mean(0, keepdims=True)
    ev = np.linalg.svd(zc, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev**2).sum())


def summarize(name, z, ep, prog, k, seed):
    purity, purity_ch, phase, phase_ch = neighbour_metrics(z, ep, prog, k)
    acc, acc_ch, r2 = linear_probes(z, ep, prog, seed)
    return {
        "repr": name,
        "knn_purity": purity,
        "knn_purity_chance": purity_ch,
        "cross_phase_err": phase,
        "cross_phase_chance": phase_ch,
        "episode_acc": acc,
        "episode_acc_chance": acc_ch,
        "progress_r2_heldout_episode": r2,
        "kendall_tau": kendall_tau(z, ep, prog),
        "episode_eta2": episode_eta2(z, ep),
        "participation_ratio": participation_ratio(z),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-root", required=True, type=pathlib.Path)
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="Subset of step dirs; default all.")
    ap.add_argument("--num-episodes", type=int, default=24)
    ap.add_argument("--per-episode", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/latent_semantics.png"))
    args = ap.parse_args()

    steps = sorted(int(p.name) for p in args.checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.steps:
        steps = [s for s in steps if s in set(args.steps)]
    if not steps:
        raise SystemExit(f"no checkpoints in {args.checkpoint_root}")

    train_config = _config.get_config(args.config)
    over = infer_overrides(args.checkpoint_root.name)
    logger.info(f"variant flags recovered from directory name: {over or '(none)'}")
    model_config = dataclasses.replace(train_config.model, **over)
    data_config = train_config.data.create(train_config.assets_dirs, model_config)

    idx, ep, prog = sample_frames(data_config.repo_id, args.num_episodes, args.per_episode, args.seed)
    logger.info(f"probe set: {len(idx)} frames over {len(np.unique(ep))} episodes")

    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    batches = [
        jax.tree.map(lambda *xs: np.stack(xs), *[dataset[int(i)] for i in idx[s : s + args.batch_size]])
        for s in range(0, len(idx), args.batch_size)
    ]

    rows = []
    for step in steps:
        model = model_config.load(
            _model.restore_params(args.checkpoint_root / str(step) / "params", dtype=jnp.bfloat16)
        )
        model.eval()

        @jax.jit
        def extract(obs, _m=model):
            obs = _model.preprocess_observation(None, obs, train=False)
            z, img_mask, _, _ = _m._prefix_forward(obs)
            z_rl = _m._encode_rl_token(z, img_mask, obs.state)
            m = img_mask[..., None].astype(jnp.float32)
            pooled = jnp.sum(z * m, axis=1) / (jnp.sum(m, axis=1) + 1e-6)
            return z_rl, pooled

        outs = [extract(_model.Observation.from_dict(b)) for b in batches]
        z_rl = np.concatenate([np.asarray(o[0], np.float32) for o in outs], 0)
        pooled = np.concatenate([np.asarray(o[1], np.float32) for o in outs], 0)
        # The jit cache entry holds the closure, which holds this checkpoint's parameters on device;
        # without clearing it every checkpoint stays resident and the run OOMs a few in.
        del model, extract, outs
        jax.clear_caches()
        gc.collect()

        for name, z in (("z_rl", z_rl), ("prefix_pooled", pooled)):
            row = {"step": step, **summarize(name, z, ep, prog, args.knn, args.seed)}
            rows.append(row)
            logger.info(
                f"{step:>7} {name:<14} purity {row['knn_purity']:.3f} (chance {row['knn_purity_chance']:.3f})"
                f"  cross-phase {row['cross_phase_err']:.3f} (chance {row['cross_phase_chance']:.3f})"
                f"  ep-acc {row['episode_acc']:.3f}  prog R2 {row['progress_r2_heldout_episode']:+.3f}"
                f"  PR {row['participation_ratio']:.1f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))

    fig, axes = plt.subplots(1, 5, figsize=(21, 3.8), dpi=150)
    for name, color in (("z_rl", "tab:blue"), ("prefix_pooled", "tab:gray")):
        r = [x for x in rows if x["repr"] == name]
        xs = [x["step"] for x in r]
        axes[0].plot(xs, [x["knn_purity"] for x in r], "-o", color=color, label=name)
        axes[1].plot(xs, [x["cross_phase_err"] for x in r], "-o", color=color, label=name)
        axes[2].plot(xs, [x["episode_acc"] for x in r], "-o", color=color, label=name)
        axes[3].plot(xs, [x["progress_r2_heldout_episode"] for x in r], "-o", color=color, label=name)
        axes[4].plot(xs, [x["kendall_tau"] for x in r], "-o", color=color, label=name)
    r0 = rows[0]
    axes[0].axhline(r0["knn_purity_chance"], ls="--", c="k", lw=0.8)
    axes[0].set_title(f"{args.knn}-NN episode purity\n(dashed = chance)", fontsize=9)
    axes[1].axhline(r0["cross_phase_chance"], ls="--", c="k", lw=0.8)
    axes[1].set_title("cross-episode neighbour |Δprogress|\nlower than chance = phase structure", fontsize=9)
    axes[2].axhline(r0["episode_acc_chance"], ls="--", c="k", lw=0.8)
    axes[2].set_title("episode-ID linear probe accuracy", fontsize=9)
    axes[3].axhline(0.0, ls="--", c="k", lw=0.8)
    axes[3].set_title("progress R2, held out by episode", fontsize=9)
    axes[4].axhline(0.0, ls="--", c="k", lw=0.8)
    axes[4].set_title("cross-episode Kendall tau\ncan a frame be located in another demo", fontsize=9)
    for a in axes:
        a.set_xlabel("training step")
        a.grid(visible=True, lw=0.4, alpha=0.4)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
