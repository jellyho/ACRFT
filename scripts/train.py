import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, entity=config.wandb_entity)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            entity=config.wandb_entity,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # Some models (e.g. Pi0RLT) return (per-sample loss, aux-metrics dict); the rest return just the
        # per-sample loss. Normalize to (loss, aux) so extra diagnostics can be logged for free.
        out = model.compute_loss(rng, observation, actions, train=True)
        chunked_loss, aux = out if isinstance(out, tuple) else (out, {})
        return jnp.mean(chunked_loss), aux

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        # Emitted by every config so BC and RLT runs are directly comparable on one chart: for a
        # plain BC model this is the whole loss, while for Pi0RLT the top-level `loss` also carries
        # the RLT term and only this key isolates the policy objective.
        "bc_loss": aux.pop("bc_loss", loss),
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **aux,
    }
    return new_state, info


def compute_action_dist_metrics(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    num_samples: int,
) -> dict[str, at.Array]:
    """Measure how diverse the policy's action distribution is for a given observation.

    pi05 is a flow-matching policy: for a fixed observation it maps different noise samples to
    different action chunks. We draw ``num_samples`` chunks per observation (using the EMA params,
    i.e. what inference uses) and report the spread across those samples. If this spread keeps
    shrinking toward zero, the policy is collapsing to a single action per state (overfitting);
    healthy training keeps some spread. ``data_std`` is the marginal spread of the ground-truth
    actions across the batch, as a scale reference (actions are normalized, so ~1).
    """
    observation, actions = batch
    params = state.ema_params if state.ema_params is not None else state.params
    graphdef = state.model_def

    def sample_one(sample_rng: at.KeyArrayLike) -> _model.Actions:
        model = nnx.merge(graphdef, params)
        model.eval()
        return model.sample_actions(sample_rng, observation)

    # Sequential map over the K noise seeds (small compiled graph, bounded memory) rather than
    # unrolling K full sampling passes.
    rngs = jax.random.split(rng, num_samples)
    samples = jax.lax.map(sample_one, rngs)  # (num_samples, b, action_horizon, action_dim)
    std_over_samples = jnp.std(samples, axis=0)  # per (obs, horizon, dim) spread across samples
    sample_std = jnp.mean(std_over_samples)
    data_std = jnp.mean(jnp.std(actions, axis=0))
    return {
        "action_dist/sample_std": sample_std,
        "action_dist/sample_std_median": jnp.median(std_over_samples),
        "action_dist/data_std": data_std,
        "action_dist/sample_to_data_ratio": sample_std / (data_std + 1e-8),
    }


def compute_rlt_metrics(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,  # noqa: ARG001  (kept for a uniform monitor signature)
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Cheap RLT embedding-quality diagnostics (device-side).

    Reports the ``participation ratio`` of the RL-token batch covariance — (Σλ)²/Σλ², the effective
    number of dimensions the tokens span. It falls toward 1 if the token collapses to a single
    direction (bad) and rises toward min(batch, dim) if the tokens use many dimensions (rich). Uses
    the EMA params (what downstream RL would consume) when available.
    """
    observation, _ = batch
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    diag = model.rlt_bypass_diagnostics(observation)
    z = diag["z_rl"]  # [b, D]
    b = z.shape[0]
    zc = z - jnp.mean(z, axis=0, keepdims=True)
    # participation ratio without forming the DxD covariance: use the [b,b] Gram matrix.
    #   Σλ = tr(C) = ||zc||²/b ;  Σλ² = ||C||_F² = ||zc zcᵀ||_F² / b²
    gram = zc @ zc.T  # [b, b]
    tr_c = jnp.sum(zc * zc) / b
    frob2_c = jnp.sum(gram * gram) / (b * b)
    part_ratio = (tr_c * tr_c) / (frob2_c + 1e-12)
    metrics = {
        # Depends only on the token itself, so it stays meaningful for every objective — and matters
        # most when the objective is progress-only, where a single scalar target makes collapse the
        # main risk.
        "rlt/participation_ratio": part_ratio,
    }
    if "recon_real" in diag:
        # ~1 means the decoder reconstructs fine with somebody else's RL token, i.e. it is ignoring
        # the bottleneck and a low recon loss is NOT evidence the token carries anything.
        metrics["rlt/bypass_ratio"] = diag["recon_shuffled"] / (diag["recon_real"] + 1e-9)
        metrics["rlt/target_abs_mean"] = diag["target_abs_mean"]
    return metrics


def extract_rlt_embeddings(
    state: training_utils.TrainState, batch: tuple[_model.Observation, _model.Actions]
) -> tuple[at.Array, at.Array]:
    """Return (z_rl [b, D], proprio [b, ad]) for host-side visualization / probing."""
    observation, _ = batch
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    return model.extract_rl_token(observation), observation.state


def _obs_thumbnails(observation: _model.Observation, n: int, stride: int = 2) -> np.ndarray:
    """Small uint8 previews of the camera view(s), one per frame: [n, h, w*cams, 3].

    Used to show *what the robot was looking at* next to each embedding point. Prefers the wrist
    camera — it is the view that actually distinguishes task phases (the fixed exterior cameras look
    nearly identical for most of an episode). Falls back to every view if there is no wrist key.
    Subsampled by ``stride`` because these are thumbnails, not training data.
    """
    wrist = [v for k, v in observation.images.items() if "wrist" in k and "right" not in k]
    chosen = wrist or list(observation.images.values())
    views = []
    for img in chosen:
        x = np.asarray(img[:n], dtype=np.float32)[:, ::stride, ::stride]
        # Model-space images are normalized; map whichever convention back to bytes.
        if x.min() < -0.01:  # [-1, 1]
            x = (x + 1.0) * 127.5
        elif x.max() <= 1.001:  # [0, 1]
            x = x * 255.0
        views.append(np.clip(x, 0, 255).astype(np.uint8))
    return np.concatenate(views, axis=2) if views else np.zeros((n, 1, 1, 3), np.uint8)


def build_rlt_trajectory_probe(
    config: _config.TrainConfig, *, n_traj: int, n_frames: int, pad_to: int
) -> tuple[_model.Observation, at.Array, np.ndarray, np.ndarray, int, np.ndarray]:
    """A FIXED probe set of consecutive frames from a few episodes, for trajectory-path visualization.

    Training batches are shuffled across episodes and timesteps, so they carry no temporal structure.
    To see how the RL token evolves *along* a rollout we need ordered frames, so we sample ``n_frames``
    evenly-spaced frames from each of ``n_traj`` episodes and keep that set fixed for the whole run
    (so successive visualizations are comparable).

    The batch is padded (by repeating the last frame) to a multiple of ``pad_to`` so the extraction can
    reuse the already-compiled extractor instead of triggering a recompile per chunk.

    Returns (observation, actions, episode_ids, t_norm, n_valid).
    """
    from lerobot.datasets import lerobot_dataset

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    episodes = meta.episodes

    # Deterministic, spread across the dataset so the probe isn't all near-identical episodes.
    chosen = np.unique(np.linspace(0, meta.total_episodes - 1, n_traj).astype(int))

    idxs: list[int] = []
    ep_of: list[int] = []
    t_of: list[float] = []
    for e in chosen:
        lo = int(episodes["dataset_from_index"][int(e)])
        hi = int(episodes["dataset_to_index"][int(e)])
        # Keep the sampled action chunk inside the episode.
        hi = max(lo + 1, hi - config.model.action_horizon)
        sel = np.unique(np.linspace(lo, hi - 1, n_frames).astype(int))
        span = max(1, hi - 1 - lo)
        idxs.extend(int(i) for i in sel)
        ep_of.extend([int(e)] * len(sel))
        t_of.extend(float((i - lo) / span) for i in sel)

    n_valid = len(idxs)
    padded = idxs + [idxs[-1]] * ((-n_valid) % pad_to)
    items = [dataset[i] for i in padded]
    batch = jax.tree.map(lambda *xs: np.stack(xs), *items)
    observation = _model.Observation.from_dict(batch)
    return (
        observation,
        batch["actions"],
        np.asarray(ep_of),
        np.asarray(t_of),
        n_valid,
        _obs_thumbnails(observation, n_valid),
    )


def log_rlt_embedding_vis(
    z_traj: np.ndarray,
    proprio_traj: np.ndarray,
    ep_ids: np.ndarray,
    t_norm: np.ndarray,
    z_rand: np.ndarray,
    step: int,
    thumbs: np.ndarray | None = None,
) -> dict:
    """Trajectory-aware RLT embedding visualization.

    Projects the RL tokens onto their top-2 principal components and draws each probe episode as a
    *path* (line + markers colored by normalized time t), over a grey scatter of the current random
    training batch for context. A healthy token traces smooth, ordered paths (embedding moves
    consistently as the task progresses); a collapsed or uninformative token gives tangled blobs.

    PC1 is sign-oriented to correlate positively with t, so "progress runs left→right" and successive
    visualizations stay comparable.

    Also reports held-out-EPISODE linear-probe R²: fit z -> t on some episodes, score on unseen ones.
    That measures whether task progress is linearly decodable from the token and generalizes across
    episodes — the property the downstream RL critic actually needs.

    Returns a wandb-loggable dict (static image, interactive table, optional interactive plotly, R²s).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_traj = np.asarray(z_traj, dtype=np.float64)
    z_rand = np.asarray(z_rand, dtype=np.float64)
    t_norm = np.asarray(t_norm, dtype=np.float64)
    ep_ids = np.asarray(ep_ids)

    # Two projections, answering two different questions — neither alone is trustworthy:
    #   PCA        : unbiased. "What dominates the representation?" Used for every metric, because a
    #                metric read off a supervised axis would be circular.
    #   progress   : targeted. "Is progress in there at all?" Fit to the label, so it flatters by
    #                construction — a clean left-to-right march here proves much less than it looks.
    # Both are recomputed per call, so neither is strictly comparable across steps (the basis rotates
    # as the embedding moves); they are snapshots of structure, not a fixed coordinate system.
    z_all = np.concatenate([z_traj, z_rand], axis=0)
    mean = z_all.mean(0, keepdims=True)
    _, sv, vt = np.linalg.svd(z_all - mean, full_matrices=False)
    evr = float((sv[:2] ** 2).sum() / ((sv**2).sum() + 1e-12))
    zc_traj, zc_rand = z_traj - mean, z_rand - mean

    axes_pca = vt[:2].copy()
    if np.corrcoef(zc_traj @ axes_pca[0], t_norm)[0, 1] < 0:
        axes_pca[0] = -axes_pca[0]
    pca_traj, pca_rand = zc_traj @ axes_pca.T, zc_rand @ axes_pca.T

    k = int(min(64, max(2, len(z_traj) - 1)))
    feats = zc_traj @ vt[:k].T
    w = np.linalg.solve(feats.T @ feats + 1e-2 * np.eye(k), feats.T @ (t_norm - t_norm.mean()))
    ax1 = vt[:k].T @ w
    if np.linalg.norm(ax1) < 1e-9:
        axes_prog = axes_pca
    else:
        ax1 = ax1 / np.linalg.norm(ax1)
        resid = vt[:2] - np.outer(vt[:2] @ ax1, ax1)
        j2 = int(np.argmax(np.linalg.norm(resid, axis=1)))
        ax2 = resid[j2] / (np.linalg.norm(resid[j2]) + 1e-12)
        axes_prog = np.stack([ax1, ax2])
        if np.corrcoef(zc_traj @ axes_prog[0], t_norm)[0, 1] < 0:
            axes_prog[0] = -axes_prog[0]
    prog_traj, prog_rand = zc_traj @ axes_prog.T, zc_rand @ axes_prog.T

    # Metrics and the hover panel default to the UNBIASED projection.
    pc_traj, pc_rand = pca_traj, pca_rand

    # Nonlinear projections. They can expose structure a linear map folds away, but they are read
    # differently: distances between far-apart points are not meaningful, and the layout is redrawn
    # from scratch each call, so they show shape WITHIN a snapshot rather than movement across steps.
    projections = {"PCA": (pca_traj, pca_rand), "progress-aligned": (prog_traj, prog_rand)}
    n_all = len(z_all)
    try:
        from sklearn.manifold import TSNE

        emb = TSNE(
            n_components=2,
            # Larger perplexity keeps a continuous path continuous; small values shatter it
            # into false islands, which is exactly the artefact that would mislead here.
            perplexity=float(max(10, min(50, (n_all - 1) // 3))),
            init="pca",  # far more stable than random init, and keeps some global structure
            random_state=0,
        ).fit_transform(z_all)
        projections["t-SNE"] = (emb[: len(z_traj)], emb[len(z_traj) :])
    except Exception:  # noqa: BLE001
        pass
    try:
        import umap

        emb = umap.UMAP(
            n_components=2,
            n_neighbors=int(max(5, min(40, n_all - 1))),  # higher = more global/continuous
            min_dist=0.1,
            random_state=0,
        ).fit_transform(z_all)
        projections["UMAP"] = (emb[: len(z_traj)], emb[len(z_traj) :])
    except Exception:  # noqa: BLE001
        pass

    def _heldout_r2(target: np.ndarray) -> float:
        """Ridge on top-k PCs, fit on half the episodes and scored on the held-out half.

        Features are standardized so the ridge penalty actually bites (raw PCA scores carry the
        singular-value scale, against which a small absolute penalty is a no-op), and k is kept well
        below the number of training frames so the fit measures generalization rather than memorization.
        """
        uniq = np.unique(ep_ids)
        if len(uniq) < 2:
            return float("nan")
        tr = np.isin(ep_ids, uniq[::2])  # interleaved, so both halves span the dataset
        va = ~tr
        if tr.sum() < 4 or va.sum() < 2:
            return float("nan")
        k = int(min(16, max(2, tr.sum() // 3)))
        feats = (z_traj - mean) @ vt[:k].T
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
        f = (feats - mu) / sd
        ftr, fva = f[tr], f[va]
        y_mu = target[tr].mean(0, keepdims=True)
        w = np.linalg.solve(ftr.T @ ftr + np.eye(k), ftr.T @ (target[tr] - y_mu))
        pred = fva @ w + y_mu
        ss_res = float(np.sum((target[va] - pred) ** 2))
        ss_tot = float(np.sum((target[va] - target[va].mean(0, keepdims=True)) ** 2)) + 1e-9
        return float(1.0 - ss_res / ss_tot)

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    progress_r2 = _heldout_r2(t_norm[:, None])
    # Fit-free companion: within each episode, does PC1 advance monotonically with time? This is the
    # robust signal (no tiny-sample regression), and is what the trajectory paths show visually.
    per_ep_rho = [_spearman(pc_traj[ep_ids == e, 0], t_norm[ep_ids == e]) for e in np.unique(ep_ids)]
    progress_spearman = float(np.nanmean(per_ep_rho)) if len(per_ep_rho) else float("nan")

    # --- static figure: the same trajectories under both projections ------------------------------
    cmap = plt.get_cmap("tab10")

    def _draw(ax, tr, rnd, title, xlabel):
        ax.scatter(rnd[:, 0], rnd[:, 1], c="#d9dce1", s=14, linewidth=0, zorder=1, label="random batch")
        sc = None
        for j, e in enumerate(np.unique(ep_ids)):
            m = ep_ids == e
            p, tt = tr[m], t_norm[m]
            o = np.argsort(tt)
            p, tt = p[o], tt[o]
            ax.plot(p[:, 0], p[:, 1], "-", color=cmap(j % 10), lw=1.4, alpha=0.85, zorder=2)
            sc = ax.scatter(p[:, 0], p[:, 1], c=tt, cmap="viridis", s=24, vmin=0, vmax=1, zorder=3,
                            edgecolor="white", linewidth=0.3)
            ax.scatter(p[0, 0], p[0, 1], marker="o", s=90, facecolor="none", edgecolor=cmap(j % 10), lw=1.8, zorder=4)
            ax.scatter(p[-1, 0], p[-1, 1], marker="*", s=160, color=cmap(j % 10), zorder=4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        return sc

    out_cont = {}
    n_p = len(projections)
    fig, axes_f = plt.subplots(1, n_p, figsize=(5.6 * n_p, 5.0), dpi=130, squeeze=False)
    _NOTE = {
        "PCA": "unbiased · distances faithful",
        "progress-aligned": "SUPERVISED axis (flatters by construction)",
        "t-SNE": "local neighbourhoods faithful · far-apart distances are not",
        "UMAP": "local neighbourhoods faithful · far-apart distances are not",
    }
    sc = None
    for ax_i, (name, (tr, rnd)) in zip(axes_f[0], projections.items()):
        cont = _temporal_continuity(tr, ep_ids, t_norm)
        sc = _draw(ax_i, tr, rnd, f"{name}   (temporal continuity {cont:.2f})\n{_NOTE.get(name, '')}", name)
        out_cont[name] = cont
    if sc is not None:
        fig.colorbar(sc, ax=axes_f[0][-1], label="normalized time t")
    fig.suptitle(
        f"RLT z_rl (step {step:,})   progress ρ={progress_spearman:.2f} (on PC1)  ·  "
        f"held-out probe R²={progress_r2:.2f}  ·  top-2 PCs = {evr:.1%} of variance   (○ start, ★ end)",
        fontsize=10,
    )
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)

    # --- projection-free structure plots ---------------------------------------------------------
    # A 2-D projection only shows structure once the token has some; early in training it is noise.
    # These two read the geometry directly, so they stay informative before the PCA plot does.
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=130)

    # (a) Distance from each frame's token to that episode's final (near-success) token. If the token
    #     tracks progress at all, these curves fall monotonically toward 0 as the task completes.
    for j, e in enumerate(np.unique(ep_ids)):
        m = ep_ids == e
        zz, tt = z_traj[m], t_norm[m]
        o = np.argsort(tt)
        zz, tt = zz[o], tt[o]
        d = np.linalg.norm(zz - zz[-1], axis=-1)
        ax_a.plot(tt, d / (d.max() + 1e-9), "-", color=cmap(j % 10), lw=1.4, alpha=0.85, label=f"ep {e}")
    ax_a.set_xlabel("normalized time t")
    ax_a.set_ylabel("‖z_t − z_final‖  (per-episode normalized)")
    ax_a.set_title("distance to the episode's final token\n(monotone ↓ = token tracks progress)", fontsize=9)
    ax_a.grid(True, lw=0.5, alpha=0.4)

    # (b) Cosine-similarity matrix over all probe frames, ordered by (episode, t). Bright blocks on the
    #     diagonal mean the token mostly encodes WHICH episode; a smooth gradient inside each block
    #     (and bright off-diagonal bands between episodes) means it encodes the task PHASE instead.
    o = np.lexsort((t_norm, ep_ids))
    zn = z_traj[o]
    zn = zn / (np.linalg.norm(zn, axis=-1, keepdims=True) + 1e-9)
    sim = zn @ zn.T
    im = ax_b.imshow(sim, cmap="magma", interpolation="nearest")
    fig2.colorbar(im, ax=ax_b, label="cosine similarity")
    bounds = np.cumsum([int((ep_ids == e).sum()) for e in np.unique(ep_ids)])[:-1]
    for bnd in bounds:
        ax_b.axhline(bnd - 0.5, color="#5cf", lw=0.6)
        ax_b.axvline(bnd - 0.5, color="#5cf", lw=0.6)
    ax_b.set_title("token cosine similarity, ordered by (episode, t)\n(blocks = episode identity, "
                   "bands = shared phase)", fontsize=9)
    ax_b.set_xlabel("frame (episode, then time)")
    fig2.tight_layout()
    structure_img = wandb.Image(fig2)
    plt.close(fig2)

    out = {
        # One continuity number, on the unbiased projection. The per-projection values are still
        # printed in the panel titles, which is where they are actually compared.
        "rlt/continuity": out_cont.get("PCA", float("nan")),
        "rlt/embedding_structure": structure_img,
        "rlt/embedding_pca": img,
        "rlt/probe_progress_r2": progress_r2,
    }

    # --- interactive: a wandb.Table is queryable/chartable in the UI without extra deps ---
    # Each trajectory row carries the camera view it came from, so you can see WHICH SCENE a point in
    # embedding space corresponds to (plotly hover cannot render images; a media column can).
    # The `embedding` column additionally feeds W&B's Embedding Projector panel, which projects in the
    # browser and shows the image on hover. The probe frames are fixed, so W&B dedupes the media by
    # content hash across steps instead of re-uploading them.
    try:
        cols = ["pc1", "pc2", "kind", "episode", "t", "embedding"]
        if thumbs is not None:
            cols.append("view")
        table = wandb.Table(columns=cols)
        for i, ((x, y), e, tt) in enumerate(zip(pc_traj, ep_ids, t_norm)):
            row = [float(x), float(y), "trajectory", int(e), float(tt), z_traj[i].astype(np.float32).tolist()]
            if thumbs is not None:
                row.append(wandb.Image(thumbs[i]) if i < len(thumbs) else None)
            table.add_data(*row)
        for x, y in pc_rand:
            row = [float(x), float(y), "random", -1, float("nan"), None]
            if thumbs is not None:
                row.append(None)
            table.add_data(*row)
        out["rlt/embedding_table"] = table
    except Exception:  # noqa: BLE001
        pass

    # --- interactive scatter with IMAGE-ON-HOVER -------------------------------------------------
    # Plotly (and W&B's plotly panel) can only put text in a hover label, so this is a small
    # self-contained HTML panel instead: the camera view for each frame is inlined as base64 and
    # shown next to the cursor. That is the whole point of the panel — seeing *which scene* a region
    # of embedding space corresponds to.
    try:
        out["rlt/embedding_hover"] = wandb.Html(
            _embedding_hover_html(projections, ep_ids, t_norm, thumbs, step), inject=False
        )
    except Exception:  # noqa: BLE001
        pass

    return out



def _temporal_continuity(xy: np.ndarray, ep_ids: np.ndarray, t_norm: np.ndarray, k: int = 3) -> float:
    """How often a frame's next-in-time frame lands among its k nearest neighbours in the projection.

    This is the "does the trajectory trace a smooth path?" question stated as a number, and it is
    exactly the property t-SNE/UMAP do preserve (local neighbourhoods), so it is comparable across
    linear and nonlinear projections even though their distances are not. 1.0 = every step lands next
    to its predecessor; ~k/n = no better than chance.
    """
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    knn = np.argsort(d, axis=1)[:, :k]
    hits = tot = 0
    for e in np.unique(ep_ids):
        idx = np.flatnonzero(ep_ids == e)[np.argsort(t_norm[ep_ids == e])]
        for a, b in zip(idx[:-1], idx[1:]):
            hits += int(b in knn[a])
            tot += 1
    return float(hits / max(tot, 1))

def _embedding_hover_html(projections, ep_ids, t_norm, thumbs, step: int) -> str:
    """Self-contained interactive scatter: switch projection, hover a point to see its camera view.

    Plotly (and W&B's plotly panel) can only put text in a hover label, so this is hand-rolled HTML:
    the wrist view for each frame is inlined as base64 and shown next to the cursor. Every projection
    is embedded, so PCA / progress-aligned / t-SNE / UMAP can be compared without re-logging.
    """
    import base64
    import io
    import json

    from PIL import Image

    def thumb_b64(i: int) -> str:
        # JPEG, not PNG: these are camera photos and the whole set is inlined into one HTML panel.
        if thumbs is None or i >= len(thumbs):
            return ""
        buf = io.BytesIO()
        Image.fromarray(thumbs[i]).resize((96, 96)).save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    views = {}
    for name, (tr, rnd) in projections.items():
        allp = np.concatenate([tr, rnd], axis=0)
        lo, hi = allp.min(0), allp.max(0)
        span = np.maximum(hi - lo, 1e-9)
        views[name] = {
            "traj": [[float(x), float(y)] for x, y in (tr - lo) / span],
            "rand": [[float(x), float(y)] for x, y in (rnd - lo) / span],
        }
    meta = [{"e": int(e), "t": float(tt), "img": thumb_b64(i)} for i, (e, tt) in enumerate(zip(ep_ids, t_norm))]
    order = [
        {"e": int(e), "idx": [int(i) for i in np.flatnonzero(ep_ids == e)[np.argsort(t_norm[ep_ids == e])]]}
        for e in np.unique(ep_ids)
    ]
    payload = json.dumps({"views": views, "meta": meta, "eps": order, "step": int(step)})
    return (
        """
<style>
 #rltwrap{position:relative;font:12px system-ui,sans-serif;color:#333}
 #rlttip{position:absolute;display:none;pointer-events:none;background:#fff;border:1px solid #bbb;
   border-radius:6px;padding:4px;box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:10}
 #rlttip img{display:block;width:96px;height:96px;image-rendering:pixelated}
 #rlttip div{text-align:center;margin-top:2px}
 #rltbar{margin-bottom:6px}
 #rltbar button{font:12px system-ui;padding:4px 10px;margin-right:6px;border:1px solid #ccc;
   background:#f6f7f9;border-radius:6px;cursor:pointer}
 #rltbar button.on{background:#2f6df0;color:#fff;border-color:#2f6df0}
</style>
<div id="rltwrap"><div id="rltbar"></div><svg id="rltsvg" width="760" height="520"></svg>
<div id="rlttip"></div></div>
<script>
const D = """
        + payload
        + """;
const svg=document.getElementById('rltsvg'), tip=document.getElementById('rlttip'), bar=document.getElementById('rltbar');
const W=760,H=520,P=34, NS='http://www.w3.org/2000/svg';
const X=p=>P+p*(W-2*P), Y=p=>H-P-p*(H-2*P);
const COL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function draw(name){
  svg.innerHTML='';
  const V=D.views[name];
  svg.appendChild(el('rect',{x:0,y:0,width:W,height:H,fill:'#fff'}));
  V.rand.forEach(p=>svg.appendChild(el('circle',{cx:X(p[0]),cy:Y(p[1]),r:3,fill:'#dcdfe4'})));
  D.eps.forEach((ep,j)=>{
    const c=COL[j%10];
    svg.appendChild(el('polyline',{points:ep.idx.map(i=>X(V.traj[i][0])+','+Y(V.traj[i][1])).join(' '),
      fill:'none',stroke:c,'stroke-width':1.6,opacity:.85}));
    ep.idx.forEach((i,k)=>{
      const m=D.meta[i], last=k===ep.idx.length-1;
      const dot=el('circle',{cx:X(V.traj[i][0]),cy:Y(V.traj[i][1]),r:(k===0||last)?7:4.5,
        fill:k===0?'#fff':c,stroke:c,'stroke-width':k===0?2.5:1});
      dot.style.cursor='pointer';
      dot.addEventListener('mousemove',ev=>{
        tip.innerHTML=(m.img?'<img src="data:image/jpeg;base64,'+m.img+'">':'')
          +'<div>ep '+m.e+' &middot; t='+m.t.toFixed(2)+'</div>';
        tip.style.display='block';
        tip.style.left=Math.min(ev.offsetX+14,W-120)+'px';
        tip.style.top=Math.max(ev.offsetY-110,0)+'px';
      });
      dot.addEventListener('mouseleave',()=>{tip.style.display='none';});
      svg.appendChild(dot);
    });
  });
  const t=el('text',{x:P,y:20,'font-size':13,'font-family':'system-ui',fill:'#222'});
  t.textContent=name+' \u2014 step '+D.step.toLocaleString()+' \u2014 hover a point for its wrist view';
  svg.appendChild(t);
}
Object.keys(D.views).forEach((name,i)=>{
  const b=document.createElement('button'); b.textContent=name;
  b.onclick=()=>{[...bar.children].forEach(c=>c.className=''); b.className='on'; draw(name);};
  if(i===0) b.className='on';
  bar.appendChild(b);
});
draw(Object.keys(D.views)[0]);
</script>
"""
    )


def build_probe_eval(config, mesh, replicated_sharding, train_state_sharding):
    """Build an in-process RoboCasa sim eval of the full VLA policy AND the latent BC-probe head.

    Returns ``eval_fn(train_state, seed) -> {"eval/vla_success":.., "eval/probe_success":..}`` or None
    if unavailable (not a RoboCasa RLT config, or the sim deps are missing). The env is created once
    and reused; rollouts run headless (no video). Any failure disables the eval rather than the run.
    """
    import functools as _ft

    try:
        import examples.robocasa.rollout as _ro
    except Exception as e:  # noqa: BLE001
        logging.warning(f"probe eval unavailable (rollout import): {e}")
        return None

    task = getattr(config, "_task_name", None) or _infer_robocasa_task(config)
    if task is None:
        logging.warning("probe eval: could not infer RoboCasa task from config; disabled.")
        return None

    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = data_config.norm_stats
    # NO repack here: repack maps raw LeRobot keys -> observation/* for training, but the rollout
    # element (from examples/robocasa/rollout.py) is already in observation/* form — exactly what
    # serving does (create_trained_policy defaults repack to an empty Group).
    input_tf = _transforms.compose(
        [
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_tf = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )

    # One compiled sampler per mode; reused across evals (params change, shapes don't). Shardings are
    # applied only when provided (the training path passes them; a single-device smoke may not).
    _jit_kw = {"static_argnames": ("mode",)}
    if train_state_sharding is not None:
        _jit_kw |= {
            "in_shardings": (train_state_sharding, replicated_sharding),
            "out_shardings": replicated_sharding,
        }

    @_ft.partial(jax.jit, **_jit_kw)
    def _sample(state, rng, obs, *, mode):
        params = state.ema_params if state.ema_params is not None else state.params
        model = nnx.merge(state.model_def, params)
        model.eval()
        if mode == "probe":
            return model.probe_sample_actions(rng, obs)
        return model.sample_actions(rng, obs)

    env_box = {}  # lazily create the env on first use (heavy import + scene build)

    def _policy_fn(state, mode):
        def fn(element):
            inp = input_tf(element)
            inp = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inp)
            obs = _model.Observation.from_dict(inp)
            rng = jax.random.key(0)
            with sharding.set_mesh(mesh):
                actions = _sample(state, rng, obs, mode=mode)
            out = output_tf({"state": np.asarray(inp["state"][0]), "actions": np.asarray(actions[0])})
            return np.asarray(out["actions"])
        return fn

    def eval_fn(train_state, seed):
        if "env" not in env_box:
            env_box["env"] = _ro.make_env(task, camera_size=256, seed=seed)
        env = env_box["env"]
        n = config.rlt_probe_eval_trials
        vla = _ro.run_trials(env, _policy_fn(train_state, "vla"), task=task, num_trials=n, seed=seed)
        probe = _ro.run_trials(env, _policy_fn(train_state, "probe"), task=task, num_trials=n, seed=seed)
        return {"eval/vla_success": vla["success_rate"], "eval/probe_success": probe["success_rate"]}

    return eval_fn


def _infer_robocasa_task(config) -> str | None:
    """Recover the RoboCasa task name from the data repo_id (…/robocasa365-<Task>)."""
    repo = getattr(config.data, "repo_id", "") or ""
    if "robocasa365-" in repo:
        return repo.split("robocasa365-")[-1]
    return None


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pcompute_action_dist = None
    if config.action_dist_interval > 0:
        pcompute_action_dist = jax.jit(
            functools.partial(compute_action_dist_metrics, config, num_samples=config.action_dist_num_samples),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    pcompute_rlt_metrics = None
    if config.rlt_monitor_interval > 0:
        pcompute_rlt_metrics = jax.jit(
            functools.partial(compute_rlt_metrics, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
    pextract_rlt = None
    rlt_probe = None
    if config.rlt_vis_interval > 0:
        pextract_rlt = jax.jit(
            extract_rlt_embeddings,
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        # Fixed episode-ordered probe set, so the visualization can draw trajectory paths and stays
        # comparable across steps. Padded to a multiple of batch_size to reuse the compiled extractor.
        try:
            rlt_probe = build_rlt_trajectory_probe(
                config,
                n_traj=config.rlt_vis_num_trajectories,
                n_frames=config.rlt_vis_frames_per_trajectory,
                pad_to=config.batch_size,
            )
            logging.info(
                f"Built RLT trajectory probe: {rlt_probe[4]} frames from {len(np.unique(rlt_probe[2]))} episodes."
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Could not build the RLT trajectory probe; embedding vis disabled. ({e})")
            pextract_rlt = None

    probe_eval_fn = None
    if config.rlt_probe_eval_interval > 0:
        try:
            probe_eval_fn = build_probe_eval(config, mesh, replicated_sharding, train_state_sharding)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Could not set up probe eval; disabled. ({e})")

    start_step = int(train_state.step)
    # Iterate one step past num_train_steps so the run ends ON a round step (100_000, not 99_999).
    # That final step is a multiple of the save/log/diagnostic intervals, so the last checkpoint gets
    # its visualizations and metrics for free — no special-casing needed anywhere below.
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps + 1),
        initial=start_step,
        total=config.num_train_steps + 1,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        diag_info = {}
        # Diagnostics are monitors, not part of training: on failure, log once and disable each one.
        if pcompute_action_dist is not None and step % config.action_dist_interval == 0:
            try:
                with sharding.set_mesh(mesh):
                    d = pcompute_action_dist(jax.random.fold_in(train_rng, step), train_state, batch)
                diag_info.update(jax.device_get(d))
            except Exception as e:  # noqa: BLE001
                logging.warning(f"action-dist diagnostic failed at step {step}; disabling it. ({e})")
                pcompute_action_dist = None

        if pcompute_rlt_metrics is not None and step % config.rlt_monitor_interval == 0:
            try:
                with sharding.set_mesh(mesh):
                    d = pcompute_rlt_metrics(jax.random.fold_in(train_rng, step), train_state, batch)
                diag_info.update(jax.device_get(d))
            except Exception as e:  # noqa: BLE001
                logging.warning(f"rlt monitor failed at step {step}; disabling it. ({e})")
                pcompute_rlt_metrics = None

        if pextract_rlt is not None and step % config.rlt_vis_interval == 0:
            try:
                probe_obs, probe_act, ep_ids, t_norm, n_valid, probe_thumbs = rlt_probe
                zs, props = [], []
                for s in range(0, probe_act.shape[0], config.batch_size):
                    sl = slice(s, s + config.batch_size)
                    chunk = (jax.tree.map(lambda x, sl=sl: x[sl], probe_obs), probe_act[sl])
                    with sharding.set_mesh(mesh):
                        zc, pc = pextract_rlt(train_state, chunk)
                    zs.append(np.asarray(zc))
                    props.append(np.asarray(pc))
                z_traj = np.concatenate(zs)[:n_valid]
                proprio_traj = np.concatenate(props)[:n_valid]
                with sharding.set_mesh(mesh):
                    z_rand, _ = pextract_rlt(train_state, batch)
                diag_info.update(
                    log_rlt_embedding_vis(
                        z_traj, proprio_traj, ep_ids, t_norm, np.asarray(z_rand), step, probe_thumbs
                    )
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(f"rlt embedding vis failed at step {step}; disabling it. ({e})")
                pextract_rlt = None

        # In-process sim eval of the VLA + latent-probe policies (headless). Heavier than the other
        # monitors (spins the sim), so it is guarded and disabled on failure like the rest.
        if probe_eval_fn is not None and step % config.rlt_probe_eval_interval == 0:
            try:
                t_ev = time.perf_counter()
                res = probe_eval_fn(train_state, seed=0)
                diag_info.update(res)
                logging.info(
                    f"probe eval @ {step}: vla={res['eval/vla_success']:.0%} "
                    f"probe={res['eval/probe_success']:.0%} ({time.perf_counter() - t_ev:.0f}s)"
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(f"probe eval failed at step {step}; disabling it. ({e})")
                probe_eval_fn = None

        diag_now = bool(diag_info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info.update(diag_info)
            # Only scalar entries are printable (diagnostics may include e.g. a wandb.Image).
            info_str = ", ".join(
                f"{k}={float(v):.4f}"
                for k, v in reduced_info.items()
                if isinstance(v, (int, float)) or getattr(v, "ndim", None) == 0
            )
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        elif diag_now:
            wandb.log(diag_info, step=step)
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
