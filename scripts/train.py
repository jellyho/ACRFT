import dataclasses
import functools
import logging
import platform
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
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
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
    z = model.extract_rl_token(observation)  # [b, D]
    b = z.shape[0]
    zc = z - jnp.mean(z, axis=0, keepdims=True)
    # participation ratio without forming the DxD covariance: use the [b,b] Gram matrix.
    #   Σλ = tr(C) = ||zc||²/b ;  Σλ² = ||C||_F² = ||zc zcᵀ||_F² / b²
    gram = zc @ zc.T  # [b, b]
    tr_c = jnp.sum(zc * zc) / b
    frob2_c = jnp.sum(gram * gram) / (b * b)
    part_ratio = (tr_c * tr_c) / (frob2_c + 1e-12)
    return {
        "rlt/participation_ratio": part_ratio,
        "rlt/eval_z_batch_std": jnp.mean(jnp.std(z, axis=0)),
        "rlt/eval_z_norm_mean": jnp.mean(jnp.linalg.norm(z, axis=-1)),
    }


def extract_rlt_embeddings(
    state: training_utils.TrainState, batch: tuple[_model.Observation, _model.Actions]
) -> tuple[at.Array, at.Array]:
    """Return (z_rl [b, D], proprio [b, ad]) for host-side visualization / probing."""
    observation, _ = batch
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    return model.extract_rl_token(observation), observation.state


def log_rlt_embedding_vis(z: np.ndarray, proprio: np.ndarray, step: int) -> dict:
    """Host-side RLT embedding visualization: PCA-2D scatter + a held-out linear-probe R².

    Returns a wandb-loggable dict {"rlt/embedding_pca": Image, "rlt/probe_proprio_r2": float}.
    Best-effort: any failure returns {} (never touches the training loop).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=np.float64)
    proprio = np.asarray(proprio, dtype=np.float64)
    b = z.shape[0]

    # --- PCA-2D of the RL tokens (center + top-2 right singular vectors) ---
    zc = z - z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    pcs = zc @ vt[:2].T  # [b, 2]
    # Color by the dominant proprio direction (a rough pose/progress proxy).
    pc_prop = proprio - proprio.mean(0, keepdims=True)
    color = pc_prop @ np.linalg.svd(pc_prop, full_matrices=False)[2][0] if proprio.shape[1] else np.arange(b)

    # --- Held-out linear probe: can proprio be linearly read off z? (generalization, not fit) ---
    probe_r2 = float("nan")
    try:
        half = max(1, b // 2)
        ztr, zval = zc[:half], zc[half:]
        ytr, yval = pc_prop[:half], pc_prop[half:]
        if zval.shape[0] > 0 and ytr.shape[0] > 0:
            # ridge on the top PCA components (compact, avoids the D>>b degenerate exact fit)
            k = min(half, 64)
            ftr, fval = ztr @ vt[:k].T, zval @ vt[:k].T
            w = np.linalg.solve(ftr.T @ ftr + 1e-2 * np.eye(k), ftr.T @ ytr)
            pred = fval @ w
            ss_res = np.sum((yval - pred) ** 2)
            ss_tot = np.sum((yval - yval.mean(0, keepdims=True)) ** 2) + 1e-9
            probe_r2 = float(1.0 - ss_res / ss_tot)
    except Exception:  # noqa: BLE001
        pass

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=color, cmap="viridis", s=28, edgecolor="white", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="proprio PC1")
    ax.set_title(f"RLT z_rl — PCA-2D (step {step:,})\nheld-out proprio probe R²={probe_r2:.2f}", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return {"rlt/embedding_pca": img, "rlt/probe_proprio_r2": probe_r2}


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
    if config.rlt_vis_interval > 0:
        pextract_rlt = jax.jit(
            extract_rlt_embeddings,
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
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
                with sharding.set_mesh(mesh):
                    z, proprio = pextract_rlt(train_state, batch)
                diag_info.update(log_rlt_embedding_vis(np.asarray(z), np.asarray(proprio), step))
            except Exception as e:  # noqa: BLE001
                logging.warning(f"rlt embedding vis failed at step {step}; disabling it. ({e})")
                pextract_rlt = None

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

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
