"""Loading the annotated task into device memory, normalised and terminal-indexed."""

import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Data:
    """The annotated task, resident on the accelerator. Actions are stored NORMALISED (see load_data)."""

    token: jax.Array  # [T, D]      rl_token (+ z-scored proprio)
    chunk: jax.Array  # [T, H, A]   executed demo chunk - the `a` in Q(s, a)
    cand: jax.Array  # [T, N, H, A] VLA candidates - what the td bootstrap maxes over
    reward: jax.Array  # [T]
    episode: jax.Array  # [T]
    mc_return: jax.Array  # [T]     discounted return the behaviour policy actually collected
    done: jax.Array  # [T]          1 at the terminal frame
    done_cum: jax.Array  # [T]      running terminal count, for "any terminal inside [t, t+h-1]?"
    alive: jax.Array  # [T]         frame is at or before its episode's terminal
    horizon: int
    action_dim: int
    num_samples: int
    action_mean: np.ndarray  # [A]  normalisation applied to chunk/cand; recorded in config.json
    action_std: np.ndarray  # [A]
    proprio_mean: np.ndarray
    proprio_std: np.ndarray


def _terminals(done, episode):
    """`done_cum` and `alive`, the two lookups the target's terminal handling needs."""
    done = np.asarray(done, np.int64)
    alive = np.ones(len(done), np.float32)
    for e in np.unique(episode):
        w = np.flatnonzero(episode == e)
        fired = w[done[w] > 0]
        if len(fired):
            alive[fired[0] + 1 :][: w[-1] - fired[0]] = 0.0  # frames after the terminal: not decisions
    return {
        "done": jnp.asarray(done.astype(np.int32)),
        "done_cum": jnp.asarray(np.cumsum(done, dtype=np.int32)),
        "alive": jnp.asarray(alive),
    }


def load_data(path: pathlib.Path, *, max_frames: int = 0) -> Data:
    """Load the annotation, z-score proprio, and normalise actions per dimension.

    Everything the network consumes is normalised here, and the statistics are saved into
    config.json so evaluation applies the identical transform (load_trained handles it):
      - rl_token: left raw - the critic's first op is a LayerNorm.
      - proprio: z-scored per dim (metres, quaternions and gripper qpos share one vector).
      - actions: z-scored per dim with stats from the candidate pool, so chunk and candidates live
        on the same scale and no downstream constant (noise scales, embeddings) depends on raw units.
    """
    meta = json.loads((path / "meta.json").read_text())
    if meta["stride"] != 1:
        raise ValueError(f"stride={meta['stride']}: per-prefix reward sums need every frame (re-annotate with 1).")
    T, D, H, A, N = (meta[k] for k in ("num_frames", "token_dim", "horizon", "action_dim", "num_samples"))
    full = T
    if max_frames:
        T = min(T, max_frames)

    import ml_dtypes

    store = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]

    def rd(name, shape, dtype=None):
        arr = np.memmap(path / f"{name}.dat", dtype=store if dtype is None else dtype, mode="r", shape=shape)[:T]
        return np.asarray(arr, dtype=np.float32) if dtype is None else np.asarray(arr)

    pd_ = meta.get("proprio_dim")
    if not pd_ or not (path / "proprio.dat").exists():
        raise ValueError(f"no proprio.dat in {path} - run annotate_rlt.py (it writes proprio) or extract_proprio.py")
    pro = rd("proprio", (full, pd_), np.float32)
    p_mu, p_sd = pro.mean(0), pro.std(0)
    pro = np.where(p_sd > 1e-6, (pro - p_mu) / np.where(p_sd > 1e-6, p_sd, 1.0), 0.0).astype(np.float32)
    obs = np.concatenate([rd("rl_token", (full, D)), pro], axis=1)

    chunk = rd("action_chunk", (full, H, A))
    cand = rd("base_action", (full, N, H, A))
    a_mu = cand.reshape(-1, A).mean(0)
    a_sd = cand.reshape(-1, A).std(0)
    a_sd = np.where(a_sd > 1e-6, a_sd, 1.0)
    chunk = (chunk - a_mu) / a_sd
    cand = (cand - a_mu) / a_sd

    data = Data(
        token=jnp.asarray(obs),
        chunk=jnp.asarray(chunk),
        cand=jnp.asarray(cand),
        reward=jnp.asarray(rd("reward", (full,))),
        episode=jnp.asarray(rd("episode_index", (full,), np.int32)),
        mc_return=jnp.asarray(rd("mc_return", (full,), np.float32)),
        **_terminals(rd("done", (full,), np.int8), rd("episode_index", (full,), np.int32)),
        horizon=H,
        action_dim=A,
        num_samples=N,
        action_mean=a_mu,
        action_std=a_sd,
        proprio_mean=p_mu,
        proprio_std=p_sd,
    )
    gb = sum(x.size * x.itemsize for x in (data.token, data.chunk, data.cand)) / 1e9
    logger.info(f"loaded {T} frames ({gb:.2f} GB): token {obs.shape[1]}, chunk {H}x{A}, N={N}, actions normalised")
    return data
