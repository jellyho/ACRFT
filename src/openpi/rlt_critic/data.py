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
    # VLA candidates - what the td bootstrap maxes over. Stored SPLIT along N so no single device
    # array crosses 2**31 elements: XLA's gather codegen segfaults at compile on sm_86/Blackwell
    # once the source buffer exceeds int32 element count (mixed data: 807634*16*16*12 = 2.48e9).
    cand_parts: tuple  # k x [T, N/k, H, A]
    reward: jax.Array  # [T]
    episode: jax.Array  # [T]
    mc_return: jax.Array  # [T]     discounted return the behaviour policy actually collected
    done: jax.Array  # [T]          1 at the terminal frame
    done_cum: jax.Array  # [T]      running terminal count, for "any terminal inside [t, t+h-1]?"
    alive: jax.Array  # [T]         frame is at or before its episode's terminal
    ep_start: jax.Array  # [T]      first frame index of this frame's episode (history clamp)
    horizon: int
    action_dim: int
    num_samples: int
    action_mean: np.ndarray  # [A]  normalisation applied to chunk/cand; recorded in config.json
    action_std: np.ndarray  # [A]
    proprio_mean: np.ndarray
    proprio_std: np.ndarray

    def cand_at(self, idx: jax.Array) -> jax.Array:
        """cand[idx] -> [*idx.shape, N, H, A], gathering each sub-int32 part separately."""
        return jnp.concatenate([p[idx] for p in self.cand_parts], axis=idx.ndim)

    def obs_at(self, idx: jax.Array, *, history: int = 0, history_stride: int = 8, token_dim: int = 2048) -> jax.Array:
        """Observation for frame idx: [hist_K..hist_1, token+proprio] -> [*idx.shape, K*token_dim + D].

        History frames look back ``history_stride`` env steps apart, clamped at the episode start
        (Robo-ValueRL-style short history: single-frame values are ambiguous under occlusion and
        repeated motions — a press-approach and a post-failure retreat can share the frame).
        """
        if history <= 0:
            return self.token[idx]
        hists = []
        for k in range(history, 0, -1):
            back = jnp.clip(idx - k * history_stride, self.ep_start[idx], None)
            hists.append(self.token[back][..., :token_dim])
        return jnp.concatenate([*hists, self.token[idx]], axis=-1)


_DATA_ARRAYS = (
    "token",
    "chunk",
    "cand_parts",
    "reward",
    "episode",
    "mc_return",
    "done",
    "done_cum",
    "alive",
    "ep_start",
)
_DATA_STATS = ("action_mean", "action_std", "proprio_mean", "proprio_std")


def _data_flatten(d: Data):
    # Registered as a pytree so the training step takes Data as a traced ARGUMENT - a closed-over
    # array is baked into the compiled program as a constant, duplicating the dataset on device.
    children = tuple(getattr(d, f) for f in _DATA_ARRAYS)
    aux = (
        d.horizon,
        d.action_dim,
        d.num_samples,
        *(tuple(np.asarray(getattr(d, f), np.float64).tolist()) for f in _DATA_STATS),
    )
    return children, aux


def _data_unflatten(aux, children):
    h, a, n, *stats = aux
    return Data(
        **dict(zip(_DATA_ARRAYS, children, strict=True)),
        horizon=h,
        action_dim=a,
        num_samples=n,
        **{f: np.asarray(v, np.float32) for f, v in zip(_DATA_STATS, stats, strict=True)},
    )


jax.tree_util.register_pytree_node(Data, _data_flatten, _data_unflatten)


def _terminals(done, episode):
    """`done_cum` and `alive`, the two lookups the target's terminal handling needs."""
    done = np.asarray(done, np.int64)
    alive = np.ones(len(done), np.float32)
    for e in np.unique(episode):
        w = np.flatnonzero(episode == e)
        fired = w[done[w] > 0]
        if len(fired):
            alive[fired[0] + 1 :][: w[-1] - fired[0]] = 0.0  # frames after the terminal: not decisions
    ep = np.asarray(episode)
    start = np.zeros(len(ep), np.int32)
    starts = np.flatnonzero(np.concatenate([[True], ep[1:] != ep[:-1]]))
    for i, s0 in enumerate(starts):
        e0 = starts[i + 1] if i + 1 < len(starts) else len(ep)
        start[s0:e0] = s0
    return {
        "done": jnp.asarray(done.astype(np.int32)),
        "done_cum": jnp.asarray(np.cumsum(done, dtype=np.int32)),
        "alive": jnp.asarray(alive),
        "ep_start": jnp.asarray(start),
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

    # Split candidates along N into parts each under 2**31 elements (see Data.cand_parts).
    n_parts = int(np.ceil(cand.size / (2**31 - 1)))
    n_parts = next(k for k in range(n_parts, N + 1) if N % k == 0)
    if n_parts > 1:
        logger.info(f"cand {cand.size:,} elements > int32: splitting into {n_parts} device arrays")
    data = Data(
        token=jnp.asarray(obs),
        chunk=jnp.asarray(chunk),
        cand_parts=tuple(jnp.asarray(np.ascontiguousarray(c)) for c in np.split(cand, n_parts, axis=1)),
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
    gb = sum(x.size * x.itemsize for x in (data.token, data.chunk, *data.cand_parts)) / 1e9
    logger.info(f"loaded {T} frames ({gb:.2f} GB): token {obs.shape[1]}, chunk {H}x{A}, N={N}, actions normalised")
    return data
