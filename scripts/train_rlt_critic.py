"""Stage 2: train a QC or ARQ critic on the precomputed RL tokens (see annotate_rlt.py).

The whole dataset is numeric and small (a few GB), so it is loaded ONCE onto the GPU and never
touched by a data loader again: batches are on-device gathers, and many update steps are fused into
a single dispatch with `lax.scan`. For a critic this matters more than the network itself — the
model is tiny, so Python/dispatch overhead and host->device traffic are what would otherwise
dominate the step time.

Targets, for a chunk of `h` steps starting at frame t:

    y_h = sum_{i<h} gamma^i r_{t+i}  +  gamma^h * (not terminal) * V(z_{t+h})
    V(s') = max over the VLA's candidate chunks (and, for ARQ, over prefix lengths) of
            min over the ensemble of Q_target(s', a')

`Q_target` is an EMA copy of the critic (Polyak), not the online network.

  * **QC** trains the single full-chunk value, so there is one target per transition.
  * **ARQ** trains every prefix head against its own horizon-matched target, which is what puts the
    per-prefix values on a common discounted scale and makes them comparable at deployment.

Usage:

    srun --gres=gpu:1 uv run scripts/train_rlt_critic.py \
        --data data/rlt_critic/PrepareCoffee --kind arq --steps 200000
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.rlt_critic import critic as _critic

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Data:
    """The annotated task, resident on the accelerator."""

    token: jax.Array  # [T, D]
    chunk: jax.Array  # [T, H, A]   executed demo chunk (the `a` in Q(s, a))
    cand: jax.Array  # [T, N, H, A] VLA candidates (the `a'` the bootstrap maximises over)
    reward: jax.Array  # [T]
    episode: jax.Array  # [T]
    horizon: int
    action_dim: int
    num_samples: int


def load_data(path: pathlib.Path, *, max_frames: int = 0) -> Data:
    meta = json.loads((path / "meta.json").read_text())
    if meta["stride"] != 1:
        raise ValueError(
            f"stride={meta['stride']}: per-prefix reward sums need every frame. Re-run annotate_rlt.py with --stride 1."
        )
    T, D, H, A, N = (meta[k] for k in ("num_frames", "token_dim", "horizon", "action_dim", "num_samples"))
    if max_frames:
        T = min(T, max_frames)

    # Honour the dtype the annotation actually wrote; 16-bit storage is cast up on the way to the
    # device so the critic still computes in float32.
    import ml_dtypes

    store = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]

    def rd(name, shape, dtype=None):
        arr = np.memmap(path / f"{name}.dat", dtype=store if dtype is None else dtype, mode="r", shape=shape)[:T]
        return np.asarray(arr, dtype=np.float32) if dtype is None else np.asarray(arr)

    full = json.loads((path / "meta.json").read_text())["num_frames"]
    d = Data(
        token=jnp.asarray(rd("rl_token", (full, D))),
        chunk=jnp.asarray(rd("action_chunk", (full, H, A))),
        cand=jnp.asarray(rd("base_action", (full, N, H, A))),
        reward=jnp.asarray(rd("reward", (full,))),
        episode=jnp.asarray(rd("episode_index", (full,), np.int32)),
        horizon=H,
        action_dim=A,
        num_samples=N,
    )
    gb = sum(x.size * x.itemsize for x in (d.token, d.chunk, d.cand)) / 1e9
    logger.info(f"loaded {T} frames onto {jax.devices()[0].platform} ({gb:.2f} GB): token {D}, chunk {H}x{A}, N={N}")
    return d


def make_update(data: Data, cfg, net, hl):
    """One jitted critic update, written so it can be scanned over many steps."""
    T = data.token.shape[0]
    H, g = data.horizon, cfg.macro_group_size
    # Prefix lengths in real steps: ARQ trains all of them, QC only the full chunk.
    prefixes = jnp.arange(g, H + 1, g) if cfg.kind == "arq" else jnp.array([H])
    disc = cfg.discount ** jnp.arange(H, dtype=jnp.float32)  # [H]

    def targets(idx, tgt_params):
        """Per-prefix (cum_reward, next_value, valid) for the sampled transitions."""
        ep = data.episode[idx]  # [B]
        # Rewards over the chunk, zeroed once the episode ends.
        off = jnp.arange(H)[None, :]
        r_idx = jnp.clip(idx[:, None] + off, 0, T - 1)
        same = data.episode[r_idx] == ep[:, None]
        rew = data.reward[r_idx] * same  # [B, H]
        cum_all = jnp.cumsum(rew * disc[None, :], axis=-1)  # [B, H]
        cum = cum_all[:, prefixes - 1]  # [B, P]

        nxt = jnp.clip(idx[:, None] + prefixes[None, :], 0, T - 1)  # [B, P]
        valid = (data.episode[nxt] == ep[:, None]) & (idx[:, None] + prefixes[None, :] < T)

        # V(s') = max over candidates (and prefixes, for ARQ) of the ensemble-min target Q.
        z_next = data.token[nxt]  # [B, P, D]
        a_next = data.cand[nxt]  # [B, P, N, H, A]
        B, P, N = a_next.shape[0], a_next.shape[1], a_next.shape[2]
        q = net.apply(tgt_params, jnp.repeat(z_next[:, :, None], N, axis=2), a_next)
        q = hl.from_logits(q) if cfg.num_atoms > 1 else q  # [K, B, P, N(, prefix)]
        q = jnp.min(q, axis=0)  # ensemble min -> [B, P, N(, prefix)]
        v_next = jnp.max(q.reshape(B, P, -1), axis=-1)  # joint max over candidates (+prefixes)

        gam = cfg.discount ** prefixes.astype(jnp.float32)  # [P]
        y = cum + gam[None, :] * valid * v_next
        return y, valid

    def loss_fn(params, tgt_params, idx):
        y, valid = targets(idx, tgt_params)
        y = jax.lax.stop_gradient(y)
        pred = net.apply(params, data.token[idx], data.chunk[idx])  # [K, B(, P)(, atoms)]
        if cfg.kind == "qc":
            pred = pred[:, :, None] if cfg.num_atoms == 1 else pred[:, :, None, :]

        w = valid.astype(jnp.float32)[None]  # [1, B, P]
        if cfg.num_atoms > 1:
            probs = hl.to_probs(jnp.clip(y, hl.v_min, hl.v_max))[None]  # [1, B, P, atoms]
            per = -jnp.sum(probs * jax.nn.log_softmax(pred, axis=-1), axis=-1)
            q_mean = jnp.mean(hl.from_logits(pred))
        else:
            per = jnp.square(pred - y[None])
            q_mean = jnp.mean(pred)
        loss = jnp.sum(per * w) / (jnp.sum(w) * pred.shape[0] + 1e-8)
        return loss, {"loss": loss, "q_mean": q_mean, "target_mean": jnp.mean(y * valid), "valid": jnp.mean(w)}

    tx = optax.adam(cfg.lr)

    def step(carry, rng):
        params, tgt_params, opt_state = carry
        idx = jax.random.randint(rng, (cfg.batch_size,), 0, T)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, tgt_params, idx)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        tgt_params = optax.incremental_update(params, tgt_params, cfg.target_tau)
        return (params, tgt_params, opt_state), info

    return step, tx


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    ap.add_argument("--kind", choices=["qc", "arq"], default="arq")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument("--num-critics", type=int, default=2)
    ap.add_argument("--num-atoms", type=int, default=1, help="1 = scalar Q; >1 = HL-Gauss distributional")
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=1.0)
    ap.add_argument("--macro-group-size", type=int, default=2, help="ARQ: steps per macro token")
    # Capacity. Defaults follow the reference ACSAC critic (3 layers, 8 heads x 48 = 384 wide) and a
    # standard 3x512 MLP for QC. The input is already a learned 2048-d token, so the critic only has
    # to fit a fairly simple function on top of it — but sweep these rather than trust the default.
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512], help="QC: MLP widths")
    ap.add_argument("--num-layers", type=int, default=3, help="ARQ: transformer layers")
    ap.add_argument("--num-heads", type=int, default=8, help="ARQ: attention heads")
    ap.add_argument("--head-dim", type=int, default=48, help="ARQ: per-head dim (n_embd = heads*head_dim)")
    ap.add_argument("--mlp-dim", type=int, default=1024, help="ARQ: transformer MLP width")
    ap.add_argument("--steps-per-dispatch", type=int, default=100, help="update steps fused into one scan")
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    data = load_data(cfg.data, max_frames=cfg.max_frames)
    if data.horizon % cfg.macro_group_size:
        raise ValueError(f"macro_group_size {cfg.macro_group_size} must divide horizon {data.horizon}")

    net = _critic.Ensemble(
        make_critic=lambda: _critic.make_critic(
            cfg.kind,
            action_dim=data.action_dim,
            horizon=data.horizon,
            num_atoms=cfg.num_atoms,
            **(
                {
                    "macro_group_size": cfg.macro_group_size,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "head_dim": cfg.head_dim,
                    "mlp_dim": cfg.mlp_dim,
                }
                if cfg.kind == "arq"
                else {"hidden_dims": tuple(cfg.hidden_dims)}
            ),
        ),
        num_critics=cfg.num_critics,
    )
    hl = _critic.HLGauss(v_min=cfg.v_min, v_max=cfg.v_max, num_atoms=max(cfg.num_atoms, 2))

    rng = jax.random.key(cfg.seed)
    params = net.init(rng, data.token[:1], data.chunk[:1])
    n_param = sum(x.size for x in jax.tree.leaves(params))
    logger.info(f"{cfg.kind.upper()} critic: {n_param / 1e6:.2f}M params, {cfg.num_critics} ensemble members")

    step_fn, tx = make_update(data, cfg, net, hl)
    opt_state = tx.init(params)
    tgt_params = params

    @jax.jit
    def run_chunk(carry, rng):
        return jax.lax.scan(step_fn, carry, jax.random.split(rng, cfg.steps_per_dispatch))

    carry = (params, tgt_params, opt_state)
    t0 = time.perf_counter()
    for s in range(0, cfg.steps, cfg.steps_per_dispatch):
        carry, infos = run_chunk(carry, jax.random.fold_in(rng, s))
        if s % cfg.log_interval == 0:
            info = jax.tree.map(lambda x: float(jnp.mean(x)), infos)
            rate = (s + cfg.steps_per_dispatch) / max(time.perf_counter() - t0, 1e-6)
            logger.info(
                f"step {s + cfg.steps_per_dispatch}/{cfg.steps}  {rate:.0f} it/s  "
                + "  ".join(f"{k}={v:.4f}" for k, v in info.items())
            )

    out = cfg.out or (cfg.data / f"critic_{cfg.kind}")
    out.mkdir(parents=True, exist_ok=True)
    import flax.serialization as fser

    (out / "params.msgpack").write_bytes(fser.to_bytes(carry[0]))
    (out / "config.json").write_text(json.dumps(vars(cfg) | {"data": str(cfg.data), "out": str(out)}, indent=2))
    logger.info(f"saved to {out} ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
