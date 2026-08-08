"""HILP-style readout on frozen RL tokens: phi such that -||phi(s)-phi(g)|| is a goal-conditioned
value learned by TD alone.

Worker A's finding (hub 08-07): a TD-only readout over the frozen RLT token collapses the
episode-identity pathology (linear ep-probe 1.00 -> 0.28) and unlocks stitching (rho 0.78), because
reachability is inherently episode-invariant. This reimplements that recipe in-repo so a critic can
be trained ON TOP of phi with everything else held fixed (single-variable embedding swap).

Objective (goal-conditioned expectile TD, gamma-discounted reachability):
    V(s,g) = -||phi(s) - phi(g)||
    target(s,g) = 0                          if s == g (goal reached)
                = -1 + gamma * V(s', g)      otherwise
    expectile-weighted squared error, target network on phi.

Goals are sampled CROSS-EPISODE (random future of the same episode with p_future, else any random
frame), which is exactly what makes the geometry episode-invariant.

    uv run scripts/train_hilp_readout.py --data $CACHE_DIR/annot/mixed --out .../phi_mixed
"""

import argparse
import json
import logging
import pathlib
import time

import flax.linen as nn
import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


class Phi(nn.Module):
    dim: int = 128
    hidden: tuple = (512, 512)

    @nn.compact
    def __call__(self, z):
        x = nn.LayerNorm()(z)
        for h in self.hidden:
            x = nn.gelu(nn.Dense(h)(x))
        return nn.Dense(self.dim)(x)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True, help="annotation dir (rl_token/episode_index)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--discount", type=float, default=0.98)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--p-future", type=float, default=0.7, help="P(goal from same-episode future) vs random frame")
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use-proprio",
        action="store_true",
        help="append z-scored proprio to the token before phi - the reachability geometry then has "
        "access to joint state (whether that helps or hurts episode-invariance is the ablation)",
    )
    cfg = ap.parse_args()

    meta = json.loads((cfg.data / "meta.json").read_text())
    T, D = meta["num_frames"], meta["token_dim"]
    tok = np.asarray(np.memmap(cfg.data / "rl_token.dat", dtype=np.float32, mode="r", shape=(T, D)))
    ep = np.asarray(np.memmap(cfg.data / "episode_index.dat", dtype=np.int32, mode="r", shape=(T,)))
    if cfg.use_proprio:
        pd_ = meta["proprio_dim"]
        pro = np.asarray(np.memmap(cfg.data / "proprio.dat", dtype=np.float32, mode="r", shape=(T, pd_)))
        mu, sd = pro.mean(0), pro.std(0)
        pro = np.where(sd > 1e-6, (pro - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0).astype(np.float32)
        tok = np.concatenate([tok, pro], axis=1)
        D = tok.shape[1]
    # per-frame episode end (inclusive) for future-goal sampling
    ends = np.zeros(T, np.int64)
    idxs = np.flatnonzero(np.concatenate([ep[1:] != ep[:-1], [True]]))
    prev = 0
    for e in idxs:
        ends[prev : e + 1] = e
        prev = e + 1
    logger.info(f"{T} frames, {ep.max() + 1} episodes, token {D} -> phi {cfg.dim}")

    tok_d = jnp.asarray(tok)
    ends_d = jnp.asarray(ends)
    net = Phi(dim=cfg.dim)
    rng = jax.random.key(cfg.seed)
    params = net.init(rng, tok_d[:1])
    tgt = params
    # distance-parameterised TD can diverge without clipping (phi_mixed run 1: loss->nan)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    opt = tx.init(params)

    def loss_fn(p, tgt_p, s_idx, g_idx, key):
        s, s2, g = tok_d[s_idx], tok_d[jnp.minimum(s_idx + 1, ends_d[s_idx])], tok_d[g_idx]
        at_goal = (s_idx == g_idx) | (s_idx == ends_d[s_idx])
        ps, pg = net.apply(p, s), net.apply(jax.lax.stop_gradient(tgt_p), g)
        v = -jnp.linalg.norm(ps - pg, axis=-1)
        ps2 = net.apply(jax.lax.stop_gradient(tgt_p), s2)
        v_next = -jnp.linalg.norm(ps2 - pg, axis=-1)
        y = jnp.where(at_goal, 0.0, -1.0 + cfg.discount * v_next)
        u = jax.lax.stop_gradient(y) - v
        w = jnp.abs(cfg.expectile - (u < 0).astype(jnp.float32))
        return jnp.mean(w * jnp.square(u)), jnp.mean(v)

    @jax.jit
    def step(p, tgt_p, opt_state, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        s_idx = jax.random.randint(k1, (cfg.batch_size,), 0, T)
        fut = jax.random.randint(k2, (cfg.batch_size,), 0, 1 << 30) % jnp.maximum(ends_d[s_idx] - s_idx, 1) + s_idx + 1
        fut = jnp.minimum(fut, ends_d[s_idx])
        rand_g = jax.random.randint(k3, (cfg.batch_size,), 0, T)
        g_idx = jnp.where(jax.random.uniform(k4, (cfg.batch_size,)) < cfg.p_future, fut, rand_g)
        (loss, vbar), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, tgt_p, s_idx, g_idx, key)
        updates, opt_state = tx.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        tgt_p = optax.incremental_update(p, tgt_p, cfg.target_tau)
        return p, tgt_p, opt_state, loss, vbar

    t0 = time.perf_counter()
    for i in range(cfg.steps):
        rng, k = jax.random.split(rng)
        params, tgt, opt, loss, vbar = step(params, tgt, opt, k)
        if i % 2000 == 0:
            logger.info(f"step {i}/{cfg.steps}  loss={float(loss):.4f}  v_mean={float(vbar):.2f}")

    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "phi.msgpack").write_bytes(fser.to_bytes(params))
    (cfg.out / "config.json").write_text(
        json.dumps({**vars(cfg), "data": str(cfg.data), "out": str(cfg.out), "token_dim": D}, indent=2, default=str)
    )
    logger.info(f"saved phi to {cfg.out} ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
