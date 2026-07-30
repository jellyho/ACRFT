"""Predict the RL token h steps ahead from the token and the chunk that was executed.

The critic cannot rank the policy's candidates at a state, and 7x more data made that worse rather
than better: the within-state spread of Q shrank to 3.6 times its own noise, which is what sixteen
draws of noise produce anyway, while the state-value fit improved. That is the expected outcome of
the objective, not a training failure. The online loss is evaluated on ONE action per state - the
demonstrated chunk - and its target is read at the state the demonstration reached, so it carries the
same value whichever candidate is under consideration. No candidate ever has a target of its own, so
nothing asks the critic to separate them, and more data only sharpens the answer to the question it
IS asked, which is V(s).

A dynamics model in token space breaks that, because it turns the missing quantity into a supervised
one. The annotation is a stride-1 dump, so it already holds (z_t, a_t, z_{t+h}) for every frame and
every prefix length:

    f_w(z_t, a_1:h)  ~  z_{t+h}          trained on the demonstrated chunk
    Q(z_t, a)        ~  cum_r + gamma^h V(f_w(z_t, a))     evaluated on ANY chunk

f_w is supervised by exactly the same one-action-per-state data, so this is not something for
nothing - the trade is that its target is a 2048-dimensional vector rather than one scalar, which is
three orders of magnitude more supervision per frame for the action-dependent part. Whether that is
enough is what this measures: a model that ignores its action argument predicts the demonstrated
next state as well as an action-conditioned one, so the reported baselines are the point.

    identity        predict z_{t+h} = z_t                 (how much changes in h steps at all)
    no-action       f(z_t) alone, same capacity           (what the state already determines)
    action-conditioned                                     (what the action adds)

The gap between the last two is the entire hypothesis. If it is near zero the token pair carries no
action information either and the offline route is closed.

Usage:
    uv run scripts/train_latent_dynamics.py --data .scratch/annot_full --out .scratch/dyn_reconprog
"""

import argparse
import json
import logging
import pathlib
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


class Dynamics(nn.Module):
    """(z, flattened chunk prefix, prefix length) -> predicted z' as a residual on z."""

    token_dim: int
    width: int = 1024
    depth: int = 3
    use_action: bool = True

    @nn.compact
    def __call__(self, z, a, h):
        # h is one-hot over prefix positions so the model knows how far ahead it is predicting.
        x = jnp.concatenate([z, a, h] if self.use_action else [z, h], axis=-1)
        for _ in range(self.depth):
            x = nn.gelu(nn.Dense(self.width)(x))
        # Residual: most of the token is unchanged over a chunk, so predicting the delta puts the
        # model's capacity on what actually moves instead of on copying.
        return z + nn.Dense(self.token_dim)(x)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--macro-group-size", type=int, default=2, help="Prefix lengths are multiples of this.")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    meta = json.loads((cfg.data / "meta.json").read_text())
    T, H, A, D = meta["num_frames"], meta["horizon"], meta["action_dim"], meta["token_dim"]
    if cfg.max_frames:
        T = min(T, cfg.max_frames)
    import ml_dtypes

    store = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]

    def rd(name, shape, dtype=None):
        arr = np.memmap(cfg.data / f"{name}.dat", dtype=store if dtype is None else dtype, mode="r", shape=shape)[:T]
        return np.asarray(arr, np.float32) if dtype is None else np.asarray(arr)

    full = meta["num_frames"]
    tok = jnp.asarray(rd("rl_token", (full, D)))
    chunk = jnp.asarray(rd("action_chunk", (full, H, A)))
    ep = jnp.asarray(rd("episode_index", (full,), np.int32))
    done = jnp.asarray(rd("done", (full,), np.bool_).astype(np.int32))
    dcum = jnp.cumsum(done)
    prefixes = jnp.arange(cfg.macro_group_size, H + 1, cfg.macro_group_size)
    P = len(prefixes)
    logger.info(f"{T} frames, token {D}, chunk {H}x{A}, {P} prefix lengths")

    # Scale the target error by the token's own spread so the numbers mean the same thing across
    # annotations with different token statistics.
    zc = np.asarray(tok[: min(T, 20000)])
    z_var = float(np.mean(np.var(zc, axis=0)))
    logger.info(f"token per-dim variance {z_var:.4f}")

    def batch(rng):
        k1, k2 = jax.random.split(rng)
        i = jax.random.randint(k1, (cfg.batch_size,), 0, T)
        pi = jax.random.randint(k2, (cfg.batch_size,), 0, P)
        h = prefixes[pi]
        nxt = jnp.clip(i + h, 0, T - 1)
        # Only transitions that stay inside the episode and stop at or before its terminal.
        crossed = dcum[nxt] - jnp.where(i > 0, dcum[i - 1], 0)
        ok = ((crossed == 0) | ((crossed == 1) & (done[nxt] > 0))) & (ep[nxt] == ep[i]) & (i + h < T)
        # Zero out the actions past the prefix so the model sees only what was committed to.
        step = jnp.arange(H)[None, :]
        a = chunk[i] * (step < h[:, None])[..., None]
        return tok[i], a.reshape(cfg.batch_size, H * A), jax.nn.one_hot(pi, P), tok[nxt], ok.astype(jnp.float32)

    results = {}
    for name, use_action in (("action_conditioned", True), ("no_action", False)):
        net = Dynamics(token_dim=D, width=cfg.width, depth=cfg.depth, use_action=use_action)
        rng = jax.random.key(cfg.seed)
        z0, a0, h0, _, _ = batch(rng)
        params = net.init(rng, z0, a0, h0)
        tx = optax.adam(cfg.lr)
        opt = tx.init(params)

        def loss_fn(p, z, a, h, zn, w, _net=net):
            pred = _net.apply(p, z, a, h)
            per = jnp.mean(jnp.square(pred - zn), axis=-1)
            return jnp.sum(per * w) / (jnp.sum(w) + 1e-8)

        @jax.jit
        def step_fn(carry, r, _lf=loss_fn, _tx=tx):
            p, o = carry
            z, a, h, zn, w = batch(r)
            lo, g = jax.value_and_grad(_lf)(p, z, a, h, zn, w)
            u, o = _tx.update(g, o)
            return (optax.apply_updates(p, u), o), lo

        @jax.jit
        def run(carry, r):
            return jax.lax.scan(step_fn, carry, jax.random.split(r, 100))

        carry, t0 = (params, opt), time.perf_counter()
        for s in range(0, cfg.steps, 100):
            carry, losses = run(carry, jax.random.fold_in(rng, s))
            if s % 2000 == 0:
                logger.info(f"[{name}] step {s + 100}/{cfg.steps}  loss {float(jnp.mean(losses)):.5f}")
        params = carry[0]

        # Held-out evaluation on frames the loop also sampled from, but with a fixed key, plus the
        # identity baseline on the same batches so all three numbers share the transitions.
        errs, ident = [], []
        for j in range(40):
            z, a, h, zn, w = batch(jax.random.key(10_000 + j))
            pred = net.apply(params, z, a, h)
            errs.append(float(jnp.sum(jnp.mean(jnp.square(pred - zn), -1) * w) / (jnp.sum(w) + 1e-8)))
            ident.append(float(jnp.sum(jnp.mean(jnp.square(z - zn), -1) * w) / (jnp.sum(w) + 1e-8)))
        results[name] = float(np.mean(errs))
        results["identity"] = float(np.mean(ident))
        logger.info(
            f"[{name}] mse {results[name]:.5f}   identity {results['identity']:.5f}  ({(time.perf_counter() - t0) / 60:.1f} min)"
        )

    ac, na, idt = results["action_conditioned"], results["no_action"], results["identity"]
    print("\n=== latent dynamics ===")
    print(f"  identity  (z' = z)            {idt:.5f}   {idt / z_var:.3f} of token variance")
    print(f"  no-action f(z)                {na:.5f}   {na / idt:.3f} of identity")
    print(f"  action-conditioned f(z, a)    {ac:.5f}   {ac / idt:.3f} of identity")
    gain = (na - ac) / max(na, 1e-9)
    print(f"\n  action information: {gain:+.1%} of the no-action error")
    if gain < 0.02:
        print("  -> the chunk barely helps predict the next token. The token pair carries no action")
        print("     signal either, and a dynamics-model route to per-candidate targets is closed.")
    else:
        print("  -> the chunk measurably determines where the token goes; per-candidate targets are")
        print("     available through f, and the critic can be given a value for each candidate.")
    if cfg.out:
        cfg.out.mkdir(parents=True, exist_ok=True)
        (cfg.out / "results.json").write_text(json.dumps(results | {"token_var": z_var, "action_gain": gain}, indent=1))
        import flax.serialization as fser

        (cfg.out / "params.msgpack").write_bytes(fser.to_bytes(params))
        print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
