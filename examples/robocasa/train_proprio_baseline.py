"""Proprio-only baseline policy: the latent BC probe with the RL token removed.

The probe measures how much of the policy is recoverable from the frozen RL token, but it is also
conditioned on proprioception (deliberately - the downstream critic gets proprio separately, so the
probe must too). That makes the probe's success rate an upper bound on the token's contribution
rather than a measurement of it: a task that is largely solvable from joint states alone would score
well with a token that carries nothing.

This trains the identical head - same DiT, same width/depth/heads, same flow-matching objective, same
normalization - with only the `z_rl` conditioning term deleted. Its rollout success rate is the
baseline against which every probe number should be read:

    token contribution  =  probe(z_rl, proprio)  -  this baseline

Because it needs neither images nor the VLA, it reads proprio and actions straight out of the LeRobot
parquet files (no video decoding) and trains in well under an hour on a single small GPU.

Usage:
    uv run examples/robocasa/train_proprio_baseline.py --task PrepareCoffee --steps 30000
"""

import argparse
import json
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.pi0 import posemb_sincos
import openpi.models.pi0_rlt as pi0_rlt
import openpi.training.config as _config
import openpi.transforms as _transforms


class ProprioDiT(nnx.Module):
    """The probe head with the z_rl term removed: velocity(x_t, time, proprio)."""

    def __init__(self, *, horizon, action_dim, state_dim, width, depth, heads, mlp_ratio, rngs):
        self.horizon, self.action_dim, self.depth = horizon, action_dim, depth
        self.in_proj = nnx.Linear(action_dim, width, rngs=rngs)
        self.pos = nnx.Param(pi0_rlt._sincos_posemb(horizon, width))
        self.time = nnx.Linear(width, width, rngs=rngs)
        self.state = nnx.Linear(state_dim, width, rngs=rngs)
        self.blocks = nnx.Dict(
            {f"blk_{i}": pi0_rlt._DiTBlock(width, heads, width * mlp_ratio, rngs=rngs) for i in range(depth)}
        )
        z = nnx.initializers.zeros_init()
        self.out_ada = nnx.Linear(width, 2 * width, kernel_init=z, bias_init=z, rngs=rngs)
        self.out = nnx.Linear(width, action_dim, kernel_init=z, bias_init=z, rngs=rngs)

    def velocity(self, x_t, time, state):
        w = self.time.in_features
        cond = nnx.swish(self.time(posemb_sincos(time, w, min_period=4e-3, max_period=4.0))) + self.state(state)
        h = self.in_proj(x_t) + self.pos.value
        for i in range(self.depth):
            h = self.blocks[f"blk_{i}"](h, cond)
        shift, scale = jnp.split(self.out_ada(nnx.swish(cond))[:, None, :], 2, axis=-1)
        return self.out(pi0_rlt._layernorm(h) * (1 + scale) + shift)

    def sample(self, rng, state, *, num_steps=10):
        b = state.shape[0]
        dt = -1.0 / num_steps

        def step(carry):
            x, t = carry
            return x + dt * self.velocity(x, jnp.broadcast_to(t, b), state), t + dt

        noise = jax.random.normal(rng, (b, self.horizon, self.action_dim))
        x0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
        return x0


def load_proprio_actions(repo_id: str, horizon: int):
    """(state, action-chunk) pairs straight from the LeRobot parquet - no video, no VLA."""
    from lerobot.datasets import lerobot_dataset
    import pyarrow.dataset as pads

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    files = sorted(pathlib.Path(meta.root).glob("data/chunk-*/file-*.parquet"))
    table = pads.dataset(files, format="parquet").to_table(columns=["index", "observation.state", "action"])
    idx = table["index"].to_numpy()
    state = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    action = np.stack(table["action"].to_pylist()).astype(np.float32)
    order = np.argsort(idx)
    state, action = state[order], action[order]

    lo = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    hi = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)
    # Chunk within episodes, clamping at the end (repeat the last action), as the loader does.
    chunks = np.empty((len(state), horizon, action.shape[-1]), dtype=np.float32)
    for a, b in zip(lo, hi, strict=True):
        for k in range(horizon):
            chunks[a:b, k] = action[np.minimum(np.arange(a, b) + k, b - 1)]
    return state, chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-interval", type=int, default=10_000)
    ap.add_argument("--eval-trials", type=int, default=20)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Where to write results.json.")
    args = ap.parse_args()

    cfg = _config.get_config(f"pi05_robocasa_{args.task}_rlt")
    m = cfg.model
    H, AD = m.action_horizon, m.rlt_probe_action_dim or m.action_dim
    data_config = cfg.data.create(cfg.assets_dirs, m)

    print(f"task={args.task} horizon={H} action_dim={AD} padded={m.action_dim}", flush=True)
    state_raw, act_raw = load_proprio_actions(data_config.repo_id, H)
    print(f"loaded {len(state_raw)} frames  state{state_raw.shape[1:]}  action{act_raw.shape[1:]}", flush=True)

    # Exactly the transforms the probe's inputs go through: normalize, then pad to the model dim.
    norm = _transforms.compose(
        [
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            _transforms.PadStatesAndActions(m.action_dim),
        ]
    )
    out_tf = _transforms.compose(
        [
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    normed = norm({"state": state_raw, "actions": act_raw})
    S = jnp.asarray(np.asarray(normed["state"]))  # [N, 32]
    A = jnp.asarray(np.asarray(normed["actions"])[..., :AD])  # [N, H, AD]
    print(f"normalized: state{S.shape} actions{A.shape}", flush=True)

    model = ProprioDiT(
        horizon=H,
        action_dim=AD,
        state_dim=m.action_dim,
        width=m.rlt_probe_width,
        depth=m.rlt_probe_depth,
        heads=m.rlt_probe_heads,
        mlp_ratio=m.rlt_probe_mlp_ratio,
        rngs=nnx.Rngs(args.seed),
    )
    opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr)))

    @nnx.jit
    def train_step(model, opt, rng, s, a):
        def loss_fn(mdl):
            nr, tr = jax.random.split(rng)
            noise = jax.random.normal(nr, a.shape)
            t = jax.random.beta(tr, 1.5, 1, a.shape[:1]) * 0.999 + 0.001
            te = t[..., None, None]
            x_t, u_t = te * noise + (1 - te) * a, noise - a
            return jnp.mean(jnp.square(mdl.velocity(x_t, t, s) - u_t))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(grads)
        return loss

    def policy_fn(element):
        """Raw env observation -> raw action chunk, mirroring the serving output chain."""
        inp = norm({"state": np.asarray(element["observation/state"])[None]})
        s = jnp.asarray(np.asarray(inp["state"]))
        a = np.asarray(model.sample(jax.random.key(int(time.time_ns()) & 0x7FFFFFFF), s)[0])
        a = np.pad(a, ((0, 0), (0, m.action_dim - AD)))
        return np.asarray(out_tf({"state": np.asarray(inp["state"][0]), "actions": a})["actions"])

    results = []
    rng = jax.random.key(args.seed + 1)
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        rng, srng, brng = jax.random.split(rng, 3)
        i = jax.random.randint(brng, (args.batch_size,), 0, S.shape[0])
        loss = train_step(model, opt, srng, S[i], A[i])
        if step % 1000 == 0:
            print(f"step {step:>6}  loss {float(loss):.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            import examples.robocasa.rollout as ro

            model.eval()
            env = ro.make_env(args.task, camera_size=256, seed=args.eval_seed)
            np.random.seed(args.eval_seed)
            r = ro.run_trials(
                env, policy_fn, task=args.task, num_trials=args.eval_trials, seed=args.eval_seed, replan_steps=H
            )
            model.train()
            print(f"PROPRIO-ONLY @ {step}: success {r['success_rate']:.0%}", flush=True)
            results.append({"step": step, "success_rate": r["success_rate"]})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"task": args.task, "results": results}, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
