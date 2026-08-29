"""LPS / LPSD extraction for pi0.5: latent policy steering over the frozen alpha-Flow one-step base.

Provenance — the author's own LPS codebase (/data5/jellyho/LPS, paper arXiv 2603.05296), per the
user's directive: only `lps` (extract_method='ddpg') and `lpsd` ('onestep_ddpg') are taken from
it; the one-step base policy is OUR alpha-Flow pi0.5 (u-parametrization, per user: no jit_mf
reformulation), checkpoint yam_alphaflow_200k/200000 -- one-step action f(obs, z) = z - u(z, r=0,
t=1) (pi0_alphaflow sample path), differentiable in z with all base params frozen.

  LPS  (agents/lps.py:185-199): z = pi_L(obs); a = f(obs, z);
       loss = alpha * sg(1/|Q|_mean) * (-mean Q(s, a)); ensemble reduced by MEAN (:190-195);
       alpha = 1.0 (get_config :521).
  LPSD (agents/lps.py:155-166, :201-224): e ~ latent; z = pi_L(obs, e); anchor a_e = f(obs, e);
       loss = sg(1/|Q|)*(-mean Q(s, a_z)) + alpha * MSE(a_z, a_e).
       NOTE a faithful-line deviation: the code's line 205 computes `z_pred -
       compute_flow_actions(obs, z_pred)`, which under both mf branches equals the DISPLACEMENT
       u rather than an action; we evaluate Q on the one-step ACTION a_z = f(obs, z_pred) (the
       quantity the sampler executes, lps.py:294-327) and flagged the discrepancy to the author.
  latent_actor: plain MLP, hidden (256, 256) (get_config :506), no layer norm
       (actor_layer_norm=False, :509 via :426-428), raw output (utils/networks.py:301).
  latent_dist: their default is 'sphere' (meanflow_utils.py:54-58); our base was TRAINED with
       z ~ N(0, I) (openpi convention), so we use 'normal' -- the latent prior must match the
       base policy's training distribution (documented deviation).
  Observation rep for pi_L: frozen pooled-DINO(384) + proprio pos-14 -- the same frozen-encoder
       pattern their impala/state encoders play, and identical to the critic's rep (CacheView).
Optim: adam lr 3e-4, batch reduced for the VLA-scale backprop through the frozen expert.
"""

# ruff: noqa: PLC0415

import argparse
import functools
import pathlib
import time

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lps", "lpsd"], required=True)
    ap.add_argument(
        "--af-ckpt",
        type=pathlib.Path,
        default=pathlib.Path("/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000"),
    )
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_g5_tau9_min"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--alpha", type=float, default=1.0, help="lps get_config:521")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256], help="lps get_config:506")
    ap.add_argument("--lr", type=float, default=3e-4, help="lps get_config:503")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    out = a.out or pathlib.Path(f"/data1/jellyho/acrft_ckpts/extraction/{a.mode}_run1")

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.extraction import critic_q
    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    import openpi.training.config as _config
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    critic = critic_q.load(a.critic)
    cache = critic_q.CacheView(a.cache)
    robot_ad = critic.config["action_dim"]

    afc = _config.get_config("pi05_yam_lego_taxi_alphaflow")
    dataset, _bc_cfg = exdata.make_bc_dataset(str(a.af_ckpt / "assets"))
    ds = exdata.AnnotatedBC(dataset, {})
    it = exdata.make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)
    H, AD = afc.model.action_horizon, afc.model.action_dim
    LAT = H * AD

    model = afc.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = CheckpointWeightLoaderKeepMissing(str(a.af_ckpt / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    graphdef = nnx.graphdef(model)
    base_params = jax.device_put(nnx.state(model))  # frozen; passed as a jit ARG (closure constants OOM in XLA)

    # ---- latent actor: plain MLP (networks.py:301), raw output --------------------------------
    def init_mlp(key, dims):
        ks = jax.random.split(key, len(dims) - 1)
        return [
            (jax.random.normal(ks[i], (dims[i], dims[i + 1])) * jnp.sqrt(2.0 / dims[i]), jnp.zeros(dims[i + 1]))
            for i in range(len(dims) - 1)
        ]

    in_dim = 384 + len(critic.proprio_idx) + (LAT if a.mode == "lpsd" else 0)
    pi_l = init_mlp(jax.random.key(1), [in_dim, *a.hidden, LAT])

    def mlp(p, x):
        for w, b in p[:-1]:
            x = jax.nn.relu(x @ w + b)
        w, b = p[-1]
        return x @ w + b

    def one_step(bp, obs, z):
        """alpha-Flow one-step action: z - u(z, r=0, t=1) with the frozen base (differentiable in z)."""
        m = nnx.merge(graphdef, bp)
        prefix_mask, kv = m._prefix_forward(obs)
        b = z.shape[0]
        t = jnp.ones((b,), jnp.float32)
        r = jnp.zeros((b,), jnp.float32)
        u = m._u(obs, prefix_mask, kv, z.reshape(b, H, AD), t, r)
        return (z.reshape(b, H, AD) - u).reshape(b, LAT)

    def loss_fn(pi_l, bp, rng, obs, feats_pool, proprio, feats):
        b = feats_pool.shape[0]
        rep = jnp.concatenate([feats_pool, proprio], axis=-1)
        if a.mode == "lps":
            z = mlp(pi_l, rep)  # deterministic latent actor (lps.py:294-305 no-e branch)
            act = one_step(bp, obs, z)
            q = critic.q_mean(feats, act.reshape(b, H, AD)[..., :robot_ad], proprio)
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))  # lps.py:198
            loss = a.alpha * lam * (-q.mean())  # lps.py:197-199
            return loss, {"q_pi": q.mean(), "z_norm": jnp.mean(jnp.square(z))}
        # lpsd
        e = jax.random.normal(rng, (b, LAT))  # 'normal' latent (deviation: base trained w/ N(0,I))
        z = mlp(pi_l, jnp.concatenate([rep, e], axis=-1))
        a_e = jax.lax.stop_gradient(one_step(bp, obs, e))  # anchor from the raw latent (lps.py:176)
        a_z = one_step(bp, obs, z)
        q = critic.q_mean(feats, a_z.reshape(b, H, AD)[..., :robot_ad], proprio)
        lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
        q_loss = lam * (-q.mean())  # lps.py:214-216
        mse = jnp.mean(jnp.square(a_z - a_e))  # lps.py:217 anchor
        return q_loss + a.alpha * mse, {"q_pi": q.mean(), "mse": mse, "z_norm": jnp.mean(jnp.square(z))}

    tx = optax.adam(a.lr)
    opt = tx.init(pi_l)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(pi_l, opt, bp, rng, obs, feats_pool, proprio, feats):
        obs = _model.preprocess_observation(None, obs, train=False)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(pi_l, bp, rng, obs, feats_pool, proprio, feats)
        upd, opt = tx.update(grads, opt, pi_l)
        return optax.apply_updates(pi_l, upd), opt, loss, info

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(
            project="yam-rlt",
            entity="RSS-PFT_RLLAB",
            name=f"extract_{a.mode}_run1",
            group="extraction",
            config={k: str(v) for k, v in vars(a).items()} | {"method": a.mode},
        )
        print(f"wandb: {run.url}", flush=True)

    import numpy as np

    def save(step_i):
        import flax.serialization

        out.mkdir(parents=True, exist_ok=True)
        (out / f"latent_actor_{step_i}.msgpack").write_bytes(
            # msgpack can't pack tuples: serialize as list-of-[w, b] lists (round-2 smoke)
            flax.serialization.msgpack_serialize([[np.asarray(w), np.asarray(b)] for w, b in pi_l])
        )
        print(f"saved {out}/latent_actor_{step_i}.msgpack", flush=True)

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        obs, _actions, ann = next(it)
        f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
        fp = f.mean(axis=1)  # pooled DINO rep for the latent actor
        rng, k = jax.random.split(rng)
        pi_l, opt, loss, info = step(pi_l, opt, base_params, k, obs, jnp.asarray(fp), jnp.asarray(pr), jnp.asarray(f))
        if s % 50 == 0:
            msg = "  ".join(f"{k2} {float(v2):.4f}" for k2, v2 in info.items())
            print(f"step {s:6d}  loss {float(loss):.4f}  {msg}  ({(s + 1) / (time.time() - t0):.3f} it/s)", flush=True)
            if run is not None:
                run.log({k2: float(v2) for k2, v2 in info.items()} | {"loss": float(loss)}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print(f"{a.mode.upper()} extraction done.", flush=True)


if __name__ == "__main__":
    main()
