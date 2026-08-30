"""DQL (Diffusion-QL) extraction for pi0.5: BC + Q-maximization with BPTT through the sampler.

Provenance — "Diffusion Policies as an Expressive Policy Class for Offline RL" (arXiv
2208.06193), OFFICIAL code Zhendong-Wang/Diffusion-Policies-for-Offline-RL agents/ql_diffusion.py:
  - actor_loss = bc_loss + eta * q_loss (ql_diffusion.py:148), eta=1.0 default (:58,:98);
  - q_loss = -Q_i(s, a_new).mean() / |Q_j(s, a_new)|.mean().detach() with a randomly chosen
    twin i and the OTHER twin j providing the scale (:143-147) -- our K=2 ensemble members play
    the twins;
  - a_new is sampled from the CURRENT policy with gradients THROUGH the full denoising chain
    (their diffusion.sample keeps the graph; this BPTT is exactly what QAM/FlowDPG later avoid,
    which is why this arm matters as the canonical reference);
  - bc_loss is the policy's own generative loss (their diffusion loss; here pi0.5's flow loss,
    pi0.py:189-214).
Port notes: the chain is pi0.5's 10-step Euler sampler unrolled with a cached frozen prefix;
gradients flow through 10 suffix passes into the action expert only (backbone frozen,
arm-comparability convention). Batch is small for memory. lr 3e-4 (their run scripts), grad
clip: their code has optional grad_norm (main.py --grad_norm), commonly 1.0+; we keep global
norm 1.0 consistent with the other gradient-through-critic arms.
"""

# ruff: noqa: PLC0415

import argparse
import functools
import pathlib
import time

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init-ckpt",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/100000"
        ),
    )
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_g5_tau9_min"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--eta", type=float, default=1.0, help="ql_diffusion.py:58")
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4, help="BPTT through 10 suffix passes -- small")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/dql_run1"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="extract_dql_run1")
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.extraction import critic_q
    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    from openpi.models.pi0 import make_attn_mask
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    critic = critic_q.load(a.critic)
    cache = critic_q.CacheView(a.cache)
    robot_ad = critic.config["action_dim"]

    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    ds = exdata.AnnotatedBC(dataset, {})
    it = exdata.make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)
    H = cfg.model.action_horizon

    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    graphdef = nnx.graphdef(model)
    params = nnx.state(model)

    expert_filter = nnx.Any(
        nnx_utils.PathRegex(".*llm.*_1.*"),
        nnx_utils.PathRegex(".*(action_(in|out)_proj|time_mlp_(in|out)|state_proj).*"),
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt = tx.init(params.filter(expert_filter))
    N = a.ode_steps
    dt = 1.0 / N

    def velocity(model, obs, kv, pm, x, tau):
        import einops

        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x, tau)
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
        pref_attn = einops.repeat(pm, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn = jnp.concatenate([pref_attn, suffix_attn], axis=-1)
        pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn, positions=pos, kv_cache=kv, adarms_cond=[None, adarms]
        )
        return model.action_out_proj(out[:, -H:])

    def q_ensemble(feats, chunk, proprio):
        logits = critic.net.apply({"params": critic.params}, feats, jnp.clip(chunk, -1, 1), proprio)
        return critic.hl.from_logits(logits)[..., -1]  # [K, B]

    def loss_fn(model, rng, obs, actions, feats, proprio, swap):
        prng, brng, srng = jax.random.split(rng, 3)
        obs = _model.preprocess_observation(prng, obs, train=True)
        # bc_loss: pi0.5's own flow loss (ql_diffusion.py:140 uses the policy's generative loss)
        bc = jnp.mean(model.compute_loss(brng, obs, actions, train=True))
        # a_new: full sampler WITH gradients (their diffusion.sample keeps the graph)
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
        x = jax.random.normal(srng, actions.shape)
        for i in range(N):
            t = 1.0 - i * dt
            x = x - dt * velocity(model, obs, kv, prefix_mask, x, jnp.full((x.shape[0],), t))
        qk = q_ensemble(feats, x[..., :robot_ad], proprio)  # [2, B]
        # random twin for the objective, the other for the detached scale (ql_diffusion.py:143-147)
        qi = jnp.where(swap, qk[0], qk[1])
        qj = jnp.where(swap, qk[1], qk[0])
        q_loss = -qi.mean() / jax.lax.stop_gradient(jnp.abs(qj).mean() + 1e-8)
        loss = bc + a.eta * q_loss  # ql_diffusion.py:148
        return loss, {"bc": bc, "q_loss": q_loss, "q_pi": qi.mean()}

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt, rng, obs, actions, feats, proprio, swap):
        model = nnx.merge(graphdef, params)
        (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, expert_filter), has_aux=True)(
            model, rng, obs, actions, feats, proprio, swap
        )
        p = params.filter(expert_filter)
        upd, opt = tx.update(grads, opt, p)
        nnx.update(model, optax.apply_updates(p, upd))
        return nnx.state(model), opt, loss, info

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(
            project="yam-rlt",
            entity="RSS-PFT_RLLAB",
            name=a.wandb_name,
            group="extraction",
            config={k: str(v) for k, v in vars(a).items()} | {"method": "dql"},
        )
        print(f"wandb: {run.url}", flush=True)

    def save(step_i):
        import orbax.checkpoint as ocp

        path = (a.out / f"{step_i}").absolute()
        with ocp.StandardCheckpointer() as c:
            c.save(path, {"expert": params.filter(expert_filter).to_pure_dict()}, force=True)
        print(f"saved {path}", flush=True)

    import numpy as np

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        obs, actions, ann = next(it)
        f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
        rng, k, ks = jax.random.split(rng, 3)
        swap = jax.random.bernoulli(ks)
        params, opt, loss, info = step(params, opt, k, obs, jnp.asarray(actions), jnp.asarray(f), jnp.asarray(pr), swap)
        if s % 100 == 0:
            print(
                f"step {s:6d}  loss {float(loss):.4f}  bc {float(info['bc']):.4f}  "
                f"q_pi {float(info['q_pi']):.1f}  ({(s + 1) / (time.time() - t0):.3f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log({k2: float(v2) for k2, v2 in info.items()} | {"loss": float(loss)}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print("DQL extraction done.", flush=True)


if __name__ == "__main__":
    main()
