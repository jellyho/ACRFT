"""QC-FQL one-step distillation as an EXTRACTION arm: the FQL actor loss against a FROZEN critic.

Why this exists next to scripts/train_fql.py: that script is the full QC-FQL recipe, which
CO-TRAINS its own critic (stage 1 MC warm-up, stage 2 TD with the MC floor). That makes it a fine
method but a poor member of this comparison ring, where every other arm extracts from the SAME
frozen patch critic -- a difference in results there could come from the critic rather than from
the extraction mechanism. This arm keeps FQL's actor objective and drops the co-training, so
"QC-FQL one-step distillation" is measured under the ring's method-only-diff convention.

Objective, from the official FQL actor loss (as reproduced in train_fql.py:225-236, itself
following seohongpark/fql agents/fql.py):

    a_theta = stop_grad( multi-step flow ODE of the FROZEN teacher )   # the BC policy's own sample
    a_omega = ONE Euler step of the trainable student from the same noise
    l_distill = mean (a_omega - a_theta)^2
    l_q       = -E[Q(s, a_omega)] / sg(E|Q(s, a_omega)|)               # scale-free, official
    loss      = l_q + alpha * l_distill

The Q term uses the frozen patch critic (openpi.extraction.critic_q), and the normalization is
what keeps alpha meaningful at our |Q| ~ 1e3 value scale -- without it the distillation term is
negligible and the actor leaves the data manifold (observed in the co-trained run: l_distill
0.002 -> 2.35 within 10 steps).

Teacher/student share the frozen backbone and differ only in the action expert (same structure as
train_qam.py), so one non-expert parameter set is held, not two.
"""

# ruff: noqa: PLC0415

import argparse
import functools
import pathlib
import time


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
    ap.add_argument("--alpha", type=float, default=10.0, help="distillation weight (fql.py get_config)")
    ap.add_argument("--teacher-steps", type=int, default=10, help="teacher ODE steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/fqlx_run1"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="jellyho_")
    ap.add_argument("--wandb-name", default="extract_fqlx_run1")
    a = ap.parse_args()

    import einops
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
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
    H, AD = cfg.model.action_horizon, cfg.model.action_dim

    expert_filter = nnx.Any(
        nnx_utils.PathRegex(".*llm.*_1.*"),
        nnx_utils.PathRegex(".*(action_(in|out)_proj|time_mlp_(in|out)|state_proj).*"),
    )
    model = cfg.model.create(jax.random.key(0))
    graphdef_m, p_exp, p_rest = nnx.split(model, expert_filter, ...)
    loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(
        nnx.State.merge(p_exp, p_rest).to_pure_dict()
    )
    full = nnx.state(model)
    full.replace_by_pure_dict(loaded)
    p_exp_t, p_rest = full.filter(expert_filter), full.filter(nnx.Not(expert_filter))
    p_exp_t = jax.device_put(p_exp_t)  # teacher expert: frozen BC
    p_rest = jax.device_put(p_rest)  # shared frozen backbone
    p_exp_s = jax.tree.map(lambda x: x.copy(), p_exp_t)  # student starts at the teacher

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt = tx.init(p_exp_s)
    N = a.teacher_steps
    dt = 1.0 / N

    def prefix(p_e, obs):
        m = nnx.merge(graphdef_m, p_e, p_rest)
        tok, mask, ar = m.embed_prefix(obs)
        _, kv = m.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1)
        return kv, mask

    def velocity(p_e, obs, kv, pm, x, tau):
        m = nnx.merge(graphdef_m, p_e, p_rest)
        st, sm, sar, adarms = m.embed_suffix(obs, x, tau)
        full_attn = jnp.concatenate(
            [einops.repeat(pm, "b p -> b s p", s=st.shape[1]), make_attn_mask(sm, sar)], axis=-1
        )
        pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, out), _ = m.PaliGemma.llm(
            [None, st], mask=full_attn, positions=pos, kv_cache=kv, adarms_cond=[None, adarms]
        )
        return m.action_out_proj(out[:, -H:])

    velocity = jax.checkpoint(velocity)  # the teacher unrolls N of these; recompute in backward

    def loss_fn(p_e_s, obs, feats, proprio, z):
        # teacher: the frozen BC policy's own multi-step sample from this noise (stop-grad)
        kv_t, pm = prefix(p_exp_t, obs)
        x = z
        for i in range(N):
            x = x - dt * velocity(p_exp_t, obs, kv_t, pm, x, jnp.full((x.shape[0],), 1.0 - i * dt))
        a_theta = jax.lax.stop_gradient(x)
        # student: ONE Euler step from the same noise (the "one-step actor")
        kv_s, pm_s = prefix(p_e_s, obs)
        a_omega = z - velocity(p_e_s, obs, kv_s, pm_s, z, jnp.ones((z.shape[0],)))
        l_distill = jnp.mean(jnp.square(a_omega - a_theta))
        q = critic.q_mean(feats, jnp.clip(a_omega[..., :robot_ad], -1, 1), proprio)
        l_q = -jnp.mean(q) / jax.lax.stop_gradient(jnp.mean(jnp.abs(q)) + 1e-6)  # official normalization
        return l_q + a.alpha * l_distill, {"l_distill": l_distill, "l_q": l_q, "q_pi": jnp.mean(q)}

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(p_e_s, opt, rng, obs, feats, proprio):
        obs = _model.preprocess_observation(None, obs, train=False)
        z = jax.random.normal(rng, (obs.state.shape[0], H, AD))
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p_e_s, obs, feats, proprio, z)
        upd, opt = tx.update(grads, opt, p_e_s)
        return optax.apply_updates(p_e_s, upd), opt, loss, info

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(
            project="yam-rlt",
            entity=a.wandb_entity,
            name=a.wandb_name,
            group="extraction",
            config={k: str(v) for k, v in vars(a).items()} | {"method": "fqlx"},
        )
        print(f"wandb: {run.url}", flush=True)

    def save(step_i):
        import orbax.checkpoint as ocp

        path = (a.out / f"{step_i}").absolute()
        with ocp.StandardCheckpointer() as c:
            c.save(path, {"expert": p_exp_s.to_pure_dict()}, force=True)
        print(f"saved {path}", flush=True)

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        obs, _actions, ann = next(it)
        f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
        rng, k = jax.random.split(rng)
        p_exp_s, opt, loss, info = step(p_exp_s, opt, k, obs, jnp.asarray(f), jnp.asarray(pr))
        if s % 100 == 0:
            print(
                f"step {s:6d}  loss {float(loss):.4f}  l_distill {float(info['l_distill']):.5f}  "
                f"q_pi {float(info['q_pi']):.1f}  ({(s + 1) / (time.time() - t0):.3f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log({k2: float(v2) for k2, v2 in info.items()} | {"loss": float(loss)}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print("FQL-X extraction done.", flush=True)


if __name__ == "__main__":
    main()
