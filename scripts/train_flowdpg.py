"""FlowDPG extraction for pi0.5: critic gradient distilled into the velocity field, no BPTT.

Provenance — "FlowDPG: Deterministic Policy Gradient on Flow Matching Policies for Real-World
Manipulation" (arXiv 2606.22303), Section 3.2, Eqs. 4-9 and Algorithm 1 lines 7-13. There is NO
official code release (checked: abs page and flowdpg.github.io carry no repo), so this is a
paper-faithful implementation, appendix-checked. Their time convention is t=1 data / t=0 noise;
openpi's is the reverse (pi0.py:246 comment), so with our tau = 1 - t:
  Eq. 4  a_hat = x_tau - tau * v_theta(x_tau, tau)          (their x_t + (1-t) v)
  Eq. 5  g = grad_{a_hat} min_i Q_i(s, a_hat)               (twin min -> our K=2 ensemble min)
  Eq. 6  Delta = alpha * ||u|| / (||g|| + eps) * g          (per-sample flattened-chunk norms)
  Eq. 7  a_star = a_hat + Delta (stop-grad target), target velocity u* = eps_noise - a_star
  Eq. 8  L = lambda ||v - u*||^2 + (1 - lambda) ||v - u_bc||^2, lambda linearly warmed to
         lambda_max over N_warmup steps (Algorithm 1 line 13)
  Eq. 9  L_cons = ||a_hat - x1||^2 with gradients THROUGH a_hat (keeps the projection honest)
Hyperparameters follow Table 3 / Appendix B.2: alpha=0.5, eps=1e-6, lambda_max=0.5,
N_warmup=2000, mu_cons=1.0, AdamW lr 1e-4 wd 1e-4, grad clip 1.0. Critic queries use the frozen
patch critic on cached DINO features addressed by the loader's shared index
(openpi.extraction.data docstring), replacing their Sec.-3.1 IQL critic (ours is external).
Trainable: action expert only, backbone frozen (arm-comparability convention).
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
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--alpha", type=float, default=0.5, help="guidance scale, Table 3")
    ap.add_argument("--lambda-max", type=float, default=0.5, help="Table 3")
    ap.add_argument("--warmup", type=int, default=2000, help="N_warmup, Table 3")
    ap.add_argument("--mu-cons", type=float, default=1.0, help="Table 3")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/flowdpg_run1")
    )
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="jellyho_")
    ap.add_argument("--wandb-name", default="extract_flowdpg_run1")
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.extraction import critic_q
    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    critic = critic_q.load(a.critic)
    cache = critic_q.CacheView(a.cache)
    grad_q = critic_q.grad_qmin_chunk(critic)  # Eq. 5, twin-min
    robot_ad = critic.config["action_dim"]  # 14; policy chunks are padded to 32

    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    ds = exdata.AnnotatedBC(dataset, {})
    it = exdata.make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)

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
    # AdamW + global-norm clip 1.0 (Table 3)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=a.weight_decay))
    opt = tx.init(params.filter(expert_filter))
    H, AD = cfg.model.action_horizon, cfg.model.action_dim

    def velocity(model, obs, x_t, tau):
        """One pi0.5 velocity evaluation at (x_tau, tau) — the pieces of pi0.py:200-213."""
        from openpi.models.pi0 import make_attn_mask

        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, tau)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions, adarms_cond=[None, adarms]
        )
        return model.action_out_proj(suffix_out[:, -H:])

    def loss_fn(model, rng, obs, actions, feats, proprio, lam):
        prng, nrng, trng = jax.random.split(rng, 3)
        obs = _model.preprocess_observation(prng, obs, train=True)
        b = actions.shape[0]
        noise = jax.random.normal(nrng, actions.shape)
        tau = jax.random.beta(trng, 1.5, 1, (b,)) * 0.999 + 0.001  # pi0.5 native time (pi0.py:197)
        te = tau[..., None, None]
        x_t = te * noise + (1 - te) * actions
        u_bc = noise - actions

        v = velocity(model, obs, x_t, tau)

        # Eq. 4: projection to the clean endpoint (stop-grad branch for the target)
        a_hat_sg = jax.lax.stop_gradient(x_t - te * v)
        # Eq. 5-6: critic gradient at a_hat, rescaled to the demo-velocity norm (flattened chunk)
        g = grad_q(a_hat_sg[..., :robot_ad], feats, proprio)  # [b, H, 14]
        g = jnp.concatenate([g, jnp.zeros((b, H, AD - robot_ad))], axis=-1)  # pad dims get no push
        gn = jnp.linalg.norm(g.reshape(b, -1), axis=-1, keepdims=True)[..., None]
        un = jnp.linalg.norm(u_bc.reshape(b, -1), axis=-1, keepdims=True)[..., None]
        delta = a.alpha * (un / (gn + 1e-6)) * g  # Eq. 6, eps=1e-6 (Table 3)
        a_star = a_hat_sg + delta  # Eq. 7 (constant target)
        u_star = noise - a_star  # their u* = a* - x0, mapped to our convention

        l_dpg = lam * jnp.mean(jnp.square(v - u_star)) + (1 - lam) * jnp.mean(jnp.square(v - u_bc))  # Eq. 8
        a_hat = x_t - te * v  # Eq. 9: gradients THROUGH the projection
        l_cons = jnp.mean(jnp.square(a_hat - actions))
        loss = l_dpg + a.mu_cons * l_cons
        return loss, {"l_dpg": l_dpg, "l_cons": l_cons, "gnorm": jnp.mean(gn)}

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt, rng, obs, actions, feats, proprio, lam):
        model = nnx.merge(graphdef, params)
        (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, expert_filter), has_aux=True)(
            model, rng, obs, actions, feats, proprio, lam
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
            entity=a.wandb_entity,
            name=a.wandb_name,
            group="extraction",
            config={k: str(v) for k, v in vars(a).items()} | {"method": "flowdpg"},
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
        lam = a.lambda_max * min(s / a.warmup, 1.0)  # Algorithm 1 line 13
        rng, k = jax.random.split(rng)
        params, opt, loss, info = step(
            params, opt, k, obs, jnp.asarray(actions), jnp.asarray(f), jnp.asarray(pr), jnp.float32(lam)
        )
        if s % 100 == 0:
            print(
                f"step {s:6d}  loss {float(loss):.5f}  l_dpg {float(info['l_dpg']):.5f}  "
                f"l_cons {float(info['l_cons']):.5f}  lam {lam:.2f}  ({(s + 1) / (time.time() - t0):.2f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log({k2: float(v2) for k2, v2 in info.items()} | {"loss": float(loss), "lam": lam}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print("FlowDPG extraction done.", flush=True)


if __name__ == "__main__":
    main()
