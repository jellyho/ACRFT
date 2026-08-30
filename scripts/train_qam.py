"""QAM policy extraction for pi0.5: adjoint matching against a frozen patch critic.

Provenance — "Q-learning with Adjoint Matching" (arXiv 2601.14234), OFFICIAL code
ColinQiyangLi/qam agents/qam.py; every rule below carries its source line.
  - Two velocity fields: actor_slow = the frozen BC flow (here: the pi0.5 BC checkpoint,
    frozen -- our critic is external so no BC drift is needed, per the repo's own structure
    where actor_slow receives only the flow loss, qam.py:113-122) and actor_fast = the
    trainable fine-tuned field (a full copy of the action expert), qam.py:356-363.
  - Reference rollout (qam.py:49-77): x ~ N(0,I); deterministic time grid t_i = i/T (:63);
    sigma_t = sqrt(2(1-t+h)/(t+h)) (:64); Euler-Maruyama x <- x + h(2 v_fast - x/(t+h)) +
    sqrt(h) sigma noise (:71); LAST step deterministic with actor_slow only (:72-73). The
    rollout is constant w.r.t. gradients (adj_matching never sees grad_params, :130).
  - Adjoint: terminal adj = -grad_a Q(s, x_T) * inv_temp (:85, ensemble MEAN :81-83, action
    clipped INSIDE the Q call :80-81); backward recursion adj <- adj + h * vjp_{fn}(adj) with
    fn(x) = 2 actor_slow(x, t+h) - x/(t+h) (:92-101).
  - Loss (:140-145, residual=False default): sum_dims((v_fast - v_slow) * 2/sigma + sigma*adj)^2,
    summed over the step axis, meaned over batch.
  - Stability: global-norm clip 1.0 (:387-389), inv_temp default 0.3 (:434).
Time convention: QAM uses t=0 noise -> t=1 data; openpi is reversed (pi0.py:246). We keep QAM's
internal grid and evaluate pi0.5 velocities at tau = 1 - t with the sign flip
v_openpi(x, tau) = -v_qam(x, t) (openpi's u = noise - data).
Deviation: actor_fast duplicates only the ACTION EXPERT (suffix stack); both fields share the
frozen VLM prefix KV, so a step costs 1 prefix + 2T suffix passes + T VJPs.
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
    ap.add_argument("--flow-steps", type=int, default=10, help="qam.py:429")
    ap.add_argument("--inv-temp", type=float, default=0.3, help="qam.py:434")
    ap.add_argument("--lr", type=float, default=3e-4, help="qam.py:414")
    ap.add_argument("--batch", type=int, default=8, help="VLA-scale; official 256 at MLP scale")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/qam_run1"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="jellyho_")
    ap.add_argument("--wandb-name", default="extract_qam_run1")
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
    H, AD = cfg.model.action_horizon, cfg.model.action_dim

    # actor_slow: the BC checkpoint, frozen. actor_fast: an independent copy of the model whose
    # ACTION EXPERT is trainable (qam.py:356-363 registers slow/fast as separate modules).
    def build():
        model = cfg.model.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
        state.replace_by_pure_dict(loaded)
        return nnx.merge(graphdef, state)

    expert_filter = nnx.Any(
        nnx_utils.PathRegex(".*llm.*_1.*"),
        nnx_utils.PathRegex(".*(action_(in|out)_proj|time_mlp_(in|out)|state_proj).*"),
    )
    # slow and fast differ ONLY in the (trainable) action expert; two full fp32 param sets
    # (~14GB each) plus the training program blew the 44GB L40S (round-5 smoke OOM). Share one
    # non-expert state and keep two expert subtrees — mathematically identical to qam.py's two
    # registered modules because the backbone is frozen in both.
    graphdef_m, p_exp_s, p_rest = nnx.split(build(), expert_filter, ...)
    p_exp_s = jax.device_put(p_exp_s)
    p_rest = jax.device_put(p_rest)
    p_exp_f = jax.tree.map(lambda x: x.copy(), jax.device_put(p_exp_s))  # fast expert starts at slow

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))  # qam.py:387-389,:414
    opt = tx.init(p_exp_f)

    T = a.flow_steps
    h = 1.0 / T

    def prefix_kv(model, obs):
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
        return kv, prefix_mask

    def suffix_v(model, obs, kv, pm, x, tau):
        """pi0.5 velocity via the cached prefix (pi0.py sample_actions step body)."""
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

    def _v_qam_p(p_exp, p_rest, obs, kv, pm, x, t_qam):
        # QAM's data-pointing velocity from openpi's noise-pointing one: v_qam = -v_openpi(1 - t)
        tau = jnp.broadcast_to(1.0 - t_qam, x.shape[0])
        return -suffix_v(nnx.merge(graphdef_m, p_exp, p_rest), obs, kv, pm, x, tau)

    # remat: T fast + T slow forward passes plus the VJP recursion held every suffix activation
    # live at once (round-3 smoke OOM). Recompute per step instead; math unchanged.
    v_qam = jax.checkpoint(_v_qam_p)

    def rollout(exp_f_const, exp_s, p_rest, obs, kv_s, pm, rng):
        """qam.py:49-77 verbatim structure; runs under stop-grad (constants for the loss)."""
        b = pm.shape[0]
        x = jax.random.normal(rng, (b, H, AD))
        xs, ts = [], []
        for i in range(T):
            t = i * h
            sig = jnp.sqrt(2 * (1 - t + h) / (t + h))  # qam.py:64
            xs.append(x)
            ts.append(t)
            if i < T - 1:
                v = v_qam(exp_f_const, p_rest, obs, kv_s, pm, x, t)  # rollout uses the CURRENT fast field (:68-70)
                rng, nk = jax.random.split(rng)
                noise = jax.random.normal(nk, x.shape)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sig * noise  # qam.py:71
            else:
                v = v_qam(exp_s, p_rest, obs, kv_s, pm, x, t)  # deterministic last step, slow field (:72-73)
                x = x + h * v
        # ts is a STATIC python time grid (i*h): keep it host-side — jnp.asarray stages it into a
        # tracer under jit and float(ts[i]) then fails (round-2 smoke)
        return jax.lax.stop_gradient(jnp.stack(xs)), tuple(ts), jax.lax.stop_gradient(x)

    grad_q = critic_q.grad_q_chunk(critic)  # ensemble MEAN grad + in-call clip (qam.py:80-83)

    def adjoints(exp_s, p_rest, obs, kv_s, pm, xs, ts, x_final, feats, proprio):
        """Terminal adj = -grad Q * inv_temp (qam.py:85); backward VJP recursion (:92-101)."""
        g = grad_q(x_final[..., :robot_ad], feats, proprio)
        g = jnp.concatenate([g, jnp.zeros((*g.shape[:-1], AD - robot_ad))], axis=-1)
        adj = -g * a.inv_temp
        adjs = [adj]
        for i in range(T - 2, -1, -1):
            t = float(ts[i])

            def fn(x, t=t):
                return 2 * v_qam(exp_s, p_rest, obs, kv_s, pm, x, t) - x / (t + h)  # qam.py:95-97

            _, vjp = jax.vjp(fn, xs[i])
            adj = adj + h * vjp(adj)[0]  # qam.py:98-100
            adjs.append(adj)
        adjs.reverse()  # align with xs[0..T-1]; matches qam.py:102 ordering
        return jax.lax.stop_gradient(jnp.stack(adjs))

    def loss_fn(exp_f_train, exp_s, p_rest, obs, kv_s, pm, xs, ts, adjs):
        total = 0.0
        for i in range(T):
            t = float(ts[i])
            sig = jnp.sqrt(2 * (1 - t + h) / (t + h))
            vf = v_qam(exp_f_train, p_rest, obs, kv_s, pm, xs[i], t)
            vs = jax.lax.stop_gradient(v_qam(exp_s, p_rest, obs, kv_s, pm, xs[i], t))
            # qam.py:140-142 (residual=False): ((vf - vs) * 2/sigma + sigma * adj)^2, per-dim sum
            per = jnp.sum(jnp.square((vf - vs) * (2.0 / sig) + sig * adjs[i]), axis=(-2, -1))
            total = total + per  # summed over the step axis (qam.py:145)
        return jnp.mean(total), {"adj_absmax": jnp.abs(adjs).max()}

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(p_exp_f, opt, p_exp_s, p_rest, rng, obs, feats, proprio):
        obs = _model.preprocess_observation(None, obs, train=False)
        slow_m = nnx.merge(graphdef_m, p_exp_s, p_rest)
        kv_s, pm = prefix_kv(slow_m, obs)  # one frozen prefix, shared by both fields
        rng, rk = jax.random.split(rng)
        xs, ts, x_final = rollout(p_exp_f, p_exp_s, p_rest, obs, kv_s, pm, rk)
        adjs = adjoints(p_exp_s, p_rest, obs, kv_s, pm, xs, ts, x_final, feats, proprio)
        # grads wrt the fast EXPERT subtree only (a full-state grad pytree was the round-4 OOM)
        (loss, info), grads = jax.value_and_grad(
            lambda pe: loss_fn(pe, p_exp_s, p_rest, obs, kv_s, pm, xs, ts, adjs), has_aux=True
        )(p_exp_f)
        upd, opt = tx.update(grads, opt, p_exp_f)
        return optax.apply_updates(p_exp_f, upd), opt, loss, info

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(
            project="yam-rlt",
            entity=a.wandb_entity,
            name=a.wandb_name,
            group="extraction",
            config={k: str(v) for k, v in vars(a).items()} | {"method": "qam"},
        )
        print(f"wandb: {run.url}", flush=True)

    def save(step_i):
        import orbax.checkpoint as ocp

        path = (a.out / f"{step_i}").absolute()
        with ocp.StandardCheckpointer() as c:
            c.save(path, {"expert": p_exp_f.to_pure_dict()}, force=True)
        print(f"saved {path}", flush=True)

    import numpy as np

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        obs, _actions, ann = next(it)
        f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
        rng, k = jax.random.split(rng)
        p_exp_f, opt, loss, info = step(p_exp_f, opt, p_exp_s, p_rest, k, obs, jnp.asarray(f), jnp.asarray(pr))
        if s % 50 == 0:
            print(
                f"step {s:6d}  adj_loss {float(loss):.4f}  adj_absmax {float(info['adj_absmax']):.3f}  "
                f"({(s + 1) / (time.time() - t0):.3f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log({"adj_loss": float(loss), "adj_absmax": float(info["adj_absmax"])}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print("QAM extraction done.", flush=True)


if __name__ == "__main__":
    main()
