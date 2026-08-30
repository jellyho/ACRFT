"""QPILOTS-U test-time Q-steering for pi0.5 — zero-training extraction arm.

Provenance — "QPILOTS: Efficient Test-Time Q-Steering for Flow Policies" (arXiv 2606.14801),
NO official code (neither abs page nor paper lists a repo), so this follows the paper incl.
appendices:
  - Projection (Tweedie/MMSE point estimate, Eq. 14): a_hat = x_t + (1-t_paper)*v_paper. openpi
    runs time 1->0 with x_t = t*noise + (1-t)*a and v = noise - a (pi0.py:214,258-283), so the
    identical projection here is a_hat = x_tau - tau*v.
  - Steering gradient (QPILOTS-U, Eq. 15): g = grad_x Qbar(s, clip[a_hat, -1, 1]); the clip is
    STRAIGHT-THROUGH in the backward pass, and the gradient flows through the one velocity
    evaluation inside the projection (chain rule back to x_t), not just the critic. tau_tilt = 1
    fixed (App. C.3) — absorbed into the critic scale by the norm-matching below.
  - Velocity modification with drift-magnitude rescaling (Eq. 17): v <- v + alpha * (||v|| /
    (||g|| + eps)) * g, per-sample norms; alpha is a per-domain CONSTANT, no time schedule
    (App. B/C.3: the rescaling deliberately replaces the sigma_t^2/2 schedule — "more stable").
    In openpi's reversed time (v_openpi = -v_paper) this is v <- v - alpha*(||v||/||g||)*g.
  - Steer every Euler step EXCEPT the first (i>0 only): at t_paper=0 the projection carries no
    state-dependent signal (Sec. 4 / App. B).
  - Pessimistic ensemble (Eq. 12): Qbar = mean_j Q_j - rho * std_j Q_j, rho = 0.5 default.
  - K = 10 Euler steps everywhere incl. their pi0.5-LIBERO run (App. C.4, flow_steps=10);
    alpha swept {0.1, 0.2, 0.3} on pi0.5-LIBERO (Table 4), sensitivity peaks 0.1-0.5.
  - App. C.3 computes the steering gradient against TARGET base flow + TARGET critic; our base
    policy and critic are both frozen, so live == target by construction.
  - Their Qbar is batch-normalized/clipped before the tilt (App. B) — a QPILOTS-M/theory
    concern; for U the Eq. 17 norm-matching removes scale sensitivity of the direction, so we
    query the critic raw (only relative curvature of Q enters).

This arm trains NOTHING (paper Sec. 4: "no auxiliary network, zero extra training") — the
script is a paired evaluation: same states, same noise seeds, base sampling vs steered sampling
for each alpha; reports pessimistic-Q uplift with run-level stats. The steered sampler
(sample_steered) is importable by the serving harness.
"""

# ruff: noqa: PLC0415

import argparse
import functools
import json
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
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="pi0.5 sweep, Table 4")
    ap.add_argument("--rho", type=float, default=0.5, help="ensemble pessimism, Eq. 12 / App. C.3")
    ap.add_argument("--ode-steps", type=int, default=10, help="K=10 everywhere, App. C.4")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/qpilots_eval.json")
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi.extraction import critic_q
    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    critic = critic_q.load(a.critic)
    cache = critic_q.CacheView(a.cache)
    robot_ad = critic.config["action_dim"]

    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    ds = exdata.AnnotatedBC(dataset, {})
    it = exdata.make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)
    H, AD = cfg.model.action_horizon, cfg.model.action_dim

    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    graphdef = nnx.graphdef(model)
    params = nnx.state(model)

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

    def q_pess(feats, chunk, proprio):
        """Pessimistic ensemble Qbar = mean - rho*std (Eq. 12), full-chunk prefix, per sample."""
        logits = critic.net.apply({"params": critic.params}, feats, chunk, proprio)
        q = critic.hl.from_logits(logits)[..., -1]  # [K, B]
        return q.mean(axis=0) - a.rho * q.std(axis=0)

    def st_clip(x):
        """Straight-through clip to the normalized box (Eq. 15: 'clip is straight-through')."""
        return x + jax.lax.stop_gradient(jnp.clip(x, -1.0, 1.0) - x)

    @functools.partial(jax.jit, static_argnums=(5,))
    def sample_steered(params, rng, obs, feats, proprio, alpha: float):
        """K-step Euler with Q-steering at every step but the first. alpha=0.0 == base sampler."""
        model = nnx.merge(graphdef, params)
        obs = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
        b = feats.shape[0]
        x = jax.random.normal(rng, (b, H, AD))
        for i in range(N):
            tau = 1.0 - i * dt
            tv = jnp.full((b,), tau)
            if i == 0 or alpha == 0.0:
                v = velocity(model, obs, kv, prefix_mask, x, tv)
            else:
                # Eq. 15: sum of per-sample Qbar; grad flows through velocity AND critic.
                def q_of(x_, tv_=tv):
                    v_ = velocity(model, obs, kv, prefix_mask, x_, tv_)
                    a_hat = st_clip(x_ - tv_[:, None, None] * v_)  # Eq. 14 in openpi time
                    return q_pess(feats, a_hat[..., :robot_ad], proprio).sum(), v_

                g, v = jax.grad(q_of, has_aux=True)(x)
                vn = jnp.linalg.norm(v.reshape(b, -1), axis=-1, keepdims=True)
                gn = jnp.linalg.norm(g.reshape(b, -1), axis=-1, keepdims=True)
                # Eq. 17 (openpi sign: v_openpi = -v_paper, so ascent on Q is v - alpha*ghat)
                v = v - alpha * (vn / (gn + 1e-8)).reshape(b, 1, 1) * g
            x = x - dt * v
        chunk = jnp.clip(x, -1.0, 1.0)  # final clip, Alg. 1
        return chunk, q_pess(feats, chunk[..., :robot_ad], proprio)

    arms = [0.0, *a.alphas]
    per_arm = {al: [] for al in arms}
    rng = jax.random.key(7)
    t0 = time.time()
    for bi in range(a.num_batches):
        obs, _actions, ann = next(it)
        f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
        rng, k = jax.random.split(rng)
        for al in arms:  # same states, same noise key -> paired across arms
            _, q = sample_steered(params, k, obs, jnp.asarray(f), jnp.asarray(pr), al)
            per_arm[al].append(np.asarray(q))
        done = sum(len(v) for v in per_arm.values())
        print(f"batch {bi + 1}/{a.num_batches}  ({done} samples total, {time.time() - t0:.0f}s)", flush=True)

    base = np.concatenate(per_arm[0.0])
    result = {"rho": a.rho, "ode_steps": N, "n": int(base.size), "base_q": float(base.mean()), "arms": {}}
    print(f"\nbase sampler:  Qbar {base.mean():.1f}")
    for al in a.alphas:
        q = np.concatenate(per_arm[al])
        d = q - base  # paired per-state uplift
        ci = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        result["arms"][str(al)] = {"q": float(q.mean()), "uplift": float(d.mean()), "ci95": float(ci)}
        print(f"alpha {al:4.2f}:  Qbar {q.mean():.1f}   paired uplift {d.mean():+.2f} ± {ci:.2f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
