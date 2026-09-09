"""IDQL test-time selection for pi0.5 — N samples from the frozen BC policy, critic picks.

Provenance — "IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies"
(arXiv 2304.10573), OFFICIAL code philippe-eecs/IDQL jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py:
  - eval rule (:360-374): repeat the observation N times, sample N actions from the (target)
    score model, q = compute_q(target_critic) with MIN over the ensemble (:40-43), action =
    actions[argmax(q)] (:371-372). N = 64 default (:101).
  - implicit-policy variant (:377-403, critic_objective='expectile'): adv = min-Q - V; weights
    tau where adv > 0 else 1 - tau (the expectile weights, their Sec. 4 w2^tau); sample ONE
    index ~ weights/sum (:392-394) instead of argmax.
  - the policy itself is pure BC (actor_objective='bc' branch :327 uses uniform weights); ALL
    the RL is in the selection — which is why this arm needs zero policy training here.
Port notes: policy = frozen BC pi0.5 (10-step Euler, cached prefix KV reused across the N
draws); critic = patch critic (K=2 min plays their ensemble min), V = its PatchV head; tau
defaults to 0.9 = the expectile our g5_tau9_min critic was trained with (their
critic_hyperparam is likewise the critic's own expectile). The argmax rule is mechanically our
BoN serving arm, so this script doubles as the arm-0 BoN baseline: rules reported are
  uniform (BC baseline) / idql_argmax (== BoN-N) / idql_implicit (expectile-weighted draw).
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
            "/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/200000"
        ),
    )
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--n-samples", type=int, default=64, help="N, ddpm_iql_learner.py:101")
    ap.add_argument("--tau", type=float, default=0.9, help="expectile weight = critic's own expectile")
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/idql_eval.json")
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

    N_ODE = a.ode_steps
    dt = 1.0 / N_ODE

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

    @jax.jit
    def prefix_pass(params, obs):
        model = nnx.merge(graphdef, params)
        obs = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
        return obs, prefix_mask, kv

    @jax.jit
    def sample_one(params, rng, obs, prefix_mask, kv):
        """One 10-step Euler draw reusing the cached prefix KV (one of the N IDQL samples)."""
        model = nnx.merge(graphdef, params)
        b = prefix_mask.shape[0]
        x = jax.random.normal(rng, (b, H, AD))
        for i in range(N_ODE):
            tv = jnp.full((b,), 1.0 - i * dt)
            x = x - dt * velocity(model, obs, kv, prefix_mask, x, tv)
        return jnp.clip(x, -1.0, 1.0)

    @jax.jit
    def q_min(feats, chunk, proprio):
        """compute_q: MIN over the critic ensemble (ddpm_iql_learner.py:40-43), full chunk."""
        logits = critic.net.apply({"params": critic.params}, feats, chunk, proprio)
        return critic.hl.from_logits(logits).min(axis=0)[..., -1]

    v_of = jax.jit(functools.partial(critic.v))

    rules = ["uniform", "idql_argmax", "idql_implicit"]
    per_rule = {r: [] for r in rules}
    rng = jax.random.key(11)
    t0 = time.time()
    for bi in range(a.num_batches):
        obs, _actions, ann = next(it)
        f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
        fj, prj = jnp.asarray(f), jnp.asarray(pr)
        obs_p, pm, kv = prefix_pass(params, obs)
        qs = []
        for _n in range(a.n_samples):
            rng, k = jax.random.split(rng)
            chunk = sample_one(params, k, obs_p, pm, kv)
            qs.append(np.asarray(q_min(fj, chunk[..., :robot_ad], prj)))
        qs = np.stack(qs)  # [N, B]
        v = np.asarray(v_of(fj, prj))  # [B]
        # uniform = BC baseline: the first draw (any draw; fixed for pairing)
        per_rule["uniform"].append(qs[0])
        # eval_actions :371-372 — argmax over the N min-ensemble Qs (== BoN-N serving arm)
        per_rule["idql_argmax"].append(qs.max(axis=0))
        # sample_implicit_policy :392-394 — expectile weights on adv, one categorical draw
        adv = qs - v[None, :]
        w = np.where(adv > 0, a.tau, 1.0 - a.tau)
        rng, k = jax.random.split(rng)
        pick = np.stack(
            [
                np.random.default_rng(int(jax.random.bits(k)) + j).choice(a.n_samples, p=w[:, j] / w[:, j].sum())
                for j in range(qs.shape[1])
            ]
        )
        per_rule["idql_implicit"].append(qs[pick, np.arange(qs.shape[1])])
        print(f"batch {bi + 1}/{a.num_batches}  ({time.time() - t0:.0f}s)", flush=True)

    base = np.concatenate(per_rule["uniform"])
    result = {"n_samples": a.n_samples, "tau": a.tau, "n_states": int(base.size), "rules": {}}
    print(f"\nuniform (BC):   Q {base.mean():.1f}")
    for r in rules[1:]:
        q = np.concatenate(per_rule[r])
        d = q - base
        ci = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        result["rules"][r] = {"q": float(q.mean()), "uplift": float(d.mean()), "ci95": float(ci)}
        print(f"{r:14s}  Q {q.mean():.1f}   paired uplift {d.mean():+.2f} ± {ci:.2f}")
    result["rules"]["uniform"] = {"q": float(base.mean())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
