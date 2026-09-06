"""Unified offline evaluation for all policy-extraction arms — one protocol, one JSON per arm.

Protocol: a DETERMINISTIC evenly-strided set of dataset frames (index == cache row, data.py
docstring), same noise seed per state across arms. Each arm produces a normalized action chunk
per state; we report
  - q_mean / q_min / q_pess (ensemble mean, min, mean - 0.5*std) from the frozen patch critic
    (the g5_tau9_min serving critic unless overridden),
  - demo_mse: MSE to the dataset's action chunk (robot dims, normalized space),
  - jerk: mean squared 2nd difference along the horizon (the chunk-smoothness metric of
    scripts/eval_onestep_bc.py, which diagnosed the JVP off-diagonal poisoning),
with run-level mean ± 95% CI over states. This is the offline leg of the comparison; the
on-robot rollout leg is served separately by the user.

Arms (--arm):
  bc          frozen BC pi0.5 sampler; with --expert-ckpt it also serves every arm that only
              fine-tunes the action expert on the same arch: awr / flowdpg / qam / dql
              (each trainer saves {"expert": ...} orbax overlays).
  cfgrl       Pi0CFGRL arch + expert/opt_embed overlay, CFG sampling with --cfg-w
              (kvfrans/cfgrl iql_diffusion.py:213 guidance form; w=1 == conditioned only).
  lps|lpsd    frozen alpha-Flow one-step base + latent-actor msgpack (train_lps.py);
              lpsd draws e ~ N(0, I) per state (lps.py:294-327 sampling path).
  flowdagger  BC sampler seeded by the steering head's DCT-expanded noise prediction
              (microsoft/FlowDAgger serving: the predicted latent replaces the seed).
  qpilots     BC sampler with test-time Q-steering (qpilots_steer.py, --alpha).
  idql|bon    N base draws, argmax of min-ensemble Q (IDQL eval rule == BoN serving arm).
"""

# ruff: noqa: PLC0415

import argparse
import json
import pathlib
import time

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", required=True, choices=["bc", "cfgrl", "lps", "lpsd", "flowdagger", "qpilots", "idql", "bon"]
    )
    ap.add_argument("--label", default=None, help="name in the output JSON (default: arm[+ckpt step])")
    ap.add_argument(
        "--init-ckpt",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/200000"
        ),
    )
    ap.add_argument("--expert-ckpt", type=pathlib.Path, default=None, help="orbax {'expert': ...} overlay dir")
    ap.add_argument("--latent-actor", type=pathlib.Path, default=None, help="lps/lpsd msgpack")
    ap.add_argument(
        "--af-ckpt",
        type=pathlib.Path,
        default=pathlib.Path("/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000"),
    )
    ap.add_argument("--steering-head", type=pathlib.Path, default=None, help="flowdagger run dir")
    ap.add_argument("--cfg-w", type=float, default=1.5, help="CFG weight (cfgrl sweep {1,1.5,3,...})")
    ap.add_argument("--alpha", type=float, default=0.2, help="qpilots steering scale")
    ap.add_argument("--rho", type=float, default=0.5, help="q_pess pessimism")
    ap.add_argument("--n-samples", type=int, default=64, help="idql/bon draws")
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--stride", type=int, default=4001, help="every stride-th frame -> ~234 eval states")
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7, help="shared across arms -> paired noise")
    ap.add_argument("--out-dir", type=pathlib.Path, default=R / ".scratch/extraction/eval")
    a = ap.parse_args()

    import flax.nnx as nnx
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    from openpi.extraction import critic_q
    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    critic = critic_q.load(a.critic)
    cache = critic_q.CacheView(a.cache)
    robot_ad = critic.config["action_dim"]

    # ---- model: arch depends on arm ------------------------------------------------------------
    use_af = a.arm in ("lps", "lpsd")
    base_ckpt = a.af_ckpt if use_af else a.init_ckpt
    dataset, bc_cfg = exdata.make_bc_dataset(str(base_ckpt / "assets"))
    H, AD = bc_cfg.model.action_horizon, bc_cfg.model.action_dim

    if a.arm == "cfgrl":
        from openpi.models.pi0_cfgrl import Pi0CFGRLConfig

        mcfg = Pi0CFGRLConfig(pi05=True, action_horizon=H, action_dim=AD)
    elif use_af:
        import openpi.training.config as _config

        mcfg = _config.get_config("pi05_yam_lego_taxi_alphaflow").model
    else:
        mcfg = bc_cfg.model

    model = mcfg.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = CheckpointWeightLoaderKeepMissing(str(base_ckpt / "params")).load(state.to_pure_dict())
    if a.expert_ckpt is not None:
        import orbax.checkpoint as ocp

        with ocp.StandardCheckpointer() as c:
            expert = c.restore(a.expert_ckpt.absolute())["expert"]

        def deep_update(d, u, path=""):
            for k, v in u.items():
                if isinstance(v, dict):
                    deep_update(d[k], v, f"{path}/{k}")
                else:
                    d[k] = v

        deep_update(loaded, expert)
        print(f"overlaid expert params from {a.expert_ckpt}")
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

    def prefix(model, obs):
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
        return kv, prefix_mask

    def euler(model, obs, kv, pm, w):
        x = w
        for i in range(N):
            x = x - dt * velocity(model, obs, kv, pm, x, jnp.full((x.shape[0],), 1.0 - i * dt))
        return x

    def q_stats(feats, chunk, proprio):
        logits = critic.net.apply({"params": critic.params}, feats, chunk, proprio)
        q = critic.hl.from_logits(logits)[..., -1]  # [K, B]
        return q.mean(axis=0), q.min(axis=0), q.mean(axis=0) - a.rho * q.std(axis=0)

    def st_clip(x):
        return x + jax.lax.stop_gradient(jnp.clip(x, -1.0, 1.0) - x)

    # ---- per-arm samplers (all return the final normalized chunk [B, H, AD]) -------------------
    if a.arm in ("bc", "flowdagger"):
        head = basis = None
        if a.arm == "flowdagger":
            d = a.steering_head
            raw = flax.serialization.msgpack_restore((d / "steering_head.msgpack").read_bytes())
            raw = list(raw.values()) if isinstance(raw, dict) else raw
            head = [(jnp.asarray(p[0]), jnp.asarray(p[1])) for p in raw]
            basis = jnp.asarray(np.load(d / "dct_basis.npy"))

        @jax.jit
        def sample(params, rng, obs, feats, proprio):
            model = nnx.merge(graphdef, params)
            obs = _model.preprocess_observation(None, obs, train=False)
            kv, pm = prefix(model, obs)
            b = feats.shape[0]
            if a.arm == "bc":
                w = jax.random.normal(rng, (b, H, AD))
            else:  # steering head predicts the DCT-compressed seed from the frozen rep
                rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
                x = rep
                for wgt, bi in head[:-1]:
                    x = jax.nn.relu(x @ wgt + bi)
                wgt, bi = head[-1]
                coeffs = (jnp.tanh(x @ wgt + bi) * 3.0).reshape(b, basis.shape[0], AD)
                w = jnp.einsum("kh,bkd->bhd", basis, coeffs)
            return jnp.clip(euler(model, obs, kv, pm, w), -1.0, 1.0)

    elif a.arm == "cfgrl":

        @jax.jit
        def sample(params, rng, obs, feats, proprio):
            model = nnx.merge(graphdef, params)
            # preprocesses internally (pi0_cfgrl.py:94); final clip = critic's box, as everywhere here
            chunk = model.sample_actions_cfg(rng, obs, cfg_w=a.cfg_w, num_steps=N)
            return jnp.clip(chunk, -1.0, 1.0)

    elif a.arm in ("lps", "lpsd"):
        raw = flax.serialization.msgpack_restore(a.latent_actor.read_bytes())
        raw = list(raw.values()) if isinstance(raw, dict) else raw
        actor = [(jnp.asarray(p[0]), jnp.asarray(p[1])) for p in raw]
        LAT = H * AD

        @jax.jit
        def sample(params, rng, obs, feats, proprio):
            model = nnx.merge(graphdef, params)
            obs = _model.preprocess_observation(None, obs, train=False)
            b = feats.shape[0]
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            if a.arm == "lpsd":
                rep = jnp.concatenate([rep, jax.random.normal(rng, (b, LAT))], axis=-1)
            x = rep
            for wgt, bi in actor[:-1]:
                x = jax.nn.relu(x @ wgt + bi)
            wgt, bi = actor[-1]
            z = (x @ wgt + bi).reshape(b, H, AD)
            pm, kv = model._prefix_forward(obs)
            u = model._u(obs, pm, kv, z, jnp.ones((b,)), jnp.zeros((b,)))
            return jnp.clip(z - u, -1.0, 1.0)

    elif a.arm == "qpilots":

        @jax.jit
        def sample(params, rng, obs, feats, proprio):
            model = nnx.merge(graphdef, params)
            obs = _model.preprocess_observation(None, obs, train=False)
            kv, pm = prefix(model, obs)
            b = feats.shape[0]
            x = jax.random.normal(rng, (b, H, AD))
            for i in range(N):
                tau = 1.0 - i * dt
                tv = jnp.full((b,), tau)
                if i == 0:
                    v = velocity(model, obs, kv, pm, x, tv)
                else:

                    def q_of(x_, tv_=tv):
                        v_ = velocity(model, obs, kv, pm, x_, tv_)
                        a_hat = st_clip(x_ - tv_[:, None, None] * v_)
                        _qm, _, qp = q_stats(feats, a_hat[..., :robot_ad], proprio)
                        return qp.sum(), v_

                    g, v = jax.grad(q_of, has_aux=True)(x)
                    vn = jnp.linalg.norm(v.reshape(b, -1), axis=-1).reshape(b, 1, 1)
                    gn = jnp.linalg.norm(g.reshape(b, -1), axis=-1).reshape(b, 1, 1)
                    v = v - a.alpha * (vn / (gn + 1e-8)) * g
                x = x - dt * v
            return jnp.clip(x, -1.0, 1.0)

    elif a.arm in ("idql", "bon"):

        @jax.jit
        def draw(params, rng, obs, kv, pm):
            model = nnx.merge(graphdef, params)
            b = pm.shape[0]
            return jnp.clip(euler(model, obs, kv, pm, jax.random.normal(rng, (b, H, AD))), -1.0, 1.0)

        @jax.jit
        def prefix_only(params, obs):
            model = nnx.merge(graphdef, params)
            obs = _model.preprocess_observation(None, obs, train=False)
            kv, pm = prefix(model, obs)
            return obs, kv, pm

        def sample(params, rng, obs, feats, proprio):
            obs, kv, pm = prefix_only(params, obs)
            best_c, best_q = None, None
            for n in range(a.n_samples):
                c = draw(params, jax.random.fold_in(rng, n), obs, kv, pm)
                _, qmin, _ = q_stats(feats, c[..., :robot_ad], proprio)
                if best_q is None:
                    best_c, best_q = c, qmin
                else:
                    better = (qmin > best_q)[:, None, None]
                    best_c = jnp.where(better, c, best_c)
                    best_q = jnp.maximum(qmin, best_q)
            return best_c

    # ---- deterministic strided eval sweep ------------------------------------------------------
    ds = exdata.AnnotatedBC(dataset, {})
    idx_all = np.arange(0, cache.meta["N"], a.stride)
    q_stats_j = jax.jit(q_stats)
    rows = {"q_mean": [], "q_min": [], "q_pess": [], "demo_mse": [], "jerk": []}
    t0 = time.time()
    for s in range(0, len(idx_all), a.batch):
        idx = idx_all[s : s + a.batch]
        batch = torch.utils.data.default_collate([ds[int(i)] for i in idx])
        obs = _model.Observation.from_dict(
            {
                k: (np.asarray(v, np.float32) if np.asarray(v).dtype == np.float64 else np.asarray(v))
                if not isinstance(v, dict)
                else {kk: np.asarray(vv) for kk, vv in v.items()}
                for k, v in batch.items()
                if k != "actions" and not k.startswith("ann/")
            }
        )
        demo = np.asarray(batch["actions"], np.float32)
        f, _st, pr = cache.rows(idx, critic)
        rng = jax.random.fold_in(jax.random.key(a.seed), s)  # keyed by position -> paired across arms
        chunk = np.asarray(sample(params, rng, obs, jnp.asarray(f), jnp.asarray(pr)))
        qm, qn, qp = (
            np.asarray(x) for x in q_stats_j(jnp.asarray(f), jnp.asarray(chunk[..., :robot_ad]), jnp.asarray(pr))
        )
        rows["q_mean"].append(qm)
        rows["q_min"].append(qn)
        rows["q_pess"].append(qp)
        rows["demo_mse"].append(np.mean((chunk[..., :robot_ad] - demo[..., :robot_ad]) ** 2, axis=(-2, -1)))
        d2 = np.diff(chunk[..., :robot_ad], n=2, axis=-3)
        rows["jerk"].append(np.mean(d2**2, axis=(-2, -1)))
        if s % (a.batch * 5) == 0:
            print(f"{s}/{len(idx_all)} states  ({(s + 1) / (time.time() - t0):.2f}/s)", flush=True)

    label = a.label or a.arm
    result = {
        "arm": a.arm,
        "label": label,
        "n_states": len(idx_all),
        "stride": a.stride,
        "seed": a.seed,
        "params": {
            "expert_ckpt": str(a.expert_ckpt),
            "latent_actor": str(a.latent_actor),
            "steering_head": str(a.steering_head),
            "cfg_w": a.cfg_w,
            "alpha": a.alpha,
            "n_samples": a.n_samples,
        },
        "metrics": {},
        "per_state": {},
    }
    print(f"\n== {label} ==")
    for k, vlist in rows.items():
        v = np.concatenate(vlist)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(v.size)
        result["metrics"][k] = {"mean": float(v.mean()), "ci95": float(ci)}
        result["per_state"][k] = [float(x) for x in v]
        print(f"{k:9s} {v.mean():10.4f} ± {ci:.4f}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{label}.json"
    out.write_text(json.dumps(result))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
