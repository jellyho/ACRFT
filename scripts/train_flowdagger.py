"""FlowDAgger (offline) for pi0.5: latent-space corrections distilled into a steering head.

Provenance — microsoft/FlowDAgger (flowdagger_pi05/*), stage for stage:
  1. corrections are represented as the sampler's INITIAL NOISE w (t=1 seed)
     (flow_matching_inverter.py:3-6) obtained by inverting the discrete Euler map with the
     default `perstep_fp` fixed-point method (flow_matching_inverter.py:97-180: per step,
     x_prev <- x_next - dt * v(x_est, t_prev), fp_per_step=5 iterations);
  2. latents are compressed to a DCT-II basis of K=noise_basis_k coefficients over the horizon
     axis (train_utils.py:60-128), K=10 default (train_flowdagger.py:430);
  3. the steering head is a small policy on FROZEN features, trained by plain MSE on the
     inverted coefficients (steering_policy.py:65-77), tanh-bounded to action_magnitude=3.0
     (steering_policy.py:150-155), Adam bc_lr=1e-4, bc_batch=256 (train_flowdagger.py:386,389);
  4. chunks whose inversion reconstruction MSE (robot dims) exceeds 0.001 are dropped
     (train_utils.py:792-800).
Documented substitutions for the OFFLINE, critic-supervised setting:
  - The corrector: their scripted/human expert acts through the BaseExpert seam
    (shared/experts/base_expert.py:7-34). Ours is the frozen patch critic: starting from the
    base policy's own sample, k steps of normalized grad-ascent a <- a + eta * g/||g||
    (g = grad_a mean Q, in-call clipped) produce the "corrected" chunk -- the digest's
    substitution option (i).
  - No environment: states are dataset frames (subsampled), DAgger's aggregation collapses to a
    single offline collect pass (their buffer only ever grows, buffer.py:286-289, so one pass
    equals the fixed point of aggregation over a fixed state distribution).
  - The frozen encoder: their vlm_pi0 feature extraction (train_utils.py:230-266) is replaced by
    the pooled-DINO + proprio rep (same frozen-encoder pattern; identical rep to the critic's).
Two phases: --phase collect writes (coeffs, rep) pairs; --phase train fits the head; all = both.
"""

# ruff: noqa: PLC0415

import argparse
import functools
import pathlib
import time

R = pathlib.Path(__file__).resolve().parents[1]


def dct_basis(horizon: int, k: int):
    """DCT-II basis over the horizon axis (train_utils.py:60-85), orthonormal rows [k, horizon]."""
    import numpy as np

    n = np.arange(horizon)
    b = np.stack([np.cos(np.pi * (n + 0.5) * f / horizon) for f in range(k)])
    b[0] *= 1.0 / np.sqrt(horizon)
    b[1:] *= np.sqrt(2.0 / horizon)
    return b.astype(np.float32)  # coeffs = B @ w[:, h, d] over h; expand = B.T @ coeffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["collect", "train", "all"], default="all")
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
    ap.add_argument("--stride", type=int, default=200, help="dataset frame subsampling for collect")
    ap.add_argument("--ode-steps", type=int, default=10, help="matches the pi0.5 sampler default")
    ap.add_argument("--fp-per-step", type=int, default=5, help="train_flowdagger.py:424")
    ap.add_argument("--corr-steps", type=int, default=5, help="critic-corrector ascent steps")
    ap.add_argument("--corr-eta", type=float, default=0.05, help="ascent step in normalized units")
    ap.add_argument("--basis-k", type=int, default=10, help="noise_basis_k, train_flowdagger.py:430")
    ap.add_argument("--mse-threshold", type=float, default=0.001, help="train_utils.py:792-800 gate")
    ap.add_argument("--action-magnitude", type=float, default=3.0, help="steering tanh bound, :492")
    ap.add_argument("--bc-lr", type=float, default=1e-4, help="train_flowdagger.py:386")
    ap.add_argument("--bc-batch", type=int, default=256, help="train_flowdagger.py:389")
    ap.add_argument("--bc-steps", type=int, default=8000)
    ap.add_argument("--collect-batch", type=int, default=16)
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/flowdagger_run1")
    )
    ap.add_argument("--wandb", action="store_true")
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
    grad_q = critic_q.grad_q_chunk(critic)

    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    H, AD = cfg.model.action_horizon, cfg.model.action_dim
    B = dct_basis(H, a.basis_k)

    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    graphdef = nnx.graphdef(model)
    base_params = nnx.state(model)

    def prefix(obs):
        m = nnx.merge(graphdef, base_params)
        prefix_tokens, prefix_mask, prefix_ar = m.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = m.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)
        return kv, prefix_mask

    def velocity(obs, kv, pm, x, tau):
        import einops

        m = nnx.merge(graphdef, base_params)
        suffix_tokens, suffix_mask, suffix_ar, adarms = m.embed_suffix(obs, x, tau)
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
        pref_attn = einops.repeat(pm, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn = jnp.concatenate([pref_attn, suffix_attn], axis=-1)
        pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, out), _ = m.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn, positions=pos, kv_cache=kv, adarms_cond=[None, adarms]
        )
        return m.action_out_proj(out[:, -H:])

    N = a.ode_steps
    dt = 1.0 / N

    def sample_from(obs, kv, pm, w):
        """pi0.5 Euler sampler from a given seed (pi0.py:244-260 convention: t 1 -> 0)."""
        x = w
        for i in range(N):
            t = 1.0 - i * dt
            x = x - dt * velocity(obs, kv, pm, x, jnp.full((x.shape[0],), t))
        return x

    def invert(obs, kv, pm, target):
        """perstep_fp inversion of the discrete Euler map (flow_matching_inverter.py:97-180):
        walk the steps backwards; at each, fixed-point iterate x_prev = x_next + dt * v(x_est)."""
        x = target
        for i in range(N - 1, -1, -1):
            t = 1.0 - i * dt
            tv = jnp.full((x.shape[0],), t)
            est = x
            for _ in range(a.fp_per_step):
                est = x + dt * velocity(obs, kv, pm, est, tv)
            x = est
        return x

    def correct(chunk, feats, proprio):
        """Critic-corrector (BaseExpert substitution): k normalized-ascent steps on mean Q."""
        c = chunk
        for _ in range(a.corr_steps):
            g = grad_q(c[..., :robot_ad], feats, proprio)
            gn = jnp.linalg.norm(g.reshape(g.shape[0], -1), axis=-1)[:, None, None] + 1e-8
            c = c.at[..., :robot_ad].add(a.corr_eta * g / gn)
        return c

    @jax.jit
    def collect_step(obs, w0, feats, proprio):
        obs = _model.preprocess_observation(None, obs, train=False)
        kv, pm = prefix(obs)
        a_pi = sample_from(obs, kv, pm, w0)
        a_star = correct(a_pi, feats, proprio)
        w_star = invert(obs, kv, pm, a_star)
        recon = sample_from(obs, kv, pm, w_star)
        rec_mse = jnp.mean(jnp.square((recon - a_star)[..., :robot_ad]), axis=(-2, -1))
        q0 = critic.q_mean(feats, a_pi[..., :robot_ad], proprio)
        q1 = critic.q_mean(feats, a_star[..., :robot_ad], proprio)
        return w_star, rec_mse, q0, q1

    a.out.mkdir(parents=True, exist_ok=True)
    coeff_f = a.out / "coeffs.npy"
    rep_f = a.out / "reps.npy"

    if a.phase in ("collect", "all"):
        n = cache.meta["N"]
        idx_all = np.arange(0, n, a.stride)
        ds = exdata.AnnotatedBC(dataset, {})
        coeffs, reps, kept = [], [], 0
        rng = jax.random.key(0)
        t0 = time.time()
        for s in range(0, len(idx_all), a.collect_batch):
            idx = idx_all[s : s + a.collect_batch]
            items = [ds[int(i)] for i in idx]
            import torch

            batch = torch.utils.data.default_collate(items)
            ann = {k[4:]: np.asarray(v) for k, v in batch.items() if k.startswith("ann/")}
            obs = _model.Observation.from_dict(
                {
                    k: (np.asarray(v, np.float32) if np.asarray(v).dtype == np.float64 else np.asarray(v))
                    if not isinstance(v, dict)
                    else {kk: np.asarray(vv) for kk, vv in v.items()}
                    for k, v in batch.items()
                    if k != "actions" and not k.startswith("ann/")
                }
            )
            f, _st, pr = cache.rows(np.asarray(ann["idx"], np.int64), critic)
            rng, k = jax.random.split(rng)
            w0 = jax.random.normal(k, (len(idx), H, AD))
            w_star, rec_mse, q0, q1 = collect_step(obs, w0, jnp.asarray(f), jnp.asarray(pr))
            keep = np.asarray(rec_mse) < a.mse_threshold  # train_utils.py:792-800
            c = np.einsum("kh,bhd->bkd", B, np.asarray(w_star))
            coeffs.append(c[keep])
            reps.append(np.concatenate([f.mean(axis=1), pr], axis=-1)[keep])
            kept += int(keep.sum())
            if s % (a.collect_batch * 10) == 0:
                print(
                    f"collect {s}/{len(idx_all)}  kept {kept}  dQ {float((q1 - q0).mean()):+.1f}  "
                    f"rec_mse {float(rec_mse.mean()):.5f}  ({(s + 1) / (time.time() - t0):.2f} states/s)",
                    flush=True,
                )
        np.save(coeff_f, np.concatenate(coeffs))
        np.save(rep_f, np.concatenate(reps))
        print(f"collected {kept} corrected latents -> {coeff_f}", flush=True)

    if a.phase in ("train", "all"):
        import optax

        run = None
        if a.wandb:
            import wandb

            run = wandb.init(
                project="yam-rlt",
                entity="RSS-PFT_RLLAB",
                name="extract_flowdagger_run1",
                group="extraction",
                config={k: str(v) for k, v in vars(a).items()} | {"method": "flowdagger"},
            )
            print(f"wandb: {run.url}", flush=True)

        coeffs = np.load(coeff_f)
        reps = np.load(rep_f)
        print(f"steering BC on {len(coeffs)} pairs")
        kdim = coeffs.shape[1] * coeffs.shape[2]

        def init_mlp(key, dims):
            ks = jax.random.split(key, len(dims) - 1)
            return [
                (jax.random.normal(ks[i], (dims[i], dims[i + 1])) * jnp.sqrt(2.0 / dims[i]), jnp.zeros(dims[i + 1]))
                for i in range(len(dims) - 1)
            ]

        head = init_mlp(jax.random.key(2), [reps.shape[1], 512, 512, kdim])

        def fwd(p, x):
            for w, b in p[:-1]:
                x = jax.nn.relu(x @ w + b)
            w, b = p[-1]
            # tanh bound to action_magnitude (steering_policy.py:150-155)
            return jnp.tanh(x @ w + b) * a.action_magnitude

        tx = optax.adam(a.bc_lr)
        opt = tx.init(head)

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def bstep(head, opt, x, y):
            def lf(p):
                return jnp.mean(jnp.square(fwd(p, x) - y))  # steering_policy.py:65-77

            loss, g = jax.value_and_grad(lf)(head)
            upd, opt = tx.update(g, opt, head)
            return optax.apply_updates(head, upd), opt, loss

        perm = np.random.default_rng(0).permutation(len(coeffs))
        for s in range(a.bc_steps):
            bi = perm[(s * a.bc_batch) % len(coeffs) : (s * a.bc_batch) % len(coeffs) + a.bc_batch]
            if len(bi) < a.bc_batch:
                perm = np.random.default_rng(s).permutation(len(coeffs))
                bi = perm[: a.bc_batch]
            head, opt, loss = bstep(head, opt, jnp.asarray(reps[bi]), jnp.asarray(coeffs[bi].reshape(len(bi), -1)))
            if s % 500 == 0:
                print(f"bc step {s}  mse {float(loss):.5f}", flush=True)
                if run is not None:
                    run.log({"steering_mse": float(loss)}, step=s)
        import flax.serialization

        (a.out / "steering_head.msgpack").write_bytes(
            flax.serialization.msgpack_serialize(jax.tree.map(np.asarray, head))
        )
        np.save(a.out / "dct_basis.npy", B)
        print("FlowDAgger steering head saved.", flush=True)


if __name__ == "__main__":
    main()
