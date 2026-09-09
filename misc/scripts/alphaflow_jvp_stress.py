"""Stress test for the alpha-Flow JVP (MeanFlow) regime -- does the loss explode, and when?

Motivated by a previously observed failure: switching a flow model onto the JVP MeanFlow target made
the loss blow up mid-run, cause never pinned down (bf16 numerics vs. the objective itself). This
script reproduces the three suspect regimes under REAL training dynamics (Adam updates at the
production lr, not single loss evals) and logs the watchdogs that tell the failure modes apart:

    delta2        raw squared error (the real progress signal; the adaptive-weighted loss is ~1)
    dudt_absmax   the JVP term. An exploding du/dt with finite weights = the OBJECTIVE is unstable.
    u_tgt_absmax  should sit at <= clamp_u_target once dudt spikes -- if the explosion persists
                  with the clip active, clipping is not the fix.
    grad_norm     global gradient norm. bf16-only spikes here (with f32 clean) = NUMERICS.

Modes:
    jvp        pinned alpha = 0 from step 0 (pure MeanFlow) -- the harshest case.
    floor      pinned alpha = 5e-3 discrete (the default, JVP-free) -- the control.
    transition meanflow_jvp with the official schedule compressed so the run crosses the
               discrete -> JVP boundary mid-run (the exact place the old explosion happened).

Run both dtypes on a GPU node and diff:
    srun -p debug --gres=gpu:1 ... uv run python scripts/alphaflow_jvp_stress.py --dtype bfloat16
    srun -p debug --gres=gpu:1 ... uv run python scripts/alphaflow_jvp_stress.py --dtype float32

Random-init lora backbone by default so it fits an L40S; the numerics question (bf16 JVP noise
compounding through 18 transformer layers) does not need the pretrained weights, but --full runs
the real thing when a B200 is free.
"""

import argparse

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import pi0_alphaflow

KEY = jax.random.key(0)


def build(mode, dtype, full):
    kw = {
        "pi05": True,
        "dtype": dtype,
        "action_horizon": 16,
        "discrete_state_input": False,
        "paligemma_variant": "gemma_2b" if full else "gemma_2b_lora",
        "action_expert_variant": "gemma_300m" if full else "gemma_300m_lora",
    }
    if mode == "jvp":
        return pi0_alphaflow.Pi0AlphaFlowConfig(**kw, alpha_init=0.0, alpha_final=0.0, meanflow_jvp=True)
    if mode == "floor":
        return pi0_alphaflow.Pi0AlphaFlowConfig(**kw)
    if mode == "transition":
        return pi0_alphaflow.Pi0AlphaFlowConfig(**kw, meanflow_jvp=True)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--modes", nargs="+", default=["floor", "jvp", "transition"])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-5, help="production peak lr")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--full", action="store_true", help="full variants instead of lora (needs a B200)")
    ap.add_argument("--explode-ratio", type=float, default=100.0, help="delta2 > ratio * early median => explosion")
    a = ap.parse_args()

    for mode in a.modes:
        run_mode(mode, a)


def run_mode(mode, a):
    cfg = build(mode, a.dtype, a.full)
    model = cfg.create(KEY)
    obs = cfg.fake_obs(batch_size=a.batch)
    # structured, not white-noise, actions: a white-noise target makes v_t unlearnable and hides
    # optimisation pathologies under an irreducible floor.
    tt = jnp.linspace(0, 1, cfg.action_horizon)[None, :, None]
    act = 0.5 * jnp.sin(2 * jnp.pi * (tt + jnp.linspace(0, 1, cfg.action_dim)[None, None, :]))
    act = jnp.broadcast_to(act, (a.batch, cfg.action_horizon, cfg.action_dim))

    # Optimize only what training would optimize (the lora adapters + the new r MLP, under the
    # config's freeze filter). Adam moments for all 3B params would be ~24GB of pure overhead --
    # that is what train.py's trainable_filter avoids, and the stress test must match it anyway to
    # reproduce training dynamics.
    freeze = cfg.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze))
    graphdef, params, frozen, rest = nnx.split(model, trainable_filter, nnx.Param, ...)
    tx = optax.adam(a.lr)
    opt_state = tx.init(params)

    @jax.jit
    def step_fn(params, opt_state, rng, progress):
        def loss_fn(p):
            m = nnx.merge(graphdef, p, frozen, rest)
            loss, aux = m.compute_loss(rng, obs, act, train=True, progress=progress)
            return jnp.mean(loss), aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        gnorm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux, gnorm

    # `transition` compresses the official schedule so the boundary (~0.712 progress) falls
    # mid-run: sweep progress 0.55 -> 0.85 over the steps. Pinned modes hold progress at 0.5.
    def progress_of(k):
        if mode == "transition":
            return 0.55 + 0.30 * k / max(a.steps - 1, 1)
        return 0.5

    print(f"\n== mode={mode} dtype={a.dtype} lr={a.lr} steps={a.steps} ==")
    early = []
    exploded = None
    for k in range(a.steps):
        rng = jax.random.fold_in(KEY, k)
        params, opt_state, loss, aux, gnorm = step_fn(params, opt_state, rng, jnp.asarray(progress_of(k), jnp.float32))
        d2 = float(aux["delta2"])
        if k < 20:
            early.append(d2)
        baseline = sorted(early)[len(early) // 2]
        bad = not jnp.isfinite(loss) or (k >= 20 and d2 > a.explode_ratio * max(baseline, 1e-8))
        if k % 10 == 0 or bad:
            print(
                f"  step {k:>4}  prog {progress_of(k):.3f}  alpha {float(aux['alpha']):.4f}"
                f"  jvp {int(float(aux['jvp_active']))}  delta2 {d2:10.4f}"
                f"  dudt_max {float(aux['dudt_absmax']):10.3f}"
                f"  u_tgt_max {float(aux['u_tgt_absmax']):8.3f}"
                f"  grad_norm {float(gnorm):10.3f}" + ("  <-- EXPLODED" if bad else "")
            )
        if bad and exploded is None:
            exploded = k
    verdict = f"EXPLODED at step {exploded}" if exploded is not None else "stable"
    print(f"  [{mode} / {a.dtype}] {verdict}")
    del model, params, opt_state
    jax.clear_caches()


if __name__ == "__main__":
    main()
