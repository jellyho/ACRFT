"""End-to-end checks for alpha-Flow pi05 that need a real backbone (GPU).

Three properties decide whether this is a *finetune* of pi05 or an accidental retrain, and none of
them can be checked without running the actual model:

  1. r-independence at init. The r MLP's output layer is zero-initialised, so a freshly built
     alpha-Flow model must return the SAME mean velocity for every r -- i.e. it is still predicting
     pi05's instantaneous velocity.
  2. ODE equivalence at init. Reading the model with r = t at each step has to reproduce plain pi05
     sampling bit-for-bit from the same noise. If it does not, the adaRMS conditioning was changed
     rather than extended, and every pretrained weight is being asked to mean something new.
  3. Every phase's loss runs and is finite -- including the alpha = 0 JVP branch, which is the one
     path with a real chance of blowing up memory on a VLA.

    uv run --no-sync python scripts/alphaflow_smoke.py
"""

import gc

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_alphaflow
import openpi.models.pi0_config as pi0_config

COMMON = {
    "pi05": True,
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
    "action_horizon": 16,
    "discrete_state_input": False,
}
KEY = jax.random.key(0)


def _u_of_r(model, obs, x_t, t, r):
    prefix_mask, kv = model._prefix_forward(obs)
    return model._u(obs, prefix_mask, kv, x_t, t, r)


def check_r_independence(model, obs, act):
    b = act.shape[0]
    x_t = jax.random.normal(jax.random.key(1), act.shape)
    t = jnp.full((b,), 0.7)
    f = nnx.jit(_u_of_r)
    u_rt = f(model, obs, x_t, t, t)
    u_r0 = f(model, obs, x_t, t, jnp.zeros_like(t))
    gap = float(jnp.max(jnp.abs(u_rt - u_r0)))
    print(f"  [1] max |u(z,t,t) - u(z,0,t)| = {gap:.3e}   (0 => r conditioning is zero-init)")
    return gap


def check_ode_equivalence(obs, noise):
    """alpha-Flow read with r = t must be the pi05 ODE, on the same weights and the same noise.

    The two models are built and freed one at a time: a 2B backbone plus its jit'd sampler is most of
    an L40S, and holding both at once OOMs before either result exists.
    """
    pi0_model = pi0_config.Pi0Config(**COMMON).create(KEY)
    a_pi0 = np.asarray(
        nnx.jit(lambda m, o, n: m.sample_actions(jax.random.key(3), o, num_steps=10, noise=n))(pi0_model, obs, noise)
    )
    del pi0_model
    jax.clear_caches()
    gc.collect()

    af_model = pi0_alphaflow.Pi0AlphaFlowConfig(**COMMON, alpha_init=1.0, alpha_final=1.0).create(KEY)
    a_af = np.asarray(
        nnx.jit(lambda m, o, n: m.sample_actions_ode(jax.random.key(3), o, num_steps=10, noise=n))(af_model, obs, noise)
    )
    # one-step sampling must actually differ from the 10-step ODE, or the sampler is not using the
    # mean velocity at all and this check would pass for the wrong reason.
    a1 = np.asarray(
        nnx.jit(lambda m, o, n: m.sample_actions(jax.random.key(3), o, num_steps=1, noise=n))(af_model, obs, noise)
    )
    del af_model
    jax.clear_caches()
    gc.collect()

    gap = float(np.max(np.abs(a_pi0 - a_af)))
    print(f"  [2] max |pi05_ode - alphaflow_ode| = {gap:.3e}   (0 => same model, extended not altered)")
    print(f"  [3] 1-step vs 10-step ODE spread   = {float(np.max(np.abs(a1 - a_af))):.3e}  shape {a1.shape}")
    return gap


def check_phase(name, cfg, obs, act, progresses):
    model = cfg.create(KEY)
    fn = nnx.jit(lambda m, o, a, q: m.compute_loss(jax.random.key(4), o, a, train=True, progress=q))
    for q in progresses:
        loss, aux = fn(model, obs, act, jnp.asarray(q, jnp.float32))
        finite = bool(jnp.all(jnp.isfinite(loss)))
        print(
            f"  [{name}] progress {q:>5.2f}  loss {float(jnp.mean(loss)):8.4f}  finite={finite}  "
            f"alpha={float(aux['alpha']):.4f}  fm_ratio={float(aux['fm_ratio']):.3f}  "
            f"loss_tfm={float(aux['loss_tfm']):.4f}"
        )
        assert finite, f"{name} produced a non-finite loss at progress {q}"
    del model
    jax.clear_caches()
    gc.collect()


def main():
    cfg_tfm = pi0_alphaflow.Pi0AlphaFlowConfig(**COMMON, alpha_init=1.0, alpha_final=1.0)
    obs = cfg_tfm.fake_obs(batch_size=2)
    act = cfg_tfm.fake_act(batch_size=2)
    noise = jax.random.normal(jax.random.key(2), (act.shape[0], COMMON["action_horizon"], cfg_tfm.action_dim))

    print("== init-time equivalence to pi05 ==")
    model = cfg_tfm.create(KEY)
    gap_r = check_r_independence(model, obs, act)
    del model
    jax.clear_caches()
    gc.collect()
    gap_ode = check_ode_equivalence(obs, noise)

    print("\n== per-phase losses ==")
    check_phase("tfm     ", cfg_tfm, obs, act, [0.0, 0.5])
    check_phase(
        "anneal  ",
        pi0_alphaflow.Pi0AlphaFlowConfig(**COMMON),
        obs,
        act,
        # official whole-run sigmoid: clamped to 1 early, 0.5 at mid-run, on the 5e-3 floor late
        [0.1, 0.5, 0.9],
    )
    check_phase(
        "meanflow",
        pi0_alphaflow.Pi0AlphaFlowConfig(**COMMON, alpha_init=0.0, alpha_final=0.0, meanflow_jvp=True),
        obs,
        act,
        [0.0],
    )

    ok = gap_r < 1e-5 and gap_ode < 1e-5
    print(f"\n{'PASS' if ok else 'FAIL'}: init-time equivalence to pi05 (r-independence and ODE)")
    assert ok, f"alpha-Flow is not pi05 at init: r-gap {gap_r:.3e}, ode-gap {gap_ode:.3e}"


if __name__ == "__main__":
    main()
