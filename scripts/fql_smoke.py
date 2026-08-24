"""Structural smoke test for Pi0FQL: build the 4-expert model and check the three forwards' shapes."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.models.pi0_fql import Pi0FQLConfig
import openpi.shared.array_typing as at

B, AH, AD = 2, 8, 14
cfg = Pi0FQLConfig(action_dim=AD, action_horizon=AH, pi05=True, fql_num_atoms=51, fql_flow_ode_steps=2)
print("creating Pi0FQL (paligemma 2b + flow/actor/critic 300m + siglip So400m)...", flush=True)
model = cfg.create(jax.random.key(0))
nnx.eval_shape(lambda: model)  # touch

res = _model.IMAGE_RESOLUTION
with at.disable_typechecking():
    obs = _model.Observation(
        images={
            k: jnp.zeros((B, *res, 3), jnp.float32) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        image_masks={k: jnp.ones((B,), bool) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        state=jnp.zeros((B, AD), jnp.float32),
        tokenized_prompt=jnp.zeros((B, cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, cfg.max_token_len), bool),
    )
noise = jax.random.normal(jax.random.key(1), (B, AH, AD))

print("flow_ode (frozen mu_theta distillation target)...", flush=True)
tgt = model.flow_ode(obs, noise)
print("  flow_ode ->", tgt.shape, "expect", (B, AH, AD))

print("actor (one-step mu_omega)...", flush=True)
act = model.actor(obs, noise)
print("  actor ->", act.shape, "expect", (B, AH, AD))

print("critic_logits Q_phi(s,a)...", flush=True)
ql = model.critic_logits(obs, act)
print("  critic ->", ql.shape, "expect", (B, cfg.fql_num_atoms))

assert tgt.shape == (B, AH, AD)
assert act.shape == (B, AH, AD)
assert ql.shape == (B, cfg.fql_num_atoms)
print("SMOKE OK: all three forwards produce correct shapes.", flush=True)
