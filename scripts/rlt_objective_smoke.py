"""Shape/pytree smoke test for the RLT objective variants.

Builds a small Pi0RLT for each objective string, runs one compute_loss on fake data, and prints the
aux dict. This is a cheap way to catch the two failure modes that only show up at runtime:

  * a head that is built but never fed (or fed but never built) — the "+action"/"+behsim" heads
    follow the same build-only-when-named rule as the progress head so that old checkpoints keep
    loading, and getting that wrong is a pytree-structure error at load time, not at init;
  * a reshape that only works for one action_horizon / probe_action_dim combination.

Uses gemma_2b_lora so it fits comfortably on one GPU; the objective code paths are identical.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import pi0_rlt

OBJECTIVES = [
    "reconstruction",
    "reconstruction+progress",
    "reconstruction+action",
    "reconstruction+behsim",
    "reconstruction+action+behsim",
    "reconstruction+progress+action",
]


def main():
    for obj in OBJECTIVES:
        cfg = pi0_rlt.Pi0RLTConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=16,
            discrete_state_input=False,
            rlt_objective=obj,
            rlt_decoder_mode="parallel",
            rlt_include_proprio=False,
            rlt_mask_ratio=0.5,
            rlt_bc_probe=True,
            rlt_probe_action_dim=12,
        )
        model = cfg.create(jax.random.key(0))
        # ones, not zeros: the masks must be True or there are no image tokens to reconstruct and
        # every masked mean falls back to its epsilon guard, which hides real shape bugs.
        obs = cfg.fake_obs(batch_size=4)
        act = cfg.fake_act(batch_size=4)
        # Progress is only populated by the data config; supply it here for the objectives that need
        # it, and vary it per sample so a degenerate constant does not hide a broadcasting bug.
        obs = obs.replace(progress=jnp.linspace(0.0, 1.0, 4))
        # Vary actions per sample too: behsim's target distribution is uniform (and its gradient
        # zero) if every chunk is identical, which would make a broken distance matrix look fine.
        act = act + jax.random.normal(jax.random.key(1), act.shape) * 0.5

        loss, aux = nnx.jit(lambda m, o, a: m.compute_loss(jax.random.key(2), o, a, train=True))(model, obs, act)
        print(f"\n=== {obj} ===")
        print(f"  loss {jnp.mean(loss):.4f}  shape {loss.shape}")
        for k in sorted(aux):
            print(f"  {k:32s} {float(aux[k]):.4f}")
        del model
        jax.clear_caches()

    # The validator must reject anything that would leave the bottleneck without reconstruction.
    for bad in ("progress", "action", "reconstruction+bogus", "reconstruction+action+action"):
        try:
            pi0_rlt.Pi0RLTConfig(pi05=True, rlt_objective=bad)
        except ValueError as e:
            print(f"\nrejected {bad!r}: {str(e)[:70]}...")
        else:
            raise AssertionError(f"{bad!r} should have been rejected")

    print("\nOK")


if __name__ == "__main__":
    main()
