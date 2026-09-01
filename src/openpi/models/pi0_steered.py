"""Value-steered flow sampling, as a Pi0 subclass.

QPILOTS-U (arXiv 2606.14801) is not a different model -- it is the same pi0.5 flow policy with a
value gradient added to the drift at every Euler step. So it belongs here, next to the sampler it
modifies, rather than in the serving layer where it lived as a hand-copy of pi0.py's own
machinery: a copy that drifts from the served policy does not fail loudly, it silently steers a
base that is no longer the base being served.

The value function is INJECTED. That is the whole point of the shape: this file never learns that
a critic exists, which is what lets one critic score a BC, an alpha-Flow and an RLT base without
any of them knowing about it -- and what lets a test steer with a toy analytic value instead of a
3B checkpoint and a DINOv2 backbone.
"""

import copy
from typing import Protocol

import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.models.pi0 import Pi0
from openpi.shared import array_typing as at


class ValueFn(Protocol):
    """`a_hat [b, H, AD] -> scalar`, differentiable in `a_hat` and summed over the batch.

    The caller owns what it means. `PatchCriticSelectPolicy` passes the pessimistic ensemble mean
    of its patch critic; a test passes something with a known optimum.
    """

    def __call__(self, a_hat: jnp.ndarray) -> jnp.ndarray: ...


class Pi0Steered(Pi0):
    """`Pi0` plus one sampler. No parameters, no state, no config of its own.

    That is deliberate: it has to be able to wrap a checkpoint that was trained as a plain `Pi0`,
    because steering is a decision made at serving time about an already-trained policy.
    """

    @classmethod
    def wrap(cls, base: Pi0) -> "Pi0Steered":
        """Re-tag an already-loaded `Pi0` as this subclass, sharing its parameters.

        The subclass adds no fields, so the instance is structurally identical and the re-tag is
        exact. The alternative -- building a second module from the config just to obtain a
        graphdef carrying the right type -- allocates 3B parameters to change a label.

        The copy is shallow (`copy.copy`), so the served policy's own object is left untouched:
        its jitted graphs are keyed on a graphdef that includes the type, and re-tagging in place
        would silently invalidate every one of them.
        """
        if isinstance(base, cls):
            return base
        obj = copy.copy(base)
        obj.__class__ = cls
        return obj

    def sample_steered(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        value_fn: ValueFn,
        alpha: float,
        num_steps: int = 10,
        noise: jnp.ndarray | None = None,
        preprocess: bool = True,
    ) -> _model.Actions:
        """One chunk, with the drift steered toward higher `value_fn` at every Euler step.

        `alpha` is the steering strength, drift-norm-matched (Eq. 17): the value gradient is
        rescaled to the velocity's own magnitude before it is applied. That rescaling is what
        replaces the paper's sigma schedule, and it is why alpha is a per-domain constant rather
        than a function of t.

        `alpha=0` reproduces the unsteered sampler THROUGH THIS SAME PATH -- not through
        `sample_actions` -- which is how a caller measures what steering displaced. Passing the
        same `noise` to both draws makes the difference the steering term and nothing else.

        `preprocess=False` for a caller that has already run `preprocess_observation`, which the
        serving wrapper has (it needed the observation to compute the critic's own inputs).
        """
        if preprocess:
            observation = _model.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))

        prefix_mask, kv_cache = self._prefix_forward(observation)
        dt = 1.0 / num_steps
        x = noise

        # A plain Python loop, deliberately, and this is the record of why so it does not get
        # re-litigated: a lax.fori_loop version compiled faster (24 s against 42 s) but changed the
        # answer by 5.9e-03 in bf16 -- and, unlike the joint/cached-attention difference measured
        # elsewhere in this repo, that did NOT collapse in fp32 (7.7e-04, only 8x smaller, where
        # pure accumulation order fell by 8400x). Unexplained numerical drift in a sampler is not
        # worth a compile-time saving, and it bought no runtime: the 25x speedup that made this arm
        # servable came entirely from not doing eager work around the jit (serving.py, __call__).
        for i in range(num_steps):
            t = jnp.full((batch,), 1.0 - i * dt)
            if i == 0:
                # No state-dependent signal at t=0 (paper Sec. 4). Every alpha takes this branch,
                # including the alpha=0 twin, so the two draws stay symmetric here.
                v = self._velocity(observation, prefix_mask, kv_cache, x, t)
            else:
                # NOT short-circuited at alpha == 0, though that would save a gradient on every
                # unsteered step. `v` below comes out of jax.grad's forward pass, and a direct
                # _velocity call is the same math in a different accumulation ORDER -- measured in
                # this repo at 1.2e-02 in bf16 (1.4e-06 in fp32). Branching on alpha would put that
                # difference between a steered draw and the twin it is measured against, compound
                # it over every step, and report the sum as steering displacement.
                def scored(x_, t_=t):
                    v_ = self._velocity(observation, prefix_mask, kv_cache, x_, t_)
                    # Tweedie/MMSE point estimate (Eq. 14). openpi integrates time 1->0 with
                    # x_t = t*noise + (1-t)*a and v = noise - a, so the paper's
                    # a_hat = x_t + (1-t_paper)*v_paper is a_hat = x_t - t*v in these conventions.
                    a_hat = x_ - t_[:, None, None] * v_
                    # Straight-through clip: the value is read inside the box the critic was
                    # trained on, while the gradient still flows for samples outside it.
                    a_hat = a_hat + jax.lax.stop_gradient(jnp.clip(a_hat, -1, 1) - a_hat)
                    return value_fn(a_hat), v_

                g, v = jax.grad(scored, has_aux=True)(x)
                vn = jnp.linalg.norm(v.reshape(batch, -1), axis=-1).reshape(-1, 1, 1)
                gn = jnp.linalg.norm(g.reshape(batch, -1), axis=-1).reshape(-1, 1, 1)
                v = v - alpha * (vn / (gn + 1e-8)) * g  # Eq. 17
            x = x - dt * v
        return jnp.clip(x, -1.0, 1.0)
