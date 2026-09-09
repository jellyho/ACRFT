"""The two control arms actually control something.

`negated` and `random` exist so the re-run can tell a WRONG DIRECTION apart from an OVER-LARGE
INJECTION: if `negated` matches `critic`, the gradient carried no usable direction; if `random`
matches it, the displacement did the work and the critic was decoration. Everything the campaign
concludes from those two arms rests on them differing from `critic` in the way their names claim.

Before this file nothing executed either branch. `test_every_steer_value_builds` asserts
`spec.steer_value == mode` after `default_spec` set it with `setattr` on a non-frozen dataclass, so
it reads back its own input and passes with the field deleted; `qpilots_drift_test` and
`steer_space_test` only `inspect.getsource` the loop. A review verifier proved the gap by mutation:
forcing `sign = 1.0`, which makes `negated` serve the SAME gradient as `critic`, broke no test.

So these tests are written to fail under exactly that mutation. They run the real `_steer_jit` on a
dummy-sized Pi0 with an analytic critic, which is what makes the assertions about the integration
rather than about the plumbing around it.
"""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.extraction import serving
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0_steered import Pi0Steered

H, AD = 4, 8
#: The critic's optimum. Q = -||a - TARGET||^2, so grad_a Q points straight at it from anywhere
#: and "did steering ascend the critic" has an answer that does not depend on the network.
TARGET = 0.6


class _AnalyticCritic:
    """A stand-in whose Q is a known function of the action and ignores features and proprio.

    Shaped [K, B] like the real ensemble so `_q`'s reduction runs unchanged: two members that
    agree, so mean and mean - rho*std coincide and the test does not depend on the gate.
    """

    class _HL:
        @staticmethod
        def from_logits(x):
            return x

    class _Net:
        @staticmethod
        def apply(_params, _feats, chunk, _proprio):
            q = -jnp.sum((chunk - TARGET) ** 2, axis=(1, 2))  # [B]
            return jnp.stack([q, q])[..., None]  # [K, B, 1]; _q takes [..., -1]

    def __init__(self):
        self.hl, self.net, self.params = self._HL(), self._Net(), {}
        self.config = {"action_dim": AD, "horizon": H}


@pytest.fixture(scope="module")
def sampler():
    cfg = pi0_config.Pi0Config(
        pi05=True, action_horizon=H, action_dim=AD, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    base = cfg.create(jax.random.key(0))
    model = Pi0Steered.wrap(base)
    obj = object.__new__(serving.ArmChunkSampler)
    obj.spec = serving.ArmSpec(arm="qpilots", base_ckpt=serving.BC_CKPT, alpha=0.4, ode_steps=4)
    obj._warned_rho = False
    obj.model = model
    obj.graphdef = nnx.graphdef(model)
    obj.params = nnx.state(base)
    obj.H, obj.AD = H, AD
    obj.critic = _AnalyticCritic()
    obj.critic_horizon = H
    obj.to_critic_space = (np.ones((H, AD), np.float32), np.zeros((H, AD), np.float32))
    obs = _model.preprocess_observation(None, cfg.fake_obs(batch_size=1), train=False)
    feats = jnp.zeros((1, 4, 3), jnp.float32)
    proprio = jnp.zeros((1, 4), jnp.float32)
    return obj, obs, feats, proprio


def _draw(s, mode, alpha):
    obj, obs, feats, proprio = s
    obj.spec = dataclasses.replace(obj.spec, steer_value=mode)
    # a fresh jit per mode: steer_value is read inside the traced closure, not passed as an arg
    obj.__dict__.pop("_steer_jit", None)
    k, c = obj.to_critic_space
    paired = False  # one chunk, not the arm's twin -- alpha=0 is a rung of the ladder here
    out = obj._steer_jit(obj.params, jax.random.key(3), obs, feats, proprio, AD, float(alpha), paired, k, c)
    return np.asarray(out[0], np.float64)


def _displacement(s, mode, alpha=0.4):
    return _draw(s, mode, alpha) - _draw(s, mode, 0.0)


def test_critic_ascends_the_critic(sampler):
    """The baseline the two controls are compared against: steering moves the chunk toward TARGET."""
    base = _draw(sampler, "critic", 0.0)
    steered = _draw(sampler, "critic", 0.4)
    assert np.sum((steered - TARGET) ** 2) < np.sum((base - TARGET) ** 2)


def test_negated_moves_the_opposite_way(sampler):
    """This is the test that dies under `sign = 1.0`.

    `negated` ascends -Q, so its displacement must be anti-correlated with `critic`'s. Comparing
    displacements rather than endpoints keeps the shared alpha=0 draw out of the comparison.
    """
    d_c = _displacement(sampler, "critic").ravel()
    d_n = _displacement(sampler, "negated").ravel()
    cos = float(d_c @ d_n / (np.linalg.norm(d_c) * np.linalg.norm(d_n) + 1e-12))
    assert cos < -0.9, f"negated should oppose critic, cosine {cos:+.3f}"
    # and it must move AWAY from the critic's optimum, which `critic` moves toward
    base = _draw(sampler, "critic", 0.0)
    assert np.sum((_draw(sampler, "negated", 0.4) - TARGET) ** 2) > np.sum((base - TARGET) ** 2)


def test_random_is_a_direction_the_critic_did_not_choose(sampler):
    """`random` ascends a fixed unit direction, so it must not track the value gradient."""
    d_c = _displacement(sampler, "critic").ravel()
    d_r = _displacement(sampler, "random").ravel()
    cos = abs(float(d_c @ d_r / (np.linalg.norm(d_c) * np.linalg.norm(d_r) + 1e-12)))
    assert cos < 0.8, f"random tracks the critic gradient too closely, |cosine| {cos:.3f}"
    assert np.linalg.norm(d_r) > 1e-6, "random did not move the chunk at all"


def test_every_arm_injects_the_same_magnitude(sampler):
    """Eq. 17 rescales to the drift's own norm, so the CONTROLS are controls: they differ from
    `critic` in direction, not in how hard they push. Without this the campaign could not read a
    difference in outcome as a difference in direction."""
    n = {m: np.linalg.norm(_displacement(sampler, m)) for m in serving.STEER_VALUES}
    lo, hi = min(n.values()), max(n.values())
    assert lo > 1e-6, f"an arm did not move: {n}"
    assert hi / lo < 3.0, f"injected magnitudes differ too much to be controls: {n}"


def test_alpha_zero_is_the_same_draw_for_every_arm(sampler):
    """The unsteered twin cannot depend on which value function was not used."""
    draws = [_draw(sampler, m, 0.0) for m in serving.STEER_VALUES]
    for other in draws[1:]:
        np.testing.assert_allclose(draws[0], other, rtol=1e-5, atol=1e-6)
