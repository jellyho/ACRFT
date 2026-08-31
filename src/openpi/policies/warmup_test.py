"""The warm-up has to warm everything an inference touches.

This is the third time it did not, and each time it failed the same way: the server logged
`warm-up done in Ns`, spent the full compile, and the first real request paid it again anyway. A
partial warm-up is indistinguishable from a working one at the log, which is exactly the shape of
bug this repo keeps producing -- an artifact that describes the property without checking it.

    warmed                                      first request
    nothing                                     26749 ms
    sampler, on fake_obs' state width (32/42)   26749 ms   <- warmed a graph nothing could reuse
    sampler, correct width                       6360 ms   <- patchify and score still cold
    patchify + score + v + sampler               6747 ms   <- with --drift-samples: refs cold
    ...and the reference draws                    555 ms

So these tests assert on CALLS, not on log lines: whatever an inference invokes, the warm-up must
invoke too, with the shapes the real request will use.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.policies import patch_critic_policy as pcp


class _Recorder:
    #: _candidate_count asks the arm sampler whether it draws a twin, so a stand-in for it has to
    #: answer. (The warm-up swallowed the AttributeError and logged "warm-up skipped" -- correct
    #: behaviour, and the reason a test had to look at calls rather than at the log.)
    pair_unsteered = False

    def __init__(self, ret=None):
        self.calls = []
        self._ret = ret

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self._ret


def _policy(*, arm: bool, drift: int, state_dim: int = 42):
    """A PatchCriticSelectPolicy with only the attributes warmup() reads, each one a recorder."""
    p = pcp.PatchCriticSelectPolicy.__new__(pcp.PatchCriticSelectPolicy)
    p._rng = jax.random.key(0)
    p._patch_shape = (256, 384)
    p._proprio_idx = np.arange(14)
    p._camera_keys = ("a", "b", "c")
    p._img_size = 224
    p._critic_horizon, p._critic_action_dim = 5, 14
    p._model_action_dim = 32
    p._flow_steps = 10
    p._default_samples = 8
    p._drift_samples = drift
    p._patchify = _Recorder(jnp.zeros((1, 256, 384)))
    p._score = _Recorder(jnp.zeros((1,)))
    p._v_of = _Recorder(jnp.zeros(()))
    p._extract = _Recorder(jnp.zeros((1, 30, 32)))
    p._arm_sampler = _Recorder(jnp.zeros((1, 30, 32))) if arm else None
    p._arm = "qpilots" if arm else None

    class _T:
        pass

    p._pol = type("P", (), {"_output_transform": _T()})()
    # the helper that reads the robot's real state width off the output transform
    p._forced_state_dim = state_dim
    return p


@pytest.fixture(autouse=True)
def _state_width(monkeypatch):
    monkeypatch.setattr(pcp._policy_mod, "_output_state_dim", lambda _t, fallback: 42)


class _Obs:
    def __init__(self, state):
        self.state = state


def _fake_obs(width):
    import dataclasses

    @dataclasses.dataclass
    class O:
        state: jnp.ndarray

    return O(jnp.zeros((1, width), jnp.float32))


def test_the_reference_draws_are_warmed_alongside_the_arm():
    """The regression that cost 6.7 s: `_extract` sat behind an `elif`, so with --drift-samples the
    arm's graph was warmed and the reference draws' graph was not. They are two compilations and an
    inference with the readout on runs both."""
    p = _policy(arm=True, drift=4)
    p.warmup(_fake_obs(42))
    assert p._arm_sampler.calls, "the arm's own graph must be warmed"
    assert p._extract.calls, "so must the reference draws -- an inference with drift on runs both"


def test_the_references_are_warmed_at_the_count_that_will_be_requested():
    """num_samples is a static argument: warming at 8 and serving 4 compiles twice, which is the
    whole failure mode again in a subtler form."""
    p = _policy(arm=True, drift=4)
    p.warmup(_fake_obs(42))
    assert p._extract.calls[0][1]["num_samples"] == 4


def test_without_drift_the_sampler_is_warmed_at_the_default_count():
    p = _policy(arm=False, drift=0)
    p.warmup(_fake_obs(42))
    assert p._extract.calls[0][1]["num_samples"] == 8


def test_every_jitted_graph_an_inference_touches_is_warmed():
    """patchify, score and V, not only the sampler. Warming one of three spent the compile and left
    the first request paying for the other two (6.4 s)."""
    p = _policy(arm=True, drift=0)
    p.warmup(_fake_obs(42))
    assert p._patchify.calls, "the DINOv2 patchifier"
    assert p._score.calls, "the critic scorer"
    assert p._v_of.calls, "the value head"


def test_the_state_width_comes_from_the_robot_not_from_fake_obs():
    """fake_obs derives state from the model's ACTION dim (32 on pi05); YAM's state is 42. Warming
    at 32 compiles a graph the real request cannot reuse -- the warm-up costs its full time and
    saves exactly nothing, which is the version of this bug that was hardest to see."""
    p = _policy(arm=True, drift=0)
    p.warmup(_fake_obs(32))  # what fake_obs hands over
    warmed = p._arm_sampler.calls[0][0][1]  # (rng, observation, feats, proprio)
    assert warmed.state.shape[-1] == 42, "the arm was warmed on the wrong state width"


def test_a_warm_up_failure_is_not_a_startup_failure(caplog):
    """It is an optimisation. If it throws, the server must still come up and the first inference
    just pays the compile as it did before."""
    p = _policy(arm=True, drift=0)

    def boom(*_a, **_k):
        raise RuntimeError("no GPU today")

    p._patchify = boom
    p.warmup(_fake_obs(42))  # must not raise
    assert any("warm-up skipped" in r.message for r in caplog.records)
