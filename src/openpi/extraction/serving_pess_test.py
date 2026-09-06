"""The pessimistic read is refused below K=10, and that changes what QPILOTS steers along.

serving.py's `_q(reduce="pess")` is the value function the steering arm differentiates. It was
`mean - rho*std` with rho=0.5 over K=2 members -- a verbatim transplant of QAM (agents/qam.py:33)
minus the ensemble QAM sizes it for (num_qs=10, qam.py:424). At K=2, jnp.std with ddof=0 is
|q1-q2|/2, and measured on 64 on-manifold states that term carried 59.6 of gradient magnitude
against 123.5 from the mean, in a direction near-orthogonal to it (cos -0.078).

So the alpha sweep that fell to 0.30 was following value-gradient + a third noise. These tests pin
the gate that stops that, because the re-run's whole point is that it steers along something
different from what the original sweep did.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.extraction import serving


class _FakeHL:
    def from_logits(self, x):
        return x  # the test supplies values directly, not logits


class _FakeNet:
    def __init__(self, q):
        self._q = q

    def apply(self, _params, _feats, _chunk, _proprio):
        return self._q


def _sampler(qvals):
    """An ArmChunkSampler with just enough wired to exercise _q, and no 3B model loaded."""
    obj = object.__new__(serving.ArmChunkSampler)
    obj.spec = dataclasses.replace(serving.ArmSpec(arm="qpilots", base_ckpt=serving.BC_CKPT), rho=0.5)
    obj._warned_rho = False
    obj.critic = type("C", (), {"net": _FakeNet(jnp.asarray(qvals)), "params": {}, "hl": _FakeHL()})()
    return obj


# q[K, B, prefixes]; _q reads [..., -1]
Q2 = np.array([[[-100.0], [-200.0]], [[-140.0], [-160.0]]])  # K=2: means -120/-180, stds 20/20
Q10 = np.stack([Q2[0]] * 5 + [Q2[1]] * 5)  # K=10 with the same two values


def test_k2_refuses_pessimism_and_returns_the_mean():
    s = _sampler(Q2)
    got = np.asarray(s._q(None, None, None, reduce="pess"))
    np.testing.assert_allclose(got, [-120.0, -180.0], rtol=1e-6)


def test_k10_applies_it():
    s = _sampler(Q10)
    got = np.asarray(s._q(None, None, None, reduce="pess"))
    np.testing.assert_allclose(got, [-120.0 - 0.5 * 20.0, -180.0 - 0.5 * 20.0], rtol=1e-6)


def test_min_is_untouched_by_the_gate():
    """The ranking arms reduce by min, not by this -- the gate must not move `bon`."""
    s = _sampler(Q2)
    got = np.asarray(s._q(None, None, None, reduce="min"))
    np.testing.assert_allclose(got, [-140.0, -200.0], rtol=1e-6)


def test_rho_zero_is_unaffected_by_the_gate():
    s = _sampler(Q2)
    s.spec = dataclasses.replace(s.spec, rho=0.0)
    np.testing.assert_allclose(np.asarray(s._q(None, None, None, reduce="pess")), [-120.0, -180.0])


def test_the_warning_names_k_and_fires_once(caplog):
    s = _sampler(Q2)
    with caplog.at_level("WARNING"):
        s._q(None, None, None, reduce="pess")
        s._q(None, None, None, reduce="pess")
    assert caplog.text.count("refusing the rho") == 1
    assert "K=2" in caplog.text


@pytest.mark.parametrize("mode", serving.STEER_VALUES)
def test_every_steer_value_builds(mode):
    spec = serving.default_spec("qpilots", steer_value=mode, alpha=0.1)
    assert spec.steer_value == mode
    assert spec.alpha == 0.1
