"""The steering gradient is taken in the critic's space, not in a displaced copy of it.

The selection path always routed candidates through PHYSICAL units before scoring them: the
sampler's output is normalized by the POLICY's statistics and the critic was trained under its
own, so handing the raw array over scores a displaced trajectory. The comment saying so has been
in `infer` the whole time, and the steering path did exactly that anyway --

    proprio   built with the critic's stats     correct
    actions   passed through as-is              NOT

-- so one run computed its recorded `critic_scores` in the critic's space and followed a gradient
taken in the policy's. On the checkpoint pair this was found with, the two normalizations happen to
be close (spans within 3%, displacement 0.2% of the critic's box), which is luck, not design: the
proprio version of this same mistake moved grad_a Q by cosine 0.85 mean and -0.69 at worst.

The map is affine, so it is CALIBRATED by probing the real transforms rather than by re-deriving
the quantile algebra -- a hand-written copy is the same kind of duplicate that caused this.
"""

import numpy as np
import pytest


class _Pre:
    """A critic preproc with its own action statistics, and a delta convention."""

    def __init__(self, q01, q99, ref):
        self.q01, self.q99, self.ref = q01, q99, ref

    def actions(self, chunk, base_state):
        a = np.array(chunk, np.float64, copy=True)
        for i, r in enumerate(self.ref):
            if r >= 0:
                a[..., i] -= np.asarray(base_state)[..., None, r]
        return ((a - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)


class _OutputTransform:
    """Unnormalize with the POLICY's statistics, then add the PHYSICAL state.

    openpi's output transform runs Unnormalize before JointAbsoluteActions, so the value added is
    the state in real units -- the same quantity the critic's preproc subtracts again. State
    normalization is the identity in this fixture, so the tests pass one array for both and the
    cancellation is real rather than arranged.
    """

    def __init__(self, q01, q99, ref):
        self.q01, self.q99, self.ref = q01, q99, ref

    def __call__(self, d):
        a = (np.asarray(d["actions"], np.float64) + 1) / 2 * (self.q99 - self.q01) + self.q01
        for i, r in enumerate(self.ref):
            if r >= 0:
                a[..., i] += np.asarray(d["state"])[..., None, r]
        return {"actions": a.astype(np.float32)}


AD, H, S = 4, 3, 6
REF = np.arange(AD)


def _policy(pol_q01, pol_q99, cr_q01, cr_q99, *, critic_horizon=H):
    from openpi.policies import patch_critic_policy as pcp

    p = pcp.PatchCriticSelectPolicy.__new__(pcp.PatchCriticSelectPolicy)
    p._action_horizon, p._model_action_dim = H, AD
    p._critic_horizon, p._critic_action_dim = critic_horizon, AD
    p._pre = _Pre(cr_q01, cr_q99, REF)
    p._pol = type("P", (), {"_output_transform": _OutputTransform(pol_q01, pol_q99, REF)})()
    return p


def _reference(p, x, norm_state, state):
    """What the SELECTION path computes -- the answer steering has to reproduce."""
    dec = p._pol._output_transform({"state": norm_state[None].copy(), "actions": x})["actions"]
    return np.asarray(p._pre.actions(dec, state), np.float32)[:, : p._critic_horizon, : p._critic_action_dim]


@pytest.mark.parametrize("shifted", [False, True])
def test_the_calibrated_map_reproduces_the_selection_path(shifted):
    """Identical statistics or not, `k * a + c` must land where the selection path lands."""
    pol_q01, pol_q99 = np.full(AD, -1.0), np.full(AD, 1.0)
    cr_q01 = pol_q01 + (0.3 if shifted else 0.0)
    cr_q99 = pol_q99 * (1.4 if shifted else 1.0)
    p = _policy(pol_q01, pol_q99, cr_q01, cr_q99)

    rng = np.random.default_rng(0)
    state = rng.normal(size=S).astype(np.float32)
    norm_state = state  # identity state normalization in this fixture (see _OutputTransform)
    x = rng.uniform(-0.9, 0.9, size=(1, H, AD)).astype(np.float32)

    k, c = p._critic_space_affine(norm_state, state)
    got = x[0, : p._critic_horizon] * k + c
    assert np.allclose(got, _reference(p, x, norm_state, state)[0], atol=1e-4)


def test_passing_the_chunk_through_raw_is_wrong_when_the_stats_differ():
    """The bug, stated as a measurement: with different statistics the uncalibrated array is NOT
    what the critic was trained to read, so a test that only checked "it runs" would pass."""
    p = _policy(np.full(AD, -1.0), np.full(AD, 1.0), np.full(AD, -0.7), np.full(AD, 1.4))
    rng = np.random.default_rng(1)
    state = rng.normal(size=S).astype(np.float32)
    norm_state = state
    x = rng.uniform(-0.9, 0.9, size=(1, H, AD)).astype(np.float32)

    want = _reference(p, x, norm_state, state)[0]
    raw = x[0, : p._critic_horizon]
    assert np.abs(raw - want).max() > 0.1, "this fixture must actually exercise a difference"

    k, c = p._critic_space_affine(norm_state, state)
    assert np.abs((raw * k + c) - want).max() < 1e-4


def test_identical_stats_make_the_map_the_identity():
    """When the two were normalized alike the calibration must not move anything -- otherwise it
    would be a new source of displacement for every pair that was previously fine."""
    q01, q99 = np.full(AD, -1.0), np.full(AD, 1.0)
    p = _policy(q01, q99, q01, q99)
    rng = np.random.default_rng(2)
    st = rng.normal(size=S).astype(np.float32)
    k, c = p._critic_space_affine(st, st)
    assert np.allclose(k, 1.0, atol=1e-4)
    assert np.allclose(c, 0.0, atol=1e-4)


def test_the_state_cancels_for_a_pi05_space_critic():
    """The policy adds the physical state and the critic subtracts it again, so the map does not
    depend on the state -- which is why recomputing it per inference is cheap insurance rather than
    a requirement. It is NOT true for a legacy critic scoring absolute targets, and that is exactly
    why it is recomputed rather than cached at construction."""
    q01, q99 = np.full(AD, -1.0), np.full(AD, 1.0)
    p = _policy(q01, q99, q01 - 0.2, q99 * 1.3)
    rng = np.random.default_rng(4)
    a = rng.normal(size=S).astype(np.float32)
    b = rng.normal(size=S).astype(np.float32)
    ka, ca = p._critic_space_affine(a, a)
    kb, cb = p._critic_space_affine(b, b)
    assert np.allclose(ka, kb, atol=1e-5)
    assert np.allclose(ca, cb, atol=1e-5)

    p._pre = None  # legacy: the critic scores absolute joint targets, so the state stays in
    _k, ca = p._critic_space_affine(a, a)
    _k, cb = p._critic_space_affine(b, b)
    assert not np.allclose(ca, cb, atol=1e-3)


def test_a_shorter_critic_is_not_handed_the_whole_chunk():
    """An h30 critic under an h50 policy: the selection path slices, the steering path did not, and
    the critic was silently read outside the horizon it was trained on."""
    p = _policy(np.full(AD, -1.0), np.full(AD, 1.0), np.full(AD, -1.0), np.full(AD, 1.0), critic_horizon=2)
    rng = np.random.default_rng(3)
    st = rng.normal(size=S).astype(np.float32)
    k, _c = p._critic_space_affine(st, st)
    assert k.shape == (2, AD), k.shape


def test_the_serving_layer_applies_it_rather_than_recording_it():
    """Behaviour lives in `value_fn`; a calibration computed and not used is the shape of bug this
    whole file exists for."""
    import inspect

    from openpi.extraction.serving import ArmChunkSampler

    fun = inspect.getsource(ArmChunkSampler._steer_jit.func)
    assert "* k + c" in fun, "the steering gradient must be taken in the critic's space"
    assert "[:, :ch, :ad]" in fun, "and over the critic's horizon"

    from openpi.policies import patch_critic_policy as pcp

    src = inspect.getsource(pcp.PatchCriticSelectPolicy.infer)
    assert "_set_critic_space(" in src, "the wrapper owns both transforms, so it computes the map"
