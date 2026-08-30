"""Tests for the alpha-Flow objective's algebra and curriculum.

These run on CPU in seconds: they exercise the pure pieces of the loss (the alpha schedule, the
target blend, the adaptive weight) rather than a PaliGemma forward. The end-to-end checks that need
a real backbone -- that a freshly built alpha-Flow model IS the pretrained pi05, and that its
one-step sample tracks the 10-step ODE -- live in scripts/alphaflow_smoke.py, which needs a GPU.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_alphaflow as af

OFFICIAL = {"init": 1.0, "final": 0.0, "p_start": 0.0, "p_end": 1.0, "gamma": 25.0}


def test_alpha_schedule_matches_official_recipe():
    """Official runs: sigmoid over the WHOLE run, gamma 25, centred mid-run (B/2: [0, 400k] of
    400k; B/2-cfg: [0, 1.2M] of 1.2M). The raw sigmoid before clamping."""
    assert af._ratio_schedule(0.0, **OFFICIAL) == pytest.approx(1.0, abs=1e-5)
    assert af._ratio_schedule(0.5, **OFFICIAL) == pytest.approx(0.5, abs=1e-6)
    assert af._ratio_schedule(1.0, **OFFICIAL) == pytest.approx(0.0, abs=1e-5)
    # monotone decreasing
    vals = [float(af._ratio_schedule(q / 50, **OFFICIAL)) for q in range(51)]
    assert all(b <= a + 1e-6 for a, b in itertools.pairwise(vals))


def test_clamps_carve_the_official_three_phases():
    """With clamp eta = 5e-3, the whole-run sigmoid crosses 1 - eta at progress ~0.288 and eta at
    ~0.712: the paper's warm-up / anneal / floor phases fall out of the clamps, with no explicit
    window. sigmoid(x * 25) = 5e-3  <=>  x = -ln(199)/25 ~ -0.2118."""
    x = 0.2118
    for q, expect in ((0.5 - x - 0.02, 1.0), (0.5 + x + 0.02, 5e-3)):
        a = af.clamp_alpha(af._ratio_schedule(q, **OFFICIAL), eta=5e-3, floor=5e-3)
        assert float(a) == pytest.approx(expect, abs=1e-4)
    a_mid = af.clamp_alpha(af._ratio_schedule(0.5, **OFFICIAL), eta=5e-3, floor=5e-3)
    assert float(a_mid) == pytest.approx(0.5, abs=1e-6)


def test_schedule_rescales_with_run_length():
    """The whole point of progress-based scheduling: the same config, run for 240 steps or 60k
    steps, must trace the same curriculum -- equal progress, equal alpha."""
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        short = af._ratio_schedule(frac * 240 / 240, **OFFICIAL)
        long = af._ratio_schedule(frac * 60_000 / 60_000, **OFFICIAL)
        assert float(short) == pytest.approx(float(long), abs=1e-7)


def test_clamp_alpha_snaps_both_ends():
    # above 1-eta the discrete target is within eta of v_t, so we take the cheaper L_TFM path
    assert af.clamp_alpha(jnp.asarray(0.999), eta=5e-3, floor=5e-3) == pytest.approx(1.0)
    # below eta it lands on the floor -- 5e-3 keeps training discrete (no JVP), 0.0 opts into MeanFlow
    assert af.clamp_alpha(jnp.asarray(1e-4), eta=5e-3, floor=5e-3) == pytest.approx(5e-3)
    assert af.clamp_alpha(jnp.asarray(1e-4), eta=5e-3, floor=0.0) == pytest.approx(0.0)
    # untouched in between
    assert af.clamp_alpha(jnp.asarray(0.5), eta=5e-3, floor=5e-3) == pytest.approx(0.5)


def _fixture(b=4, h=3, d=2, seed=0):
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.normal(size=(b, h, d)), jnp.float32)
    u_s = jnp.asarray(rng.normal(size=(b, h, d)), jnp.float32)
    return v, u_s


def test_target_at_alpha_one_is_plain_flow_matching():
    """alpha = 1 must reproduce the pi05 BC target exactly, or the curriculum's first phase is not
    a finetune of the same objective."""
    v, u_s = _fixture()
    is_border = jnp.zeros((v.shape[0],), bool)
    tgt = af.alpha_flow_target(v, u_s, jnp.asarray(1.0), is_border=is_border)
    np.testing.assert_allclose(np.asarray(tgt), np.asarray(v), rtol=1e-6)


def test_target_approaches_self_consistency_as_alpha_vanishes():
    v, u_s = _fixture()
    is_border = jnp.zeros((v.shape[0],), bool)
    tgt = af.alpha_flow_target(v, u_s, jnp.asarray(1e-3), is_border=is_border)
    np.testing.assert_allclose(np.asarray(tgt), np.asarray(u_s), atol=5e-3)


def test_border_rows_ignore_the_consistency_term():
    """Rows sampled with r == t have no interval, so they must fall back to v_t whatever alpha is."""
    v, u_s = _fixture()
    is_border = jnp.asarray([True, False, True, False])
    tgt = af.alpha_flow_target(v, u_s, jnp.asarray(0.3), is_border=is_border)
    np.testing.assert_allclose(np.asarray(tgt[0]), np.asarray(v[0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(tgt[2]), np.asarray(v[2]), rtol=1e-6)
    assert not np.allclose(np.asarray(tgt[1]), np.asarray(v[1]))


def test_intermediate_state_is_one_euler_step_of_size_alpha_times_gap():
    v, _ = _fixture()
    x_t = jnp.zeros_like(v)
    t = jnp.asarray([1.0, 0.8, 0.6, 0.4])
    r = jnp.asarray([0.0, 0.4, 0.6, 0.0])
    z_s, s = af.intermediate_state(x_t, v, t, r, jnp.asarray(0.25))
    np.testing.assert_allclose(np.asarray(s), np.asarray(t - 0.25 * (t - r)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(z_s), np.asarray(-(0.25 * (t - r))[:, None, None] * v), rtol=1e-6)
    # r == t (row 2) must not move: the border case is a no-op here
    np.testing.assert_allclose(np.asarray(z_s[2]), np.zeros_like(np.asarray(v[2])), atol=1e-7)


def test_adaptive_weight_normalises_and_carries_alpha():
    delta2 = jnp.asarray([1.0, 4.0])
    w_border = af.adaptive_weight(delta2, jnp.asarray([1.0, 1.0]), eps=1e-3)
    np.testing.assert_allclose(np.asarray(w_border), [1 / 1.001, 1 / 4.001], rtol=1e-6)
    # discrete rows carry alpha in the numerator, which is what keeps the branches comparable
    w_disc = af.adaptive_weight(delta2, jnp.asarray([0.25, 0.25]), eps=1e-3)
    np.testing.assert_allclose(np.asarray(w_disc), np.asarray(w_border) * 0.25, rtol=1e-6)


def test_two_pass_is_off_only_for_single_branch_runs():
    """A pure-TFM run and a pure-JVP run each cost one forward; anything that anneals through the
    discrete regime must pay the second."""
    assert not af.Pi0AlphaFlowConfig(alpha_init=1.0, alpha_final=1.0).two_pass
    assert af.Pi0AlphaFlowConfig().two_pass  # the official annealing run
    pinned = af.Pi0AlphaFlowConfig(alpha_init=0.0, alpha_final=0.0, meanflow_jvp=True)
    assert pinned.pinned_jvp
    assert not pinned.two_pass
    # annealing INTO the JVP tail still needs the discrete branch on the way down
    transition = af.Pi0AlphaFlowConfig(meanflow_jvp=True)
    assert transition.two_pass
    assert not transition.pinned_jvp


def test_jvp_transition_floors_alpha_at_zero():
    """meanflow_jvp with the official schedule: the clamp must land alpha exactly on 0 (the JVP
    branch's predicate), not on the discrete floor."""
    cfg = af.Pi0AlphaFlowConfig(meanflow_jvp=True)
    assert cfg.alpha_floor == 0.0
    a_late = af.clamp_alpha(
        af._ratio_schedule(0.9, init=1.0, final=0.0, p_start=0.0, p_end=1.0, gamma=25.0),
        eta=cfg.alpha_clamp,
        floor=cfg.alpha_floor,
    )
    assert float(a_late) == 0.0
    # and mid-run it is still discrete (strictly between 0 and 1)
    a_mid = af.clamp_alpha(
        af._ratio_schedule(0.5, init=1.0, final=0.0, p_start=0.0, p_end=1.0, gamma=25.0),
        eta=cfg.alpha_clamp,
        floor=cfg.alpha_floor,
    )
    assert 0.0 < float(a_mid) < 1.0


def test_config_rejects_incoherent_phases():
    with pytest.raises(ValueError, match="pi05"):
        af.Pi0AlphaFlowConfig(pi05=False)
    with pytest.raises(ValueError, match="meanflow_jvp"):
        af.Pi0AlphaFlowConfig(alpha_floor=0.0)
    with pytest.raises(ValueError, match="never reaches the JVP"):
        af.Pi0AlphaFlowConfig(meanflow_jvp=True, alpha_floor=5e-3)
    with pytest.raises(ValueError, match="time_pair"):
        af.Pi0AlphaFlowConfig(time_pair="bogus")
    with pytest.raises(ValueError, match="alpha_anneal"):
        af.Pi0AlphaFlowConfig(alpha_anneal_start=0.8, alpha_anneal_end=0.2)
    with pytest.raises(ValueError, match="fm_ratio"):
        af.Pi0AlphaFlowConfig(fm_ratio=1.5)


def test_default_floor_keeps_training_jvp_free():
    """The default has to be the cheap path: alpha bottoms out at the clamp, never at 0."""
    cfg = af.Pi0AlphaFlowConfig()
    assert cfg.meanflow_jvp is False
    assert cfg.alpha_floor == cfg.alpha_clamp > 0.0
