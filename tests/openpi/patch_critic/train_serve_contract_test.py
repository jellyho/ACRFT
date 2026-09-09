"""What the critic was trained on, and what serving hands it, are the same thing.

The critic's inputs are built twice: once by the trainer (train_patch_critic_cached.py:339-342, via
critic_q.CacheView.rows) and once by the serving wrapper. Nothing checks that the two agree, and
the failure is silent in the worst way -- Q comes back in range either way, and the only symptom is
that grad_a Q points somewhere else. That cost this ring nine retrained arms.

So these tests do not read the two paths and judge them alike. They compute both, with the real
YAM critic's spec, and compare numbers. Each also computes the plausible WRONG version and asserts
it differs, because a test that only confirms the right answer cannot tell you it would have caught
the wrong one.
"""

import json
import pathlib

import numpy as np
import pytest

from openpi.patch_critic import preproc as critic_preproc

#: The YAM proprio layout: two arms, 6 joints + gripper each, inside a 42-wide state. This is the
#: shape the order matters for -- indices 0..6 land on statistics 0..6 whichever order you use, so
#: the left arm is correct even under the bug and only 21..27 moves.
_PROPRIO_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27], np.int64)
_REF = np.array([0, 1, 2, 3, 4, 5, -1, 21, 22, 23, 24, 25, 26, -1], np.int64)
_STATE_DIM, _ACTION_DIM, _HORIZON = 42, 14, 30

#: The real critic, when it happens to be on this machine. NOT the guard for the tests below: they
#: need a SPEC (proprio indices, statistics, delta reference), not weights, and gating them on a
#: downloaded artifact meant they ran on exactly one machine and skipped everywhere else --
#: including on the machine where the training half of the contract is written. `4 skipped` reads
#: as green in a summary line, so the guard on the boundary between the two halves was inert
#: precisely where it was most needed. CI runs pre-commit only, so it never ran there either.
_REAL_CRITIC = pathlib.Path.home() / "hf_utils_downloads/acrft-yam-critics/patch_critic_yam_s347_fixed_200k"


def _synthetic_pre(seed: int = 0):
    """A Pi05Preproc with per-channel statistics that differ across the state.

    They have to differ: if channels 7..13 carried the same statistics as 21..27 the two orders
    would agree numerically and the test would pass while asserting nothing.
    """
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-3.0, -1.0, _STATE_DIM)
    # float64 arrays, exactly what load_norm_stats produces -- a list here would work in some
    # expressions and raise in others, and the test would be exercising a shape the loader
    # never hands the code under test.
    stats = {
        "state": {
            "q01": lo,
            "q99": lo + rng.uniform(2.0, 5.0, _STATE_DIM),
            "mean": rng.normal(size=_STATE_DIM),
            "std": rng.uniform(0.5, 2.0, _STATE_DIM),
        },
        "actions": {
            "q01": rng.uniform(-1.0, -0.3, _ACTION_DIM),
            "q99": rng.uniform(0.3, 1.0, _ACTION_DIM),
            "mean": rng.normal(size=_ACTION_DIM),
            "std": rng.uniform(0.2, 1.0, _ACTION_DIM),
        },
    }
    return critic_preproc.Pi05Preproc(ref=_REF, stats=stats, use_quantiles=True, delta=True)


@pytest.fixture(params=["synthetic", "real"])
def spec_and_pre(request):
    """Runs on the synthetic spec everywhere; adds the real critic where it is downloaded."""
    if request.param == "synthetic":
        return {
            "state_dim": _STATE_DIM,
            "action_dim": _ACTION_DIM,
            "horizon": _HORIZON,
            "proprio_indices": _PROPRIO_IDX.tolist(),
            "joint_delta_reference": _REF.tolist(),
        }, _synthetic_pre()
    if not _REAL_CRITIC.exists():
        pytest.skip("the real critic is not on this machine; the synthetic case covers the property")
    cfg = json.loads((_REAL_CRITIC / "config.json").read_text())
    spec = cfg["input_spec"]
    pre = critic_preproc.Pi05Preproc(
        ref=np.asarray(spec["joint_delta_reference"], np.int64),
        stats=critic_preproc.load_norm_stats(_REAL_CRITIC / spec["norm_stats_file"]),
        use_quantiles=bool(spec["use_quantiles"]),
        delta=spec["delta_mode"] == "joint",
    )
    return spec, pre


def _train_state(pre, pidx, raw):
    """train_patch_critic_cached.py:340-342 — normalize the FULL state, THEN slice."""
    s = pre.state(raw)
    return s if pidx is None else s[..., pidx]


def test_serving_builds_the_proprio_the_trainer_built(spec_and_pre):
    spec, pre = spec_and_pre
    pidx = np.asarray(spec["proprio_indices"], np.int64)
    raw = np.random.default_rng(0).normal(size=spec["state_dim"]).astype(np.float32)

    # the serving wrapper's expression (patch_critic_policy.py, pi05 branch)
    critic_state = pre.state(raw)
    serving = critic_state if pidx is None else critic_state[pidx]

    np.testing.assert_array_equal(serving, _train_state(pre, pidx, raw))


def test_slicing_first_is_a_different_answer_and_nothing_would_raise(spec_and_pre):
    """The bug itself. proprio_indices point into the 42-wide state, so slicing first pairs
    channels 21..27 with the first-14 statistics -- a shape-valid, in-range, wrong answer."""
    spec, pre = spec_and_pre
    pidx = np.asarray(spec["proprio_indices"], np.int64)
    raw = np.random.default_rng(1).normal(size=spec["state_dim"]).astype(np.float32)

    right = _train_state(pre, pidx, raw)
    wrong = pre.state(raw[pidx])  # slice first, then normalize

    assert right.shape == wrong.shape, "identical shapes -- which is why nothing catches it"
    assert not np.allclose(right, wrong), "and different numbers, which is why it matters"
    assert np.max(np.abs(wrong)) < 100, "still 'in range', so a sanity check on Q would pass"

    # And this is why it survived review. The proprio indices are [0..6, 21..27] -- two arms. Slice
    # first and the LEFT arm still lands on statistics 0..6, exactly right, difference identically
    # zero; only the right arm gets paired with 7..13. So a spot check, a plot of one arm, or any
    # eyeball on the first half of the vector shows a perfect match.
    d = np.abs(right - wrong)
    assert np.all(d[:7] == 0.0), "the left arm is correct even under the bug"
    assert np.any(d[7:] > 0.5), "and the right arm is wrong by a lot"


def test_the_action_delta_is_taken_against_the_whole_state(spec_and_pre):
    """train_patch_critic_cached.py:339 takes the delta before the state is normalized or sliced,
    because joint_delta_reference indexes 21..27 of the 42-wide state. Serving does the same
    (`self._pre.actions(robot_actions, state)` with the raw full state). Handing it the SLICED
    state would index the wrong joints -- or fall off the end."""
    spec, pre = spec_and_pre
    pidx = np.asarray(spec["proprio_indices"], np.int64)
    raw = np.random.default_rng(2).normal(size=spec["state_dim"]).astype(np.float32)
    chunk = np.random.default_rng(3).normal(size=(2, spec["horizon"], spec["action_dim"])).astype(np.float32)

    full = pre.actions(chunk, raw)
    assert np.all(np.isfinite(full))

    ref = np.asarray(spec["joint_delta_reference"], np.int64)
    assert int(ref.max()) >= len(pidx), "the reference reaches past the sliced width, so FULL is required"


def test_the_two_normalizations_are_not_interchangeable(spec_and_pre):
    """state and actions have separate statistics; using one for the other is shape-valid too."""
    _spec, pre = spec_and_pre
    a_stats, s_stats = pre.stats["actions"], pre.stats["state"]
    a_q01 = np.asarray(a_stats["q01"][:14], float)
    s_q01 = np.asarray(s_stats["q01"][:14], float)
    assert not np.allclose(a_q01, s_q01), "distinct statistics, so mixing them silently rescales"
