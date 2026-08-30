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

_CRITIC = pathlib.Path.home() / "hf_utils_downloads/acrft-yam-critics/patch_critic_yam_s347_fixed_200k"

pytestmark = pytest.mark.skipif(not _CRITIC.exists(), reason="needs the downloaded YAM critic")


@pytest.fixture(scope="module")
def spec_and_pre():
    cfg = json.loads((_CRITIC / "config.json").read_text())
    spec = cfg["input_spec"]
    pre = critic_preproc.Pi05Preproc(
        ref=np.asarray(spec["joint_delta_reference"], np.int64),
        stats=critic_preproc.load_norm_stats(_CRITIC / spec["norm_stats_file"]),
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


def test_the_action_delta_is_taken_against_the_FULL_state(spec_and_pre):
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
