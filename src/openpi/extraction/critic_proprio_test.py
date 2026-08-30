"""The proprio invariant: normalize the FULL state, THEN slice.

Behavioural, not a source grep: the order is what broke, so the test compares against the wrong
order and asserts they differ, rather than asserting the right words appear in the file.
"""

import numpy as np

from openpi.extraction.critic_q import critic_proprio
from openpi.patch_critic import preproc as critic_preproc

# 42-wide state whose proprio channels live at 0..6 and 21..27, as the YAM critics were trained
IDX = np.array([0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27])


def _pre():
    rng = np.random.default_rng(0)
    lo = rng.uniform(-3, -1, 42)
    hi = rng.uniform(1, 3, 42)
    stats = {
        "state": {"q01": lo, "q99": hi, "mean": np.zeros(42), "std": np.ones(42)},
        "actions": {"q01": lo[:14], "q99": hi[:14], "mean": np.zeros(14), "std": np.ones(14)},
    }
    return critic_preproc.Pi05Preproc(ref=np.full(14, -1), stats=stats, use_quantiles=True, delta=True)


def test_normalizes_the_full_state_before_slicing():
    pre = _pre()
    raw = np.random.default_rng(1).uniform(-2, 2, (5, 42)).astype(np.float32)
    got = critic_proprio(pre, IDX, raw)
    assert np.allclose(got, pre.state(raw)[:, IDX])


def test_slicing_first_is_a_different_answer():
    """Slice-then-normalize pairs channels 21..27 with the first-14 statistics. It returns numbers
    in a plausible range, which is exactly why the original bug survived: nothing raises."""
    pre = _pre()
    raw = np.random.default_rng(2).uniform(-2, 2, (5, 42)).astype(np.float32)
    wrong = pre.state(raw[:, IDX])  # normalize the 14-wide slice against state stats[:14]
    assert not np.allclose(critic_proprio(pre, IDX, raw), wrong)


def test_raw_units_critic_passes_the_state_through():
    raw = np.random.default_rng(3).uniform(-2, 2, (5, 42)).astype(np.float32)
    assert np.array_equal(critic_proprio(None, IDX, raw), raw[:, IDX])
    assert np.array_equal(critic_proprio(None, None, raw), raw)
