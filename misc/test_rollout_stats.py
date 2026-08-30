"""Rollout statistics: the arithmetic, not the reading.

Everything here is a pure function of arrays, so it is testable without a dataset -- which matters
because the failure mode of a stats tool is a number that is wrong rather than absent.
"""

import numpy as np
import pytest

from misc.rollout_stats import boundary_jumps, chunk_starts, mean_ci


def test_boundaries_are_where_the_chunk_id_changes():
    """Same definition the renderer draws with, so a table and a video never disagree about how
    many replans an episode had."""
    assert chunk_starts(np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])) == [0, 3, 5]
    assert chunk_starts(np.array([[0], [1], [1]])) == [0, 1]  # column may arrive shaped [T, 1]


def test_a_constant_column_is_no_information_not_one_huge_chunk():
    """A run recorded while provenance was written as a constant zero has the column and nothing in
    it. Taking it at face value reports one episode-long chunk, which reads as a real measurement."""
    assert chunk_starts(np.zeros(500)) == []
    assert chunk_starts(np.array([])) == []
    assert chunk_starts(None) == []


def test_jumps_are_split_at_the_step_into_a_new_reply():
    """The discontinuity is between the last action of one chunk and the first of the next, so it
    belongs to the step INTO the boundary frame -- off by one and the splice hides in the
    within-chunk pile, which is exactly where it would look harmless."""
    actions = np.array([[0.0], [0.1], [0.2], [5.0], [5.1]])  # a big step entering frame 3
    at_b, within = boundary_jumps(actions, starts=[0, 3])
    assert at_b.tolist() == pytest.approx([4.8])
    assert within.tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_jumps_use_the_worst_joint_not_the_average():
    """One joint stepping 0.5 rad while thirteen hold still is a jolt; averaged over 14 dims it
    reads as 0.036 and disappears."""
    actions = np.array([[0.0] * 14, [0.5] + [0.0] * 13])
    _, within = boundary_jumps(actions, starts=[0])
    assert within.tolist() == pytest.approx([0.5])


def test_one_episode_reports_no_interval_rather_than_zero():
    """A single run has no spread. Printing 0 would read as perfect agreement across runs."""
    assert mean_ci([4.0]) == {"n": 1, "mean": 4.0, "ci": None}
    assert mean_ci([]) == {"n": 0, "mean": None, "ci": None}


def test_the_interval_is_over_episodes_and_uses_t():
    """t, not 1.96: with 5 episodes the normal approximation understates the interval by ~30%."""
    m = mean_ci([10.0, 12.0, 14.0, 16.0, 18.0])
    assert m["n"] == 5
    assert m["mean"] == pytest.approx(14.0)
    assert m["ci"] == pytest.approx(2.78 * (np.std([10, 12, 14, 16, 18], ddof=1) / np.sqrt(5)), rel=1e-6)


def test_missing_values_are_dropped_not_counted_as_zero():
    """An episode that recorded no infer_ms must not drag the mean toward zero."""
    m = mean_ci([100.0, None, 200.0, float("nan")])
    assert m["n"] == 2 and m["mean"] == pytest.approx(150.0)


def test_kstar_is_the_choice_and_the_chunk_is_what_ran():
    """`critic_best_prefix` is what the critic committed to; the realized chunk can be shorter,
    cut off by the end of an episode or an intervention. Reporting only the realized length blames
    the critic for those truncations, and `cut short` is what separates them."""
    import numpy as np

    from misc.rollout_stats import chunk_starts

    # three replans: the critic asked for 30, 10, 30 steps (k* = 6, 2, 6 at macro 5)...
    chunk_ids = np.array([0] * 30 + [1] * 10 + [2] * 12)  # ...but the last ran only 12
    starts = chunk_starts(chunk_ids)
    assert starts == [0, 30, 40]
    lengths = np.diff([*starts, len(chunk_ids)])
    kstar = np.array([6, 2, 6])
    assert lengths.tolist() == [30, 10, 12]
    assert float(np.mean(lengths < kstar * 5)) == pytest.approx(1 / 3)


def test_drift_is_measured_against_the_twin_and_scaled_by_the_spread():
    """action_samples is laid out [executed, unsteered twin, uncond...]. The twin shares the
    executed draw's noise, so their distance IS the steering displacement; the unconditional spread
    is what turns it into a number with a scale. 0.1 rad is small inside a spread of 0.5 and
    enormous inside 0.02, and the raw value cannot tell you which."""
    import numpy as np

    rng = np.random.default_rng(0)
    uncond = rng.normal(size=(4, 8, 14)) * 0.5
    executed = uncond[:, 0] + 0.1
    twin = uncond[:, 0]
    samples = np.concatenate([executed[:, None], twin[:, None], uncond], axis=1)

    drift = np.abs(samples[:, 0] - samples[:, 1]).max(axis=-1)
    spread = samples[:, 2:].std(axis=1).mean(axis=-1)
    assert np.allclose(drift, 0.1, atol=1e-9)
    assert float(np.median(drift / spread)) < 0.5, "inside its own spread"

    tight = np.concatenate([executed[:, None], twin[:, None], uncond * 0.02], axis=1)
    tight_spread = tight[:, 2:].std(axis=1).mean(axis=-1)
    assert float(np.median(drift / tight_spread)) > 5.0, "the same 0.1 rad, far outside a tight one"


def test_the_advantage_tripwire_separates_the_two_critics():
    """0.204 is the fixed critic's reference; 0.277 is what the raw-proprio bug produced, and Q
    stayed in range the whole time. A deploy run near 0.28 is a signal, not a result."""
    assert abs(0.204 - 0.277) > 0.05, "the two distributions are far enough apart to act on"
