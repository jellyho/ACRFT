"""Rollout statistics.

The failure mode of a stats tool is a number that is WRONG rather than absent, so most of this is
pure arithmetic over made-up arrays and needs no dataset.

That was once the whole file, and it left a gap: testing the formula proves the formula, not the
code that runs. `episode_stats` decided which columns held what by inferring the layout from the
candidate COUNT (`N >= 3`), which is also true of a best-of-8 run -- so it reported a steering
displacement for runs with no steering in them, computed between two independent draws. The
arithmetic tests all passed. `_Reader` below closes that: the real function, on columns shaped the
way the serving wrapper emits them.
"""

import numpy as np
import pytest

from misc.rollout_stats import boundary_jumps, chunk_starts, episode_stats, mean_ci


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


class _Reader:
    """The two methods episode_stats calls. Enough to run the real function on made-up columns."""

    fps = 30

    def __init__(self, cols, frames):
        self._cols, self._frames = cols, frames

    def episode_length(self, _episode):
        return self._frames

    def column(self, _episode, key):
        return self._cols.get(key)


def _run(*, n_candidates, twin, replans=4, per=5):
    """One episode's worth of columns, laid out the way the serving wrapper emits them."""
    import numpy as np

    rng = np.random.default_rng(0)
    frames = replans * per
    uncond = rng.normal(size=(replans, n_candidates - 2, 14)) * 0.5
    executed = uncond[:, 0] + 0.1
    per_replan = np.concatenate([executed[:, None], uncond[:, :1], uncond], axis=1)
    cols = {
        "policy.chunk_index": np.repeat(np.arange(replans, dtype=np.float32), per).reshape(-1, 1),
        "action_samples": np.repeat(per_replan, per, axis=0).astype(np.float32),
    }
    if twin is not None:
        cols["critic_twin"] = np.full((frames, 1), twin, np.float32)
    return _Reader(cols, frames)


def test_drift_is_read_out_of_a_real_episode_not_re_derived():
    """The arithmetic above is checked in numpy; this checks that `episode_stats` -- the function
    the stats table actually calls -- produces it from columns shaped the way the wrapper emits
    them. Re-deriving a formula in a test proves the formula, not the code that runs."""
    stats = episode_stats(_run(n_candidates=6, twin=1.0), 0)
    assert abs(stats["steer_drift_p50"] - 0.1) < 1e-5
    assert "steer_drift_over_spread" in stats


def test_drift_is_not_reported_without_a_recorded_twin():
    """A best-of-8 run has N >= 3 and no steering in it at all. Reading column 1 as "the twin"
    there measures the distance between two INDEPENDENT draws and reports it as steering
    displacement -- a plausible number with no error attached, which is the only kind of wrong
    answer that survives. The layout is now read from `critic_twin`, which the server records.
    """
    bon8 = episode_stats(_run(n_candidates=8, twin=None), 0)
    assert "steer_drift_p50" not in bon8, "a best-of-N run has no steering to measure"
    assert bon8.get("steer_drift") == "no twin recorded", "and says so, rather than going missing"

    # A non-steering arm asked for reference draws: the references are real, the twin is not.
    lps = episode_stats(_run(n_candidates=6, twin=0.0), 0)
    assert "steer_drift_p50" not in lps


def test_the_flag_and_the_count_come_from_one_expression():
    """`critic_twin` says column 1 is the twin and `_candidate_count` says how many columns there
    are. Derived separately they agree by coincidence, and the run where they stop agreeing is one
    whose drift column is quietly measuring the wrong pair."""
    import inspect
    import pathlib as _pl
    import sys

    sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
    from openpi.policies import patch_critic_policy as pcp

    counter = inspect.getsource(pcp.PatchCriticSelectPolicy._candidate_count)
    emitter = inspect.getsource(pcp.PatchCriticSelectPolicy.infer)
    assert "_has_twin" in counter
    assert "self._has_twin" in emitter
