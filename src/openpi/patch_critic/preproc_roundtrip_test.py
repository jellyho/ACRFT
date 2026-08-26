"""Scoring a critic through physical units instead of assuming shared statistics.

The sampler's chunks are normalized by the POLICY's norm stats; the critic was trained under its
own. The wrapper used to hand the policy's arrays straight to the critic and refuse to load when
the two sets of numbers differed. It now decodes to physical units and re-normalizes with the
critic's own preprocessing, which is correct either way -- but only if that round trip is the
identity when the stats DO agree. That is what these check, with the real YAM stats on CPU.
"""

import json
import pathlib

import numpy as np
import pytest

from openpi.patch_critic import preproc as critic_preproc

_HOME = pathlib.Path.home() / "hf_utils_downloads"
_RLT = _HOME / "acrft-yam-critics/patch_critic_yam_s347_fixed_200k/pi05_norm_stats.json"
_BC = _HOME / "pi05_yam_lego_taxi_bc_s300_h30/200000/assets/jellyho/yam_lego_taxi/norm_stats.json"

pytestmark = pytest.mark.skipif(
    not (_RLT.exists() and _BC.exists()), reason="needs the downloaded YAM checkpoints"
)

# YAM: 6 joints + gripper per arm; the grippers (-1) stay absolute.
_REF = [0, 1, 2, 3, 4, 5, -1, 21, 22, 23, 24, 25, 26, -1]


def _pre(path):
    return critic_preproc.Pi05Preproc(
        ref=np.asarray(_REF, np.int64),
        stats=critic_preproc.load_norm_stats(path),
        use_quantiles=True,
        delta=True,
    )


def _unnormalize_actions(pre, norm, state):
    """The policy's output transform, in miniature: normalized delta -> absolute joint target."""
    s = pre.stats["actions"]
    q01, q99 = np.asarray(s["q01"][:14], np.float64), np.asarray(s["q99"][:14], np.float64)
    delta = (np.asarray(norm, np.float64) + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    out = delta.copy()
    for i, r in enumerate(_REF):
        if r >= 0:
            out[..., i] += np.asarray(state, np.float64)[..., None, r]
    return out


def test_round_trip_is_the_identity_when_stats_agree():
    """Decode with stats X, re-encode with stats X: the critic sees exactly what it used to."""
    pre = _pre(_RLT)
    rng = np.random.default_rng(0)
    state = rng.normal(size=42)
    norm_chunk = rng.uniform(-1, 1, size=(8, 30, 14))

    absolute = _unnormalize_actions(pre, norm_chunk, state)
    back = pre.actions(absolute, state)
    np.testing.assert_allclose(back, norm_chunk, rtol=0, atol=2e-5)


def test_the_round_trip_is_what_moves_when_stats_differ():
    """Decode with the POLICY's stats, re-encode with the CRITIC's. The result must differ from the
    old pass-through by exactly the disagreement between the two -- that difference is the bug the
    conversion removes, not noise it adds."""
    policy_pre, critic_pre = _pre(_BC), _pre(_RLT)
    rng = np.random.default_rng(1)
    state = rng.normal(size=42)
    norm_chunk = rng.uniform(-1, 1, size=(4, 30, 14))

    absolute = _unnormalize_actions(policy_pre, norm_chunk, state)
    converted = critic_pre.actions(absolute, state)

    assert not np.allclose(converted, norm_chunk, atol=1e-4), "stats differ, so the values must move"
    # ...but only by the size of the disagreement: these two arms are the same dataset estimated
    # twice, so the shift stays small. A large move here would mean the decode path itself is wrong.
    assert np.max(np.abs(converted - norm_chunk)) < 0.2

    # And the conversion is exact, not approximate: re-encoding with the SAME stats it was decoded
    # under returns the original, so all of the movement is the stats disagreement.
    np.testing.assert_allclose(policy_pre.actions(absolute, state), norm_chunk, rtol=0, atol=2e-5)


def test_state_is_normalized_before_it_is_sliced():
    """proprio_indices point into the 42-wide state. Slicing first pairs those channels with the
    first-14 statistics, which is a silent mis-scaling rather than an error."""
    pre = _pre(_RLT)
    state = np.random.default_rng(2).normal(size=42)
    idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27], np.int64)

    correct = pre.state(state)[idx]
    wrong = pre.state(state[idx])
    assert not np.allclose(correct, wrong), "the two orders must not be confusable"


def test_the_first_C_steps_of_a_long_chunk_are_a_C_step_chunk():
    """Why an h30 critic can score an h50 policy: the joint delta at step k is taken against the
    same base state whatever the chunk length, so the first 30 steps of a 50-step proposal are the
    same object the critic was fitted on -- and the critic's own (h30) action statistics are the
    right ones to re-normalize them with. The h50 statistics are wider only because they pool steps
    31..50, which is exactly the part that is not being scored."""
    pre = _pre(_RLT)
    rng = np.random.default_rng(3)
    state = rng.normal(size=42)
    absolute_50 = rng.normal(size=(4, 50, 14)) * 0.1 + state[None, None, :14]

    long_scored = pre.actions(absolute_50, state)[:, :30]
    short_scored = pre.actions(absolute_50[:, :30], state)
    np.testing.assert_array_equal(long_scored, short_scored)
