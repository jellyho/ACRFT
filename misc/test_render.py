"""Rendering logic: the parts that are pure functions of a recording.

The end-to-end tests that used to live here built a dataset with the ROBOT repo's recorder
(AsyncDatasetWriter) and rendered it. This tool no longer depends on that repo, so they went with
it; the whole pipeline is instead checked against real recordings -- most recently by rendering
the same episode from both copies and diffing the frames (identical, max |difference| 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pathlib

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("mujoco")  # forward kinematics
pytest.importorskip("mink")
pytest.importorskip("imageio")


CAMERAS = ("wrist_left", "wrist_right", "agentview")
IMG = (64, 64, 3)
HORIZON = 5
CANDIDATES = 8
FRAMES = HORIZON * 4  # four whole replans


def _args(dataset: Path, out: Path, **over) -> argparse.Namespace:
    base = dict(
        repo_id="t/samples",
        root=str(dataset.parent),
        # A path that does not exist, so the loaders fail-soft to no calibrated extrinsics (CAD
        # wrist, raw agentview) -- keeps the test hermetic instead of discovering the repo config.
        config=str(dataset.parent / "no_such_config.yaml"),
        source="samples",
        episode=0,
        wrists=["left", "right"],
        agentview_arms=["left", "right"],
        horizon=HORIZON,
        candidates=CANDIDATES,
        # The fixtures record no critic_scores, so the panel is absent either way; the field has to
        # exist because render() consults it.
        no_value_plot=False,
        no_chunk_plot=False,
        replans=0,
        hold=2,
        height=180,
        fps=10,
        out=str(out),
        fx=430.0,
        fy=430.0,
        cx=320.0,
        cy=240.0,
        agent_fx=390.0,
        agent_fy=390.0,
        agent_cx=320.0,
        agent_cy=240.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------------------- #
# The render
# --------------------------------------------------------------------------------------- #


def test_the_fan_is_coloured_by_the_critic_s_own_ranking():
    """A value-guided run records what the critic thought of each candidate, so the fan can show
    the value landscape the decision was made on instead of a spread of look-alike options.

    Normalised per replan: the absolute numbers are arbitrary (cost-to-goal runs to -2777), the
    useful question is which candidate the critic preferred here."""
    from misc.render_deploy_samples import _value_color

    worst, best = _value_color(-20.0, -20.0, -5.0), _value_color(-5.0, -20.0, -5.0)
    assert worst != best
    assert best[0] > worst[0], "the preferred candidate should read warmer"
    assert worst[2] > best[2], "...and the rejected one colder"
    # A replan the critic saw nothing to choose between must not paint a false gradient.
    flat = {_value_color(v, -7.0, -7.0) for v in (-7.0, -7.0)}
    assert len(flat) == 1


def test_the_executed_candidate_is_the_one_the_critic_picked():
    """`critic_choice` is read from the recording: highlighting index 0 would draw the wrong path
    as executed on any run where the critic picked something else."""

    from misc.render_deploy_samples import _load_critic

    class _Reader:
        def get_extra(self, ep, frame, key, shape):
            return np.array([-9.0, -3.0, -7.0]) if key == "critic_scores" else None

        def get_scalar(self, ep, frame, key):
            return 1.0 if key == "critic_choice" else None

    scores, chosen = _load_critic(_Reader(), 0, 0, 3)
    assert chosen == 1 and float(scores[chosen]) == -3.0

    class _Plain(_Reader):
        def get_extra(self, ep, frame, key, shape):
            return None

    assert _load_critic(_Plain(), 0, 0, 3) == (None, 0)


def test_a_constant_chunk_index_is_treated_as_no_information():
    """Rollouts recorded while the provenance was written as a constant 0 carry the column but
    nothing in it. Believing it would draw the entire episode as one chunk."""
    from misc.render_deploy_samples import _recorded_chunk_starts

    class _Reader:
        def __init__(self, values):
            self.values = values

        def has_feature(self, key):
            return True

        def get_scalar(self, ep, frame, key):
            return self.values[frame]

    assert _recorded_chunk_starts(_Reader([0.0] * 40), 0, 40) is None
    assert _recorded_chunk_starts(_Reader([0.0] * 10 + [1.0] * 10), 0, 20) == [0, 10]


def test_the_value_curve_is_painted_once_and_only_the_cursor_moves(qapp_free=None):
    """The curve is the same picture at every frame of a 9000-frame render; repainting it per
    frame would multiply the cost by the length of the episode for nothing."""

    from misc.render_deploy_samples import _value_panel
    from misc.render_deploy_samples import _value_panel_base

    chosen = np.array([-10.0, -9.0, -9.0, -6.0])
    series = (chosen, chosen - 2.0, chosen + 2.0)
    base = _value_panel_base(series, 320, 180)
    first = _value_panel(base, 0, len(chosen), float(chosen[0]))
    last = _value_panel(base, 3, len(chosen), float(chosen[3]))
    assert first.shape == last.shape == (180, 320, 3)
    assert not np.array_equal(first, last), "the cursor must move"
    # The base image is untouched by drawing a cursor on a copy.
    again = _value_panel(base, 0, len(chosen), float(chosen[0]))
    assert np.array_equal(first, again)


def test_no_critic_no_value_curve():
    """A plain rollout has no critic_scores; asking for the panel must not invent one."""
    from misc.render_deploy_samples import _value_series

    class _Reader:
        def has_feature(self, key):
            return False

    assert _value_series(_Reader(), 0, 10, 8) is None


def test_the_value_strip_keeps_the_frame_encodable():
    """h264 with yuv420p subsamples chroma 2x2 and refuses an odd frame dimension. The strip sets
    the frame's height together with the cameras, and 360 + 151 = 511 killed the encode."""

    from misc.render_deploy_samples import _value_panel_base

    chosen = np.array([-3.0, -2.0, -4.0])
    for panel_h in (360, 240, 300):
        height = 2 * round(panel_h * 0.42 / 2)
        assert height % 2 == 0
        img, *_ = _value_panel_base((chosen, chosen - 1, chosen + 1), 640, height)
        assert (panel_h + img.size[1]) % 2 == 0, "cameras + strip must stay even"


def test_chunk_lengths_come_from_the_whole_run_not_the_rendered_part():
    """--replans limits what is DRAWN, not what the run did. Deriving the lengths from the
    truncated list makes the last kept chunk absorb the entire remaining episode -- a 30-step
    reply plotted as 237."""

    from misc.render_deploy_samples import _chunk_series

    lengths = _chunk_series([0, 30, 60, 90], 120)
    assert list(np.unique(lengths)) == [30.0]
    # An adaptive run: each reply's own length, held across the frames it owns.
    lengths = _chunk_series([0, 5, 20], 30)
    assert lengths[0] == 5 and lengths[5] == 15 and lengths[-1] == 10


def test_a_strip_without_a_band_is_still_drawn():
    """The chunk strip is a plain line -- no spread to shade. Passing None for the band must not
    fall back to shading the line against itself."""

    from misc.render_deploy_samples import _value_panel
    from misc.render_deploy_samples import _value_panel_base

    line = np.array([30.0, 30.0, 12.0, 12.0])
    base = _value_panel_base((line, None, None), 320, 96, title="chunk length", fmt=".0f")
    img = _value_panel(base, 2, len(line), float(line[2]))
    assert img.shape == (96, 320, 3)


def _straight_paths(n_candidates=3, steps=30):
    """Candidate paths as plain pixel polylines, which is all _draw_fan consumes."""

    return [np.stack([np.linspace(10, 300, steps), np.full(steps, 40.0 + 25 * c)], axis=1) for c in range(n_candidates)]


def test_macro_group_boundaries_are_marked_on_the_chosen_path():
    """The granularity the commitment could stop at. Only on the chosen path -- putting them on
    every candidate would bury the one decision the picture is about."""

    from misc.render_deploy_samples import _draw_fan

    blank = np.zeros((120, 320, 3), np.uint8)
    plain = _draw_fan(blank, _straight_paths(), chosen=0)
    dotted = _draw_fan(blank, _straight_paths(), chosen=0, macro=5)
    assert not np.array_equal(plain, dotted), "macro=5 must mark the boundaries"
    # The dots are near-white (alpha-composited, so not pure 255); the coloured path is not.
    assert (dotted > 200).all(axis=2).sum() > (plain > 200).all(axis=2).sum()


def test_the_uncommitted_tail_is_drawn_faint():
    """Adaptive executes only a prefix of the winning chunk. Drawing the rest at full weight would
    claim the arm went somewhere it never did."""

    from misc.render_deploy_samples import _draw_fan

    blank = np.zeros((120, 320, 3), np.uint8)
    whole = _draw_fan(blank, _straight_paths(), chosen=0)
    part = _draw_fan(blank, _straight_paths(), chosen=0, committed=10)
    assert not np.array_equal(whole, part)
    # The tail is dimmer, so the committed run keeps more bright pixels than the partial one.
    assert (whole.sum(axis=2) > 300).sum() > (part.sum(axis=2) > 300).sum()


def test_a_recording_without_the_commitment_columns_draws_as_before():
    """bon runs and every older recording: no macro, nothing to dim."""
    from misc.render_deploy_samples import _load_commitment

    class _Reader:
        def get_scalar(self, ep, frame, key):
            return None

    assert _load_commitment(_Reader(), 0, 0) == (None, None)

    class _Adaptive(_Reader):
        def get_scalar(self, ep, frame, key):
            return {"critic_macro": 5.0, "critic_best_prefix": 2.0}.get(key)

    # committed = (best_prefix + 1) * macro -- three groups of five.
    assert _load_commitment(_Adaptive(), 0, 0) == (5, 15)


def test_sequential_decoding_matches_random_access():
    """Rendering walks the episode in order, so the decoder does too. Seeking per frame cost 221 ms
    for three cameras against ~1 ms streamed -- the whole cost of a render -- but it has to return
    the same pictures, or the speedup would be a different video."""
    import numpy as np

    from misc.dataset_reader import DatasetReader, SequentialImages

    root = "/home/rllab4/lerobot_rollout/yam_s300_rel_200k_g5"
    if not pathlib.Path(root, "meta").exists():
        pytest.skip("needs a recorded rollout on this machine")
    reader = DatasetReader("lerobot_rollout/yam_s300_rel_200k_g5", "/home/rllab4/lerobot_rollout")
    reader.load()
    stream = SequentialImages(root, 4)
    try:
        for index in (0, 17, 60):
            seeked, streamed = reader.get_images(4, index), stream.frame(index)
            assert set(seeked) == set(streamed)
            for camera, image in seeked.items():
                assert np.array_equal(image, streamed[camera]), camera
    finally:
        stream.close()


def test_the_stream_is_forward_only():
    """Rewinding is exactly the seek this exists to avoid, so it refuses rather than quietly
    paying for one."""
    from misc.dataset_reader import SequentialImages

    root = "/home/rllab4/lerobot_rollout/yam_s300_rel_200k_g5"
    if not pathlib.Path(root, "meta").exists():
        pytest.skip("needs a recorded rollout on this machine")
    stream = SequentialImages(root, 4)
    try:
        stream.frame(5)
        with pytest.raises(ValueError, match="forward-only"):
            stream.frame(4)
    finally:
        stream.close()
