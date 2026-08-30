"""Figures from rollout statistics: that they draw, and that they do not mislead."""

import json
import pathlib

import pytest

pytest.importorskip("matplotlib")

from misc.stats_plots import _short, commitment_distribution, commitment_length, splice  # noqa: E402


def _result(name, hist, chunk_mean=14.0, ci=2.0):
    return {
        "repo_id": name,
        "episodes": 3,
        "per_episode": [],
        "aggregate": {
            "chunk_hist": hist,
            "chunk_mean": {"n": 3, "mean": chunk_mean, "ci": ci},
            "boundary_jump_p95": {"n": 3, "mean": 0.3, "ci": 0.05},
            "within_jump_p95": {"n": 3, "mean": 0.07, "ci": 0.01},
        },
    }


def test_names_lose_only_the_shared_prefix():
    assert _short("yam_s300_h30_g5_tau9_min") == "g5_tau9_min"
    assert _short("lerobot_rollout/yam_s300_rel_200k_fixed") == "fixed"
    assert _short("something_else") == "something_else"


def test_the_three_figures_draw(tmp_path):
    results = [_result("yam_s300_h30_fixed", {5: 2, 30: 400}), _result("yam_s300_h30_g5", {5: 300, 30: 100})]
    for fn, name in ((commitment_distribution, "d"), (commitment_length, "l"), (splice, "s")):
        out = fn(results, tmp_path / f"{name}.png", "t")
        assert out.exists() and out.stat().st_size > 1000


def test_json_string_keys_are_handled(tmp_path):
    """Stats round-trip through --json, where int dict keys come back as strings. Indexing with the
    wrong type raised KeyError on the first real run."""
    results = json.loads(json.dumps([_result("a", {5: 10, 30: 90})]))
    assert commitment_distribution(results, tmp_path / "x.png").exists()


def test_shares_still_sum_to_100_with_an_other_bucket(tmp_path):
    """Rare commitment lengths (a chunk cut short by the end of an episode) are bucketed, not
    dropped -- a distribution that quietly sums to 80% invites reading the gap as something real."""
    from misc.stats_plots import _style

    _style()
    hist = {5: 500, 30: 400, 2: 3, 7: 2, 13: 1}  # three odd lengths, each under 2%
    total = sum(hist.values())
    major = [k for k, v in hist.items() if 100 * v / total >= 2.0]
    other = sum(v for k, v in hist.items() if k not in major)
    assert 100 * (sum(hist[k] for k in major) + other) / total == pytest.approx(100.0)
    assert other == 6


def test_macro_choice_needs_an_adaptive_run(tmp_path):
    """A fixed critic has one group, so there is no choice to plot -- better to say so than to draw
    a single bar at k=1 and call it a distribution."""
    from misc.stats_plots import macro_choice

    with pytest.raises(SystemExit, match="no adaptive run"):
        macro_choice([_result("a", {30: 100})], tmp_path / "m.png")


def test_macro_choice_draws_per_episode_points(tmp_path):
    """A mean k* of 2.9 hides episodes running 1.8 to 3.6, so each episode is a point behind the
    bar."""
    from misc.stats_plots import macro_choice

    r = _result("yam_s300_h30_g5", {5: 300, 30: 100})
    r["aggregate"] |= {"kstar_hist": {1: 300, 6: 100}, "kstar_mean": {"n": 2, "mean": 2.9, "ci": 0.4}, "macro": 5}
    r["per_episode"] = [{"episode": 0, "kstar_hist": {1: 150, 6: 20}}, {"episode": 1, "kstar_hist": {1: 150, 6: 80}}]
    assert macro_choice([r], tmp_path / "m.png").exists()
