"""The cache must find statistics by what they were computed on, not by where someone filed them."""

import json
import pathlib

import numpy as np
import pytest

from openpi.shared import normalize as _normalize
from openpi.training import norm_stats_cache as nsc


class _DC:
    def __init__(self, episodes, repo_id="jellyho/yam_lego_taxi"):
        self.episodes, self.repo_id = episodes, repo_id


def _write(d: pathlib.Path, prov: dict, scale: float = 1.0):
    d.mkdir(parents=True, exist_ok=True)
    _normalize.save(d, {"actions": _normalize.NormStats(mean=np.zeros(4) + scale, std=np.ones(4))})
    (d / "provenance.json").write_text(json.dumps(prov))


def test_prose_provenance_is_matched_on_its_episode_count(tmp_path):
    """The real YAM lego h30 asset describes its subset in a sentence; the count is what we can check.

    This is the case that forced launchers to hard-code --data.assets.asset-id, and a hard-coded id
    is what went stale and killed the alpha-Flow 100k run.
    """
    _write(
        tmp_path / "pi05_yam_lego_taxi_rlt" / "jellyho" / "yam_lego_taxi_s300h30",
        {"computed_on": {"repo_id": "jellyho/yam_lego_taxi", "episodes_subset": "success-only (300/347)"}},
    )
    dc = _DC(tuple(range(300)))
    hit = nsc._search(tmp_path, dc, nsc.stats_key(dc, 30), 30)
    assert hit is not None
    assert hit[0] == 1
    assert "yam_lego_taxi_s300h30" in str(hit[2])


def test_a_different_subset_size_is_not_a_match(tmp_path):
    _write(
        tmp_path / "cfg" / "jellyho" / "yam_lego_taxi",
        {"computed_on": {"repo_id": "jellyho/yam_lego_taxi", "episodes_subset": "success-only (300/347)"}},
    )
    dc = _DC(tuple(range(347)))
    assert nsc._search(tmp_path, dc, nsc.stats_key(dc, 30), 30) is None


def test_a_stated_horizon_that_disagrees_is_not_a_match(tmp_path):
    _write(
        tmp_path / "cfg" / "a",
        {"computed_on": {"repo_id": "jellyho/yam_lego_taxi", "episodes_subset": "all", "action_horizon": 50}},
    )
    dc = _DC(None)
    assert nsc._search(tmp_path, dc, nsc.stats_key(dc, 30), 30) is None


def test_two_assets_claiming_the_same_data_but_holding_different_numbers_refuse_to_resolve(tmp_path):
    """Silently picking one would reintroduce exactly the mis-normalization this cache exists to stop."""
    prov = {"computed_on": {"repo_id": "jellyho/yam_lego_taxi", "episodes_subset": "success-only (300/347)"}}
    _write(tmp_path / "cfg_a" / "x", prov, scale=1.0)
    _write(tmp_path / "cfg_b" / "x", prov, scale=2.0)
    dc = _DC(tuple(range(300)))
    with pytest.raises(ValueError, match="DIFFERENT numbers"):
        nsc._search(tmp_path, dc, nsc.stats_key(dc, 30), 30)


def test_identical_numbers_filed_twice_resolve_without_complaint(tmp_path):
    prov = {"computed_on": {"repo_id": "jellyho/yam_lego_taxi", "episodes_subset": "success-only (300/347)"}}
    _write(tmp_path / "cfg_a" / "x", prov)
    _write(tmp_path / "cfg_b" / "x", prov)
    dc = _DC(tuple(range(300)))
    assert nsc._search(tmp_path, dc, nsc.stats_key(dc, 30), 30) is not None
