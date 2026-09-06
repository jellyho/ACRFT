"""The cache key, and the one case it must not touch.

Stats are found by NAME today: asset_id falls back to the repo id, so one config trained with and
without --data.success-only resolves to the same file. Those two differ by up to 2.5% in action q99
on the YAM lego assets -- enough to mis-normalize everything, never enough to NaN.
"""

import dataclasses
import json

from openpi.training import config as _config
from openpi.training import norm_stats_cache as nsc


def _cfg(**over):
    c = _config.get_config("pi05_yam_lego_taxi")
    return dataclasses.replace(c, **over) if over else c


def test_the_key_separates_success_only_from_all_episodes():
    c = _cfg()
    dc_all = c.data.create(c.assets_dirs, c.model)
    c2 = dataclasses.replace(c, data=dataclasses.replace(c.data, success_only=True))
    dc_sub = c2.data.create(c2.assets_dirs, c2.model)
    assert dc_all.episodes is None
    assert dc_sub.episodes is not None
    assert nsc.stats_key(dc_all, c.model.action_horizon) != nsc.stats_key(dc_sub, c.model.action_horizon)


def test_the_key_is_stable_across_calls():
    c = _cfg()
    dc = c.data.create(c.assets_dirs, c.model)
    assert nsc.stats_key(dc, 30) == nsc.stats_key(dc, 30)


def test_horizon_is_part_of_the_key():
    c = _cfg()
    dc = c.data.create(c.assets_dirs, c.model)
    assert nsc.stats_key(dc, 30) != nsc.stats_key(dc, 50)


def test_an_explicit_asset_id_is_left_alone(caplog):
    """Pinning a run to another run's statistics is how a schedule-only comparison is made; the
    cache must not override it, stamped or not."""
    c = _config.get_config("pi05_yam_lego_taxi_alphaflow")
    c = dataclasses.replace(
        c, data=dataclasses.replace(c.data, assets=dataclasses.replace(c.data.assets, asset_id="jellyho/pinned"))
    )
    dc = dataclasses.replace(c.data.create(c.assets_dirs, c.model), asset_id="jellyho/pinned")
    with caplog.at_level("INFO"):
        out = nsc.ensure_norm_stats(c, dc)
    assert out is dc
    assert "named explicitly" in caplog.text


def test_a_stamped_sibling_is_a_hit(tmp_path):
    c = _cfg()
    dc = dataclasses.replace(c.data.create(c.assets_dirs, c.model), repo_id="fake/repo", asset_id="fake/repo")
    c = dataclasses.replace(c, assets_base_dir=str(tmp_path)) if hasattr(c, "assets_base_dir") else c
    key = nsc.stats_key(dc, c.model.action_horizon)
    d = tmp_path / f"fake/repo__{key}"
    d.mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps({"stats_key": key}))
    assert nsc._provenance_key(d) == key


def test_an_unstamped_directory_is_not_a_hit(tmp_path):
    d = tmp_path / "asset"
    d.mkdir()
    assert nsc._provenance_key(d) is None
    (d / "provenance.json").write_text(json.dumps({"computed_on": {"repo_id": "x"}}))
    assert nsc._provenance_key(d) is None, "a legacy provenance stamp carries no stats_key and must not count"


def test_corrupt_provenance_does_not_crash(tmp_path):
    d = tmp_path / "asset"
    d.mkdir()
    (d / "provenance.json").write_text("{not json")
    assert nsc._provenance_key(d) is None
