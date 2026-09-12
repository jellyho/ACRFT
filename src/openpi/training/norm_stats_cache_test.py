"""The cache must find statistics by what they were computed on, not by where someone filed them."""

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader
from openpi.training import norm_stats_cache as nsc
import openpi.transforms as _transforms


class _DC:
    """A DataConfig stand-in. It has to carry the transform groups: they are part of the key now,
    because the statistics are of the transforms' OUTPUT and the search crosses config directories."""

    def __init__(self, episodes, repo_id="jellyho/yam_lego_taxi", transforms=()):
        self.episodes, self.repo_id = episodes, repo_id
        self.drop_failure_homing = False
        self.repack_transforms = _transforms.Group(inputs=())
        self.data_transforms = _transforms.Group(inputs=tuple(transforms))


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


def test_naming_an_asset_that_does_not_exist_is_an_error_not_a_silent_skip(tmp_path):
    """Naming an asset is how a run is pinned to one statistics file; a missing file is a typo.

    The old behaviour logged "skipping" and trained with norm_stats=None, which looks exactly like
    training with normalization until the checkpoint is deployed. It is the failure mode a BC job
    starting before its norm-stats job finishes would have hit.
    """
    import dataclasses

    from openpi.training import config as _config

    cfg = _config.get_config("pi05_yam_cable_tie")
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, assets=dataclasses.replace(cfg.data.assets, asset_id="jellyho/no_such_asset")
        ),
    )
    # The error moved: create() must NOT raise (compute_norm_stats names the asset it is about to
    # write and calls create() only to build transforms), so the training path enforces it instead.
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert dc.norm_stats_required
    assert dc.norm_stats is None
    # ... on the TRAINING path. create_torch_dataset must stay usable: compute_norm_stats.py builds
    # a dataset through it precisely in order to WRITE the asset it names, and a raise there makes
    # the job that produces the file fail for want of the file (job 37773 died exactly this way).
    data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model, skip_videos=True)
    with pytest.raises(FileNotFoundError, match="named explicitly"):
        data_loader.create_torch_data_loader(dc, cfg.model, cfg.model.action_horizon, batch_size=1)


def test_failure_homing_is_dropped_and_success_homing_is_not(tmp_path, monkeypatch):
    """The filter must cut a failure's give-up tail and leave a success's correct retraction alone.

    A failure's homing is the operator retracting from a task that is not done -- the only
    supervision in the set that says give up, and it sits at cosine 0.956 from its nearest
    success-task frame while teaching an action 3.5x further away than that neighbour's.
    """

    from openpi.training import data_loader
    from openpi.training import progress

    class _HF:
        def __getitem__(self, k):
            # two episodes of 10 frames: ep0 success, ep1 failure, both homing from frame 7
            return {"episode_index": [0] * 10 + [1] * 10, "frame_index": list(range(10)) * 2}[k]

    class _DS:
        hf_dataset = _HF()

        def __len__(self):
            return 20

    monkeypatch.setattr(progress, "homing_onsets", lambda *a, **k: {0: 7, 1: 7})
    monkeypatch.setattr(progress, "success_episode_indices", lambda *a, **k: [0])
    cfg = dataclasses.replace(_config.DataConfig(), repo_id="x/y", drop_failure_homing=True)
    out = data_loader._drop_failure_homing(_DS(), cfg, action_horizon=1)
    kept = set(out.indices)
    assert kept == set(range(10)) | set(range(10, 17)), "ep0 kept whole; ep1 cut from its onset"
    assert len(kept) == 17


def test_the_filter_refuses_rather_than_guessing_when_control_mode_is_missing(tmp_path, monkeypatch):
    from openpi.training import data_loader
    from openpi.training import progress

    monkeypatch.setattr(progress, "homing_onsets", lambda *a, **k: None)
    cfg = dataclasses.replace(_config.DataConfig(), repo_id="x/y", drop_failure_homing=True)
    with pytest.raises(ValueError, match="observation.control_mode"):
        data_loader._drop_failure_homing(object(), cfg, action_horizon=1)


def test_delta_mode_does_not_collide_in_the_key(tmp_path):
    """Joint deltas and absolute joint targets are different numbers from the same episodes.

    The search deliberately crosses config directories, so anything a config does that changes the
    statistics has to be in the key. It was not: pi05_yam_lego_taxi (JointDeltaActions) and
    pi05_yam_lego_taxi_none (no delta transform) hashed identically, and a joint run would load the
    absolute stats as a "proven" match -- a 1.74 rad mean offset with 2.3-2.5x the std, silently.
    """
    joint = _config.get_config("pi05_yam_lego_taxi")
    absolute = _config.get_config("pi05_yam_lego_taxi_none")
    kj = nsc.stats_key(joint.data.create(joint.assets_dirs, joint.model), joint.model.action_horizon)
    ka = nsc.stats_key(absolute.data.create(absolute.assets_dirs, absolute.model), absolute.model.action_horizon)
    assert kj != ka
    # ... while two configs that really do share a transform pipeline and data must still share stats.
    rlt = _config.get_config("pi05_yam_lego_taxi_rlt")
    assert nsc.stats_key(rlt.data.create(rlt.assets_dirs, rlt.model), rlt.model.action_horizon) == kj


def test_resolving_an_asset_carries_its_provenance(tmp_path):
    """Otherwise the downstream check validates a file the run no longer uses."""
    prov = {"computed_on": {"repo_id": "jellyho/yam_lego_taxi", "episodes_subset": list(range(300))}}
    d = tmp_path / "cfg" / "asset"
    _write(d, prov)
    dc = dataclasses.replace(
        _config.DataConfig(),
        repo_id="jellyho/yam_lego_taxi",
        norm_stats_provenance={"computed_on": {"episodes_subset": "all"}},  # the stale, name-resolved one
    )
    out = nsc._resolved(dc, d, tmp_path, _normalize.load(d))
    assert out.norm_stats_provenance["computed_on"]["episodes_subset"] == list(range(300))


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
