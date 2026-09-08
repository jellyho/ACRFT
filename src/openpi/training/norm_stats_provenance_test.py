"""The subset check: a success-only asset and an all-episodes asset differ only here.

repo_id, total_episodes and total_frames are all properties of the WHOLE dataset, so stats computed
on 300 success episodes and stats computed on all 347 agree on every one of them. Measured on the
YAM lego assets, their action q99 differs by up to 2.5% -- enough to mis-normalize quietly, never
enough to NaN. `episodes_subset` is the only field that separates them, compute_norm_stats.py has
always written it, and nothing read it until now.
"""

import dataclasses
import logging

import pytest

from openpi.training import config as _config
from openpi.training import data_loader


class _Meta:
    total_episodes = 347
    total_frames = 937_993


def _cfg(*, episodes, prov):
    return dataclasses.replace(
        _config.DataConfig(),
        repo_id="jellyho/yam_lego_taxi",
        asset_id="jellyho/yam_lego_taxi",
        norm_stats={"actions": None},  # presence is all the checker needs
        norm_stats_provenance=prov,
        episodes=episodes,
    )


_FULL = {"repo_id": "jellyho/yam_lego_taxi", "total_episodes": 347, "total_frames": 937_993}


def test_all_episode_stats_on_an_all_episode_run_passes():
    data_loader._check_norm_stats_provenance(_cfg(episodes=None, prov=_FULL | {"episodes_subset": "all"}), _Meta())


def test_success_only_stats_on_an_all_episode_run_is_caught():
    prov = _FULL | {"episodes_subset": list(range(300))}
    with pytest.raises(ValueError, match="episodes_subset"):
        data_loader._check_norm_stats_provenance(_cfg(episodes=None, prov=prov), _Meta())


def test_all_episode_stats_on_a_success_only_run_is_caught():
    with pytest.raises(ValueError, match="episodes_subset"):
        data_loader._check_norm_stats_provenance(
            _cfg(episodes=tuple(range(300)), prov=_FULL | {"episodes_subset": "all"}), _Meta()
        )


def test_matching_subsets_pass():
    data_loader._check_norm_stats_provenance(
        _cfg(episodes=tuple(range(300)), prov=_FULL | {"episodes_subset": list(range(300))}), _Meta()
    )


def test_legacy_stats_without_provenance_warn_and_name_the_subset(caplog):
    """Neither YAM asset on disk carries provenance, so this is the path they take today."""
    with caplog.at_level(logging.WARNING):
        data_loader._check_norm_stats_provenance(_cfg(episodes=tuple(range(300)), prov=None), _Meta())
    assert "300 episodes" in caplog.text
    assert "indistinguishable by name" in caplog.text
