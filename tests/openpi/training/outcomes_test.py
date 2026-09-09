"""The verdict reader against a hand-built LeRobot v3 metadata layout (no lerobot needed)."""

import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import openpi.training.outcomes as oc


def _dataset(root: pathlib.Path, verdicts: list[str], *, declare: bool = True, lengths: list[int] | None = None):
    """meta/info.json + meta/episodes parquet the way LeRobot writes them: one row per episode, the
    verdict encoded as the per-episode max of the two bool features."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    feats = {"observation.state": {"dtype": "float32", "shape": [42]}}
    if declare:
        feats[oc.SUCCESS_KEY] = {"dtype": "bool", "shape": [1]}
        feats[oc.DONE_KEY] = {"dtype": "bool", "shape": [1]}
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(verdicts), "features": feats}))
    lengths = lengths or [10 + i for i in range(len(verdicts))]
    cols = {
        "episode_index": list(range(len(verdicts))),
        "length": lengths,
        f"stats/{oc.SUCCESS_KEY}/max": [[v == "success"] for v in verdicts],
        f"stats/{oc.DONE_KEY}/max": [[v in ("success", "fail")] for v in verdicts],
    }
    pq.write_table(pa.table(cols), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def test_reads_the_three_states_from_the_metadata(tmp_path):
    root = _dataset(tmp_path, ["success", "fail", "unknown", "success"])
    assert oc.episode_outcomes(root) == {0: "success", 1: "fail", 2: "unknown", 3: "success"}
    assert oc.success_episode_indices(root) == [0, 3]
    assert oc.episode_lengths(root) == {0: 10, 1: 11, 2: 12, 3: 13}
    assert oc.load_outcomes(root) == oc.episode_outcomes(root)


def test_a_dataset_without_the_features_is_none_not_empty(tmp_path):
    """None means "this dataset has no notion of a verdict" (demos successful by construction);
    it must not be confused with "every episode failed"."""
    root = _dataset(tmp_path, ["success"], declare=False)
    assert oc.episode_outcomes(root) is None
    assert oc.success_episode_indices(root) is None
    with pytest.raises(oc.OutcomeColumnsMissingError, match="migrate-outcomes"):
        oc.load_outcomes(root)


def test_legacy_sidecar_is_only_read_when_named(tmp_path, caplog):
    root = _dataset(tmp_path, ["success", "fail"])
    sidecar = tmp_path / "outcomes.jsonl"
    sidecar.write_text(
        json.dumps({"episode": 0, "outcome": "fail"})
        + "\n"
        + json.dumps({"episode": 1, "outcome": "keep"})
        + "\n"  # the pre-rename DAgger label
        + json.dumps({"episode": 2, "outcome": "discard"})
        + "\n"
    )
    # the schema wins by default...
    assert oc.load_outcomes(root) == {0: "success", 1: "fail"}
    # ...and the sidecar only when a caller names it, loudly
    with caplog.at_level("WARNING"):
        assert oc.load_outcomes(root, legacy_jsonl=sidecar) == {0: "fail", 1: "success", 2: "unknown"}
    assert "legacy sidecar" in caplog.text


def test_cache_meta_carries_the_verdict_with_a_bool_fallback():
    meta = {
        "episodes": {
            "0": {"success": True, "outcome": "success"},
            "1": {"success": False, "outcome": "unknown"},
            "2": {"success": False},
        }
    }  # an older cache: only the bool
    assert oc.cache_outcomes(meta) == {0: "success", 1: "unknown", 2: "fail"}


def test_dataset_root_joins_root_and_repo_id():
    assert oc.dataset_root("jellyho/yam_lego_taxi", "/data/lerobot") == pathlib.Path(
        "/data/lerobot/jellyho/yam_lego_taxi"
    )
