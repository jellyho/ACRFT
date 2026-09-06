"""save_servable writes what `restore_params` + `load_norm_stats` read: the BC checkpoint contract."""

import json
import pathlib

import numpy as np

from openpi.extraction.checkpoint import save_servable
from openpi.models import model as _model


def test_save_servable_round_trips_through_the_serving_readers(tmp_path: pathlib.Path):
    init = tmp_path / "init"
    (init / "assets" / "someone" / "some_dataset").mkdir(parents=True)
    (init / "assets" / "someone" / "some_dataset" / "norm_stats.json").write_text(json.dumps({"norm_stats": {}}))
    params = {"a": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}, "b": np.ones(4, dtype=np.float32)}

    step = save_servable(tmp_path / "run" / "30000", params, assets_from=init)

    assert sorted(p.name for p in step.iterdir()) == ["assets", "params"]
    back = _model.restore_params(step / "params", restore_type=np.ndarray)
    np.testing.assert_array_equal(back["a"]["w"], params["a"]["w"])
    np.testing.assert_array_equal(back["b"], params["b"])
    assert (step / "assets" / "someone" / "some_dataset" / "norm_stats.json").exists()
    assert not step.with_name("30000.tmp").exists(), "the staging directory must be renamed away"
    # overwriting an existing step replaces it wholesale
    step2 = save_servable(step, {"b": np.zeros(2, dtype=np.float32)}, assets_from=init)
    assert list(_model.restore_params(step2 / "params", restore_type=np.ndarray)) == ["b"]


def test_save_servable_rejects_a_tree_that_mixes_wrapped_and_pure_leaves(tmp_path: pathlib.Path):
    import pytest

    init = tmp_path / "init"
    (init / "assets").mkdir(parents=True)
    mixed = {"a": {"value": np.zeros(1)}, "b": np.zeros(1)}
    with pytest.raises(ValueError, match="mixes"):
        save_servable(tmp_path / "run" / "1", mixed, assets_from=init)
    assert not (tmp_path / "run" / "1").exists()
    assert not (tmp_path / "run" / "1.tmp").exists()


def test_save_servable_refuses_an_init_without_assets(tmp_path: pathlib.Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        save_servable(tmp_path / "run" / "1", {"x": np.zeros(1)}, assets_from=tmp_path / "no_such_init")
