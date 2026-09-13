"""The AWR arm's two halves have to agree: the data attaches A, the loss weights by it.

These are the checks that the config-driven arm reproduces `scripts/train_awr.py`'s arithmetic
exactly -- the z-score over the whole dataset (awr_agent.py:403) and the clipped exponential
(awr_agent.py:407,409-410) -- and that a missing annotation fails loudly instead of silently
training plain BC, which is the failure the CFGRL config already had.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0_awr as pi0_awr
import openpi.models.pi0_cfgrl as pi0_cfgrl
import openpi.training.advantage as _advantage


def _write(tmp_path, q, v):
    np.save(tmp_path / "q_data.npy", np.asarray(q, np.float32))
    np.save(tmp_path / "v_data.npy", np.asarray(v, np.float32))
    return tmp_path


def test_advantage_is_z_scored_over_the_whole_dataset(tmp_path):
    """Per-batch normalization would make a sample's weight depend on who it was batched with."""
    q = np.arange(100, dtype=np.float32)
    v = np.zeros(100, np.float32)
    a = _advantage.load_advantage(_write(tmp_path, q, v))
    assert a.shape == (100,)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a.mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(a.std(), 1.0, rtol=1e-3)
    # exactly awr_agent.py:403, eps included
    adv = q - v
    np.testing.assert_allclose(a, (adv - adv.mean()) / (adv.std() + 1e-5), rtol=1e-6)


def test_raw_mode_keeps_the_sign_the_cfgrl_threshold_reads(tmp_path):
    """CFGRL's label is 1{A > 0} on the RAW advantage (iql_diffusion.py:157). Z-scoring moves that
    threshold to 1{A > mean(A)} -- a different, larger set, and nothing downstream would notice."""
    q = np.array([-3.0, -1.0, 1.0, 9.0], np.float32)  # mean 1.5, so the two thresholds disagree
    v = np.zeros(4, np.float32)
    d = _write(tmp_path, q, v)
    raw = _advantage.load_advantage(d, normalize="raw")
    z = _advantage.load_advantage(d, normalize="zscore")
    np.testing.assert_allclose(raw, q)
    assert (raw > 0).tolist() == [False, False, True, True]
    assert (z > 0).tolist() == [False, False, False, True], "z-score really does move the threshold"


def test_an_unknown_normalization_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="zscore"):
        _advantage.load_advantage(_write(tmp_path, np.zeros(4), np.zeros(4)), normalize="minmax")


def test_mismatched_q_and_v_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="disagree"):
        _advantage.load_advantage(_write(tmp_path, np.zeros(10), np.zeros(11)))


def test_the_label_is_addressed_by_the_global_frame_index(tmp_path):
    """LeRobot's `index` is the row the annotation files are keyed by; getting this wrong pairs
    every frame with another frame's advantage, which is silent -- the loss still trains."""
    a = _advantage.load_advantage(_write(tmp_path, np.arange(50), np.zeros(50)))
    add = _advantage.AddAdvantage(a)
    out = add({"index": np.array([0, 7, 49]), "state": "untouched"})
    np.testing.assert_allclose(out["advantage"], a[[0, 7, 49]])
    assert out["state"] == "untouched"


def test_it_is_a_no_op_at_serving_time(tmp_path):
    """The same transform chain serves, where the raw item carries no index and there is no label."""
    add = _advantage.AddAdvantage(_advantage.load_advantage(_write(tmp_path, np.arange(5), np.zeros(5))))
    assert "advantage" not in add({"state": 1})


def test_an_annotation_from_a_different_dataset_is_caught(tmp_path):
    """Otherwise a shorter annotation silently wraps or reads garbage for the tail of the dataset."""
    add = _advantage.AddAdvantage(_advantage.load_advantage(_write(tmp_path, np.arange(10), np.zeros(10))))
    with pytest.raises(IndexError, match="past the 10 annotated frames"):
        add({"index": np.array([9, 10])})


def _tiny_awr(**over):
    cfg = pi0_awr.Pi0AWRConfig(
        pi05=True,
        action_horizon=4,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        **over,
    )
    return cfg, cfg.create(jax.random.key(0))


def test_the_weight_is_the_official_clipped_exponential():
    cfg, model = _tiny_awr()
    adv = jnp.array([2.0, 0.0, -2.0, 10.0])
    obs = dataclasses.replace(cfg.fake_obs(batch_size=4), advantage=adv)
    loss, aux = model.compute_loss(jax.random.key(1), obs, cfg.fake_act(batch_size=4), train=True)

    expected = np.minimum(np.exp(np.asarray(adv) / cfg.awr_temp), cfg.awr_weight_clip)
    assert loss.shape == (4,)  # per sample: the trainer takes the mean
    np.testing.assert_allclose(aux["awr/weight_mean"], expected.mean(), rtol=1e-5)
    np.testing.assert_allclose(aux["awr/weight_max"], expected.max(), rtol=1e-5)
    np.testing.assert_allclose(aux["awr/frac_at_clip"], 0.25)  # only the +10 sample clips


def test_bc_loss_is_reported_unweighted():
    """It is the key every config emits so a weighted run and a plain BC run compare on one chart."""
    cfg, model = _tiny_awr()
    obs = dataclasses.replace(cfg.fake_obs(batch_size=4), advantage=jnp.zeros(4))
    loss, aux = model.compute_loss(jax.random.key(1), obs, cfg.fake_act(batch_size=4), train=True)
    # advantage 0 -> every weight is exp(0) = 1, so the weighted mean IS the BC loss
    np.testing.assert_allclose(jnp.mean(loss), aux["bc_loss"], rtol=1e-6)


def test_a_missing_annotation_raises_instead_of_training_plain_bc():
    cfg, model = _tiny_awr()
    with pytest.raises(ValueError, match=r"needs Observation\.advantage"):
        model.compute_loss(jax.random.key(1), cfg.fake_obs(batch_size=2), cfg.fake_act(batch_size=2), train=True)


def test_the_arm_trains_the_action_expert_only():
    """Every arm gets the same budget on the same frozen features, or they are not comparable."""

    def trainable_paths(cfg, model):
        f = nnx.All(nnx.Param, nnx.Not(cfg.get_freeze_filter()))
        # flat_state() is a dict keyed by the path tuple
        return ["/".join(str(k) for k in path) for path in nnx.state(model, f).flat_state()]

    cfg, model = _tiny_awr()
    paths = trainable_paths(cfg, model)
    assert paths, "nothing trainable"
    assert not [p for p in paths if "llm" in p and "_1" not in p], "backbone llm params are trainable"

    # ...and the escape hatch really does widen it to the BC budget.
    wide = _tiny_awr(freeze_backbone=False)
    assert len(trainable_paths(*wide)) > len(paths)


def _tiny_cfgrl(**over):
    cfg = pi0_cfgrl.Pi0CFGRLConfig(
        pi05=True,
        action_horizon=4,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        **over,
    )
    return cfg, cfg.create(jax.random.key(0))


def test_cfgrl_trains_its_own_objective_not_plain_bc():
    """The defect this override fixes: Pi0CFGRL used to inherit Pi0.compute_loss, so a run launched
    as `train.py <task>_cfgrl` trained BC under a CFGRL name and nothing said so."""
    cfg, model = _tiny_cfgrl()
    obs = dataclasses.replace(cfg.fake_obs(batch_size=4), advantage=jnp.array([1.0, -1.0, 2.0, -2.0]))
    act = cfg.fake_act(batch_size=4)
    loss, aux = model.compute_loss(jax.random.key(1), obs, act, train=True)

    # the CFGRL objective reports its two branches; plain BC reports neither
    assert {"cond", "uncond", "frac_pos"} <= set(aux)
    np.testing.assert_allclose(aux["frac_pos"], 0.5)  # 1{A > 0} over [1, -1, 2, -2]
    assert loss.ndim == 0


def test_cfgrl_refuses_to_run_without_its_label():
    cfg, model = _tiny_cfgrl()
    with pytest.raises(ValueError, match=r"needs Observation\.advantage"):
        model.compute_loss(jax.random.key(1), cfg.fake_obs(batch_size=2), cfg.fake_act(batch_size=2), train=True)
