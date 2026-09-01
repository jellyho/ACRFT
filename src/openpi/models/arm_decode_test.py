"""The arms' sampling steps are the model's, and moving to them changed no math.

Three arms drew their chunk with code hand-copied out of the model they were drawing from:

    lps / lpsd   rebuilt the alpha-flow one-step decode from `_u` and a prefix pass
    flowdagger   rebuilt pi0's Euler loop to integrate its own seed
    qpilots      rebuilt the whole sampler to add a value gradient (see pi0_steered_test)

A copy does not fail when it drifts. It silently serves a base that is no longer the base being
served -- which is how this ring lost nine arms to a critic fed raw proprio where it was trained on
normalized. These tests are the guard on the move: each arm's new call reproduces the loop it
replaced, exactly where exactness is available.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_alphaflow
from openpi.models import pi0_config


@pytest.fixture(scope="module")
def flow():
    cfg = pi0_config.Pi0Config(
        pi05=True, action_horizon=4, action_dim=8, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    m = cfg.create(jax.random.key(0))
    raw = cfg.fake_obs(batch_size=1)
    return cfg, m, raw, _model.preprocess_observation(None, raw, train=False)


@pytest.fixture(scope="module")
def alphaflow():
    cfg = pi0_alphaflow.Pi0AlphaFlowConfig(
        action_horizon=4, action_dim=8, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    m = cfg.create(jax.random.key(0))
    raw = cfg.fake_obs(batch_size=1)
    return cfg, m, raw, _model.preprocess_observation(None, raw, train=False)


# ---------------------------------------------------------------------------------------------
# lps / lpsd


def test_decode_latent_matches_the_inline_version(alphaflow):
    """What the serving layer used to do, reaching into `_u` from outside."""
    _cfg, m, _raw, obs = alphaflow
    z = jax.random.normal(jax.random.key(5), (1, 4, 8))

    pm, kv = m._prefix_forward(obs)
    u = m._u(obs, pm, kv, z, jnp.ones((1,)), jnp.zeros((1,)))
    inline = np.asarray(z - u)

    assert np.array_equal(inline, np.asarray(m.decode_latent(obs, z, preprocess=False)))


def test_decode_latent_is_sample_actions_with_the_prior_replaced(alphaflow):
    """The claim that makes lps a *policy-extraction* arm rather than a different policy: it draws
    the latent differently and decodes it identically. Feed the model's own prior back in and the
    two must agree."""
    _cfg, m, raw, obs = alphaflow
    z = jax.random.normal(jax.random.key(9), (1, 4, 8))
    drawn = np.asarray(m.sample_actions(jax.random.key(0), raw, num_steps=1, noise=z))
    decoded = np.asarray(m.decode_latent(obs, z, preprocess=False))
    assert np.allclose(drawn, decoded, atol=1e-6), np.abs(drawn - decoded).max()


def test_the_actor_stays_out_of_the_model(alphaflow):
    """`decode_latent` takes an ARRAY. If it ever took the actor, or the critic features it reads,
    the model would have learned what an extraction arm is -- and one model would then have to know
    about all ten."""
    import inspect

    src = inspect.getsource(pi0_alphaflow.Pi0AlphaFlow.decode_latent)
    for word in ("actor", "critic", "proprio", "lps"):
        assert word not in src.split('"""')[2], f"{word} leaked into the model"


# ---------------------------------------------------------------------------------------------
# flowdagger


def test_seeded_sampling_matches_the_hand_copied_euler_loop(flow):
    """flowdagger integrates a DCT-parameterised seed instead of the drawn noise. That is exactly
    `sample_actions(noise=seed)`, and the loop the serving layer carried was the model's own
    sampler written out a second time: a Python loop against the model's while_loop.

    Not asserted bit-identical -- the two loop constructs accumulate in a different order. Measured
    at 1.2e-07 on a chunk of magnitude ~2, i.e. fp32 round-off. That is a different situation from
    the fori_loop rewrite rejected in pi0_steered.py, where the difference was 5.9e-03 and did NOT
    collapse in fp32: there the loop body carried a gradient, here it is a plain velocity step.
    """
    cfg, m, raw, obs = flow
    seed = jax.random.normal(jax.random.key(3), (1, cfg.action_horizon, cfg.action_dim))
    n = 10

    pm, kv = m._prefix_forward(obs)
    x, dt = seed, 1.0 / n
    for i in range(n):
        x = x - dt * m._velocity(obs, pm, kv, x, jnp.full((1,), 1.0 - i * dt))
    inline = np.asarray(x)

    served = np.asarray(m.sample_actions(jax.random.key(0), raw, num_steps=n, noise=seed))
    assert np.allclose(inline, served, atol=1e-5), np.abs(inline - served).max()


def test_the_seed_is_what_makes_the_draw_different(flow):
    """If the seed were ignored the arm would be an ordinary rollout with extra machinery, and
    every evaluation of it would be a comparison of the base policy against itself."""
    cfg, m, raw, _obs = flow
    a = m.sample_actions(jax.random.key(0), raw, num_steps=4, noise=jnp.zeros((1, 4, 8)))
    b = m.sample_actions(jax.random.key(0), raw, num_steps=4, noise=jnp.ones((1, 4, 8)) * 0.7)
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_preprocessing_twice_is_a_no_op(flow):
    """The serving layer preprocesses once for the critic's inputs and then hands the SAME
    observation to `sample_actions`, which preprocesses again. Safe only because it is idempotent
    at train=False -- resize is a no-op at the right resolution and the mask fill is a fill."""
    _cfg, _m, raw, obs = flow
    twice = _model.preprocess_observation(None, obs, train=False)
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), obs, twice))


def test_the_serving_configs_actually_have_the_methods_their_arms_call():
    """lps/lpsd and flowdagger load their base from a named training config, and their checkpoints
    are not on every machine -- so the call cannot be smoke-tested everywhere the way qpilots' can
    (its base is the served policy). What IS checkable everywhere is that the config each arm names
    produces a class carrying the method the arm calls, so a missing one is an import-time failure
    here rather than an AttributeError on a robot.
    """
    from openpi.extraction import serving
    from openpi.models.pi0 import Pi0
    from openpi.training import config as _config

    latent = _config.get_config("pi05_yam_lego_taxi_alphaflow").model
    assert isinstance(latent, pi0_alphaflow.Pi0AlphaFlowConfig)
    assert hasattr(pi0_alphaflow.Pi0AlphaFlow, "decode_latent"), "lps/lpsd decode through this"

    base = _config.get_config("pi05_yam_lego_taxi").model
    assert isinstance(base, pi0_config.Pi0Config)
    assert hasattr(Pi0, "sample_actions"), "flowdagger integrates its seed through this"

    # ...and that the arms are still routed to the sampler at all.
    for arm in ("lps", "lpsd", "flowdagger", "qpilots"):
        assert arm in serving.SAMPLER_ARMS
