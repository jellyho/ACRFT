"""Pi0.sample_n_actions: N candidates from one prefix pass.

Value-guided serving (best-of-N, adaptive commitment) needs N chunks per frame, and the patch
critics are trained against a plain pi05 BC base -- which had no such entry point, so the wrapper
refused it with "offers neither extract_token_and_base_actions nor sample_n_actions".

These run on CPU against a real (tiny) PaliGemma, because the claim worth testing is not a shape
but an equivalence: tiling the cached prefix must give exactly what re-running it would have.
"""

import jax
import numpy as np
import pytest

from openpi.models import pi0_config


@pytest.fixture(scope="module")
def model():
    cfg = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
    )
    return cfg, cfg.create(jax.random.key(0))


def _obs(cfg, batch=1):
    obs, _ = cfg.fake_obs(batch_size=batch), None
    return obs


def test_shape_is_n_candidates(model):
    cfg, m = model
    out = m.sample_n_actions(jax.random.key(1), _obs(cfg), num_samples=5, num_steps=4)
    assert out.shape == (5, cfg.action_horizon, cfg.action_dim)


def test_candidates_actually_differ(model):
    """One prefix, N noise draws. If the noise were tiled instead of drawn per candidate, best-of-N
    would be choosing among N copies of one chunk -- which is exactly the failure that looks like a
    working critic."""
    cfg, m = model
    out = np.asarray(m.sample_n_actions(jax.random.key(1), _obs(cfg), num_samples=4, num_steps=4))
    for i in range(1, 4):
        assert not np.allclose(out[0], out[i]), f"candidate {i} duplicates candidate 0"


def test_sharing_the_prefix_changes_nothing(model):
    """The equivalence the whole method rests on: a tiled KV cache is what N prefix passes over one
    unchanged frame would have produced. Same noise in, same chunk out as sample_actions."""
    cfg, m = model
    obs = _obs(cfg)
    rng = jax.random.key(7)
    n = 3
    shared = np.asarray(m.sample_n_actions(rng, obs, num_samples=n, num_steps=4))
    # sample_n_actions draws its noise as one [n, ah, ad] normal; feed the same rows back through
    # the ordinary sampler one at a time.
    noise = jax.random.normal(rng, (n, cfg.action_horizon, cfg.action_dim))
    for i in range(n):
        one = np.asarray(m.sample_actions(rng, obs, num_steps=4, noise=noise[i][None]))
        np.testing.assert_allclose(shared[i], one[0], rtol=2e-4, atol=2e-4)


def test_refuses_a_batched_observation(model):
    """The prefix is tiled across candidates, so a batch of frames would silently mix them."""
    cfg, m = model
    with pytest.raises(ValueError, match="batch-1"):
        m.sample_n_actions(jax.random.key(1), _obs(cfg, batch=2), num_samples=2, num_steps=4)


def test_the_patch_critic_wrapper_now_accepts_pi05(model):
    """The wrapper dispatches on this attribute; that dispatch is the thing that was failing."""
    _, m = model
    assert hasattr(m, "sample_n_actions")


def test_batched_shape_is_states_by_candidates(model):
    cfg, m = model
    out = m.sample_n_actions_batched(jax.random.key(1), _obs(cfg, batch=3), num_samples=5, num_steps=4)
    assert out.shape == (3, 5, cfg.action_horizon, cfg.action_dim)


def test_batched_does_not_mix_states(model):
    """The claim the offline bank rests on: state i's candidates are conditioned on state i's prefix.

    jnp.repeat interleaves, so row i*n+j is state i's j-th draw; getting the tiling wrong (repeat vs
    tile) would silently pair every state with the wrong frame's prefix, and the resulting bank would
    look perfectly well-formed. Compared against the batch-1 path, which is already pinned to
    sample_actions by test_sharing_the_prefix_changes_nothing.
    """
    cfg, m = model
    b, n = 3, 2
    obs = _obs(cfg, batch=b)
    rng = jax.random.key(11)
    batched = np.asarray(m.sample_n_actions_batched(rng, obs, num_samples=n, num_steps=4))
    # the batched path draws one [b*n, ah, ad] normal; give the single-state path the same rows
    noise = jax.random.normal(rng, (b * n, cfg.action_horizon, cfg.action_dim))
    for i in range(b):
        one_obs = jax.tree.map(lambda x, i=i: x[i : i + 1], obs)
        for j in range(n):
            one = np.asarray(m.sample_actions(rng, one_obs, num_steps=4, noise=noise[i * n + j][None]))
            np.testing.assert_allclose(batched[i, j], one[0], rtol=2e-4, atol=2e-4)
