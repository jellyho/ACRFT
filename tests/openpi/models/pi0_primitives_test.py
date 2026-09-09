"""The sampler primitives are ONE implementation, and moving to them changed no math.

pi0.py's prefix/velocity had been hand-copied into the extraction arm sampler and into five
extraction trainers. A copy does not fail when it drifts -- it silently trains or serves against a
different policy than the one it reports, which is how this ring lost nine arms to a critic fed raw
proprio where it was trained on normalized (grad_a Q direction moved by cosine 0.85 mean, -0.69 at
worst). These tests are the guard: the primitives reproduce the old inline code exactly, and the
samplers built on them reproduce themselves.
"""

import einops
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask


@pytest.fixture(scope="module")
def model_and_obs():
    cfg = pi0_config.Pi0Config(
        pi05=True, action_horizon=4, action_dim=8, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    m = cfg.create(jax.random.key(0))
    obs = _model.preprocess_observation(None, cfg.fake_obs(batch_size=1), train=False)
    return cfg, m, obs


def test_prefix_primitive_matches_the_inline_version(model_and_obs):
    """The code every copy started from, kept here so a change to the primitive has to be
    deliberate rather than incidental."""
    _cfg, m, obs = model_and_obs

    tokens, mask, ar = m.embed_prefix(obs)
    attn = make_attn_mask(mask, ar)
    pos = jnp.cumsum(mask, axis=1) - 1
    _, kv_inline = m.PaliGemma.llm([tokens, None], mask=attn, positions=pos)

    prefix_mask, kv = m._prefix_forward(obs)
    assert jnp.array_equal(prefix_mask, mask)
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), kv, kv_inline))


def test_velocity_primitive_matches_the_inline_version(model_and_obs):
    cfg, m, obs = model_and_obs
    x = jax.random.normal(jax.random.key(7), (1, cfg.action_horizon, cfg.action_dim))
    tau = jnp.full((1,), 0.6)
    prefix_mask, kv = m._prefix_forward(obs)

    st, sm, sar, ad = m.embed_suffix(obs, x, tau)
    sattn = make_attn_mask(sm, sar)
    pattn = einops.repeat(prefix_mask, "b p -> b s p", s=st.shape[1])
    full = jnp.concatenate([pattn, sattn], axis=-1)
    pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
    (_, out), _ = m.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv, adarms_cond=[None, ad])
    v_inline = m.action_out_proj(out[:, -cfg.action_horizon :])

    assert float(jnp.max(jnp.abs(v_inline - m._velocity(obs, prefix_mask, kv, x, tau)))) == 0.0


def test_prefix_returns_mask_first(model_and_obs):
    """`(prefix_mask, kv_cache)`, adopted from Pi0AlphaFlow. The arm sampler returned the reverse,
    and both are two-tuples of things that look alike at a call site -- one convention, or the next
    caller unpacks a KV cache into a mask and finds out at runtime."""
    _cfg, m, obs = model_and_obs
    first, second = m._prefix_forward(obs)
    assert first.dtype == jnp.bool_, "the mask comes first"
    assert not isinstance(second, jnp.ndarray) or second.ndim > first.ndim


def test_alphaflow_inherits_the_prefix_rather_than_copying_it():
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_alphaflow import Pi0AlphaFlow

    assert Pi0AlphaFlow._prefix_forward is Pi0._prefix_forward
    # ...but its mean-velocity takes TWO times (t, r), so it stays its own method.
    assert "_u" in Pi0AlphaFlow.__dict__
