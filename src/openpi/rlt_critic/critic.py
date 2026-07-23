"""Critic architectures over precomputed RL tokens: QC (flat) and ARQ (prefix-conditioned).

Both score an action chunk from the compact RL token, and both come in a scalar or a distributional
(HL-Gauss) flavour. They differ in what they output:

**QC — Q-chunking.** One value for the whole chunk::

    Q(z, a_{1:H})

An MLP over ``concat(z, flatten(a))``. Simple and fast; the chunk length is fixed.

**ARQ — autoregressive / prefix-conditioned Q** (the ACSAC critic). A causal transformer over
``[z, m_1, ..., m_M]`` (one state token, then the chunk cut into ``M = H / macro_group_size`` macro
tokens) that returns, in a single forward, a value for *every prefix*::

    [ Q(z, a_{1:g}), Q(z, a_{1:2g}), ..., Q(z, a_{1:H}) ]

The causal mask is what makes this sound: output ``h`` attends only to ``(z, a_{1:h})``, never to the
suffix, so each head really is the value of committing to that prefix and re-planning. Grouping H
steps into macro tokens keeps the sequence short — with H=10 and g=2 there are 5 prefixes to choose
between rather than 10 nearly-identical ones. Because every head is trained against a return target
on the same discounted scale, the values are comparable across prefix lengths, which is what lets
deployment take a joint arg-max over (candidate, prefix length) and pick how far to commit.

Both are ensembled (K critics, min-aggregated) to blunt over-estimation.
"""

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp


def _mlp(x, dims, *, out_dim):
    for d in dims:
        x = nn.gelu(nn.Dense(d)(x))
    return nn.Dense(out_dim)(x)


class QCCritic(nn.Module):
    """Flat Q-chunking critic: one value per chunk."""

    action_dim: int
    horizon: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    num_atoms: int = 1  # 1 -> scalar Q; >1 -> HL-Gauss categorical logits
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, actions):
        # obs [..., D], actions [..., H, A] (or already flat [..., H*A])
        a = actions.reshape(*actions.shape[: -2 if actions.ndim > 2 else -1], -1)
        x = jnp.concatenate([obs, a], axis=-1)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        out = _mlp(x, self.hidden_dims, out_dim=self.num_atoms)
        return out if self.num_atoms > 1 else jnp.squeeze(out, -1)


class ARQCritic(nn.Module):
    """Prefix-conditioned causal-transformer critic: one value per prefix length."""

    action_dim: int
    horizon: int
    macro_group_size: int = 2
    num_layers: int = 3
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 1
    per_position_head: bool = True

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def macro_h(self) -> int:
        return self.horizon // self.macro_group_size

    @nn.compact
    def __call__(self, obs, actions):
        d, mh = self.n_embd, self.macro_h
        a = actions.reshape(*actions.shape[:-2], mh, self.macro_group_size * self.action_dim)

        state_tok = nn.Dense(d)(nn.LayerNorm()(obs))[..., None, :]  # [..., 1, d]
        act_tok = nn.Dense(d)(a)  # [..., mh, d]
        x = jnp.concatenate([state_tok, act_tok], axis=-2)  # [..., 1+mh, d]
        x = x + self.param("pos", nn.initializers.normal(0.02), (1 + mh, d))

        # Causal: prefix h must not see the suffix, otherwise its "value of committing to h steps"
        # would be contaminated by actions that would never be executed.
        causal = jnp.tril(jnp.ones((1 + mh, 1 + mh), dtype=bool))
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, h, mask=causal)
            h = nn.LayerNorm()(x)
            x = x + nn.Dense(d)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        x = nn.LayerNorm()(x)[..., 1:, :]  # drop the state position: [..., mh, d]

        if self.per_position_head:
            # A distinct head per prefix position (ACSAC Prop. G.7).
            kernel = self.param("head_k", nn.initializers.lecun_normal(), (mh, d, self.num_atoms))
            bias = self.param("head_b", nn.initializers.zeros, (mh, self.num_atoms))
            out = jnp.einsum("...hd,hda->...ha", x, kernel) + bias
        else:
            out = nn.Dense(self.num_atoms)(x)
        return out if self.num_atoms > 1 else jnp.squeeze(out, -1)  # [..., mh(, atoms)]


class Ensemble(nn.Module):
    """K independent critics stacked on a leading axis (min-aggregated by the caller)."""

    make_critic: callable
    num_critics: int = 2

    @nn.compact
    def __call__(self, obs, actions):
        vmapped = nn.vmap(
            lambda mdl, o, a: mdl(o, a),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_critics,
        )
        return vmapped(self.make_critic(), obs, actions)


@dataclasses.dataclass(frozen=True)
class HLGauss:
    """Scalar <-> soft-histogram transforms for the distributional critic."""

    v_min: float
    v_max: float
    num_atoms: int
    sigma_frac: float = 0.75

    @property
    def edges(self):
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms + 1)

    @property
    def centers(self):
        e = self.edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def sigma(self):
        return self.sigma_frac * (self.v_max - self.v_min) / self.num_atoms

    def to_probs(self, y):
        cdf = jax.scipy.stats.norm.cdf(self.edges, loc=y[..., None], scale=self.sigma)
        p = cdf[..., 1:] - cdf[..., :-1]
        return p / (jnp.sum(p, axis=-1, keepdims=True) + 1e-8)

    def from_logits(self, logits):
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * self.centers, axis=-1)


def make_critic(kind: str, *, action_dim: int, horizon: int, num_atoms: int, **kw):
    """Build a QC or ARQ critic module. Both take (obs, actions) and return values."""
    if kind == "qc":
        return QCCritic(action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **kw)
    if kind == "arq":
        return ARQCritic(action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **kw)
    raise ValueError(f"critic kind must be 'qc' or 'arq', got {kind!r}")
