"""Standalone patch-critic (JAX/flax), independent of the VLA.

The value transformer is our validated ARQ critic (``openpi.rlt_critic.critic.ARQCritic``) with ONE
change: the observation is a grid of FROZEN dense patch tokens (Patch Policy style) instead of a
single VLA-derived token. Every patch token is a leading context token visible to all action
positions; the per-prefix distributional head is unchanged, so each commitment horizon (adaptive
chunking K) still gets its own value on the same discounted scale.

The frozen patch backbone (an independent SigLIP, see ``backbone.py``) is NOT part of this module --
callers pass already-extracted patch tokens. That keeps the critic a small trainable head and lets
backbone features be cached or computed inline.

HL-Gauss and the ensemble vmap mirror ``rlt_critic.critic``; we import HLGauss directly.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.rlt_critic.critic import HLGauss  # reuse the validated histogram head

__all__ = ["HLGauss", "PatchARQCritic", "PatchCriticEnsemble"]


class PatchARQCritic(nn.Module):
    """ARQ value transformer over patch-token observations.

    obs:     [..., P, patch_dim]  frozen patch tokens (P = num_cameras * patches_per_cam)
    proprio: [..., proprio_dim]   optional low-dim state, given its own token
    actions: [..., H, action_dim] candidate chunk
    -> [..., mh(, num_atoms)]     per-prefix value (mh = H / macro_group_size commitment horizons)
    """

    action_dim: int
    horizon: int
    macro_group_size: int = 2
    num_layers: int = 3
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 51  # distributional by default
    per_position_head: bool = True

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def macro_h(self) -> int:
        return self.horizon // self.macro_group_size

    @nn.compact
    def __call__(self, obs, actions, proprio=None):
        d, mh = self.n_embd, self.macro_h
        a = actions.reshape(*actions.shape[:-2], mh, self.macro_group_size * self.action_dim)

        # Each patch is a leading context token. A learned per-token bias is unnecessary (the patch
        # positional identity is already baked into the frozen features); we add a learned type embed.
        patch_tok = nn.Dense(d)(nn.LayerNorm()(obs))  # [..., P, d]
        patch_tok = patch_tok + self.param("patch_type", nn.initializers.normal(0.02), (1, d))
        lead = patch_tok
        if proprio is not None:
            pro = nn.Dense(d)(nn.LayerNorm()(proprio))[..., None, :]  # [..., 1, d]
            lead = jnp.concatenate([lead, pro], axis=-2)
        nl = lead.shape[-2]

        act_tok = nn.Dense(d)(a)  # [..., mh, d]
        x = jnp.concatenate([lead, act_tok], axis=-2)  # [..., nl+mh, d]
        x = x + self.param("pos", nn.initializers.normal(0.02), (nl + mh, d))

        # Causal over action positions; every leading (patch/proprio) token is visible to all.
        causal = jnp.tril(jnp.ones((nl + mh, nl + mh), dtype=bool))
        causal = causal.at[:, :nl].set(True)
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, h, mask=causal)
            h = nn.LayerNorm()(x)
            x = x + nn.Dense(d)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        x = nn.LayerNorm()(x)[..., nl:, :]  # keep only action positions -> [..., mh, d]

        if self.per_position_head:
            kernel = self.param("head_k", nn.initializers.lecun_normal(), (mh, d, self.num_atoms))
            bias = self.param("head_b", nn.initializers.zeros, (mh, self.num_atoms))
            out = jnp.einsum("...hd,hda->...ha", x, kernel) + bias
        else:
            out = nn.Dense(self.num_atoms)(x)
        return out if self.num_atoms > 1 else jnp.squeeze(out, -1)  # [..., mh(, atoms)]


class PatchV(nn.Module):
    """State value V(patches, proprio). A small self-attention pool over patches read from a CLS token.

    num_atoms==1 -> scalar V (classic IQL). num_atoms>1 -> a HL-Gauss return DISTRIBUTION for V, so
    the Q backup can bootstrap V's whole distribution (distributional Bellman), not just its mean.
    """

    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 1

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim

    @nn.compact
    def __call__(self, obs, proprio=None):
        d = self.n_embd
        x = nn.Dense(d)(nn.LayerNorm()(obs))  # [..., P, d]
        if proprio is not None:
            x = jnp.concatenate([x, nn.Dense(d)(nn.LayerNorm()(proprio))[..., None, :]], axis=-2)
        cls = self.param("cls", nn.initializers.normal(0.02), (1, d))
        x = jnp.concatenate([jnp.broadcast_to(cls, (*x.shape[:-2], 1, d)), x], axis=-2)
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, h)
            h = nn.LayerNorm()(x)
            x = x + nn.Dense(d)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        out = nn.Dense(max(self.num_atoms, 1))(nn.LayerNorm()(x)[..., 0, :])  # from the CLS token
        return out if self.num_atoms > 1 else out[..., 0]


class PatchCriticEnsemble(nn.Module):
    """N independent PatchARQCritics (vmap'd), mirroring rlt_critic.critic.Ensemble."""

    action_dim: int
    horizon: int
    num_critics: int = 2
    macro_group_size: int = 2
    num_layers: int = 3
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 51
    per_position_head: bool = True

    @nn.compact
    def __call__(self, obs, actions, proprio=None):
        def make():
            return PatchARQCritic(
                action_dim=self.action_dim,
                horizon=self.horizon,
                macro_group_size=self.macro_group_size,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                num_atoms=self.num_atoms,
                per_position_head=self.per_position_head,
            )

        vmapped = nn.vmap(
            lambda module, o, a, p: module(o, a, p),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            axis_size=self.num_critics,
        )
        return vmapped(make(), obs, actions, proprio)  # [num_critics, ..., mh(, atoms)]
