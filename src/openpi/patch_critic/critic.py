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

__all__ = [
    "HLGauss",
    "PatchARQCritic",
    "PatchActionHead",
    "PatchCriticEnsemble",
    "PatchTrunk",
    "SharedTrunkCriticEnsemble",
]


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
    per_member_actions: bool = False
    """`actions` is [K, ..., H, action_dim], one tensor per member -- see the same flag on
    SharedTrunkCriticEnsemble. Present here too so EDAC can be run against BOTH architectures."""

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
            in_axes=(None, 0, None) if self.per_member_actions else None,
            axis_size=self.num_critics,
        )
        return vmapped(make(), obs, actions, proprio)  # [num_critics, ..., mh(, atoms)]


class PatchTrunk(nn.Module):
    """The action-INDEPENDENT half of PatchARQCritic: patch + proprio tokens, encoded once.

    Splitting this out is an exact refactor, not an approximation. In ``PatchARQCritic`` the mask is
    ``tril(...)`` with ``[:, :nl]`` forced True, so a leading token at position j < nl attends to
    leading tokens and to nothing else -- the patch representations already cannot see the action.
    Computing them once instead of once per ensemble member is therefore free of any modelling
    change, and it is what makes a large ensemble affordable: at batch 256 the K=2 ensemble holds
    ~5.5 GB of activations and K=10 needs ~27.6 GB (measured: it OOMs an L40S with a single 38.6 GB
    allocation), while a shared trunk is ~2.66 GB plus ~8 MB per member.

    The independence this gives up is REPRESENTATION diversity. Measured
    (scripts/diag_ensemble_independence.py, 96 states), that is not what our members had anyway: two
    members of one checkpoint have entirely separate trunks and heads and still show the HIGHEST
    grad_a Q cosine of any pair we can form (0.332, against 0.049 for unrelated directions), while
    changing the recipe -- macro_group_size or expectile -- halves it to 0.185. Separate weights buy
    almost nothing here; structural difference does.
    """

    num_layers: int = 3
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim

    @nn.compact
    def __call__(self, obs, proprio=None):
        d = self.n_embd
        patch_tok = nn.Dense(d)(nn.LayerNorm()(obs))
        patch_tok = patch_tok + self.param("patch_type", nn.initializers.normal(0.02), (1, d))
        lead = patch_tok
        if proprio is not None:
            lead = jnp.concatenate([lead, nn.Dense(d)(nn.LayerNorm()(proprio))[..., None, :]], axis=-2)
        nl = lead.shape[-2]
        x = lead + self.param("pos", nn.initializers.normal(0.02), (nl, d))
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, h)
            h = nn.LayerNorm()(x)
            x = x + nn.Dense(d)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        return nn.LayerNorm()(x)  # [..., nl, d]


class PatchActionHead(nn.Module):
    """One ensemble member: action tokens cross-attending to a trunk encoding.

    Self-attention among the action tokens stays causal, which is what preserves ``PatchARQCritic``'s
    commitment-prefix semantics -- prefix k must not see actions past its own macro group, or the
    per-prefix values stop being values of a prefix.
    """

    action_dim: int
    horizon: int
    macro_group_size: int = 30
    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 101
    per_position_head: bool = True

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def macro_h(self) -> int:
        return self.horizon // self.macro_group_size

    @nn.compact
    def __call__(self, z, actions):
        d, mh = self.n_embd, self.macro_h
        a = actions.reshape(*actions.shape[:-2], mh, self.macro_group_size * self.action_dim)
        q = nn.Dense(d)(a) + self.param("act_pos", nn.initializers.normal(0.02), (mh, d))
        causal = jnp.tril(jnp.ones((mh, mh), dtype=bool))
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(q)
            q = q + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, h, mask=causal)
            h = nn.LayerNorm()(q)
            q = q + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, nn.LayerNorm()(z))
            h = nn.LayerNorm()(q)
            q = q + nn.Dense(d)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        q = nn.LayerNorm()(q)
        if self.per_position_head:
            kernel = self.param("head_k", nn.initializers.lecun_normal(), (mh, d, self.num_atoms))
            bias = self.param("head_b", nn.initializers.zeros, (mh, self.num_atoms))
            out = jnp.einsum("...hd,hda->...ha", q, kernel) + bias
        else:
            out = nn.Dense(self.num_atoms)(q)
        return out if self.num_atoms > 1 else jnp.squeeze(out, -1)


class SharedTrunkCriticEnsemble(nn.Module):
    """One trunk, K action heads. Drop-in for PatchCriticEnsemble: same [K, ..., mh(, atoms)] output.

    Keeping the output contract identical is deliberate -- every consumer (the trainer's HL-Gauss
    cross-entropy, CriticQ's reductions, the serving wrapper's per-prefix commitment read) works
    unchanged, so `--critic-arch shared` is a one-factor change against the independent control.
    """

    action_dim: int
    horizon: int
    num_critics: int = 10
    macro_group_size: int = 30
    trunk_layers: int = 3
    head_layers: int = 2
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 101
    per_position_head: bool = True
    per_member_actions: bool = False
    """Give each member its OWN action tensor, i.e. `actions` is [K, ..., H, action_dim].

    Only needed to differentiate each member with respect to its own copy, which is how EDAC's
    gradient-diversity penalty gets per-member grad_a Q in one backward pass (snu-mllab/EDAC
    sac.py: `actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)`).
    The parameter tree is identical either way, so the same params can be applied through both.
    """

    @nn.compact
    def __call__(self, obs, actions, proprio=None):
        z = PatchTrunk(
            num_layers=self.trunk_layers, num_heads=self.num_heads, head_dim=self.head_dim, mlp_dim=self.mlp_dim
        )(obs, proprio)

        def make():
            return PatchActionHead(
                action_dim=self.action_dim,
                horizon=self.horizon,
                macro_group_size=self.macro_group_size,
                num_layers=self.head_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                num_atoms=self.num_atoms,
                per_position_head=self.per_position_head,
            )

        vmapped = nn.vmap(
            lambda module, z_, a_: module(z_, a_),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(None, 0) if self.per_member_actions else None,
            axis_size=self.num_critics,
        )
        return vmapped(make(), z, actions)  # [num_critics, ..., mh(, atoms)]
