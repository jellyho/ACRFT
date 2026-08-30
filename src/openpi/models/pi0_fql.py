"""pi05 + FQL: a one-step RL actor and an in-VLA critic expert on a FROZEN BC flow expert.

See docs/fql_one_step_actor.md for the design. Concretely this is a 4-expert gemma MoT over ONE shared
VLM prefix (image + prompt), computed once and reused by every expert via its KV cache:

  idx 0  paligemma 2b   VLM backbone            frozen     image + prompt  -> prefix KV
  idx 1  gemma 300m     BC flow policy  mu_theta FROZEN     noisy act + t   -> flow velocity (distill target)
  idx 2  gemma 300m     one-step actor  mu_omega TRAIN      noise z         -> action chunk (1 forward, no t)
  idx 3  gemma 300m     critic          Q_phi    TRAIN      action chunk    -> distributional Q (HL-Gauss)

mu_theta stays frozen: its ODE output at t=1 for a noise z is the distillation target that keeps
mu_omega's actions on the BC manifold. mu_omega and Q_phi are new gemma experts (indices _2 / _3 in the
gemma weight names); mu_omega can be WARM-STARTED from mu_theta's weights (same gemma_300m arch) so it
inherits the visual-action grounding -- see train_fql.py.

pi05 conditions on image+prompt (no proprio state token), so "state" is the VLM prefix here: the actor
reads noise + prefix, the critic reads the action chunk + prefix. The FQL objectives (critic TD, flow
distillation, actor Q+distill) live in scripts/train_fql.py; this module exposes the forwards they need.
"""

from __future__ import annotations

import dataclasses

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True, repr=False)
class Pi0FQLConfig(pi0_config.Pi0Config):
    """pi05 with a one-step FQL actor + a critic expert over the frozen BC flow expert.

    Inherits all Pi0Config fields (pi05, action_dim, action_horizon). The base pi05 (VLM + flow expert)
    is loaded from a pretrained checkpoint and frozen; only the actor + critic experts are trained.
    """

    # extra gemma experts (must match the base experts' depth / head_dim / num_heads / num_kv_heads).
    # actor keeps 300m capacity (it must one-shot the flow ODE from scratch); the critic is a narrower
    # gemma_150m -- value regression is easier and a leaner critic is more stable / less overestimation-prone.
    actor_expert_variant: _gemma.Variant = "gemma_300m"
    critic_expert_variant: _gemma.Variant = "gemma_150m"

    # critic distributional head (HL-Gauss)
    fql_num_atoms: int = 101
    fql_v_min: float = -100.0
    fql_v_max: float = 0.0

    # FQL training knobs (used by train_fql.py; kept here so a checkpoint records them)
    fql_alpha: float = 10.0  # behavioural (distillation) coefficient in L_actor
    fql_discount: float = 0.99
    fql_target_tau: float = 0.005
    fql_flow_ode_steps: int = 10  # steps to roll the frozen flow ODE for the distillation target
    # floor the TD target with the transition's Monte-Carlo return (y = max(y, MC)) -- the same
    # anchor the patch-critic uses, and the same idea as Cal-QL's max(Q, V^mu): a bootstrapped
    # target below the return actually observed from this state is provably too pessimistic, and
    # with sparse/terminal rewards the floor is what keeps early critics from free-falling.
    fql_mc_floor: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0FQL:
        return Pi0FQL(self, rngs=nnx.Rngs(rng))


# expert indices in the gemma configs list
_PALI, _FLOW, _ACTOR, _CRITIC = 0, 1, 2, 3


class Pi0FQL(nnx.Module):
    """Frozen pi05 (VLM + flow expert) + trainable one-step actor + critic expert, one shared prefix."""

    def __init__(self, config: Pi0FQLConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.pi05 = config.pi05
        pali = _gemma.get_config(config.paligemma_variant)
        flow = _gemma.get_config(config.action_expert_variant)
        actor = _gemma.get_config(config.actor_expert_variant)
        critic = _gemma.get_config(config.critic_expert_variant)
        self._widths = [pali.width, flow.width, actor.width, critic.width]

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[pali, flow, actor, critic],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        # only the flow expert (idx 1) uses adaRMS time conditioning; actor/critic do not.
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, config.pi05, False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=pali.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        ad = config.action_dim
        # flow expert (idx 1) projections -- identical to Pi0 so a pretrained pi05 loads seamlessly.
        self.action_in_proj = nnx.Linear(ad, flow.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(flow.width, flow.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(flow.width, flow.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(flow.width, ad, rngs=rngs)

        # one-step actor (idx 2): noise chunk -> tokens -> action chunk. No timestep.
        self.actor_in_proj = nnx.Linear(ad, actor.width, rngs=rngs)
        self.actor_out_proj = nnx.Linear(actor.width, ad, rngs=rngs)

        # critic expert (idx 3): action chunk -> tokens -> distributional Q (HL-Gauss over num_atoms).
        self.critic_in_proj = nnx.Linear(ad, critic.width, rngs=rngs)
        self.critic_out_proj = nnx.Linear(critic.width, config.fql_num_atoms, rngs=rngs)

        self.deterministic = True

    # ---- prefix (image + prompt), computed once, reused by every expert -----------------------------

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask, ar_mask, tokens = [], [], []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
        if obs.tokenized_prompt is not None:
            tok = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tok)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tok.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        return tokens, input_mask, jnp.array(ar_mask)

    def _prefix_kv(self, obs: _model.Observation):
        """Run the VLM prefix once; return its KV cache + mask so every expert can cross-attend to it."""
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # experts 1..3 carry no prefix tokens
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None, None], mask=attn, positions=positions, adarms_cond=[None, None, None, None]
        )
        return kv_cache, prefix_mask

    def _run_suffix(self, kv_cache, prefix_mask, expert_idx, suffix_tokens, suffix_ar, adarms_cond):
        """Cross-attend one expert's suffix tokens to the cached prefix (mirrors Pi0.sample_actions)."""
        suffix_mask = jnp.ones(suffix_tokens.shape[:2], dtype=jnp.bool_)
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        embedded = [None, None, None, None]
        embedded[expert_idx] = suffix_tokens
        adarms = [None, None, None, None]
        adarms[expert_idx] = adarms_cond
        outs, _ = self.PaliGemma.llm(
            embedded, mask=full_attn, positions=positions, kv_cache=kv_cache, adarms_cond=adarms
        )
        return outs[expert_idx]  # [b, s, width]

    # ---- flow expert mu_theta (frozen): ODE distillation target ------------------------------------

    def _embed_flow_suffix(self, noisy_actions, timestep):
        """pi05 flow suffix: action tokens with adaRMS time conditioning (matches Pi0.embed_suffix)."""
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(self.time_mlp_in(time_emb))
        time_emb = nnx.swish(self.time_mlp_out(time_emb))
        ar = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, ar, time_emb

    @property
    def action_horizon(self) -> int:
        return self.config.action_horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def flow_ode(
        self,
        obs: _model.Observation,
        noise: at.Float[at.Array, "b ah ad"],
        num_steps: int | None = None,
        kv_cache=None,
        prefix_mask=None,
    ):
        """Roll the FROZEN flow ODE from noise (t=1) to the action (t=0). This is mu_theta(s, z)."""
        num_steps = num_steps or self.config.fql_flow_ode_steps
        if kv_cache is None:
            kv_cache, prefix_mask = self._prefix_kv(obs)
        dt = -1.0 / num_steps
        b = noise.shape[0]

        def step(carry):
            x_t, t = carry
            act_tok, ar, time_emb = self._embed_flow_suffix(x_t, jnp.broadcast_to(t, b))
            out = self._run_suffix(kv_cache, prefix_mask, _FLOW, act_tok, ar, time_emb)
            v_t = self.action_out_proj(out[:, -self.action_horizon :])
            return x_t + dt * v_t, t + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        x0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x0

    def flow_velocity(self, x_t, time, kv_cache, prefix_mask):
        """One velocity eval of the flow expert (for continued flow-matching BC training)."""
        act_tok, ar, time_emb = self._embed_flow_suffix(x_t, time)
        out = self._run_suffix(kv_cache, prefix_mask, _FLOW, act_tok, ar, time_emb)
        return self.action_out_proj(out[:, -self.action_horizon :])

    # ---- one-step actor mu_omega -------------------------------------------------------------------

    def actor(self, obs: _model.Observation, noise: at.Float[at.Array, "b ah ad"], kv_cache=None, prefix_mask=None):
        """One forward: noise z -> action chunk (no denoise loop, no timestep)."""
        if kv_cache is None:
            kv_cache, prefix_mask = self._prefix_kv(obs)
        tokens = self.actor_in_proj(noise)  # [b, ah, width]
        ar = jnp.array([True] + [False] * (self.action_horizon - 1))
        out = self._run_suffix(kv_cache, prefix_mask, _ACTOR, tokens, ar, None)
        return self.actor_out_proj(out)  # [b, ah, ad]

    # ---- critic expert Q_phi -----------------------------------------------------------------------

    def critic_logits(
        self, obs: _model.Observation, actions: at.Float[at.Array, "b ah ad"], kv_cache=None, prefix_mask=None
    ):
        """Distributional Q(s, a): HL-Gauss logits over fql_num_atoms, read from the last action token."""
        if kv_cache is None:
            kv_cache, prefix_mask = self._prefix_kv(obs)
        tokens = self.critic_in_proj(actions)  # [b, ah, width]
        ar = jnp.array([True] + [False] * (self.action_horizon - 1))
        out = self._run_suffix(kv_cache, prefix_mask, _CRITIC, tokens, ar, None)
        return self.critic_out_proj(out[:, -1])  # [b, num_atoms] (last token sees the whole chunk)
