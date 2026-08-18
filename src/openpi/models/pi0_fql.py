"""pi05 + FQL: a one-step RL actor and an in-VLA critic on top of a FROZEN BC flow expert.

See docs/fql_one_step_actor.md for the full design. Summary:

  * The pretrained pi05 flow action expert is the BC policy mu_theta and is kept FROZEN. Its ODE output
    at t=1 for a noise z is the distillation target that keeps the one-step actor's actions in-manifold.
  * one_step_actor  mu_omega(context, z, state)   -> action chunk in ONE forward (no denoising timestep).
  * critic          Q_phi(context, action, state) -> distributional value (HL-Gauss).

Both new heads are conditioned on the SAME VLM prefix (image + prompt), computed once, via cross-
attention (the RLT-style separate-module pattern in pi0_rlt.py -- lighter than adding gemma MoT experts,
and the frozen flow expert is untouched).

FQL objectives (offline, RoboCasa DEAS protocol):
  L_critic = (Q(s,a) - r - gamma * Qbar(s', mu_omega(s',z')))^2
  L_actor  = -Q(s, mu_omega(s,z)) + alpha * || mu_omega(s,z) - mu_theta(s,z) ||^2

This module is a WORK IN PROGRESS: the config + head modules are defined here; the joint forward, the
frozen-flow distillation target, and the FQL train step land in scripts/train_fql.py.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at


@dataclasses.dataclass(repr=False)
class Pi0FQLConfig(pi0_config.Pi0Config):
    """pi05 with a one-step FQL actor + a critic head over the frozen BC flow expert.

    The base pi05 (VLM + flow action expert) is loaded from a pretrained checkpoint and FROZEN; only the
    one-step actor and the critic are trained. Inherits all Pi0Config fields (pi05, action_dim, horizon).
    """

    # --- one-step actor mu_omega ---
    fql_actor_width: int = 1024
    fql_actor_depth: int = 4
    # noise dim fed to the actor; the action chunk is (action_horizon x action_dim), noise matches it.
    # (kept implicit = action_horizon * action_dim by default)

    # --- critic Q_phi (distributional, HL-Gauss) ---
    fql_critic_width: int = 1024
    fql_critic_depth: int = 4
    fql_num_critics: int = 2  # ensemble (min) to fight overestimation
    fql_num_atoms: int = 101
    fql_v_min: float = -100.0
    fql_v_max: float = 0.0

    # --- FQL training ---
    fql_alpha: float = 10.0  # behavioural (distillation) coefficient in L_actor
    fql_discount: float = 0.99
    fql_target_tau: float = 0.005
    fql_flow_ode_steps: int = 10  # steps to roll the frozen flow ODE for the distillation target
    fql_critic_cotrain: bool = True  # co-train the critic (Eq.1); False = use a fixed pretrained critic

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # reuse the pi05 transform/tokenizer stack; the FQL heads are extra.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0FQL:
        from openpi.models.pi0_fql import Pi0FQL

        return Pi0FQL(self, rngs=nnx.Rngs(rng))


class OneStepActor(nnx.Module):
    """mu_omega: (VLM prefix context, noise z, state) -> action chunk, in ONE forward.

    A small transformer that cross-attends to the frozen VLM prefix (image+prompt) and reads a noise
    token + a state token, then projects to the (action_horizon x action_dim) chunk. No flow timestep.
    """

    def __init__(self, config: Pi0FQLConfig, ctx_dim: int, *, rngs: nnx.Rngs):
        raise NotImplementedError("OneStepActor forward lands next; scaffold committed for review.")


class CriticHead(nnx.Module):
    """Q_phi: (VLM prefix context, action chunk, state) -> distributional value over fql_num_atoms.

    Ensemble of `fql_num_critics` heads (ensemble-min at read). Cross-attends to the VLM prefix, reads
    the action chunk + state, outputs HL-Gauss logits over [fql_v_min, fql_v_max].
    """

    def __init__(self, config: Pi0FQLConfig, ctx_dim: int, *, rngs: nnx.Rngs):
        raise NotImplementedError("CriticHead forward lands next; scaffold committed for review.")


class Pi0FQL(nnx.Module):
    """Frozen pi05 (VLM + flow expert) + trainable OneStepActor + CriticHead. WIP: __init__ wires the
    frozen base and the two heads; the joint forward / FQL losses live in scripts/train_fql.py."""

    def __init__(self, config: Pi0FQLConfig, *, rngs: nnx.Rngs):
        self.config = config
        raise NotImplementedError(
            "Pi0FQL wiring lands next: load frozen pi05 base, build OneStepActor + CriticHead on its "
            "VLM prefix width, expose embed_prefix + a one-step decode + a critic forward."
        )
