r"""alpha-Flow pi0.5 — turning the flow VLA into a few/one-step generator.

Implements *AlphaFlow: Understanding and Improving MeanFlow Models* (Zhang et al., ICLR 2026,
arXiv:2510.20771) on top of pi05.  The motivation here is offline RL, not sampling speed: every
actor-critic update needs an action from the policy, and paying a 10-step ODE per update is what
makes RL on a VLA expensive.  A one-step generator collapses that to a single forward, and (unlike
distillation) alpha-Flow gets there with a pure regression objective on the data -- the VLA never has
to sample during training.

The model.  pi05 predicts an INSTANTANEOUS velocity v_th(z_t, t).  alpha-Flow predicts a MEAN
velocity over an interval, u_th(z_t, r, t) ~ (1/(t-r)) \int_r^t v(z_tau, tau) dtau, so one jump
z_r = z_t - (t-r) u_th(z_t, r, t) replaces the ODE.  r enters through the same adaRMS conditioning as
t; its MLP is ZERO-INITIALISED, so at step 0 the model reproduces the pretrained pi05 exactly
(u_th(z_t, r, t) = v_pretrained(z_t, t) for every r).  That makes this a finetune, not a retrain.

The objective (paper Def. 1, and the reference implementation snap-research/alphaflow
``src/training/loss.py`` which we follow where the two disagree):

    s   = alpha * r + (1 - alpha) * t            # dt := t - s = alpha * (t - r)
    z_s = z_t - (t - s) * v_t
    u_tgt = ( (t - s) * v_t + (s - r) * u_th^-(z_s, r, s) ) / (t - r)
          = alpha * v_t + (1 - alpha) * u_th^-(z_s, r, s)
    L_alpha = || u_th(z_t, r, t) - sg(u_tgt) ||^2   weighted by  sg( alpha / (||.||^2 + eps) )

alpha is the *consistency step ratio*, and it is the whole curriculum:

  * alpha = 1     -> u_tgt = v_t exactly.  This is trajectory flow matching (L_TFM): the pi05 BC loss
                     with an extra r input.  ONE forward pass, so this phase costs what BC costs.
  * 0 < alpha < 1 -> u_tgt mixes v_t with the model's own value at the intermediate s (Shortcut-model
                     flavour at alpha=1/2).  TWO forwards, but the second is a stop-gradient target,
                     so it carries no backward pass.
  * alpha -> 0    -> the gradient equals MeanFlow's, whose target v_t - (t-r) du/dt needs a JVP.

Why the curriculum: the paper shows L_MF decomposes into L_TFM + L_TC (trajectory consistency) whose
gradients are strongly NEGATIVELY correlated, so optimising them jointly from scratch conflicts.
Annealing alpha 1 -> 0 fits the low-variance term first and only then the high-variance one.

The alpha = 0 stage is OFF by default (``meanflow_jvp=False``).  It is the only stage that needs a
JVP through the whole VLA, and the reference implementation itself supports stopping at the clamp
value instead (``discrete_training``).  Skipping it costs some 1-step quality and buys a much simpler
and cheaper training step; run it as a separate short polish finetune if it turns out to matter.

The curriculum runs INSIDE one training run: the schedule is a function of the training step, which
reaches ``compute_loss`` as progress = step / num_train_steps through the ``wants_progress`` hook in scripts/train.py, so a single config
carries alpha from 1 (pure trajectory flow matching, where the pretrained pi05 starts as an exact
fixed point thanks to the zero-init r MLP) down to the clamp floor with no checkpoint chaining.  An
annealing run pays for the second (stop-gradient, no-backward) forward even while alpha == 1 --
about 15% of a BC step, the price of not splitting the run.  A run pinned at alpha == 1 throughout
(``alpha_final=1``) skips that forward entirely and costs exactly what BC costs.

    # the default: one run, the official schedule -- sigmoid over the whole run, clamps carve the
    # phases (alpha = 1 until ~29% progress, anneal to ~71%, floor after)
    Pi0AlphaFlowConfig()
    # optional polish afterwards: pure MeanFlow, needs the JVP
    Pi0AlphaFlowConfig(alpha_init=0.0, alpha_final=0.0, meanflow_jvp=True)
"""

from __future__ import annotations

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
import openpi.models.pi0_config as pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0AlphaFlowConfig(pi0_config.Pi0Config):
    """pi05 + alpha-Flow. Only pi05=True is supported (r rides on the adaRMS conditioning)."""

    pi05: bool = True

    # --- alpha curriculum (sigmoid in TRAINING PROGRESS, not absolute steps) ---
    # The schedule is a function of progress = step / num_train_steps, which train.py passes in, so
    # changing --num-train-steps rescales the whole curriculum instead of stranding it: a 240-step
    # smoke run and the 60k production run anneal over the same FRACTION of training.
    #
    # Defaults mirror the official runs (snap-research/alphaflow, experiments-alphaflow.yaml): the
    # sigmoid spans the WHOLE run (B/2: [0, 400k] of 400k; B/2-cfg: [0, 1.2M] of 1.2M), targeting 0
    # with gamma 25 and clamp 0.005. The clamps carve the run into the paper's three phases on
    # their own: alpha stays snapped to 1 until progress ~0.29, anneals through ~0.71, then sits on
    # the floor -- no separate warm-up window needed. (The XL/2-cfg run's [0.5, 5/6] window is the
    # exception, reachable via alpha_anneal_start/end.)
    alpha_init: float = 1.0
    alpha_final: float = 0.0  # official end_value; where it lands is decided by the clamp/floor below
    alpha_anneal_start: float = 0.0
    alpha_anneal_end: float = 1.0
    alpha_gamma: float = 25.0
    # eta: alpha snaps to 1 above 1-eta (L_TFM is the same thing but cheaper) and to `alpha_floor`
    # below eta. The paper measures 1-step quality peaking near alpha=5e-3 and degrading below it.
    alpha_clamp: float = 5e-3
    # where alpha lands once it falls under the clamp: 0.0 = the MeanFlow/JVP branch (requires
    # meanflow_jvp=True; with an annealing schedule the run TRANSITIONS into JVP training when the
    # sigmoid crosses the clamp, which is the official recipe's tail), alpha_clamp = stay discrete
    # forever (the reference's `discrete_training`).
    alpha_floor: float | None = None

    # --- border-case flow matching (r = t) fraction of each batch ---
    # CONSTANT, matching the official runs: MeanFlow needs 75% here, alpha-Flow's main results use
    # 0.5 (B/2 and XL/2; the cfg-trained B/2 uses 0.25) -- the curriculum, not an fm schedule, is
    # what reduces the reliance on border-case supervision.
    fm_ratio: float = 0.5

    # --- (t, r) sampling ---
    # "minmax": draw two times from pi05's Beta(1.5,1) and take (max, min).
    # "scaled": t ~ Beta(1.5,1), r = t * Uniform(0,1)  -- more mass on large jumps (r near 0).
    time_pair: str = "minmax"

    # elementwise clip on u_tgt (reference `clamp_utgt`; official value 4.0); None disables.
    clamp_u_target: float | None = 4.0
    # MeanFlow's adaptive weight w = alpha / (||delta||^2 + eps), applied with stop-gradient.
    adaptive_loss_eps: float = 1e-3
    adaptive_loss: bool = True

    # allow the alpha = 0 limit: once the schedule's clamp floors alpha at 0, the loss switches to
    # the true MeanFlow target v_t - (t-r) du/dt, which needs a JVP through the action expert. With
    # the default whole-run schedule that switch happens at ~71% progress (the official tail); with
    # alpha_init = alpha_final = 0 the whole run is JVP MeanFlow. Off = the run floors at
    # alpha_clamp and never touches the JVP (the reference's `discrete_training`).
    meanflow_jvp: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0AlphaFlowConfig supports pi05=True only (r conditioning uses adaRMS).")
        if self.time_pair not in ("minmax", "scaled"):
            raise ValueError(f"time_pair must be 'minmax' or 'scaled', got {self.time_pair!r}")
        if not (0.0 <= self.alpha_anneal_start <= self.alpha_anneal_end <= 1.0):
            raise ValueError(
                "alpha_anneal window must satisfy 0 <= start <= end <= 1, "
                f"got [{self.alpha_anneal_start}, {self.alpha_anneal_end}]"
            )
        if not (0.0 <= self.fm_ratio <= 1.0):
            raise ValueError(f"fm_ratio must be in [0, 1], got {self.fm_ratio}")
        if self.alpha_floor is None:
            object.__setattr__(self, "alpha_floor", 0.0 if self.meanflow_jvp else self.alpha_clamp)
        if not self.meanflow_jvp and self.alpha_floor == 0.0:
            raise ValueError("alpha_floor=0 needs meanflow_jvp=True (alpha=0 has no discrete target)")
        if self.meanflow_jvp and self.alpha_floor != 0.0:
            raise ValueError("meanflow_jvp=True with a nonzero floor never reaches the JVP: set alpha_floor=0")

    @property
    def pinned_jvp(self) -> bool:
        """Is the WHOLE run the alpha = 0 JVP limit (no discrete branch ever)?"""
        return self.meanflow_jvp and self.alpha_init == 0.0 and self.alpha_final == 0.0

    @property
    def two_pass(self) -> bool:
        """Does the objective ever need the second (stop-gradient) forward at z_s?"""
        return not self.pinned_jvp and not (self.alpha_init == 1.0 and self.alpha_final == 1.0)

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0AlphaFlow:
        return Pi0AlphaFlow(self, rngs=nnx.Rngs(rng))


def _ratio_schedule(progress, *, init: float, final: float, p_start: float, p_end: float, gamma: float):
    """Reference `get_ratio` with scheduler='sigmoid', over training PROGRESS in [0, 1].

    Progress rather than absolute steps so the curriculum rescales with --num-train-steps; the
    sigmoid's temperature acts within the [p_start, p_end] window exactly as the reference's does
    within [k_start, k_end].
    """
    q = jnp.asarray(progress, jnp.float32)
    span = max(float(p_end - p_start), 1e-8)
    mid = p_start + (p_end - p_start) / 2.0
    val = init + (final - init) * jax.nn.sigmoid((q - mid) / span * gamma)
    val = jnp.where(q < p_start, init, val)
    return jnp.where(q > p_end, final, val)


def clamp_alpha(alpha, *, eta: float, floor: float):
    """Reference `get_ratio`'s clamping: snap to 1 above 1-eta, to `floor` below eta.

    Above 1-eta the discrete target is within eta of v_t anyway, and L_TFM computes it with one
    forward instead of two -- the clamp is a compute saving, not a change of objective.
    """
    alpha = jnp.where(alpha > 1.0 - eta, 1.0, alpha)
    return jnp.where(alpha < eta, floor, alpha)


def alpha_flow_target(v_t, u_s, alpha, *, is_border):
    """u_tgt = alpha * v_t + (1 - alpha) * u^-(z_s, r, s), with the r == t rows falling back to v_t.

    alpha == 1 gives trajectory flow matching (u_tgt = v_t); alpha == 1/2 with u_s read at the
    midpoint is the Shortcut model; alpha -> 0 approaches MeanFlow. `is_border` marks rows sampled
    with r == t, which have no interval to be consistent over.
    """
    a = alpha[..., None, None] if jnp.ndim(alpha) else alpha
    tgt = a * v_t + (1.0 - a) * u_s
    return jnp.where(is_border[..., None, None], v_t, tgt)


def intermediate_state(x_t, v_t, t, r, alpha):
    """z_s and s for the discrete branch: one Euler step of size dt = alpha * (t - r) along v_t."""
    dt = alpha * (t - r)
    return x_t - dt[..., None, None] * v_t, t - dt


def adaptive_weight(delta2, weight_num, *, eps: float):
    """MeanFlow's adaptive weight, stop-gradient'd: w = numerator / (||delta||^2 + eps).

    The numerator is alpha on discrete rows and 1 on border/JVP rows (reference `weight_c`/`weight_d`),
    which is what keeps the two branches on a common scale as alpha shrinks.
    """
    return jax.lax.stop_gradient(weight_num / (delta2 + eps))


class Pi0AlphaFlow(Pi0):
    """pi05 whose action expert predicts a MEAN velocity u(z_t, r, t) instead of v(z_t, t)."""

    # train.py passes progress = step / num_train_steps so the curriculum is a function of it.
    wants_progress = True

    def __init__(self, config: Pi0AlphaFlowConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        width = self.action_in_proj.out_features
        # r embedding, ZERO-initialised at the output so a fresh alpha-Flow model IS the pretrained
        # pi05: u(z_t, r, t) = v_pi05(z_t, t) for every r, which is exactly the alpha=1, r=t case.
        self.r_mlp_in = nnx.Linear(width, width, rngs=rngs)
        self.r_mlp_out = nnx.Linear(
            width, width, kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.af_config = config

    # ------------------------------------------------------------------ forward

    def _adarms(self, t: at.Array, r: at.Array) -> at.Array:
        """pi05's time conditioning plus the (zero-init) r term."""
        width = self.action_in_proj.out_features
        t_emb = posemb_sincos(t, width, min_period=4e-3, max_period=4.0)
        t_emb = nnx.swish(self.time_mlp_out(nnx.swish(self.time_mlp_in(t_emb))))
        r_emb = posemb_sincos(r, width, min_period=4e-3, max_period=4.0)
        r_emb = self.r_mlp_out(nnx.swish(self.r_mlp_in(r_emb)))
        return t_emb + r_emb

    def _u(self, observation, prefix_mask, kv_cache, x_t, t, r):
        """Mean velocity u(z_t, r, t) from a cached prefix. [b, ah, ad]"""
        action_tokens = self.action_in_proj(x_t)
        suffix_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        suffix_ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=action_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, action_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, self._adarms(t, r)],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    # ------------------------------------------------------------------ loss

    def _sample_t_r(self, rng, batch_shape, fm_ratio):
        cfg = self.af_config
        k1, k2, k3 = jax.random.split(rng, 3)
        t1 = jax.random.beta(k1, 1.5, 1, batch_shape) * 0.999 + 0.001
        if cfg.time_pair == "minmax":
            t2 = jax.random.beta(k2, 1.5, 1, batch_shape) * 0.999 + 0.001
            t, r = jnp.maximum(t1, t2), jnp.minimum(t1, t2)
        else:  # "scaled"
            t = t1
            r = t1 * jax.random.uniform(k2, batch_shape)
        # a `fm_ratio` share of the batch trains the border case r = t (plain flow matching), which
        # is the boundary condition that keeps the consistency term from collapsing.
        is_fm = jax.random.uniform(k3, batch_shape) < fm_ratio
        return t, jnp.where(is_fm, t, r), is_fm

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress: at.Array | float | None = None,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        cfg = self.af_config
        progress = 0.0 if progress is None else progress
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        alpha = _ratio_schedule(
            progress,
            init=cfg.alpha_init,
            final=cfg.alpha_final,
            p_start=cfg.alpha_anneal_start,
            p_end=cfg.alpha_anneal_end,
            gamma=cfg.alpha_gamma,
        )
        alpha = clamp_alpha(alpha, eta=cfg.alpha_clamp, floor=cfg.alpha_floor)
        fm_ratio = jnp.asarray(cfg.fm_ratio, jnp.float32)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        t, r, is_fm = self._sample_t_r(time_rng, batch_shape, fm_ratio)
        x_t = t[..., None, None] * noise + (1 - t[..., None, None]) * actions
        v_t = noise - actions  # the instantaneous (conditional) velocity

        prefix_mask, kv_cache = self._prefix_forward(observation)

        def f(x, tt, rr):
            return self._u(observation, prefix_mask, kv_cache, x, tt, rr)

        def jvp_parts(_):
            # alpha = 0: u_tgt = v_t - (t - r) du/dt, with du/dt the total derivative along the
            # trajectory -- tangents (v_t, 1, 0) for (z_t, t, r). No gradient flows into the target.
            # Border rows (r = t) degrade gracefully: (t - r) = 0 makes their target exactly v_t.
            u_p, dudt = jax.jvp(f, (x_t, t, r), (v_t, jnp.ones_like(t), jnp.zeros_like(r)))
            u_tgt_p = v_t - (t - r)[..., None, None] * jax.lax.stop_gradient(dudt)
            dudt_mag = jnp.max(jnp.abs(dudt))
            return u_p, u_tgt_p, jnp.ones_like(t), dudt_mag

        def discrete_parts(_):
            # z_s is one Euler step of size dt = alpha*(t-r) along v_t; the target blends v_t with
            # the model's own mean velocity on the remaining [r, s] leg.
            u_p = f(x_t, t, r)
            z_s, s_mid = intermediate_state(x_t, v_t, t, r, alpha)
            u_s = jax.lax.stop_gradient(f(z_s, s_mid, r))
            is_border = is_fm | (alpha >= 1.0)
            u_tgt_p = alpha_flow_target(v_t, u_s, alpha, is_border=is_border)
            return u_p, u_tgt_p, jnp.where(is_border, 1.0, alpha), jnp.zeros(())

        if cfg.pinned_jvp:
            u, u_tgt, weight_num, dudt_mag = jvp_parts(None)
        elif cfg.meanflow_jvp:
            # The official tail: the run anneals discretely and SWITCHES to the JVP target once the
            # clamp floors alpha at 0 (~71% progress on the default schedule). alpha is traced, so
            # the switch is a lax.cond -- both branches compile once, one executes per step.
            u, u_tgt, weight_num, dudt_mag = jax.lax.cond(alpha <= 0.0, jvp_parts, discrete_parts, None)
        elif cfg.two_pass:
            u, u_tgt, weight_num, dudt_mag = discrete_parts(None)
        else:
            u = f(x_t, t, r)
            u_tgt, weight_num, dudt_mag = v_t, jnp.ones_like(t), jnp.zeros(())

        if cfg.clamp_u_target is not None:
            u_tgt = jnp.clip(u_tgt, -cfg.clamp_u_target, cfg.clamp_u_target)
        delta2 = jnp.mean(jnp.square(u - jax.lax.stop_gradient(u_tgt)), axis=(-1, -2))  # [b]

        if cfg.adaptive_loss:
            weight = adaptive_weight(delta2, weight_num, eps=cfg.adaptive_loss_eps)
        else:
            weight = jax.lax.stop_gradient(weight_num)
        loss = weight * delta2

        # WATCH delta2, NOT loss. MeanFlow's adaptive weight makes the reported loss
        # ||delta||^2 / (||delta||^2 + eps), which saturates at ~1 whenever ||delta||^2 >> eps=1e-3 --
        # so the loss curve sits flat near 1.0 for the whole run and says nothing. That is by design
        # (the stop-gradient weight turns the update into a normalised gradient, ~ d log ||delta||),
        # not a stall. delta2 is the raw error, and loss_tfm is the pi05 BC loss with an extra r
        # input -- logged raw so alpha-Flow runs stay comparable to plain BC finetunes, whose number
        # it is on the alpha = 1 phase.
        aux = {
            "alpha": alpha,
            "fm_ratio": fm_ratio,
            "delta2": jnp.mean(delta2),
            "loss_tfm": jnp.mean(jnp.square(u - v_t)),
            "fm_frac": jnp.mean(is_fm.astype(jnp.float32)),
            "t_mean": jnp.mean(t),
            "gap_mean": jnp.mean(t - r),
            # JVP-tail watchdogs: dudt_mag spiking (or u_tgt riding its clip) is the early signature
            # of the known MeanFlow JVP loss explosion -- visible well before delta2 goes non-finite.
            "jvp_active": (alpha <= 0.0).astype(jnp.float32),
            "dudt_absmax": dudt_mag,
            "u_tgt_absmax": jnp.max(jnp.abs(u_tgt)),
            "u_absmax": jnp.max(jnp.abs(u)),
        }
        return loss, aux

    # ------------------------------------------------------------------ sampling

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 1,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Few-step mean-velocity sampling: z_r = z_t - (t-r) u(z_t, r, t) on a uniform grid.

        num_steps=1 is the one-step jump this whole model exists for. Larger values subdivide [1, 0];
        num_steps=10 with r=t at every step would instead be the original pi05 ODE, which the model
        still represents (u(z,t,t) is the instantaneous velocity) -- see `sample_actions_ode`.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_mask, kv_cache = self._prefix_forward(observation)

        # trace-compatible (num_steps may be a tracer when the caller jits sample_kwargs, as
        # Policy.infer does) -- a fori_loop, not a Python loop.
        dt = 1.0 / jnp.asarray(num_steps, jnp.float32)

        def body(i, x):
            t = jnp.full((batch_size,), 1.0 - i * dt)
            r = jnp.full((batch_size,), 1.0 - (i + 1) * dt)
            return x - dt * self._u(observation, prefix_mask, kv_cache, x, t, r)

        return jax.lax.fori_loop(0, jnp.asarray(num_steps, jnp.int32), body, noise)

    def sample_n_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_samples: int,
        num_steps: int = 1,
    ) -> at.Float[at.Array, "n ah ad"]:
        """N candidate chunks from ONE prefix pass -- the alpha-Flow analogue of Pi0RLT's
        ``extract_token_and_base_actions``, for value-guided (best-of-N / adaptive-commit) serving.

        The expensive VLM prefix runs once (batch 1); its KV cache and mask are tiled across the
        N noise draws, which then cost N suffix forwards each of `num_steps` (default 1 -- with the
        one-step gate passed, BoN-16 costs about 1.6 pi05 draws instead of 16). Model-space output
        [N, H, action_dim]; the caller unnormalizes/scores.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        if observation.state.shape[0] != 1:
            raise ValueError("sample_n_actions expects a batch-1 observation (one live frame)")
        prefix_mask, kv_cache = self._prefix_forward(observation)
        kv_n = jax.tree.map(lambda x: jnp.repeat(x, num_samples, axis=1), kv_cache)  # KVCache is [layers, BATCH, ...]
        mask_n = jnp.repeat(prefix_mask, num_samples, axis=0)
        noise = jax.random.normal(rng, (num_samples, self.action_horizon, self.action_dim))

        dt = 1.0 / jnp.asarray(num_steps, jnp.float32)

        def body(i, x):
            t = jnp.full((num_samples,), 1.0 - i * dt)
            r = jnp.full((num_samples,), 1.0 - (i + 1) * dt)
            return x - dt * self._u(observation, mask_n, kv_n, x, t, r)

        return jax.lax.fori_loop(0, jnp.asarray(num_steps, jnp.int32), body, noise)

    def sample_actions_ode(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """The original pi05 ODE, read off the same weights via r = t (instantaneous velocity).

        Useful as a control: it isolates "did alpha-Flow damage the policy" from "is one step enough".
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_mask, kv_cache = self._prefix_forward(observation)

        dt = 1.0 / jnp.asarray(num_steps, jnp.float32)

        def body(i, x):
            t = jnp.full((batch_size,), 1.0 - i * dt)
            return x - dt * self._u(observation, prefix_mask, kv_cache, x, t, t)

        return jax.lax.fori_loop(0, jnp.asarray(num_steps, jnp.int32), body, noise)
