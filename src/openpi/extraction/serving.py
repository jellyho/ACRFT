"""Sampler variants for the policy-extraction arms that need the CRITIC's inputs at inference.

There is one serving entry point in this repo -- ``scripts/serve_policy.py`` -- and arms reach it
two ways, both of them existing conventions:

  weight-only arms (awr, cfgrl, flowdpg, qam, dql)
      exported to ordinary openpi checkpoints by ``scripts/export_extraction_checkpoint.py``, so
      they serve as any checkpoint does:  --policy.config <name> --policy.dir <exported>.
      Nothing in the serving path knows they came from an extraction run.
  critic-consuming arms (qpilots, idql/bon, lps, lpsd, flowdagger)
      served through the existing ``--critic`` wrapper (PatchCriticSelectPolicy), which already
      owns the frozen DINOv2 backbone and the critic, selected by ``--critic-mode``. This module
      supplies only the per-mode sampling math those modes call.

The samplers below take the pooled patch features + proprio the wrapper already computes:

  expert-overlay arms (awr, cfgrl, flowdpg, qam, dql)
      the trainer saved an orbax {"expert": ...} subtree; we overlay it on the BC params and
      sample normally. CFGRL additionally carries ``opt_embed`` and samples with classifier-free
      guidance at weight ``cfg_w`` (kvfrans/cfgrl iql_diffusion.py:205-216).
  latent-actor arms (lps, lpsd)
      the frozen alpha-Flow base stays as-is; a small MLP picks the latent z and the action is
      the one-step map z - u(z, r=0, t=1) (lps.py:294-327). lpsd draws e ~ N(0, I) per call.
  seed-steering arm (flowdagger)
      the steering head predicts DCT coefficients of the sampler's initial noise; the base
      sampler then runs from that seed (microsoft/FlowDAgger serving path).
  test-time arms (qpilots, idql/bon)
      the policy weights are the BC ones; the frozen patch critic acts at inference — QPILOTS-U
      steers each Euler step (arXiv 2606.14801 Eq. 14/15/17), IDQL/BoN draw N chunks and keep the
      argmax of min-ensemble Q (ddpm_iql_learner.py:360-374).

The critic-using arms need the DINOv2 patch features of the live camera images, which is what
``patch_critic_policy`` already builds for BoN/adaptive serving; we reuse its preprocessing so a
served arm sees exactly the critic's training-time inputs.

    from openpi.extraction import serving
    policy = serving.load_arm("qpilots", alpha=0.2)      # or "dql", "lps", ...
    action = policy.infer(obs)["actions"]
"""

# ruff: noqa: PLC0415  (heavy/optional deps imported lazily, as elsewhere in extraction/)

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_steered as _pi0_steered
from openpi.training import config as _config
from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

# 200000, not 100000: every robot evaluation in this project ran the 200k step (user, 2026-09-06), and until now every default here said 100k -- so an arm trained from 100k would have had its expert subtree overlaid on a base it was never fine-tuned against.
BC_CKPT = pathlib.Path("/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/200000")
AF_CKPT = pathlib.Path("/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000")
CRITIC = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k")
CKPT_ROOT = pathlib.Path("/data1/jellyho/acrft_ckpts/extraction")

EXPERT_ARMS = ("awr", "cfgrl", "flowdpg", "qam", "dql")
LATENT_ARMS = ("lps", "lpsd")
#: Arms whose CHUNK comes from ArmChunkSampler rather than the served policy's own sampler. The
#: serving wrapper reuses everything else (patch features, robot-space decode, scoring, HUD), so
#: it needs to know which modes route here -- and that list belongs with the arms, not in the
#: wrapper, where it was a second hand-maintained copy of the same four names.
SAMPLER_ARMS = ("qpilots", *LATENT_ARMS, "flowdagger")
#: Arms that draw by steering a flow, and so have an alpha=0 twin of the same draw to compare
#: against. Only qpilots today; the constant exists so that adding a second steering arm is a
#: one-line change here rather than a hunt for `== "qpilots"` across the serving stack.
STEERING_ARMS = frozenset({"qpilots"})
#: What a steering arm ascends. See ArmSpec.steer_value -- `negated` and `random` are the controls
#: that separate a wrong-direction gradient from an over-large injection.
STEER_VALUES = ("critic", "negated", "random")

CRITIC_ARMS = ("qpilots", "idql", "bon")
ALL_ARMS = (*EXPERT_ARMS, *LATENT_ARMS, "flowdagger", *CRITIC_ARMS, "bc")


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _restore_mlp(path: pathlib.Path):
    """[w, b] lists as written by train_lps / train_flowdagger."""
    raw = flax.serialization.msgpack_restore(pathlib.Path(path).read_bytes())
    raw = list(raw.values()) if isinstance(raw, dict) else raw
    return [(jnp.asarray(p[0]), jnp.asarray(p[1])) for p in raw]


def _mlp(params, x, *, tanh_scale: float | None = None):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    out = x @ w + b
    return jnp.tanh(out) * tanh_scale if tanh_scale else out


@dataclasses.dataclass
class ArmSpec:
    """What a served arm is made of — everything needed to reproduce it."""

    arm: str
    base_ckpt: pathlib.Path
    expert_ckpt: pathlib.Path | None = None
    latent_actor: pathlib.Path | None = None
    steering_head: pathlib.Path | None = None
    critic: pathlib.Path | None = None
    cfg_w: float = 1.5
    alpha: float = 0.2
    rho: float = 0.5
    n_samples: int = 8
    ode_steps: int = 10

    #: WHAT the steering ascends. The robot alpha sweep (1.80, 1.10, 1.30, 1.00, 1.20, 0.30 at
    #: alpha = 0, .005, .01, .025, .05, .1) is not a dose-response: the middle four are mutually
    #: indistinguishable (Kruskal p=0.45) across an 8x range of injected displacement, and only
    #: alpha=0.1 collapses. A magnitude threshold is DIRECTION-AGNOSTIC, so that sweep alone cannot
    #: tell "the critic's gradient points the wrong way" from "an injection that large damages the
    #: action whatever its direction". These modes are the control that separates them, and they
    #: cost one substitution because the value function was already injected rather than baked in:
    #:   critic   ascend the pessimistic ensemble Q (QPILOTS-U as published)
    #:   negated  ascend -Q. If this BEATS `critic`, the gradient is anti-correlated with quality.
    #:   random   ascend <a_hat, u> for one fixed random unit direction u per chunk, supported on
    #:            exactly the sub-array the critic reads. Eq. 17 rescales every gradient to the
    #:            drift norm, so this arm injects the SAME displacement magnitude at the same alpha
    #:            and differs only in direction. If it also collapses at alpha=0.1, the damage is
    #:            the injection, not the critic.
    #: `random` is coherent across the chunk rather than resampled per Euler step, deliberately: a
    #: per-step redraw random-walks and partly cancels, which would understate a systematically
    #: wrong direction and make the control easier to pass than the hypothesis it is testing.
    #:
    #: Caveat, measured in pi0_steered_test.py: the arms are magnitude-matched at INJECTION, not
    #: exactly at the output. `sample_steered` returns clip(x, -1, 1), so where a coordinate sits on
    #: the box boundary an outward push is truncated and an inward one is not, and the realized
    #: displacement becomes direction-dependent (~1.6x across directions in that fixture, where 37.5%
    #: of coordinates were on the boundary). Report the realized displacement per arm from the
    #: unsteered twin rather than assuming alpha fixes it.
    steer_value: str = "critic"


def default_spec(arm: str, step: int | None = None, **over: Any) -> ArmSpec:
    """Conventional checkpoint layout of the extraction runs (`<arm>_run1/<step>`)."""
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {ALL_ARMS}")
    spec = ArmSpec(arm=arm, base_ckpt=AF_CKPT if arm in LATENT_ARMS else BC_CKPT)
    run = CKPT_ROOT / f"{arm}_run1"
    # An arm saved in the BC layout carries its own base (arm_meta.json) and is servable as-is; a
    # LEGACY expert-only arm is a subtree of absolute weights co-adapted with the backbone it was
    # trained on, so serving it on any other base is silently wrong. Neither can be inferred from a
    # module-level constant, which is exactly the mistake this guards: BC_CKPT said 100000 for
    # months while the robot ran 200000, and moving the constant to 200000 would have re-based every
    # already-trained expert arm without a word.
    LEGACY_EXPERT_BASE = BC_CKPT.with_name("100000")
    if arm in EXPERT_ARMS:
        steps = sorted((int(p.name) for p in run.iterdir() if p.name.isdigit()), reverse=True)
        if not steps:
            raise FileNotFoundError(f"no checkpoints under {run}")
        spec.expert_ckpt = run / str(step or steps[0])
        meta = spec.expert_ckpt / "arm_meta.json"
        if meta.exists():
            spec.base_ckpt = pathlib.Path(json.loads(meta.read_text())["init_ckpt"])
        else:
            spec.base_ckpt = LEGACY_EXPERT_BASE
            logging.warning(
                "%s has no arm_meta.json (pre-2026-09-06 checkpoint); assuming it was trained from %s, "
                "which was the trainers' --init-ckpt default at the time. Serving it on any other base "
                "pairs an expert subtree with a backbone it was never fine-tuned against.",
                spec.expert_ckpt,
                LEGACY_EXPERT_BASE.name,
            )
    elif arm in LATENT_ARMS:
        cands = sorted(run.glob("latent_actor_*.msgpack"), key=lambda p: int(p.stem.split("_")[-1]), reverse=True)
        if not cands:
            raise FileNotFoundError(f"no latent actor under {run}")
        spec.latent_actor = run / f"latent_actor_{step}.msgpack" if step else cands[0]
    elif arm == "flowdagger":
        spec.steering_head = run
    # every feature-consuming arm loads the critic bundle: CRITIC_ARMS use its Q, while
    # lps/lpsd/flowdagger need its feature layout AND proprio slice (their MLPs were trained on
    # exactly that rep -- see train_lps.py / train_flowdagger.py)
    if arm in CRITIC_ARMS or arm in LATENT_ARMS or arm == "flowdagger":
        spec.critic = CRITIC
    for k, v in over.items():
        setattr(spec, k, v)
    if spec.steer_value not in STEER_VALUES:
        raise ValueError(f"unknown steer_value {spec.steer_value!r}; known: {STEER_VALUES}")
    return spec


class ArmChunkSampler:
    """Produces the action chunk for one critic-consuming arm.

    Called by ``PatchCriticSelectPolicy`` (the ``--critic-mode`` wrapper), which already owns the
    frozen DINOv2 backbone, the critic and the robot-space decode — this only decides the chunk.
    ``__call__(rng, observation, patches, proprio) -> [N, H, AD]`` in the model's normalized space,
    the same array the base sampler would have returned, so nothing downstream changes.

    For lps/lpsd the base model is the frozen alpha-Flow checkpoint (a different network from the
    served BC policy), so it is loaded here; for qpilots the served policy's own model is used.
    """

    #: Also return the unsteered twin, so a deploy recording can say how far steering displaced
    #: the action. Off by default: it changes the returned N, and every caller that only wants the
    #: chunk should keep getting one. Only meaningful where `offers_unsteered_twin` is true.
    pair_unsteered: bool = False

    #: `(k, c)` mapping a policy-normalized chunk into the critic's action space, and the critic's
    #: own horizon. Set by the serving wrapper, which owns both transforms; the identity default is
    #: correct only when the two were normalized alike, so the wrapper always sets it rather than
    #: leaving it to chance.
    to_critic_space: tuple = (1.0, 0.0)
    critic_horizon: int | None = None

    @property
    def offers_unsteered_twin(self) -> bool:
        """Whether an alpha=0 draw of this arm is a meaningful reference for its steered one.

        Asked by the serving wrapper so that it does not have to know which arms steer. It knows
        it wants a twin; which arms HAVE one is a property of the arm, and the arms live here.
        """
        return self.spec.arm in STEERING_ARMS

    def __init__(self, spec: ArmSpec, served_model=None):
        self.spec = spec
        self._warned_rho = False
        if spec.arm == "qpilots" and served_model is not None:
            # Steer the policy that is actually being served, whatever checkpoint it came from.
            # Re-tagged as the subclass that owns the steered sampler; it shares the served
            # model's parameters and leaves the served object itself untouched (see wrap()).
            self.model = _pi0_steered.Pi0Steered.wrap(served_model)
            self.graphdef = nnx.graphdef(self.model)
            self.params = jax.device_put(nnx.state(served_model))
            self.H = served_model.action_horizon
            self.AD = served_model.action_dim
        else:
            self._load_base(spec)

        self.actor = _restore_mlp(spec.latent_actor) if spec.latent_actor else None
        self.head = self.basis = None
        if spec.steering_head:
            d = pathlib.Path(spec.steering_head)
            self.head = _restore_mlp(d / "steering_head.msgpack")
            self.basis = jnp.asarray(np.load(d / "dct_basis.npy"))
        self.critic = None
        if spec.critic:
            from openpi.extraction import critic_q

            self.critic = critic_q.load(spec.critic)

    def _load_base(self, spec: ArmSpec):
        mcfg = (
            _config.get_config("pi05_yam_lego_taxi_alphaflow").model
            if spec.arm in LATENT_ARMS
            else _config.get_config("pi05_yam_lego_taxi").model
        )
        self.H, self.AD = mcfg.action_horizon, mcfg.action_dim
        model = mcfg.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(
            CheckpointWeightLoaderKeepMissing(str(spec.base_ckpt / "params")).load(state.to_pure_dict())
        )
        self.model = nnx.merge(graphdef, state)
        self.graphdef = nnx.graphdef(self.model)
        self.params = jax.device_put(nnx.state(self.model))

    # No sampler pieces live here any more. _prefix / _velocity / _euler were hand-copies of
    # pi0.py's sampler, and every arm now calls the model instead: qpilots through
    # Pi0Steered.sample_steered, lps/lpsd through Pi0AlphaFlow.decode_latent, flowdagger through
    # Pi0.sample_actions(noise=seed). A copy does not fail when it drifts -- it silently serves a
    # base that is no longer the base being served, which is how this ring once lost nine arms.

    #: Below this many members, the `mean - rho*std` read is refused and falls back to the mean.
    #: rho=0.5 is transplanted from QAM (agents/qam.py:33) and so is the reduction -- but QAM sizes
    #: its ensemble for it at num_qs=10 (qam.py:424; RLPD's config.py:7 likewise). At K=2, jnp.std
    #: with ddof=0 is just |q1-q2|/2: a single-sample estimate carrying ~75% relative sampling error,
    #: and it is DIFFERENTIATED, because this same read is the value function QPILOTS steers along.
    #: Measured on 64 on-manifold states: the std term contributes 59.6 of gradient magnitude against
    #: 123.5 from the mean term -- a third of every steering step -- in a direction near-orthogonal to
    #: it (cos -0.078). That is not pessimism, it is noise injected into the drift, and the alpha
    #: sweep it produced fell to 0.30 at alpha=0.1. Ten members is where the transplant came from, so
    #: ten is where it is allowed.
    MIN_MEMBERS_FOR_PESSIMISM = 10

    def _q(self, feats, chunk, proprio, *, reduce: str):
        logits = self.critic.net.apply({"params": self.critic.params}, feats, chunk, proprio)
        q = self.critic.hl.from_logits(logits)[..., -1]  # [K, B]
        if reduce == "min":
            return q.min(axis=0)
        k = q.shape[0]
        if self.spec.rho and k < self.MIN_MEMBERS_FOR_PESSIMISM:
            if not self._warned_rho:
                object.__setattr__(self, "_warned_rho", True)
                logging.warning(
                    "critic has K=%d members; refusing the rho=%.2f pessimistic read (needs K>=%d) and "
                    "using the ensemble MEAN. At K=%d, std is |q1-q2|/2 -- a single-sample estimate that "
                    "contributed ~34%% of the steering gradient in a direction near-orthogonal to the "
                    "value gradient. Train a K>=%d critic to enable it.",
                    k,
                    self.spec.rho,
                    self.MIN_MEMBERS_FOR_PESSIMISM,
                    k,
                    self.MIN_MEMBERS_FOR_PESSIMISM,
                )
            return q.mean(axis=0)
        return q.mean(axis=0) - self.spec.rho * q.std(axis=0)  # pessimistic, QPILOTS Eq. 12

    @functools.cached_property
    def _steer_jit(self):
        """`(state, rng, obs, feats, proprio, ad, alpha, paired) -> chunk`, compiled once.

        Follows nnx_utils.module_jit's shape -- split the module, pass the state, merge inside --
        because that is what the rest of the repo does and why the other paths compile as one XLA
        module instead of a pile of eager kernels. `alpha` stays traced, so the steered draw and
        its alpha=0 twin share a single compilation; `ad` and `paired` are static.

        The integration itself is NOT here. It is `Pi0Steered.sample_steered`, next to the sampler
        it modifies. What this layer contributes is the one thing the model must not know: what
        the value IS.
        """
        graphdef = self.graphdef

        ch = self.critic_horizon

        def fun(state, rng, obs, feats, proprio, ad, alpha, paired, k, c):
            model = nnx.merge(graphdef, state)
            x0 = jax.random.normal(rng, (obs.state.shape[0], self.H, self.AD))

            sign = -1.0 if self.spec.steer_value == "negated" else 1.0
            # One direction per chunk, drawn from the SAME rng the noise came from so a rerun of
            # this arm is bit-reproducible, and folded so it is independent of x0.
            u = jax.random.normal(jax.random.fold_in(rng, 0x5EED), (ch, ad))
            u = u / (jnp.linalg.norm(u) + 1e-8)

            def value_fn(a_hat):
                # Into the CRITIC's space before scoring, and this is the whole reason the caller
                # passes k/c: `a_hat` is normalized by the POLICY's statistics, and the critic was
                # trained under its own. The selection path has always routed candidates through
                # physical units for exactly this; steering fed the raw array straight in, so the
                # gradient it followed was taken in a displaced copy of the critic's space while
                # the scores RECORDED for the same chunk were taken in the right one.
                #
                # Sliced to the critic's horizon too, which the selection path also does: a critic
                # shorter than the policy's chunk (h30 critic, h50 policy) was otherwise being
                # handed 50 steps it was never trained to read.
                if self.spec.steer_value == "random":
                    # Supported on exactly the sub-array the critic reads, so the control does not
                    # get to push on padding dimensions the critic never touches. No k/c: an
                    # affine map of the argument cannot change a random direction into a less
                    # random one, and leaving it out keeps u a unit vector in the space it lives in.
                    return jnp.sum(a_hat[:, :ch, :ad] * u)
                a = a_hat[:, :ch, :ad] * k + c
                return sign * self._q(feats, a, proprio, reduce="pess").sum()

            def draw(a):
                # preprocess=False: __call__ has already preprocessed, and doing it twice would
                # put a second normalisation between the two draws.
                return model.sample_steered(
                    rng,
                    obs,
                    value_fn=value_fn,
                    alpha=a,
                    num_steps=self.spec.ode_steps,
                    noise=x0,
                    preprocess=False,
                )

            steered = draw(alpha)
            if not paired:
                return steered
            # The twin: same x0, same observation, same everything but alpha. Both draws recompute
            # the prefix rather than sharing one pass, which reads as waste and is not: they are
            # the identical pure subcomputation inside a single jit, so XLA folds them together.
            return jnp.concatenate([steered, draw(0.0)], axis=0)  # index 0 is what executes

        return jax.jit(fun, static_argnums=(5, 7))

    def __call__(self, rng, observation, patches, proprio):
        """-> chunk [N, H, AD] in the model's normalized space (N=1 for these arms).

        `patches` are the critic's pooled DINOv2 features and `proprio` its state slice, both
        already computed by the serving wrapper for scoring — we reuse them rather than recomputing.
        """
        spec = self.spec
        obs = _model.preprocess_observation(None, observation, train=False)
        b = obs.state.shape[0]
        feats = patches if patches.ndim == 3 else patches[None]
        proprio = proprio if proprio.ndim == 2 else proprio[None]

        if spec.arm == "qpilots":
            # FIRST, and before any eager work: everything this arm needs happens inside
            # _steer_jit. Reconstructing the 3B model with nnx.merge and running an un-jitted
            # prefix pass here -- only to have the jitted function compute the prefix again --
            # cost 3.2 s per inference against 150 ms for best-of-N through the same wrapper.
            # Measured: with the steering loop reduced to ZERO gradient steps it still took
            # 3229 ms, which is how the overhead was found to be entirely outside the loop.
            ad = self.critic.config["action_dim"]
            k, c = self.to_critic_space
            if np.ndim(k) != 2:
                # The scalar default would compile a SECOND graph -- a different shape is a
                # different jit -- so the warm-up would have warmed one nothing serves and the real
                # compile would land on a live request. Refuse rather than silently recompile.
                raise RuntimeError(
                    "the policy->critic action map was never set; PatchCriticSelectPolicy sets it "
                    "in _set_critic_space, from both infer and warmup, and both must run."
                )
            return self._steer_jit(self.params, rng, obs, feats, proprio, ad, spec.alpha, self.pair_unsteered, k, c)

        # The remaining arms still build the model here; they are single-forward paths, so the
        # merge is not repeated inside a loop the way qpilots' prefix was.
        model = nnx.merge(self.graphdef, self.params)

        if spec.arm in LATENT_ARMS:
            # The arm's contribution is the latent; decoding it is the model's, and is called
            # rather than rebuilt here out of `_u` and a prefix pass.
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            if spec.arm == "lpsd":
                rep = jnp.concatenate([rep, jax.random.normal(rng, (b, self.H * self.AD))], axis=-1)
            z = _mlp(self.actor, rep).reshape(b, self.H, self.AD)
            return model.decode_latent(obs, z, preprocess=False)

        if spec.arm == "flowdagger":
            # Likewise: the arm's contribution is the SEED (a DCT-parameterised displacement of the
            # noise the policy would otherwise have drawn), and integrating it is the base
            # sampler's job. `sample_actions` already accepts a seed, so the hand-copied Euler loop
            # that used to be here was the model's own sampler written out a second time --
            # measured identical to 1.2e-07, i.e. fp32 round-off, on the dummy variant.
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            coeffs = _mlp(self.head, rep, tanh_scale=3.0).reshape(b, self.basis.shape[0], self.AD)
            seed = jnp.einsum("kh,bkd->bhd", self.basis, coeffs)
            # obs is already preprocessed; preprocess_observation is idempotent at train=False
            # (resize is a no-op at the right resolution, the mask fill is a fill).
            return model.sample_actions(rng, obs, num_steps=spec.ode_steps, noise=seed)

        raise ValueError(
            f"{spec.arm!r} is not sampled here: bon/idql are the wrapper's own selection path, and "
            "the weight-only arms serve as exported checkpoints (export_extraction_checkpoint.py)."
        )
