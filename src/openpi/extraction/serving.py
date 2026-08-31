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
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

BC_CKPT = pathlib.Path("/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/100000")
AF_CKPT = pathlib.Path("/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000")
CRITIC = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k")
CKPT_ROOT = pathlib.Path("/data1/jellyho/acrft_ckpts/extraction")

EXPERT_ARMS = ("awr", "cfgrl", "flowdpg", "qam", "dql")
LATENT_ARMS = ("lps", "lpsd")
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


def default_spec(arm: str, step: int | None = None, **over: Any) -> ArmSpec:
    """Conventional checkpoint layout of the extraction runs (`<arm>_run1/<step>`)."""
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {ALL_ARMS}")
    spec = ArmSpec(arm=arm, base_ckpt=AF_CKPT if arm in LATENT_ARMS else BC_CKPT)
    run = CKPT_ROOT / f"{arm}_run1"
    if arm in EXPERT_ARMS:
        steps = sorted((int(p.name) for p in run.iterdir() if p.name.isdigit()), reverse=True)
        if not steps:
            raise FileNotFoundError(f"no checkpoints under {run}")
        spec.expert_ckpt = run / str(step or steps[0])
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

    #: Also return the unsteered twin (qpilots only), so a deploy recording can say how far
    #: steering displaced the action. Off by default: it changes the returned N, and every caller
    #: that only wants the chunk should keep getting one.
    pair_unsteered: bool = False

    def __init__(self, spec: ArmSpec, served_model=None):
        self.spec = spec
        if spec.arm == "qpilots" and served_model is not None:
            # steer the policy that is actually being served, whatever checkpoint it came from
            self.model = served_model
            self.graphdef = nnx.graphdef(served_model)
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

    # ---- pi0.5 sampler pieces (same math as the trainers / eval harness) ----------------------
    # _prefix / _velocity used to be hand-copies of pi0.py's sampler. They are now the model's own
    # primitives: a copy that drifts from the served policy does not fail, it silently steers away
    # from a base that is not the policy being served -- and the same math is duplicated again in
    # five extraction trainers, so the copies have to converge on ONE implementation, not two.
    def _prefix(self, model, obs):
        """``(kv_cache, prefix_mask)`` -- reversed from the model's ``(prefix_mask, kv_cache)``,
        kept only because every call site below unpacks it this way."""
        prefix_mask, kv = model._prefix_forward(obs)
        return kv, prefix_mask

    def _velocity(self, model, obs, kv, pm, x, tau):
        return model._velocity(obs, pm, kv, x, tau)

    def _euler(self, model, obs, kv, pm, x):
        n = self.spec.ode_steps
        dt = 1.0 / n
        for i in range(n):
            x = x - dt * self._velocity(model, obs, kv, pm, x, jnp.full((x.shape[0],), 1.0 - i * dt))
        return x

    def _q(self, feats, chunk, proprio, *, reduce: str):
        logits = self.critic.net.apply({"params": self.critic.params}, feats, chunk, proprio)
        q = self.critic.hl.from_logits(logits)[..., -1]  # [K, B]
        if reduce == "min":
            return q.min(axis=0)
        return q.mean(axis=0) - self.spec.rho * q.std(axis=0)  # pessimistic, QPILOTS Eq. 12

    @functools.cached_property
    def _steer_jit(self):
        """`(state, rng, obs, feats, proprio, ad, alpha) -> chunk`, compiled once.

        Follows nnx_utils.module_jit's shape -- split the module, pass the state, merge inside --
        because that is what the rest of the repo does and why the other paths compile as one XLA
        module instead of a pile of eager kernels. `alpha` stays traced, so the steered draw and
        its alpha=0 twin share a single compilation; `ad` and the step count are static.
        """
        graphdef = self.graphdef

        def fun(state, rng, obs, feats, proprio, ad, alpha, paired):
            model = nnx.merge(graphdef, state)
            kv, pm = self._prefix(model, obs)  # ONE prefix pass, shared by both draws
            b = obs.state.shape[0]
            x0 = jax.random.normal(rng, (b, self.H, self.AD))
            steered = self._steer(model, obs, kv, pm, feats, proprio, ad, x0, alpha)
            if not paired:
                return steered
            # The twin, from the same x0 and the same cached prefix, differing only in alpha.
            base = self._steer(model, obs, kv, pm, feats, proprio, ad, x0, 0.0)
            return jnp.concatenate([steered, base], axis=0)  # index 0 is what executes

        return jax.jit(fun, static_argnums=(5, 7))

    def _steer(self, model, obs, kv, pm, feats, proprio, ad, x, alpha):
        """QPILOTS-U Euler integration at steering strength `alpha`. alpha=0 is the base sampler.

        A plain Python loop, deliberately. A lax.fori_loop version compiled faster (24 s against
        42 s) but changed the answer by 5.9e-03 in bf16 -- and, unlike the joint/cached-attention
        difference measured elsewhere in this repo, that did NOT collapse in fp32 (7.7e-04, only
        8x smaller, where pure accumulation order fell by 8400x). Unexplained numerical drift in
        the sampler is not worth a compile-time saving, and it bought no runtime: the 25x speedup
        came entirely from not doing eager work in __call__ (see there).

        Kept as ONE path rather than a steered and an unsteered variant: the twin only measures the
        steering term if everything else about the two draws is identical, and two code paths
        cannot be identical by inspection for long.
        """
        n = self.spec.ode_steps
        dt = 1.0 / n
        for i in range(n):
            tv = jnp.full((x.shape[0],), 1.0 - i * dt)
            if i == 0:
                # No state-dependent signal at t=0 (paper Sec. 4). Both draws take this branch, so
                # it is symmetric between them.
                v = self._velocity(model, obs, kv, pm, x, tv)
            else:
                # NOT short-circuited at alpha == 0, though it would save the gradient on every
                # base step. `v` here comes out of jax.grad's forward pass, and a direct velocity
                # call is the same math in a different accumulation ORDER -- measured in this repo
                # at 1.2e-02 in bf16 (1.4e-06 in fp32). Branching on alpha would put that between
                # the steered draw and the twin it is measured against, compounded over every step,
                # and report the sum as steering displacement.

                def q_of(x_, tv_=tv):
                    v_ = self._velocity(model, obs, kv, pm, x_, tv_)
                    a_hat = x_ - tv_[:, None, None] * v_  # Tweedie projection, Eq. 14
                    a_hat = a_hat + jax.lax.stop_gradient(jnp.clip(a_hat, -1, 1) - a_hat)  # straight-through
                    return self._q(feats, a_hat[..., :ad], proprio, reduce="pess").sum(), v_

                g, v = jax.grad(q_of, has_aux=True)(x)
                vn = jnp.linalg.norm(v.reshape(x.shape[0], -1), axis=-1).reshape(-1, 1, 1)
                gn = jnp.linalg.norm(g.reshape(x.shape[0], -1), axis=-1).reshape(-1, 1, 1)
                v = v - alpha * (vn / (gn + 1e-8)) * g  # drift-norm-matched, Eq. 17
            x = x - dt * v
        return jnp.clip(x, -1.0, 1.0)

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
            return self._steer_jit(self.params, rng, obs, feats, proprio, ad, spec.alpha, self.pair_unsteered)

        # The remaining arms still build the model here; they are single-forward paths, so the
        # merge is not repeated inside a loop the way qpilots' prefix was.
        model = nnx.merge(self.graphdef, self.params)

        if spec.arm in LATENT_ARMS:
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            if spec.arm == "lpsd":
                rep = jnp.concatenate([rep, jax.random.normal(rng, (b, self.H * self.AD))], axis=-1)
            z = _mlp(self.actor, rep).reshape(b, self.H, self.AD)
            pm, kv = model._prefix_forward(obs)
            u = model._u(obs, pm, kv, z, jnp.ones((b,)), jnp.zeros((b,)))
            return z - u

        kv, pm = self._prefix(model, obs)

        if spec.arm == "flowdagger":
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            coeffs = _mlp(self.head, rep, tanh_scale=3.0).reshape(b, self.basis.shape[0], self.AD)
            seed = jnp.einsum("kh,bkd->bhd", self.basis, coeffs)
            return self._euler(model, obs, kv, pm, seed)

        raise ValueError(
            f"{spec.arm!r} is not sampled here: bon/idql are the wrapper's own selection path, and "
            "the weight-only arms serve as exported checkpoints (export_extraction_checkpoint.py)."
        )
