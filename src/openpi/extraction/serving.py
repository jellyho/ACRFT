"""Inference for every policy-extraction arm — one loader, one servable ``Policy``.

Each arm changes exactly one thing about how an action chunk is produced from the frozen BC
pi0.5, so this module keeps the ordinary openpi serving path (transforms + norm stats from the
BC checkpoint's assets) and swaps only the sampler:

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
import pathlib
from typing import Any

import einops
import flax.nnx as nnx
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy as _policy_mod
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

BC_CKPT = pathlib.Path("/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/100000")
AF_CKPT = pathlib.Path("/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000")
CRITIC = pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_g5_tau9_min")
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


class _ArmSampler:
    """Holds the (possibly overlaid) model and exposes a Policy-compatible sample_actions."""

    def __init__(self, spec: ArmSpec):
        self.spec = spec
        self.train_config = _config.get_config("pi05_yam_lego_taxi")
        if spec.arm == "cfgrl":
            from openpi.models.pi0_cfgrl import Pi0CFGRLConfig

            mcfg = Pi0CFGRLConfig(
                pi05=True,
                action_horizon=self.train_config.model.action_horizon,
                action_dim=self.train_config.model.action_dim,
            )
        elif spec.arm in LATENT_ARMS:
            mcfg = _config.get_config("pi05_yam_lego_taxi_alphaflow").model
        else:
            mcfg = self.train_config.model
        self.H, self.AD = mcfg.action_horizon, mcfg.action_dim

        model = mcfg.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        loaded = CheckpointWeightLoaderKeepMissing(str(spec.base_ckpt / "params")).load(state.to_pure_dict())
        if spec.expert_ckpt is not None:
            import orbax.checkpoint as ocp

            with ocp.StandardCheckpointer() as c:
                _deep_update(loaded, c.restore(pathlib.Path(spec.expert_ckpt).absolute())["expert"])
        state.replace_by_pure_dict(loaded)
        self.model = nnx.merge(graphdef, state)
        self.graphdef = nnx.graphdef(self.model)
        self.params = jax.device_put(nnx.state(self.model))

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

    # ---- pi0.5 sampler pieces (same math as the trainers / eval harness) ----------------------
    def _prefix(self, model, obs):
        tokens, mask, ar = model.embed_prefix(obs)
        attn = make_attn_mask(mask, ar)
        pos = jnp.cumsum(mask, axis=1) - 1
        _, kv = model.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
        return kv, mask

    def _velocity(self, model, obs, kv, pm, x, tau):
        st, sm, sar, adarms = model.embed_suffix(obs, x, tau)
        sattn = make_attn_mask(sm, sar)
        pattn = einops.repeat(pm, "b p -> b s p", s=st.shape[1])
        full = jnp.concatenate([pattn, sattn], axis=-1)
        pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, out), _ = model.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv, adarms_cond=[None, adarms])
        return model.action_out_proj(out[:, -self.H :])

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

    def sample_actions(self, rng, observation, **kw):
        """Signature-compatible with model.sample_actions; critic arms take feats/proprio in kw."""
        spec = self.spec
        model = nnx.merge(self.graphdef, self.params)
        obs = _model.preprocess_observation(None, observation, train=False)
        b = obs.state.shape[0]

        if spec.arm == "cfgrl":
            return model.sample_actions_cfg(rng, observation, cfg_w=spec.cfg_w, num_steps=spec.ode_steps)

        if spec.arm in LATENT_ARMS:
            feats, proprio = kw["feats"], kw["proprio"]
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            if spec.arm == "lpsd":
                rep = jnp.concatenate([rep, jax.random.normal(rng, (b, self.H * self.AD))], axis=-1)
            z = _mlp(self.actor, rep).reshape(b, self.H, self.AD)
            pm, kv = model._prefix_forward(obs)
            u = model._u(obs, pm, kv, z, jnp.ones((b,)), jnp.zeros((b,)))
            return z - u

        kv, pm = self._prefix(model, obs)

        if spec.arm == "flowdagger":
            feats, proprio = kw["feats"], kw["proprio"]
            rep = jnp.concatenate([feats.mean(axis=1), proprio], axis=-1)
            coeffs = _mlp(self.head, rep, tanh_scale=3.0).reshape(b, self.basis.shape[0], self.AD)
            seed = jnp.einsum("kh,bkd->bhd", self.basis, coeffs)
            return self._euler(model, obs, kv, pm, seed)

        if spec.arm == "qpilots":
            feats, proprio = kw["feats"], kw["proprio"]
            ad = self.critic.config["action_dim"]
            n = spec.ode_steps
            dt = 1.0 / n
            x = jax.random.normal(rng, (b, self.H, self.AD))
            for i in range(n):
                tv = jnp.full((b,), 1.0 - i * dt)
                if i == 0:  # no state-dependent signal at t=0 (paper Sec. 4)
                    v = self._velocity(model, obs, kv, pm, x, tv)
                else:

                    def q_of(x_, tv_=tv):
                        v_ = self._velocity(model, obs, kv, pm, x_, tv_)
                        a_hat = x_ - tv_[:, None, None] * v_  # Tweedie projection, Eq. 14
                        a_hat = a_hat + jax.lax.stop_gradient(jnp.clip(a_hat, -1, 1) - a_hat)  # straight-through
                        return self._q(feats, a_hat[..., :ad], proprio, reduce="pess").sum(), v_

                    g, v = jax.grad(q_of, has_aux=True)(x)
                    vn = jnp.linalg.norm(v.reshape(b, -1), axis=-1).reshape(b, 1, 1)
                    gn = jnp.linalg.norm(g.reshape(b, -1), axis=-1).reshape(b, 1, 1)
                    v = v - spec.alpha * (vn / (gn + 1e-8)) * g  # drift-norm-matched, Eq. 17
                x = x - dt * v
            return jnp.clip(x, -1.0, 1.0)

        if spec.arm in ("idql", "bon"):
            feats, proprio = kw["feats"], kw["proprio"]
            ad = self.critic.config["action_dim"]
            best_c = best_q = None
            for i in range(spec.n_samples):
                c = jnp.clip(
                    self._euler(
                        model, obs, kv, pm, jax.random.normal(jax.random.fold_in(rng, i), (b, self.H, self.AD))
                    ),
                    -1.0,
                    1.0,
                )
                q = self._q(feats, c[..., :ad], proprio, reduce="min")  # ddpm_iql_learner.py:40-43
                if best_q is None:
                    best_c, best_q = c, q
                else:
                    take = (q > best_q)[:, None, None]
                    best_c, best_q = jnp.where(take, c, best_c), jnp.maximum(q, best_q)
            return best_c

        # bc and the expert-overlay arms: the ordinary sampler
        return self._euler(model, obs, kv, pm, jax.random.normal(rng, (b, self.H, self.AD)))


class ExtractionPolicy(_policy_mod.BasePolicy):
    """Serves one arm through the BC transform chain; adds critic features when the arm needs them."""

    def __init__(self, spec: ArmSpec, *, default_prompt: str | None = None):
        self._spec = spec
        self._sampler = _ArmSampler(spec)
        # the ordinary served BC policy supplies transforms + norm stats (and, for arms that need
        # no critic, the whole inference path); we only borrow its transform chain
        self._policy = _policy_config.create_trained_policy(
            self._sampler.train_config, spec.base_ckpt, default_prompt=default_prompt
        )
        self._input_transform = self._policy._input_transform
        self._output_transform = self._policy._output_transform
        self._rng = jax.random.key(0)
        self._needs_feats = spec.critic is not None or spec.arm in (*LATENT_ARMS, "flowdagger")
        self._patchify = None
        if self._needs_feats:
            self._build_patchify()

    def _build_patchify(self, img_size: int = 224):
        """DINOv2 + the critic's 2x2 mean pooling — patch_critic_policy.py:212-231 verbatim, so a
        served arm sees exactly the feature layout the critic (and the cache) was built with."""
        from openpi.patch_critic.backbone import DinoV2Backbone

        cc = self._sampler.critic.config
        bb = DinoV2Backbone(cc["backbone"])
        grid = int(bb.num_patches(img_size) ** 0.5)
        pooled = grid // 2
        from openpi.policies import patch_critic_policy as _pcp

        self._camera_keys = _pcp.YAM_CAMERA_KEYS
        ncam = len(self._camera_keys)
        npatch = ncam * pooled * pooled
        self._img_size = img_size

        def pool(p):
            b, _, d = p.shape
            return (
                p.reshape(b, ncam, grid, grid, d)
                .reshape(b, ncam, pooled, 2, pooled, 2, d)
                .mean((3, 5))
                .reshape(b, npatch, d)
            )

        @jax.jit
        def patchify(imgs_nchw):  # [1, ncam, 3, S, S] -> [1, npatch, D]
            return pool(bb(imgs_nchw))

        self._patchify = patchify

    def _critic_inputs(self, obs: dict):
        """DINOv2 patch features + the critic's proprio slice, exactly as the critic was trained."""
        from openpi.patch_critic.backbone import to_nchw
        from openpi.policies import patch_critic_policy as _pcp

        imgs = np.stack([_pcp._parse_image(obs[k], self._img_size) for k in self._camera_keys])
        feats = self._patchify(jnp.asarray(to_nchw(imgs))[None])  # [1, npatch, D]
        state = np.asarray(obs[_pcp.YAM_STATE_KEY], np.float32)[None]
        critic = self._sampler.critic
        idx = critic.proprio_idx if critic is not None else None
        proprio = state if idx is None else state[:, idx]
        return jnp.asarray(feats, jnp.float32), jnp.asarray(proprio, jnp.float32)

    def infer(self, obs: dict, **_) -> dict:
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, key = jax.random.split(self._rng)
        kw = {}
        if self._needs_feats:
            feats, proprio = self._critic_inputs(obs)
            kw = {"feats": feats, "proprio": proprio}
        actions = self._sampler.sample_actions(key, _model.Observation.from_dict(batched), **kw)
        out = {"state": inputs["state"], "actions": np.asarray(actions[0])}
        return self._output_transform(out)

    @property
    def metadata(self) -> dict:
        return {"arm": self._spec.arm, **{k: str(v) for k, v in dataclasses.asdict(self._spec).items()}}


def load_arm(arm: str, *, step: int | None = None, default_prompt: str | None = None, **over: Any):
    """Build a servable policy for one extraction arm (see ALL_ARMS)."""
    return ExtractionPolicy(default_spec(arm, step, **over), default_prompt=default_prompt)
