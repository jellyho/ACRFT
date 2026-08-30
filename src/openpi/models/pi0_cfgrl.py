"""pi0.5 + CFGRL: optimality-conditioned flow policy with classifier-free guidance sampling.

Provenance (official CFGRL, kvfrans/cfgrl — the value-based variant, which matches our fixed
external critic):
  - Conditioning variable: binary optimality index O in {0,1}; the effective training label is
    the hard indicator O = 1{A > 0} (rlbase/algs_offline/iql_diffusion.py:157).
  - Both branches trained on every sample: conditional loss masked by the indicator, plus the
    unconditional branch at weight 0.1 (iql_diffusion.py:170-179).
  - Sampling: at every Euler step, v = v_uncond + w * (v_cond - v_uncond) (iql_diffusion.py:213),
    guidance weight w swept in {1, 1.5, 3, 5, 10, 30, 100} at eval (iql_diffusion.py:342).
  - Their conditioning is a learned 2-entry embedding concatenated into an MLP input
    (iql_diffusion.py:105-106). pi0.5 has no concat seam; the native conditioning stream is the
    adaRMS vector (pi0.py:169 `adarms_cond = time_emb`), so we ADD a learned 2-entry table there
    -- the same injection style pi0_alphaflow uses for its r input. Documented deviation #1.
  - Deviation #2: time sampling keeps pi0.5's Beta(1.5,1) (pi0.py:197) instead of their
    discretized uniform grid (iql_diffusion.py:164) -- time sampling is a policy-family detail,
    the CFGRL mechanism is the conditioning/guidance.
  - The embedding table is ZERO-initialized so step 0 is bit-identical to the BC checkpoint and
    cond == uncond (guidance starts as a no-op). Their table is randomly initialized (deviation
    #3, in favor of warm-start fidelity).
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0CFGRLConfig(Pi0Config):
    # Guidance weight the SAMPLER runs at: v_u + w (v_c - v_u) (iql_diffusion.py:213). It belongs
    # to the model, not to a serving flag, so a checkpoint served by --policy.config samples the
    # way the config says. w = 1 reduces to the conditioned policy (no guidance); the official
    # sweep is {1, 1.5, 3, 5, 10, 30, 100}.
    cfg_w: float = 1.5

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    def create(self, rng: at.KeyArrayLike) -> "Pi0CFGRL":
        return Pi0CFGRL(self, rngs=nnx.Rngs(rng))


class Pi0CFGRL(Pi0):
    def __init__(self, config: Pi0Config, *, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        # sampling weight carried on the model: the ordinary serving path calls sample_actions(),
        # so without this a served CFGRL checkpoint would silently run UNGUIDED (w = 1) while the
        # config said otherwise -- the failure returns well-formed chunks, so nothing would catch it
        self._cfg_w = float(getattr(config, "cfg_w", 1.5))
        width = self.action_in_proj.out_features
        # 2-entry optimality table (iql_diffusion.py:105 nn.Embed(2, 32)); zero-init (deviation #3)
        self.opt_embed = nnx.Param(jnp.zeros((2, width), jnp.float32))

    def _suffix_opt(self, obs, x_t, time, opt_idx):
        tokens, mask, ar_mask, adarms = self.embed_suffix(obs, x_t, time)
        # inject optimality into the adaRMS conditioning stream (deviation #1; pi0.py:169)
        adarms = adarms + self.opt_embed[opt_idx]
        return tokens, mask, ar_mask, adarms

    def compute_loss_cfgrl(self, rng, observation, actions, label):
        """Both CFGRL branches in ONE llm pass by doubling the batch (equivalent to the official
        two passes at iql_diffusion.py:168-177: identical x_t/time feed both heads).
        Returns mean( 1{A>0} * L_cond + 0.1 * L_uncond ) (iql_diffusion.py:170-179)."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=True)
        b = actions.shape[0]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b,)) * 0.999 + 0.001  # deviation #2
        te = time[..., None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions

        obs2 = jax.tree.map(lambda x: jnp.concatenate([x, x], axis=0), observation)
        x2 = jnp.concatenate([x_t, x_t], axis=0)
        t2 = jnp.concatenate([time, time], axis=0)
        opt2 = jnp.concatenate([jnp.ones(b, jnp.int32), jnp.zeros(b, jnp.int32)], axis=0)

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(obs2)
        suffix_tokens, suffix_mask, suffix_ar, adarms = self._suffix_opt(obs2, x2, t2, opt2)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions, adarms_cond=[None, adarms]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        per = jnp.mean(jnp.square(v_t - jnp.concatenate([u_t, u_t], axis=0)), axis=(-2, -1))  # [2b]
        per_cond, per_uncond = per[:b], per[b:]
        loss = jnp.mean(label * per_cond + 0.1 * per_uncond)  # iql_diffusion.py:170-179
        return loss, {"cond": jnp.mean(label * per_cond), "uncond": jnp.mean(per_uncond), "frac_pos": jnp.mean(label)}

    @override
    def sample_actions(self, rng, observation, *, num_steps: int = 10, **kw):
        """The serving entry point: guided sampling at the config's weight (see Pi0CFGRLConfig)."""
        return self.sample_actions_cfg(rng, observation, cfg_w=kw.pop("cfg_w", self._cfg_w), num_steps=num_steps)

    def sample_actions_cfg(self, rng, observation, *, cfg_w, num_steps=10):
        """Euler sampler with per-step CFG combination v_u + w (v_c - v_u) (iql_diffusion.py:205-216).
        No intermediate clipping, matching the official sampler (final clip is env-specific there;
        pi0.5 actions are unnormalized downstream, so none is applied here)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        b = observation.state.shape[0]
        noise = jax.random.normal(rng, (b, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn, positions=positions)
        # double the prefix KV across the (cond, uncond) pair; KVCache layout is [layers, batch, ...]
        kv2 = jax.tree.map(lambda x: jnp.concatenate([x, x], axis=1), kv_cache)
        pm2 = jnp.concatenate([prefix_mask, prefix_mask], axis=0)
        obs2 = jax.tree.map(lambda x: jnp.concatenate([x, x], axis=0), observation)
        opt2 = jnp.concatenate([jnp.ones(b, jnp.int32), jnp.zeros(b, jnp.int32)], axis=0)

        def step(carry):
            x_t, time = carry
            x2 = jnp.concatenate([x_t, x_t], axis=0)
            t2 = jnp.broadcast_to(time, 2 * b)
            suffix_tokens, suffix_mask, suffix_ar, adarms = self._suffix_opt(obs2, x2, t2, opt2)
            suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
            pref_attn = einops.repeat(pm2, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn = jnp.concatenate([pref_attn, suffix_attn], axis=-1)
            pos = jnp.sum(pm2, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn, positions=pos, kv_cache=kv2, adarms_cond=[None, adarms]
            )
            v2 = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            v_c, v_u = v2[:b], v2[b:]
            v = v_u + cfg_w * (v_c - v_u)  # iql_diffusion.py:213
            return x_t + dt * v, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        x0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x0
