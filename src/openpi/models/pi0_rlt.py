"""
Pi0 RLT — "RL Token" compact-representation bottleneck on top of pi05.

Implements the representation-learning stage of *RL Token: Bootstrapping Online
RL with Vision-Language-Action Models* (Xu et al., Physical Intelligence,
arXiv:2604.23073): a small encoder-decoder bottleneck that compresses the VLA's
internal features into one compact ``RL token`` (z_rl) via an autoregressive
reconstruction objective (Fig. 2 / Eq. 1-2).  z_rl is a deterministic readout
(NOT a VAE: no KL, no sampling).

Differences from the reference frozen-VLA stage — chosen for the RoboCasa port:

  * **Language-INCLUDED token (default).**  z_{1:M} are the final-layer *image*
    hidden states of the FULL (image+language) prefix forward, so the token is
    instruction-conditioned (needed for RoboCasa's per-task prompts).  The
    reference's older image-only variant is intentionally not ported.

  * **Joint training with BC finetuning.**  ``compute_loss`` returns the pi05
    flow-matching (BC) loss PLUS ``rlt_loss_weight`` * RLT loss, from a SINGLE
    backbone forward — the RLT bottleneck is learned while the VLA is being
    BC-finetuned (pi05_base -> RoboCasa), not on a frozen VLA.

  * **Variant switches** for experimentation:
      - ``rlt_backbone_gradient``: if False (default) the RLT loss gradient is
        stopped before the VLM backbone (z_rl is a pure readout head; the
        backbone is shaped only by BC).  If True, the RLT loss also reshapes the
        VLM features.  (BC always flows into the backbone regardless.)
      - ``rlt_target_stop_gradient``: whether the reconstruction *target* z̄ is
        stop-gradient'd (default True; disabling risks feature collapse).
      - ``rlt_objective``: "reconstruction" (inc1).  "progress" /
        "reconstruction+progress" are reserved for a later increment (a
        reward/success task-progress head + label plumbing).

Monitoring hooks: ``compute_loss`` returns an ``aux`` dict of per-step scalars
(loss components, z collapse/norm stats).  ``extract_rl_token`` gives z_rl for a
batch so the trainer can compute participation-ratio / linear-probe diagnostics
and PCA scatter images.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.shared import array_typing as at


# ---------------------------------------------------------------------------
# Small standalone transformer blocks (separate from the Gemma backbone)
# ---------------------------------------------------------------------------


def _sincos_posemb(length: int, dim: int) -> jax.Array:
    """Fixed sinusoidal positional embedding, shape [length, dim] (dim even)."""
    pos = jnp.arange(length, dtype=jnp.float32)[:, None]
    i = jnp.arange(dim // 2, dtype=jnp.float32)[None, :]
    freq = jnp.exp(-jnp.log(10000.0) * (2.0 * i / dim))
    ang = pos * freq
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # [length, dim]


class _Mlp(nnx.Module):
    def __init__(self, dim: int, hidden: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(x)))


class _Block(nnx.Module):
    """Pre-norm Transformer block (self-attention + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, decode=False, dropout_rate=0.0, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = _Mlp(dim, mlp_hidden, rngs=rngs)

    def __call__(self, x, mask):
        # mask: bool, broadcastable to [b, num_heads, q, kv]; True = attend.
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pi0RLTConfig(pi0_config.Pi0Config):
    """Config for Pi0RLT.  Inherits all Pi0Config fields; requires ``pi05=True``.

    Train it as a normal pi05 BC finetune (e.g. from ``pi05_base``): the VLA is
    finetuned by the BC loss and the ``rlt_*`` bottleneck is learned jointly.  Do
    NOT set a freeze filter unless you specifically want the frozen-VLA stage.
    """

    # Bottleneck size of the RL token (paper value: 2048).
    rlt_token_dim: int = 2048
    # Hidden width of the encoder/decoder transformer (d_model). Must be even (sincos posemb).
    rlt_width: int = 1024
    rlt_encoder_depth: int = 4
    rlt_decoder_depth: int = 4
    rlt_num_heads: int = 8
    rlt_mlp_ratio: int = 4

    # --- RLT loss weighting ---
    rlt_loss_weight: float = 1.0  # weight of the whole RLT loss vs the BC loss
    proprio_loss_weight: float = 1.0  # weight of the proprio-reconstruction term inside the RLT loss

    # --- Variant switches ---
    # If False (default): stop-gradient z before the encoder, so the RLT loss does NOT reshape the
    # VLM backbone (z_rl is a pure readout head; the backbone is shaped only by BC). If True: let the
    # RLT loss flow into the backbone too.
    rlt_backbone_gradient: bool = False
    # Whether the reconstruction TARGET z̄ is stop-gradient'd (default True; disabling is collapse-prone).
    rlt_target_stop_gradient: bool = True
    # Objective for the bottleneck. "progress" / "reconstruction+progress" need a reward/success
    # task-progress label plumbed into the batch (a later increment); only reconstruction is wired now.
    rlt_objective: str = "reconstruction"
    # How the reconstruction is decoded from z_rl. "autoregressive" is the paper's Eq. 2 (teacher
    # forced on the true previous embeddings); "parallel" decodes every position from z_rl alone, so
    # the token cannot be bypassed via context. See Pi0RLT._decode.
    rlt_decoder_mode: str = "autoregressive"

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0RLTConfig requires pi05=True")
        if self.rlt_width % 2 != 0:
            raise ValueError(f"rlt_width must be even (sincos posemb), got {self.rlt_width}")
        if self.rlt_decoder_mode not in ("autoregressive", "parallel"):
            raise ValueError(f"rlt_decoder_mode must be 'autoregressive' or 'parallel', got {self.rlt_decoder_mode!r}")
        if self.rlt_objective != "reconstruction":
            raise ValueError(
                f"rlt_objective={self.rlt_objective!r} is not wired yet; only 'reconstruction' is "
                "supported in this increment (the task-progress head + reward/success label plumbing "
                "comes in the next increment)."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RLT":
        return Pi0RLT(self, rngs=nnx.Rngs(rng))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Pi0RLT(Pi0):
    """pi05 + an encoder-decoder RL-token bottleneck, trained jointly with BC."""

    def __init__(self, config: Pi0RLTConfig, rngs: nnx.Rngs):
        # Build the full pi05 (PaliGemma img+llm, action expert).
        super().__init__(config, rngs)

        vlm_width = _gemma.get_config(config.paligemma_variant).width  # W (e.g. 2048)
        d = config.rlt_width
        mlp_hidden = d * config.rlt_mlp_ratio

        self._vlm_width = vlm_width
        self.rlt_width = d
        self.rlt_token_dim = config.rlt_token_dim
        self.rlt_loss_weight = config.rlt_loss_weight
        self.proprio_loss_weight = config.proprio_loss_weight
        self.rlt_backbone_gradient = config.rlt_backbone_gradient
        self.rlt_target_stop_gradient = config.rlt_target_stop_gradient
        self.rlt_decoder_mode = config.rlt_decoder_mode
        self._enc_depth = config.rlt_encoder_depth
        self._dec_depth = config.rlt_decoder_depth

        # ── Encoder ─────────────────────────────────────────────────────────
        # Blocks live in an nnx.Dict (string keys) rather than a Python list: the weight loader
        # flattens the param tree with sep="/", which cannot join integer list indices.
        self.rlt_enc_in_proj = nnx.Linear(vlm_width, d, rngs=rngs)  # W → d
        self.rlt_proprio_in_proj = nnx.Linear(config.action_dim, d, rngs=rngs)  # proprio → d
        self.rlt_token_embed = nnx.Param(jax.random.normal(rngs.params(), (d,)) * 0.02)
        self.rlt_encoder = nnx.Dict(
            {f"blk_{i}": _Block(d, config.rlt_num_heads, mlp_hidden, rngs=rngs) for i in range(self._enc_depth)}
        )
        self.rlt_out_proj = nnx.Linear(d, config.rlt_token_dim, rngs=rngs)  # d → bottleneck

        # ── Decoder (autoregressive reconstruction) ─────────────────────────
        self.rlt_dec_rl_proj = nnx.Linear(config.rlt_token_dim, d, rngs=rngs)  # z_rl → d (start token)
        self.rlt_dec_tgt_proj = nnx.Linear(vlm_width, d, rngs=rngs)  # z̄ → d (teacher forcing)
        self.rlt_decoder = nnx.Dict(
            {f"blk_{i}": _Block(d, config.rlt_num_heads, mlp_hidden, rngs=rngs) for i in range(self._dec_depth)}
        )
        self.rlt_dec_out_proj = nnx.Linear(d, vlm_width, rngs=rngs)  # d → W (reconstruction h_φ)
        # Proprio reconstruction head straight off the bottleneck (forces proprio into z_rl, since the
        # downstream critic sees only z_rl).
        self.rlt_proprio_out_proj = nnx.Linear(config.rlt_token_dim, config.action_dim, rngs=rngs)

    # ------------------------------------------------------------------
    # Encoder g_φ : (z_img, proprio, <rl>) → z_rl bottleneck
    # ------------------------------------------------------------------

    def _encode_rl_token(self, z, img_mask, state):
        """Compress the VLA image embeddings + proprio into one RL token [b, rlt_token_dim]."""
        b, M, _ = z.shape
        d = self.rlt_width

        zt = self.rlt_enc_in_proj(z)  # [b, M, d]
        pt = self.rlt_proprio_in_proj(state)[:, None, :]  # [b, 1, d]
        rl = jnp.broadcast_to(self.rlt_token_embed.value[None, None], (b, 1, d))  # [b, 1, d]
        x = jnp.concatenate([zt, pt, rl], axis=1)  # [b, M+2, d]
        x = x + _sincos_posemb(M + 2, d)[None]  # positional

        # Bidirectional: every query attends to every valid key. Proprio and <rl> are always valid.
        valid = jnp.concatenate([img_mask, jnp.ones((b, 2), dtype=jnp.bool_)], axis=1)  # [b, M+2]
        mask = valid[:, None, None, :]  # [b, 1, 1, M+2]

        for i in range(self._enc_depth):
            x = self.rlt_encoder[f"blk_{i}"](x, mask)
        return self.rlt_out_proj(x[:, -1])  # [b, rlt_token_dim]

    # ------------------------------------------------------------------
    # Decoder d_φ : autoregressive reconstruction of z̄_{1:M} from z_rl
    # ------------------------------------------------------------------

    def _decode(self, z_rl, z_tgt, img_mask):
        """Reconstruct z̄_{1:M} [b, M, W] from the single RL token z_rl [b, rlt_token_dim].

        ``autoregressive`` (default) is the paper's Eq. 2: teacher-forced over the sequence
        [z_rl, z̄_1, ..., z̄_{M-1}] with a causal mask, so position j predicts z̄_{j+1} having seen
        z_rl and the *true* z̄_{1:j}.

        ``parallel`` drops the teacher forcing entirely: every output position is decoded from z_rl
        plus a positional query, so the token is the ONLY route by which information can reach the
        reconstruction. Neighbouring SigLIP tokens are highly correlated, so the autoregressive
        decoder can score well while ignoring z_rl (see ``rlt_bypass_diagnostics``); this mode makes
        that impossible, at the cost of a much harder reconstruction task (expect a higher loss —
        what matters is what ends up inside z_rl, not the loss value).
        """
        b, M, _ = z_tgt.shape
        d = self.rlt_width
        start = self.rlt_dec_rl_proj(z_rl)[:, None, :]  # [b, 1, d]

        if self.rlt_decoder_mode == "parallel":
            # Broadcast the token to every position; positions differ only by the positional code.
            x = jnp.broadcast_to(start, (b, M, d)) + _sincos_posemb(M, d)[None]
            mask = jnp.ones((b, 1, 1, M), dtype=jnp.bool_)  # full attention, nothing to hide
            for i in range(self._dec_depth):
                x = self.rlt_decoder[f"blk_{i}"](x, mask)
            return self.rlt_dec_out_proj(x)  # [b, M, W]

        shifted = self.rlt_dec_tgt_proj(z_tgt[:, : M - 1])  # [b, M-1, d]
        x = jnp.concatenate([start, shifted], axis=1)  # [b, M, d]
        x = x + _sincos_posemb(M, d)[None]

        causal = jnp.tril(jnp.ones((M, M), dtype=jnp.bool_))  # [M, M]
        # Key validity: position 0 (z_rl) always valid; position j (z̄_j) valid iff image token j valid.
        kv_valid = jnp.concatenate([jnp.ones((b, 1), dtype=jnp.bool_), img_mask[:, : M - 1]], axis=1)  # [b, M]
        mask = causal[None, None] & kv_valid[:, None, None, :]  # [b, 1, M, M]

        for i in range(self._dec_depth):
            x = self.rlt_decoder[f"blk_{i}"](x, mask)
        return self.rlt_dec_out_proj(x)  # [b, M, W]

    # ------------------------------------------------------------------
    # Image-token embeddings from the (language-conditioned) prefix
    # ------------------------------------------------------------------

    def _split_image_tokens(self, observation: _model.Observation, prefix_hidden, prefix_mask):
        """Slice the IMAGE-token hidden states out of the full prefix (images come first)."""
        num_lang = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
        num_img = prefix_hidden.shape[1] - num_lang
        z = prefix_hidden[:, :num_img].astype(jnp.float32)  # [b, M, W]
        img_mask = prefix_mask[:, :num_img]  # [b, M]
        return z, img_mask

    def _rlt_losses(self, z, img_mask, state):
        """Reconstruction + proprio losses and z_rl, from image embeddings z [b, M, W]."""
        z_enc_in = z if self.rlt_backbone_gradient else jax.lax.stop_gradient(z)
        z_tgt = jax.lax.stop_gradient(z) if self.rlt_target_stop_gradient else z

        z_rl = self._encode_rl_token(z_enc_in, img_mask, state)  # [b, rlt_token_dim]
        recon = self._decode(z_rl, z_tgt, img_mask)  # [b, M, W]

        # Masked reconstruction MSE (mean over feature dim, masked mean over tokens).
        err = jnp.mean(jnp.square(recon - z_tgt), axis=-1)  # [b, M]
        m = img_mask.astype(jnp.float32)
        recon_loss = jnp.sum(err * m, axis=-1) / (jnp.sum(m, axis=-1) + 1e-6)  # [b]

        proprio_recon = self.rlt_proprio_out_proj(z_rl)  # [b, ad]
        proprio_loss = jnp.mean(jnp.square(proprio_recon - state), axis=-1)  # [b]

        rlt_loss = recon_loss + self.proprio_loss_weight * proprio_loss  # [b]
        return rlt_loss, z_rl, z_tgt, recon_loss, proprio_loss

    # ------------------------------------------------------------------
    # Inference: extract the RL token (used by monitoring + downstream RL)
    # ------------------------------------------------------------------

    def rlt_bypass_diagnostics(self, observation: _model.Observation) -> dict[str, at.Array]:
        """Does the decoder actually USE the RL token, or is it reconstructing from context alone?

        The decoder is teacher-forced on z̄_{1:i-1}, and neighbouring SigLIP image tokens are highly
        correlated, so it can reach a very low reconstruction loss by interpolating from the previous
        tokens while ignoring z_rl — the bottleneck would look "trained" while carrying nothing.

        We measure that directly: reconstruct once with the true token and once with the tokens
        rolled across the batch (so every sample gets *another* sample's token). If the loss barely
        moves, the token is being bypassed.

            bypass_ratio = loss(shuffled z_rl) / loss(true z_rl)
              ~1   -> token ignored (bad; reconstruction is coming from context)
              >>1  -> token carries the information the decoder depends on (good)
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outs, _ = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        z, img_mask = self._split_image_tokens(observation, outs[0], prefix_mask)

        z_rl = self._encode_rl_token(z, img_mask, observation.state)
        m = img_mask.astype(jnp.float32)

        def _recon_loss(token):
            recon = self._decode(token, z, img_mask)
            err = jnp.mean(jnp.square(recon - z), axis=-1)
            return jnp.sum(err * m, axis=-1) / (jnp.sum(m, axis=-1) + 1e-6)

        real = jnp.mean(_recon_loss(z_rl))
        shuffled = jnp.mean(_recon_loss(jnp.roll(z_rl, 1, axis=0)))
        return {"z_rl": z_rl, "recon_real": real, "recon_shuffled": shuffled}

    def extract_rl_token(self, observation: _model.Observation) -> at.Float[at.Array, "b t"]:
        """Language-conditioned forward → RL token z_rl [b, rlt_token_dim]."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outs, _ = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        z, img_mask = self._split_image_tokens(observation, outs[0], prefix_mask)
        return self._encode_rl_token(z, img_mask, observation.state)

    # ------------------------------------------------------------------
    # compute_loss: BC flow-matching (pi05) + RLT reconstruction, one forward
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        """Returns (per-sample total loss [b], aux dict).

        total = mean_H(BC flow-matching) + rlt_loss_weight * (recon + proprio_w * proprio).
        Both terms come from ONE backbone forward (the RLT token reuses the prefix hidden states).
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One forward pass of prefix + suffix (same as Pi0.compute_loss).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # BC flow-matching loss (per sample: mean over horizon + action dim).
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        bc_chunked = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [b, H]
        bc_loss = jnp.mean(bc_chunked, axis=-1)  # [b]

        # RLT loss from the image-token hidden states of this same forward (language-conditioned).
        z, img_mask = self._split_image_tokens(observation, prefix_out, prefix_mask)
        rlt_loss, z_rl, z_tgt, recon_loss, proprio_loss = self._rlt_losses(z, img_mask, observation.state)

        total = bc_loss + self.rlt_loss_weight * rlt_loss  # [b]

        aux = {
            "rlt/bc_loss": jnp.mean(bc_loss),
            "rlt/loss_recon": jnp.mean(recon_loss),
            "rlt/loss_proprio": jnp.mean(proprio_loss),
            "rlt/loss_rlt_total": jnp.mean(rlt_loss),
            "rlt/z_abs_mean": jnp.mean(jnp.abs(z_rl)),
            "rlt/z_norm_mean": jnp.mean(jnp.linalg.norm(z_rl, axis=-1)),
            "rlt/z_batch_std": jnp.mean(jnp.std(z_rl, axis=0)),  # ~0 ⇒ token collapse
            "rlt/target_abs_mean": jnp.mean(jnp.abs(z_tgt)),
        }
        return total, aux
