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
      - ``rlt_objective``: "reconstruction", "progress", or both.  The progress
        target is time-to-success derived from the sparse success reward (see
        training/progress.py); its head is HL-Gauss distributional by
        default, or plain regression.
      - ``rlt_decoder_mode``: "autoregressive" (the paper's Eq. 2) or "parallel"
        (no teacher forcing, so the token cannot be bypassed via context).

Monitoring hooks: ``compute_loss`` returns an ``aux`` dict of per-step scalars
(loss components, z collapse/norm stats).  ``extract_rl_token`` gives z_rl for a
batch so the trainer can compute participation-ratio / linear-probe diagnostics
and PCA scatter images.
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
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
        return x + self.mlp(self.norm2(x))


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
    # Whether proprio participates in the bottleneck: fed to the encoder as an extra token AND
    # reconstructed back from z_rl. Set False to make the RL token purely image+language derived —
    # which is what the paper actually does, since its critic takes (z_rl, s^p) and concatenates
    # proprio at the critic rather than squeezing it through the bottleneck. Turning this off means
    # whatever consumes z_rl downstream has to supply proprio itself.
    rlt_include_proprio: bool = True
    # Objective for the bottleneck: "reconstruction", "progress", or "reconstruction+progress".
    # Anything with "progress" needs the data config to inject a `progress` label
    # (LeRobotRoboCasaDataConfig(include_progress=True); see training/progress.py).
    rlt_objective: str = "reconstruction"
    # Progress head: "distributional" (HL-Gauss histogram + cross-entropy) or "regression" (MSE).
    # Distributional gives the token a K-dim target instead of a single scalar, and can represent the
    # genuine ambiguity in "how long until success" rather than collapsing to a conditional mean.
    rlt_progress_head: str = "distributional"
    # Number of histogram bins over progress in [0, 1] (distributional head only).
    rlt_progress_bins: int = 51
    # Gaussian smoothing of the histogram target, in units of bin width (HL-Gauss).
    rlt_progress_sigma_frac: float = 0.75
    # Weight of the progress term inside the RLT loss.
    progress_loss_weight: float = 1.0
    # How the reconstruction is decoded from z_rl. "autoregressive" is the paper's Eq. 2 (teacher
    # forced on the true previous embeddings); "parallel" decodes every position from z_rl alone, so
    # the token cannot be bypassed via context. See Pi0RLT._decode.
    rlt_decoder_mode: str = "autoregressive"
    # --- Latent BC probe ---
    # A small flow-matching action head trained on the (stop-gradient) RL token with the ordinary BC
    # objective. It measures how much of the policy is recoverable from the frozen latent ALONE: its
    # gradient never reaches z_rl or the backbone, so it changes nothing about RLT/BC training. At eval
    # its rollout success rate is compared against the full VLA's.
    rlt_bc_probe: bool = False
    rlt_probe_width: int = 1024
    rlt_probe_depth: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0RLTConfig requires pi05=True")
        if self.rlt_width % 2 != 0:
            raise ValueError(f"rlt_width must be even (sincos posemb), got {self.rlt_width}")
        if self.rlt_decoder_mode not in ("autoregressive", "parallel"):
            raise ValueError(f"rlt_decoder_mode must be 'autoregressive' or 'parallel', got {self.rlt_decoder_mode!r}")
        if self.rlt_objective not in ("reconstruction", "progress", "reconstruction+progress"):
            raise ValueError(
                "rlt_objective must be 'reconstruction', 'progress' or 'reconstruction+progress', "
                f"got {self.rlt_objective!r}"
            )
        if self.rlt_progress_head not in ("distributional", "regression"):
            raise ValueError(
                f"rlt_progress_head must be 'distributional' or 'regression', got {self.rlt_progress_head!r}"
            )
        if self.rlt_progress_bins < 2:
            raise ValueError(f"rlt_progress_bins must be >= 2, got {self.rlt_progress_bins}")

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
        self.rlt_include_proprio = config.rlt_include_proprio
        if self.rlt_include_proprio:
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
        # Proprio reconstruction head straight off the bottleneck: forces proprio into z_rl, which is
        # only needed when whatever consumes z_rl downstream cannot see proprio on its own.
        if self.rlt_include_proprio:
            self.rlt_proprio_out_proj = nnx.Linear(config.rlt_token_dim, config.action_dim, rngs=rngs)

        # ── Task-progress head ──────────────────────────────────────────────
        # Distributional: logits over `bins` buckets of progress in [0, 1]. Regression: one scalar.
        # Built ONLY when the objective uses progress, so a reconstruction-only model has exactly the
        # params it did before this head existed — otherwise its checkpoint (which predates the head)
        # fails to load with a pytree-structure mismatch on `rlt_progress_out_proj`.
        self.rlt_objective = config.rlt_objective
        self.rlt_progress_head = config.rlt_progress_head
        self.progress_loss_weight = config.progress_loss_weight
        # Only plain Python scalars here: nnx.Module rejects bare jax.Array attributes (they are not
        # valid graph leaves), so the HL-Gauss bin edges are rebuilt inside the loss instead. They are
        # compile-time constants, so XLA folds them away.
        self._prog_bins = config.rlt_progress_bins
        self._prog_sigma = config.rlt_progress_sigma_frac / config.rlt_progress_bins
        if "progress" in config.rlt_objective:
            n_out = config.rlt_progress_bins if config.rlt_progress_head == "distributional" else 1
            self.rlt_progress_out_proj = nnx.Linear(config.rlt_token_dim, n_out, rngs=rngs)

        # ── Latent BC probe (flow-matching action head on the frozen z_rl) ──
        # Built only when enabled, so a non-probe checkpoint keeps its old param structure.
        self.rlt_bc_probe = config.rlt_bc_probe
        if config.rlt_bc_probe:
            pw = config.rlt_probe_width
            act_flat = self.action_horizon * self.action_dim
            # velocity(x_t, t, z_rl, proprio): [flatten(x_t), time_emb, z_rl, proprio] -> flatten(v_t).
            # Proprio is fed in ALWAYS (even with rlt_include_proprio) so the probe is a fair test of
            # the token's control value: with --no-proprio the token carries no proprio, and the
            # downstream critic gets proprio separately, so the probe must too.
            self.rlt_probe_in = nnx.Linear(act_flat + pw + config.rlt_token_dim + config.action_dim, pw, rngs=rngs)
            self.rlt_probe_time = nnx.Linear(pw, pw, rngs=rngs)  # maps sincos(t) -> width
            self._probe_depth = config.rlt_probe_depth
            self.rlt_probe_hidden = nnx.Dict(
                {f"fc_{i}": nnx.Linear(pw, pw, rngs=rngs) for i in range(config.rlt_probe_depth)}
            )
            self.rlt_probe_out = nnx.Linear(pw, act_flat, rngs=rngs)

    # ------------------------------------------------------------------
    # Encoder g_φ : (z_img, proprio, <rl>) → z_rl bottleneck
    # ------------------------------------------------------------------

    def _encode_rl_token(self, z, img_mask, state):
        """Compress the VLA image embeddings + proprio into one RL token [b, rlt_token_dim]."""
        b, M, _ = z.shape
        d = self.rlt_width

        zt = self.rlt_enc_in_proj(z)  # [b, M, d]
        rl = jnp.broadcast_to(self.rlt_token_embed.value[None, None], (b, 1, d))  # [b, 1, d]
        # With proprio: [image tokens, proprio, <rl>]. Without: [image tokens, <rl>] — the token is
        # then a pure image+language readout and proprio reaches the critic by another route.
        extra = 2 if self.rlt_include_proprio else 1
        parts = [zt]
        if self.rlt_include_proprio:
            parts.append(self.rlt_proprio_in_proj(state)[:, None, :])  # [b, 1, d]
        parts.append(rl)
        x = jnp.concatenate(parts, axis=1)  # [b, M+extra, d]
        x = x + _sincos_posemb(M + extra, d)[None]  # positional

        # Bidirectional: every query attends to every valid key. Proprio and <rl> are always valid.
        valid = jnp.concatenate([img_mask, jnp.ones((b, extra), dtype=jnp.bool_)], axis=1)  # [b, M+extra]
        mask = valid[:, None, None, :]  # [b, 1, 1, M+extra]

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

    def _progress_loss(self, z_rl, target):
        """Progress loss + diagnostics. ``target`` is the scalar label in [0, 1], shape [b].

        Distributional (HL-Gauss): the scalar target becomes a Gaussian smeared over the bins, via
        differences of its CDF at the bin edges, and the head is trained with cross-entropy. That
        gives the bottleneck a ``bins``-dimensional target instead of a single number, and lets it
        express genuine ambiguity about how far away success is (the same frame can be followed by a
        fast or a slow finish) instead of being forced onto a conditional mean.
        """
        out = self.rlt_progress_out_proj(z_rl)  # [b, bins] or [b, 1]
        target = jnp.clip(target, 0.0, 1.0)

        if self.rlt_progress_head == "regression":
            pred = out[:, 0]
            return jnp.square(pred - target), pred, jnp.zeros_like(pred)

        # Soft histogram target: P(bin_k) = Phi((edge_{k+1} - y)/s) - Phi((edge_k - y)/s).
        edges = jnp.linspace(0.0, 1.0, self._prog_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        cdf = jax.scipy.stats.norm.cdf(edges[None, :], loc=target[:, None], scale=self._prog_sigma)
        probs = cdf[:, 1:] - cdf[:, :-1]
        probs = probs / (jnp.sum(probs, axis=-1, keepdims=True) + 1e-8)

        logp = jax.nn.log_softmax(out, axis=-1)
        loss = -jnp.sum(probs * logp, axis=-1)  # [b]

        p = jnp.exp(logp)
        pred = jnp.sum(p * centers[None], axis=-1)  # expected progress
        entropy = -jnp.sum(p * logp, axis=-1)  # how unsure the token is
        return loss, pred, entropy

    def _rlt_losses(self, z, img_mask, state, progress=None):
        """RLT loss terms + z_rl, from the image embeddings z [b, M, W].

        Which terms are active is set by ``rlt_objective``; the reconstruction decoder is only run
        when reconstruction is part of the objective (so "progress" alone is also notably cheaper).
        """
        z_enc_in = z if self.rlt_backbone_gradient else jax.lax.stop_gradient(z)
        z_tgt = jax.lax.stop_gradient(z) if self.rlt_target_stop_gradient else z

        z_rl = self._encode_rl_token(z_enc_in, img_mask, state)  # [b, rlt_token_dim]
        zeros = jnp.zeros(z_rl.shape[0], dtype=jnp.float32)
        aux = {}

        if "reconstruction" in self.rlt_objective:
            recon = self._decode(z_rl, z_tgt, img_mask)  # [b, M, W]
            # Masked reconstruction MSE (mean over feature dim, masked mean over tokens).
            err = jnp.mean(jnp.square(recon - z_tgt), axis=-1)  # [b, M]
            m = img_mask.astype(jnp.float32)
            recon_loss = jnp.sum(err * m, axis=-1) / (jnp.sum(m, axis=-1) + 1e-6)  # [b]
            rlt_loss = recon_loss
            aux |= {"rlt/loss_recon": jnp.mean(recon_loss)}
            if self.rlt_include_proprio:
                proprio_recon = self.rlt_proprio_out_proj(z_rl)  # [b, ad]
                proprio_loss = jnp.mean(jnp.square(proprio_recon - state), axis=-1)  # [b]
                rlt_loss = rlt_loss + self.proprio_loss_weight * proprio_loss
                aux |= {"rlt/loss_proprio": jnp.mean(proprio_loss)}
        else:
            rlt_loss = zeros

        if "progress" in self.rlt_objective:
            if progress is None:
                raise ValueError(
                    f"rlt_objective={self.rlt_objective!r} needs a `progress` label on the observation. "
                    "Set include_progress=True on LeRobotRoboCasaDataConfig."
                )
            prog_loss, prog_pred, prog_entropy = self._progress_loss(z_rl, progress)
            rlt_loss = rlt_loss + self.progress_loss_weight * prog_loss
            aux |= {
                "rlt/loss_progress": jnp.mean(prog_loss),
                # Mean absolute error in progress units: |Δ| of 0.1 ≈ 10% of the success horizon.
                "rlt/progress_mae": jnp.mean(jnp.abs(prog_pred - progress)),
                "rlt/progress_entropy": jnp.mean(prog_entropy),
            }

        return rlt_loss, z_rl, z_tgt, aux

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
        if "reconstruction" not in self.rlt_objective:
            # No reconstruction term in the objective means the decoder never receives a gradient, so
            # it stays at its random init. A bypass ratio measured through an untrained decoder says
            # nothing about the token — and would read near 1.0, i.e. a false "token is ignored"
            # alarm. Report only what still means something.
            return {"z_rl": z_rl}
        m = img_mask.astype(jnp.float32)

        def _recon_loss(token):
            recon = self._decode(token, z, img_mask)
            err = jnp.mean(jnp.square(recon - z), axis=-1)
            return jnp.sum(err * m, axis=-1) / (jnp.sum(m, axis=-1) + 1e-6)

        real = jnp.mean(_recon_loss(z_rl))
        shuffled = jnp.mean(_recon_loss(jnp.roll(z_rl, 1, axis=0)))
        return {
            "z_rl": z_rl,
            "recon_real": real,
            "recon_shuffled": shuffled,
            # Scale of the reconstruction target: the yardstick for reading loss_recon (a
            # "predict the mean" baseline sits near (1.25*target_abs_mean)^2). Nearly constant, so
            # it belongs here rather than on every step.
            "target_abs_mean": jnp.mean(jnp.abs(z)),
        }

    def extract_rl_token(self, observation: _model.Observation) -> at.Float[at.Array, "b t"]:
        """Language-conditioned forward → RL token z_rl [b, rlt_token_dim]."""
        observation = _model.preprocess_observation(None, observation, train=False)
        z, img_mask, _, _ = self._prefix_forward(observation)
        return self._encode_rl_token(z, img_mask, observation.state)

    def _prefix_forward(self, observation: _model.Observation):
        """One backbone pass over the prefix. Returns (z_img, img_mask, prefix_mask, kv_cache).

        The KV cache is handed back so callers that also need action samples can reuse this single
        3B forward instead of paying for a second one.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outs, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        z, img_mask = self._split_image_tokens(observation, outs[0], prefix_mask)
        return z, img_mask, prefix_mask, kv_cache

    def _denoise_from_cache(self, rng, observation, prefix_mask, kv_cache, *, num_samples, num_steps):
        """Flow-matching sampling of ``num_samples`` action chunks off an existing prefix KV cache.

        Same distribution as ``Pi0.sample_actions`` (π_vla), just drawn many times per state. The
        backbone prefix is NOT re-run; only the (small) action expert is, vmapped over the noises.
        """
        b = observation.state.shape[0]
        dt = -1.0 / num_steps

        def denoise(noise):
            def step(carry):
                x_t, time = carry
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, b)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
                pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn,
                    positions=pos,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t + dt * v_t, time + dt

            x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
            return x_0

        noises = jax.random.normal(rng, (num_samples, b, self.action_horizon, self.action_dim))
        chunks = jax.vmap(denoise)(noises)  # [n, b, H, D]
        return jnp.transpose(chunks, (1, 0, 2, 3))  # [b, n, H, D]

    def extract_token_and_base_actions(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, num_samples: int, num_steps: int = 10
    ) -> tuple[at.Float[at.Array, "b t"], at.Float[at.Array, "b n ah ad"]]:
        """RL token AND ``num_samples`` base-policy action chunks from ONE backbone forward.

        This is the annotation entry point: computing them separately would run the 3B prefix twice
        per frame for no reason. Actions come back in normalized model space; decode them with the
        data config's output transforms to get raw actions.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        z, img_mask, prefix_mask, kv_cache = self._prefix_forward(observation)
        z_rl = self._encode_rl_token(z, img_mask, observation.state)
        base = self._denoise_from_cache(
            rng, observation, prefix_mask, kv_cache, num_samples=num_samples, num_steps=num_steps
        )
        return z_rl, base

    # ------------------------------------------------------------------
    # Latent BC probe: a flow-matching action head on the frozen RL token
    # ------------------------------------------------------------------

    def _probe_velocity(self, x_t, time, z_rl, state):
        """Flow-matching velocity from (noisy action chunk, time, RL token, proprio). z_rl is used
        as-is; callers pass a stop-gradient'd token during training so the probe never shapes it."""
        b = x_t.shape[0]
        xf = x_t.reshape(b, -1)  # [b, H*A]
        t_emb = posemb_sincos(time, self.rlt_probe_time.in_features, min_period=4e-3, max_period=4.0)
        t_emb = nnx.swish(self.rlt_probe_time(t_emb))
        h = nnx.swish(self.rlt_probe_in(jnp.concatenate([xf, t_emb, z_rl, state], axis=-1)))
        for i in range(self._probe_depth):
            h = h + nnx.swish(self.rlt_probe_hidden[f"fc_{i}"](h))  # residual MLP
        return self.rlt_probe_out(h).reshape(x_t.shape)  # [b, H, A]

    def _probe_loss(self, x_t, time, u_t, z_rl, state):
        # proprio is a plain conditioning input, not part of the latent — no stop-gradient needed
        # (it does not touch z_rl), but the token itself is detached so the probe stays a pure probe.
        v = self._probe_velocity(x_t, time, jax.lax.stop_gradient(z_rl), state)
        return jnp.mean(jnp.square(v - u_t), axis=(-1, -2))  # [b]

    def probe_sample_actions(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, num_steps: int = 10
    ) -> _model.Actions:
        """Roll-out action chunk from the probe head alone: extract z_rl, then flow-match on it."""
        z_rl = self.extract_rl_token(observation)  # preprocesses internally
        state = observation.state  # preprocess leaves state unchanged
        b = z_rl.shape[0]
        dt = -1.0 / num_steps

        def step(carry):
            x_t, t = carry
            v = self._probe_velocity(x_t, jnp.broadcast_to(t, b), z_rl, state)
            return x_t + dt * v, t + dt

        noise = jax.random.normal(rng, (b, self.action_horizon, self.action_dim))
        x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
        return x_0

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
        rlt_loss, z_rl, z_tgt, rlt_aux = self._rlt_losses(z, img_mask, observation.state, observation.progress)

        total = bc_loss + self.rlt_loss_weight * rlt_loss  # [b]

        aux = {
            **rlt_aux,
            "bc_loss": jnp.mean(bc_loss),
            "rlt/z_batch_std": jnp.mean(jnp.std(z_rl, axis=0)),  # ~0 ⇒ token collapse
        }

        # Latent BC probe: trains its own head on the stop-gradient token (reusing the same x_t/u_t as
        # the main BC loss). It adds to `total` but, because z_rl is detached, only its own params get
        # a gradient — RLT/BC training is unchanged.
        if self.rlt_bc_probe:
            probe_loss = self._probe_loss(x_t, time, u_t, z_rl, observation.state)
            total = total + probe_loss
            aux["rlt/probe_bc_loss"] = jnp.mean(probe_loss)

        return total, aux
