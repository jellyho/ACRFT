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
      - ``rlt_objective``: "reconstruction" plus any of "+progress",
        "+action", "+behsim".  Reconstruction is always present: it is the only
        term that pressures z_rl to retain the observation.  Progress is one
        scalar per frame, so on its own it lets the bottleneck collapse to ~1
        dimension (measured: 10% probe success against 45-50% with
        reconstruction), which is why every extra term is an addition and not an
        objective in its own right.  The progress target is time-to-success
        derived from the sparse success reward (see training/progress.py); its
        head is HL-Gauss distributional by default, or plain regression.
        ``action`` and ``behsim`` both attack the same measured pathology:
        reconstruction targets SigLIP features, which are a fixed function of
        *appearance*, so every RoboCasa demo (different kitchen, different props)
        is trivially separable and z_rl ends up encoding "which episode" rather
        than "what is happening".  Diagnostically that shows up as a linear
        episode-ID probe at 100% accuracy and kNN neighbourhoods 3.6x enriched
        for same-episode frames.  Both terms ground the token in BEHAVIOUR,
        which is shared across episodes:
          * ``action``: predict the demonstrated action chunk from z_rl, with the
            gradient flowing (unlike ``rlt_bc_probe``, which is detached and is a
            measurement only).  Ni et al. (2401.08898) show a self-predictive
            objective needs a grounding partner or it degenerates; with no reward
            available at BC time, the action chunk is that partner.
          * ``behsim``: PSE-style (Agarwal et al., Contrastive Behavioral
            Similarity Embeddings) soft-target InfoNCE across the batch, where
            the target similarity between two frames is set by how similar their
            action chunks are.  Unlike ``action`` it acts on the token GEOMETRY,
            pulling behaviourally-equivalent frames from *different* demos
            together — which is exactly the cross-trajectory stitching that
            appearance-based reconstruction destroys.
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


def _layernorm(x):
    """Parameter-free layer norm, matching nnx.LayerNorm(use_scale=False, use_bias=False).

    Deliberately a function, not a module: a parameterless nnx.LayerNorm contributes an EMPTY subtree
    to the model state, which orbax does not write, so the checkpoint ends up with fewer children than
    the model expects and `BaseModel.load` fails the structure check (it can drop extra params but not
    invent missing ones). Every load path — serving included — would break on such a checkpoint.
    """
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.mean(jnp.square(x - m), axis=-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-6)


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


class _DiTBlock(nnx.Module):
    """DiT block: self-attention + MLP, both modulated by a conditioning vector via adaLN-Zero.

    The conditioning vector produces per-block (shift, scale, gate) for each sub-layer. Because the
    modulation projection is zero-initialized, every block starts as the identity and the network
    learns how much conditioning to apply - the standard adaLN-Zero recipe (Peebles & Xie), which is
    what makes a small DiT train stably. The multiplicative scale/gate is also what lets the head
    represent time-dependent rescalings (e.g. the x_t/t target on constant action dims) that a
    concatenated time embedding cannot.
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, *, rngs: nnx.Rngs):
        # No learned affine in the norms (adaLN supplies scale/shift), so they are functions.
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, decode=False, dropout_rate=0.0, rngs=rngs
        )
        self.mlp = _Mlp(dim, mlp_hidden, rngs=rngs)
        self.ada = nnx.Linear(
            dim,
            6 * dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x, cond):
        # x: [b, T, dim];  cond: [b, dim]
        shift1, scale1, gate1, shift2, scale2, gate2 = jnp.split(self.ada(nnx.swish(cond))[:, None, :], 6, axis=-1)
        h = _layernorm(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        h = _layernorm(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


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
    # How many RL tokens the encoder emits. The decoder in `parallel` mode conditions on nothing but
    # these, so one vector is the entire channel between the observation and the reconstruction -
    # and the effective rank of that vector was measured to contract from ~12 directions to ~6 over
    # training, which is a bottleneck the task may not fit through. More tokens widen the channel
    # without widening each vector, so the critic still reads fixed-size slots. Downstream sees them
    # concatenated, so the consumed dimension is rlt_num_tokens * rlt_token_dim.
    rlt_num_tokens: int = 1
    # MAE-style masking. With 0 the encoder sees every image token and the decoder reproduces every
    # one, which makes the objective compression. With a ratio > 0 the encoder sees only the kept
    # fraction and the loss is taken on the DROPPED positions, which makes it inference: the token has
    # to carry enough to predict content it was never shown. Masking applies during training only -
    # annotation and deployment always encode the full view, as in MAE.
    rlt_mask_ratio: float = 0.0
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
    # Objective for the bottleneck: "reconstruction" plus any "+"-joined subset of
    # {progress, action, behsim}. Reconstruction is not optional — it is the only term that makes
    # z_rl keep the observation, and every other term is too low-dimensional to prevent collapse on
    # its own (progress is one scalar per frame; behsim only constrains relative geometry). They are
    # therefore ADDITIONS, never objectives by themselves. "progress" needs the data config to inject
    # a `progress` label (LeRobotRoboCasaDataConfig(include_progress=True); see training/progress.py);
    # "action" and "behsim" use the action chunk that is already in every batch.
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

    # --- Action grounding ("+action") ---
    # Weight of the action-chunk regression term. The head predicts the whole demonstrated chunk
    # [H, rlt_probe_action_dim] and the gradient DOES flow into z_rl — this is what separates it from
    # rlt_bc_probe, which is the same prediction with a stop-gradient and exists only to measure.
    # Both can be on at once: the probe then reports how much of the policy a *detached* reader can
    # recover, which stays an honest metric because it has its own head and its own gradient path.
    rlt_action_loss_weight: float = 1.0

    # --- Behavioural-similarity contrastive ("+behsim") ---
    # Soft-target InfoNCE over the batch (PSE, Agarwal et al. 2021). Target similarity between frames
    # i and j is softmax(-d_ij / beta) where d_ij = mean|a_i - a_j| over the chunk; the prediction is
    # softmax(cos(g(z_i), g(z_j)) / tau) through a SimCLR-style projection head g. Because it is a
    # cross-batch loss it is computed per data-parallel shard, which is fine (it is still a valid
    # loss) but does make its effective difficulty depend on the per-device batch size.
    rlt_behsim_weight: float = 1.0
    # Temperature on the behavioural distance. Small beta => a peaky target (only the single most
    # behaviourally-similar frame counts); large beta => nearly uniform, i.e. no signal. 0.1 is
    # roughly one tenth of the typical |Δaction| spread in normalized action units.
    rlt_behsim_beta: float = 0.1
    # Temperature on the predicted cosine similarities.
    rlt_behsim_tau: float = 0.1
    # Width/output dim of the projection head. Contrastive losses are applied through a projection
    # rather than on the representation itself so that the token is not forced to *be* the
    # contrastive space — it only has to contain it (SimCLR §4.2).
    rlt_behsim_proj_dim: int = 128

    # --- Episode-adversarial invariance ("+epadv") ---
    # A domain-adversarial (DANN, Ganin & Lempitsky 2015) term that DIRECTLY attacks the measured
    # pathology: a linear probe reads "which demo" off z_rl at 100% accuracy, because reconstruction
    # targets appearance and every RoboCasa demo looks different. An episode classifier is trained on
    # z_rl, but a gradient-reversal layer flips the sign of the gradient flowing back into the token,
    # so the classifier gets better while the token is pushed to make "which demo" undecodable —
    # leaving only what is shared across demos (task structure). Unlike behsim it assumes NOTHING
    # about temporal order, so it survives demos whose subtasks happen in different orders. Needs the
    # data config to inject episode_index (include_episode_index=True).
    rlt_epadv_weight: float = 1.0
    # Gradient-reversal strength λ. The classifier minimizes its loss normally; the token receives
    # -λ times that gradient. Larger λ = stronger push to invariance (and less stable).
    rlt_epadv_lambda: float = 1.0
    # Class count of the adversary. Episodes are bucketed by (episode_index % rlt_num_episodes), so
    # this only has to be >= the dataset's episode count to avoid collisions (514 for PrepareCoffee);
    # it is a build-time constant baked into the head's shape, so keep it fixed across a checkpoint's
    # train/diagnostic lifetime. Bucketing (rather than the exact count) keeps the head dataset-agnostic.
    rlt_num_episodes: int = 1024
    # Where the auxiliary heads (progress, proprio) read their features from.
    #   "decoder" (default): the decoder's position-0 output, a NONLINEAR function of z_rl alone
    #       (position 0 attends only to itself, and its input is the projected token, so no
    #       teacher-forced context can leak in - progress/proprio cannot be bypassed either).
    #   "token": a linear map straight off z_rl - the original design, kept for ablation.
    # A linear head forces its low-dimensional target to be linearly decodable from the WHOLE 2048-d
    # token, which drags the representation onto that direction; a scalar target can then collapse
    # the bottleneck (observed as a pretty progress-coloured UMAP with bypass_ratio ~ 1 and a low
    # participation ratio). Reading through the decoder instead only asks that the information be
    # RECOVERABLE, and shares the trunk that reconstruction already needs, so collapse costs
    # reconstruction and is self-limiting.
    rlt_aux_head_source: str = "decoder"
    # How the reconstruction is decoded from z_rl. "autoregressive" is the paper's Eq. 2 (teacher
    # forced on the true previous embeddings); "parallel" decodes every position from z_rl alone, so
    # the token cannot be bypassed via context. See Pi0RLT._decode.
    rlt_decoder_mode: str = "autoregressive"
    # --- Latent BC probe ---
    # A small flow-matching action head trained on the (stop-gradient) RL token with the ordinary BC
    # objective. It measures how much of the policy is recoverable from the frozen latent ALONE: its
    # gradient never reaches z_rl or the backbone, so it changes nothing about RLT/BC training. At eval
    # its rollout success rate is compared against the full VLA's.
    # The head is a small DiT: the action chunk is a sequence of H tokens with self-attention, and
    # time + z_rl + proprio condition it through adaLN-Zero. That mirrors how the VLA's action expert
    # is conditioned (adaRMS), so a probe/VLA gap is evidence about the LATENT rather than about the
    # head being weaker. A flat MLP with concatenated time cannot represent the multiplicative
    # x_t/t interaction the flow-matching target needs, which showed up as a large loss floor.
    rlt_bc_probe: bool = False
    rlt_probe_width: int = 512
    rlt_probe_depth: int = 4
    rlt_probe_heads: int = 8
    rlt_probe_mlp_ratio: int = 4
    # Number of LEADING action dims the probe models. Actions are padded out to `action_dim` (32) to
    # fit the pretrained pi05 projections, but the padding is a constant 0 whose flow-matching target
    # is exactly x_t/t - an artificial regression problem that dominated the probe loss (20 of 32
    # dims). The probe is our own head with no pretrained constraint, so it models only the real dims
    # and pads back to `action_dim` when sampling. None = model everything.
    rlt_probe_action_dim: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0RLTConfig requires pi05=True")
        if self.rlt_width % 2 != 0:
            raise ValueError(f"rlt_width must be even (sincos posemb), got {self.rlt_width}")
        if self.rlt_aux_head_source not in ("decoder", "token"):
            raise ValueError(f"rlt_aux_head_source must be decoder|token, got {self.rlt_aux_head_source!r}")
        if self.rlt_decoder_mode not in ("autoregressive", "parallel"):
            raise ValueError(f"rlt_decoder_mode must be 'autoregressive' or 'parallel', got {self.rlt_decoder_mode!r}")
        if not 0.0 <= self.rlt_mask_ratio < 1.0:
            raise ValueError(f"rlt_mask_ratio must be in [0, 1), got {self.rlt_mask_ratio}")
        if self.rlt_num_tokens < 1:
            raise ValueError(f"rlt_num_tokens must be >= 1, got {self.rlt_num_tokens}")
        # "reconstruction" + any subset of the addition terms, in any order after the first.
        head, *additions = self.rlt_objective.split("+")
        if head != "reconstruction" or not set(additions) <= {"progress", "action", "behsim", "epadv"}:
            raise ValueError(
                "rlt_objective must start with 'reconstruction' and add any of '+progress', "
                "'+action', '+behsim', '+epadv' — every addition is too low-dimensional to hold the "
                "bottleneck open on its own (progress is one scalar per frame; behsim only "
                f"constrains relative geometry), so reconstruction is always required. Got "
                f"{self.rlt_objective!r}"
            )
        if len(additions) != len(set(additions)):
            raise ValueError(f"rlt_objective repeats a term: {self.rlt_objective!r}")
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
        self.rlt_num_tokens = config.rlt_num_tokens
        self.rlt_mask_ratio = config.rlt_mask_ratio
        # What everything downstream of the encoder actually receives.
        self.rlt_token_total = config.rlt_token_dim * config.rlt_num_tokens
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
        # One learned query per RL token; identical queries would make the encoder emit K copies of
        # the same vector. Kept at shape (d,) for a single token rather than (1, d): every existing
        # checkpoint stores it that way, and BaseModel.load checks structure, so (1, d) would refuse
        # to load any of them.
        _emb_shape = (d,) if config.rlt_num_tokens == 1 else (config.rlt_num_tokens, d)
        self.rlt_token_embed = nnx.Param(jax.random.normal(rngs.params(), _emb_shape) * 0.02)
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
        # Auxiliary heads (proprio, progress) read either the decoder's position-0 output (nonlinear
        # in z_rl, default) or z_rl itself (the original linear design). See rlt_aux_head_source.
        self.rlt_aux_head_source = config.rlt_aux_head_source
        # A dedicated decoder query is only worth building when some aux head actually reads it; with
        # no aux head the model is then byte-identical to one that never had this feature, which keeps
        # "reconstruction only, no proprio" a clean baseline across code versions.
        self._dec_has_aux = config.rlt_aux_head_source == "decoder" and (
            config.rlt_include_proprio or "progress" in config.rlt_objective or "action" in config.rlt_objective
        )
        if self._dec_has_aux:
            self.rlt_dec_aux_embed = nnx.Param(jax.random.normal(rngs.params(), (d,)) * 0.02)
        # Aux heads read the decoder's hidden state at that query (width d), or z_rl directly.
        aux_in = d if config.rlt_aux_head_source == "decoder" else self.rlt_token_total
        # Proprio reconstruction head: forces proprio into z_rl, which is only needed when whatever
        # consumes z_rl downstream cannot see proprio on its own.
        if self.rlt_include_proprio:
            self.rlt_proprio_out_proj = nnx.Linear(aux_in, config.action_dim, rngs=rngs)

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
            self.rlt_progress_out_proj = nnx.Linear(aux_in, n_out, rngs=rngs)

        # ── Behaviour-grounding heads (action / behsim) ─────────────────────
        # Same build-only-when-named rule as the progress head, for the same reason: a checkpoint
        # trained without them must keep loading.
        # Both use the same "real action dims" count as the probe: actions are padded out to
        # action_dim (32) to fit the pretrained pi05 projections, and the padding is a constant that
        # would otherwise be 20 of 32 dims of a trivially-solved regression.
        self._rlt_act_dim = config.rlt_probe_action_dim or config.action_dim
        self.rlt_action_loss_weight = config.rlt_action_loss_weight
        if "action" in config.rlt_objective:
            if not 0 < self._rlt_act_dim <= config.action_dim:
                raise ValueError(f"rlt_probe_action_dim must be in (0, {config.action_dim}]")
            # One shot at the whole chunk: a per-step autoregressive head would let the token get away
            # with encoding only the first action, and it is the CHUNK that carries the behaviour.
            self.rlt_action_out_proj = nnx.Linear(aux_in, config.action_horizon * self._rlt_act_dim, rngs=rngs)

        self.rlt_behsim_weight = config.rlt_behsim_weight
        self.rlt_behsim_beta = config.rlt_behsim_beta
        self.rlt_behsim_tau = config.rlt_behsim_tau
        if "behsim" in config.rlt_objective:
            p = config.rlt_behsim_proj_dim
            # Projection head reads z_rl directly rather than the decoder trunk: the contrastive term
            # is about the geometry of the token itself, and routing it through the shared decoder
            # would let the decoder absorb the constraint instead.
            self.rlt_behsim_proj_in = nnx.Linear(self.rlt_token_total, p, rngs=rngs)
            self.rlt_behsim_proj_out = nnx.Linear(p, p, rngs=rngs)

        # ── Episode-adversarial head (epadv) ────────────────────────────────
        self.rlt_epadv_weight = config.rlt_epadv_weight
        self.rlt_epadv_lambda = config.rlt_epadv_lambda
        self.rlt_num_episodes = config.rlt_num_episodes
        if "epadv" in config.rlt_objective:
            # A 2-layer MLP adversary — strictly stronger than the linear probe the diagnostic uses,
            # so driving IT down guarantees the linear probe (episode_acc) comes down too. Reads z_rl
            # through a gradient-reversal layer applied in the loss.
            self.rlt_epadv_hidden = nnx.Linear(self.rlt_token_total, 512, rngs=rngs)
            self.rlt_epadv_out = nnx.Linear(512, config.rlt_num_episodes, rngs=rngs)

        # ── Latent BC probe (flow-matching action head on the frozen z_rl) ──
        # Built only when enabled, so a non-probe checkpoint keeps its old param structure.
        self.rlt_bc_probe = config.rlt_bc_probe
        # Real action dims the probe models; the rest of `action_dim` is constant padding.
        self.probe_action_dim = config.rlt_probe_action_dim or self.action_dim
        if config.rlt_bc_probe:
            if not 0 < self.probe_action_dim <= self.action_dim:
                raise ValueError(f"rlt_probe_action_dim must be in (0, {self.action_dim}]")
            pw = config.rlt_probe_width
            self._probe_depth = config.rlt_probe_depth
            # Chunk as a sequence of H tokens (not a flat vector), so the head shares structure across
            # timesteps the way the VLA's action expert does.
            self.rlt_probe_in = nnx.Linear(self.probe_action_dim, pw, rngs=rngs)
            self.rlt_probe_pos = nnx.Param(_sincos_posemb(self.action_horizon, pw))
            # Conditioning: time + z_rl + proprio, summed into one vector that drives every adaLN.
            # Proprio is fed in ALWAYS (even with rlt_include_proprio) so the probe is a fair test of
            # the token's control value: with --no-proprio the token carries no proprio, and the
            # downstream critic gets proprio separately, so the probe must too.
            self.rlt_probe_time = nnx.Linear(pw, pw, rngs=rngs)  # maps sincos(t) -> width
            self.rlt_probe_tok = nnx.Linear(self.rlt_token_total, pw, rngs=rngs)
            self.rlt_probe_state = nnx.Linear(config.action_dim, pw, rngs=rngs)
            self.rlt_probe_blocks = nnx.Dict(
                {
                    f"blk_{i}": _DiTBlock(pw, config.rlt_probe_heads, pw * config.rlt_probe_mlp_ratio, rngs=rngs)
                    for i in range(config.rlt_probe_depth)
                }
            )
            # Zero-init final layer (adaLN-Zero): the head starts predicting v=0 and grows from there.
            self.rlt_probe_out_ada = nnx.Linear(
                pw,
                2 * pw,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
            self.rlt_probe_out = nnx.Linear(
                pw,
                self.probe_action_dim,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

    # ------------------------------------------------------------------
    # Encoder g_φ : (z_img, proprio, <rl>) → z_rl bottleneck
    # ------------------------------------------------------------------

    def _encode_rl_token(self, z, img_mask, state, keep=None):
        """Compress the VLA image embeddings + proprio into K RL tokens, flattened to [b, K*dim].

        ``keep`` optionally hides image tokens from the encoder (MAE): a [b, M] bool where False means
        the token is not a valid key. It never hides proprio or the RL queries.
        """
        b, M, _ = z.shape
        d = self.rlt_width
        k = self.rlt_num_tokens

        zt = self.rlt_enc_in_proj(z)  # [b, M, d]
        emb = self.rlt_token_embed.value
        rl = jnp.broadcast_to(emb.reshape(k, d)[None], (b, k, d))  # [b, K, d]
        # With proprio: [image tokens, proprio, <rl>]. Without: [image tokens, <rl>] — the token is
        # then a pure image+language readout and proprio reaches the critic by another route.
        extra = k + (1 if self.rlt_include_proprio else 0)
        parts = [zt]
        if self.rlt_include_proprio:
            parts.append(self.rlt_proprio_in_proj(state)[:, None, :])  # [b, 1, d]
        parts.append(rl)
        x = jnp.concatenate(parts, axis=1)  # [b, M+extra, d]
        x = x + _sincos_posemb(M + extra, d)[None]  # positional

        # Bidirectional: every query attends to every valid key. Proprio and <rl> are always valid.
        vis = img_mask if keep is None else (img_mask & keep)
        valid = jnp.concatenate([vis, jnp.ones((b, extra), dtype=jnp.bool_)], axis=1)  # [b, M+extra]
        mask = valid[:, None, None, :]  # [b, 1, 1, M+extra]

        for i in range(self._enc_depth):
            x = self.rlt_encoder[f"blk_{i}"](x, mask)
        # Flattened at the boundary so every consumer - decoder, probe, annotation, critic - keeps
        # taking a single vector, of rlt_num_tokens * rlt_token_dim.
        return self.rlt_out_proj(x[:, -k:]).reshape(b, k * self.rlt_token_dim)

    # ------------------------------------------------------------------
    # Decoder d_φ : autoregressive reconstruction of z̄_{1:M} from z_rl
    # ------------------------------------------------------------------

    def _decode(self, z_rl, z_tgt, img_mask):
        """Decode from the single RL token z_rl. Returns ``(recon [b, M, W], aux_feat [b, d] | None)``.

        ``autoregressive`` (default) is the paper's Eq. 2: teacher-forced over the sequence
        [z_rl, z̄_1, ..., z̄_{M-1}] with a causal mask, so position j predicts z̄_{j+1} having seen
        z_rl and the *true* z̄_{1:j}.

        ``parallel`` drops the teacher forcing entirely: every output position is decoded from z_rl
        plus a positional query, so the token is the ONLY route by which information can reach the
        reconstruction. Neighbouring SigLIP tokens are highly correlated, so the autoregressive
        decoder can score well while ignoring z_rl (see ``rlt_bypass_diagnostics``); this mode makes
        that impossible, at the cost of a much harder reconstruction task (expect a higher loss —
        what matters is what ends up inside z_rl, not the loss value).

        When the auxiliary heads read from the decoder, a DEDICATED query token is appended whose
        attention row allows only z_rl (position 0) and itself. That keeps the aux features a
        function of z_rl alone - reading them off the last position instead would let them see the
        teacher-forced targets, so progress/proprio could be predicted from the true embeddings
        without the token, exactly the bypass this design exists to prevent. A dedicated query also
        avoids overloading position 0, which already has to reconstruct z̄_1.
        """
        b, M, _ = z_tgt.shape
        d = self.rlt_width
        k = self.rlt_num_tokens
        parallel = self.rlt_decoder_mode == "parallel"
        tok = self.rlt_dec_rl_proj(z_rl.reshape(b, k, self.rlt_token_dim))  # [b, K, d]

        if k > 1:
            # With several tokens the decoder gets them as PREFIX KEYS, so a reconstruction query can
            # attend to each one separately instead of to a single summary. The query content stays
            # what it is for one token (the projected token, or the teacher-forced target) using the
            # mean, so the only thing more tokens add is what attention can reach - not a different
            # kind of conditioning. Token rows attend among themselves; reconstruction rows see every
            # token plus their usual pattern; nothing sees the aux query.
            mean = jnp.mean(tok, axis=1, keepdims=True)  # [b, 1, d]
            if parallel:
                body = jnp.broadcast_to(mean, (b, M, d))
                inner = jnp.ones((b, 1, M, M), dtype=jnp.bool_)
            else:
                shifted = self.rlt_dec_tgt_proj(z_tgt[:, : M - 1])
                body = jnp.concatenate([mean, shifted], axis=1)
                causal = jnp.tril(jnp.ones((M, M), dtype=jnp.bool_))
                kv_valid = jnp.concatenate([jnp.ones((b, 1), dtype=jnp.bool_), img_mask[:, : M - 1]], axis=1)
                inner = causal[None, None] & kv_valid[:, None, None, :]
            n_aux = 1 if self._dec_has_aux else 0
            total = k + M + n_aux
            x = jnp.concatenate(
                [tok, body]
                + ([jnp.broadcast_to(self.rlt_dec_aux_embed.value[None, None], (b, 1, d))] if n_aux else []),
                axis=1,
            )
            x = x + _sincos_posemb(total, d)[None]
            cols = jnp.arange(total)
            is_tok = (cols < k)[None, None, None, :]  # [1,1,1,total]
            tok_rows = jnp.broadcast_to(is_tok, (b, 1, k, total))
            rec_rows = jnp.concatenate(
                [
                    jnp.broadcast_to(is_tok[:, :, :, :k], (b, 1, M, k)),
                    inner,
                    *([jnp.zeros((b, 1, M, 1), dtype=jnp.bool_)] if n_aux else []),
                ],
                axis=-1,
            )
            parts = [tok_rows, rec_rows]
            if n_aux:
                aux_row = jnp.broadcast_to(((cols < k) | (cols == total - 1))[None, None, None, :], (b, 1, 1, total))
                parts.append(aux_row)
            mask = jnp.concatenate(parts, axis=2)
            for i in range(self._dec_depth):
                x = self.rlt_decoder[f"blk_{i}"](x, mask)
            recon = self.rlt_dec_out_proj(x[:, k : k + M])
            return recon, (x[:, -1] if n_aux else None)

        # --- single token: the original path, byte-for-byte, so its checkpoints keep loading -------
        start = tok  # [b, 1, d]
        if parallel:
            # Broadcast the token to every position; positions differ only by the positional code.
            x = jnp.broadcast_to(start, (b, M, d))
            base = jnp.ones((b, 1, M, M), dtype=jnp.bool_)  # full attention, nothing to hide
        else:
            shifted = self.rlt_dec_tgt_proj(z_tgt[:, : M - 1])  # [b, M-1, d]
            x = jnp.concatenate([start, shifted], axis=1)  # [b, M, d]
            causal = jnp.tril(jnp.ones((M, M), dtype=jnp.bool_))  # [M, M]
            # Key validity: position 0 (z_rl) always valid; position j (z̄_j) valid iff image token j is.
            kv_valid = jnp.concatenate([jnp.ones((b, 1), dtype=jnp.bool_), img_mask[:, : M - 1]], axis=1)
            base = causal[None, None] & kv_valid[:, None, None, :]  # [b, 1, M, M]

        if not self._dec_has_aux:
            x = x + _sincos_posemb(M, d)[None]
            for i in range(self._dec_depth):
                x = self.rlt_decoder[f"blk_{i}"](x, base)
            return self.rlt_dec_out_proj(x), None

        q = jnp.broadcast_to(self.rlt_dec_aux_embed.value[None, None], (b, 1, d))
        x = jnp.concatenate([x, q], axis=1) + _sincos_posemb(M + 1, d)[None]  # [b, M+1, d]
        # Reconstruction rows cannot see the aux query; the aux row sees ONLY z_rl and itself.
        recon_rows = jnp.concatenate([base, jnp.zeros((b, 1, M, 1), dtype=jnp.bool_)], axis=-1)
        cols = jnp.arange(M + 1)
        aux_row = jnp.broadcast_to(((cols == 0) | (cols == M))[None, None, None, :], (b, 1, 1, M + 1))
        mask = jnp.concatenate([recon_rows, aux_row], axis=2)  # [b, 1, M+1, M+1]

        for i in range(self._dec_depth):
            x = self.rlt_decoder[f"blk_{i}"](x, mask)
        # The aux heads read the decoder's HIDDEN state at the query, not the reconstruction
        # projection (whose output space is shaped to be an image embedding).
        return self.rlt_dec_out_proj(x[:, :M]), x[:, M]

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

    def _progress_loss(self, feat, target):
        """Progress loss + diagnostics. ``target`` is the scalar label in [0, 1], shape [b].

        ``feat`` is whatever ``rlt_aux_head_source`` selects: the decoder's position-0 output
        (nonlinear in z_rl) or z_rl itself.

        Distributional (HL-Gauss): the scalar target becomes a Gaussian smeared over the bins, via
        differences of its CDF at the bin edges, and the head is trained with cross-entropy. That
        gives the bottleneck a ``bins``-dimensional target instead of a single number, and lets it
        express genuine ambiguity about how far away success is (the same frame can be followed by a
        fast or a slow finish) instead of being forced onto a conditional mean.
        """
        out = self.rlt_progress_out_proj(feat)  # [b, bins] or [b, 1]
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

    def _behsim_loss(self, z_rl, actions):
        """PSE-style soft-target InfoNCE: make token similarity mirror ACTION-CHUNK similarity.

        Reconstruction pins z_rl to SigLIP features, which are a fixed function of appearance, so in
        RoboCasa (a different kitchen and different props per demo) frames cluster by episode no
        matter how similar the behaviour is. This term is the direct counterweight: for each anchor i
        it builds a target distribution over the other frames in the batch from their behavioural
        distance d_ij = mean|a_i - a_j|, and asks the token's (projected, cosine) similarities to
        match it. Two frames from *different* demos that are doing the same thing are pulled
        together; two frames from the SAME demo that are doing different things are pushed apart —
        which is exactly the invariance the appearance signal cannot express.

        Returns (per-sample loss [b], aux scalars).
        """
        a = actions[..., : self._rlt_act_dim]
        b = a.shape[0]
        # [b, b] mean absolute action difference over (H, ad).
        d = jnp.mean(jnp.abs(a[:, None] - a[None, :]), axis=(-1, -2))

        g = self.rlt_behsim_proj_out(jax.nn.gelu(self.rlt_behsim_proj_in(z_rl)))
        g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-6)
        sim = g @ g.T  # [b, b] cosine

        # Self-pairs are trivially the most similar on both sides and would dominate both softmaxes.
        off_diag = ~jnp.eye(b, dtype=bool)
        neg_inf = jnp.finfo(sim.dtype).min
        p = jax.nn.softmax(jnp.where(off_diag, -d / self.rlt_behsim_beta, neg_inf), axis=-1)
        logq = jax.nn.log_softmax(jnp.where(off_diag, sim / self.rlt_behsim_tau, neg_inf), axis=-1)
        loss = -jnp.sum(p * logq, axis=-1)  # [b]

        # Cross-entropy has a floor at H(p), so the raw loss says little on its own; report the excess
        # over that floor, which is the KL that actually gets minimized and is 0 at a perfect match.
        entropy = -jnp.sum(p * jnp.log(p + 1e-8), axis=-1)
        return loss, {
            "rlt/behsim_kl": jnp.mean(loss - entropy),
            "rlt/behsim_target_entropy": jnp.mean(entropy),
            # Mean off-diagonal cosine: ->1 means the projection collapsed and the term is vacuous.
            "rlt/behsim_mean_cos": jnp.sum(sim * off_diag) / jnp.sum(off_diag),
        }

    def _epadv_loss(self, z_rl, episode_index):
        """Domain-adversarial episode-invariance: push z_rl to be un-decodable into "which demo".

        Reconstruction makes the token encode appearance, and appearance is demo-specific in
        RoboCasa, so a linear probe reads the episode off z_rl at 100% accuracy (measured). Here an
        MLP classifier is trained to predict the episode, but a gradient-reversal layer negates the
        gradient into z_rl, so the token is optimised to DEFEAT the classifier — dropping whatever is
        demo-identifying while keeping whatever is shared across demos. Assumes nothing about
        temporal order, unlike behsim/TCC.

        Returns (per-sample loss [b], aux scalars).
        """
        # Gradient reversal: identity in the forward pass, gradient scaled by -λ backward. The
        # stop_gradient branch contributes 0 to the gradient, leaving d/dz = -λ.
        lam = self.rlt_epadv_lambda
        z_rev = jax.lax.stop_gradient((1.0 + lam) * z_rl) - lam * z_rl
        logits = self.rlt_epadv_out(jax.nn.gelu(self.rlt_epadv_hidden(z_rev)))
        target = jnp.mod(episode_index.astype(jnp.int32), self.rlt_num_episodes)
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.take_along_axis(logp, target[:, None], axis=-1)[:, 0]  # [b]
        # Adversary accuracy: HIGH means the token still leaks episode identity (invariance failing);
        # it should fall over training as z_rl becomes demo-invariant.
        adv_acc = jnp.mean((jnp.argmax(logits, axis=-1) == target).astype(jnp.float32))
        return loss, {"rlt/epadv_ce": jnp.mean(loss), "rlt/epadv_adv_acc": adv_acc}

    def _rlt_losses(self, z, img_mask, state, progress=None, actions=None, episode_index=None, rng=None):
        """RLT loss terms + z_rl, from the image embeddings z [b, M, W].

        Which terms are active is set by ``rlt_objective``; the reconstruction decoder is only run
        when reconstruction is part of the objective (so "progress" alone is also notably cheaper).
        """
        z_enc_in = z if self.rlt_backbone_gradient else jax.lax.stop_gradient(z)
        z_tgt = jax.lax.stop_gradient(z) if self.rlt_target_stop_gradient else z

        # MAE: hide a random fraction of image tokens from the encoder and score the reconstruction
        # on exactly those. Without an rng (diagnostics, annotation, deployment) the full view is used.
        keep = None
        if self.rlt_mask_ratio > 0 and rng is not None:
            u = jax.random.uniform(rng, img_mask.shape)
            keep = u >= self.rlt_mask_ratio
        z_rl = self._encode_rl_token(z_enc_in, img_mask, state, keep)  # [b, K*rlt_token_dim]
        aux = {}

        # The decoder is needed for reconstruction, and also to feed the auxiliary heads when they
        # read its position-0 output. Run it once and share the result.
        has_progress = "progress" in self.rlt_objective
        recon, dec_aux = self._decode(z_rl, z_tgt, img_mask)
        # The aux query attends only to z_rl and itself, so this is a nonlinear function of z_rl
        # ALONE - no teacher-forced context can leak into the aux heads.
        aux_feat = dec_aux if self.rlt_aux_head_source == "decoder" else z_rl

        # Reconstruction is always on (see __post_init__): masked MSE, mean over the feature dim and
        # a masked mean over tokens.
        err = jnp.mean(jnp.square(recon - z_tgt), axis=-1)  # [b, M]
        # Score only what the encoder could not see, so the loss measures inference rather than
        # copying. Rows where the draw happened to keep everything fall back to the full mask so the
        # denominator can never be zero.
        m = img_mask.astype(jnp.float32)
        if keep is not None:
            hidden = (img_mask & ~keep).astype(jnp.float32)
            m = jnp.where(jnp.sum(hidden, axis=-1, keepdims=True) > 0, hidden, m)
        recon_loss = jnp.sum(err * m, axis=-1) / (jnp.sum(m, axis=-1) + 1e-6)  # [b]
        rlt_loss = recon_loss
        aux |= {"rlt/loss_recon": jnp.mean(recon_loss)}
        if self.rlt_include_proprio:
            proprio_recon = self.rlt_proprio_out_proj(aux_feat)  # [b, ad]
            proprio_loss = jnp.mean(jnp.square(proprio_recon - state), axis=-1)  # [b]
            rlt_loss = rlt_loss + self.proprio_loss_weight * proprio_loss
            aux |= {"rlt/loss_proprio": jnp.mean(proprio_loss)}

        if has_progress:
            if progress is None:
                raise ValueError(
                    f"rlt_objective={self.rlt_objective!r} needs a `progress` label on the observation. "
                    "Set include_progress=True on LeRobotRoboCasaDataConfig."
                )
            prog_loss, prog_pred, prog_entropy = self._progress_loss(aux_feat, progress)
            rlt_loss = rlt_loss + self.progress_loss_weight * prog_loss
            aux |= {
                "rlt/loss_progress": jnp.mean(prog_loss),
                # Mean absolute error in progress units: |Δ| of 0.1 ≈ 10% of the success horizon.
                "rlt/progress_mae": jnp.mean(jnp.abs(prog_pred - progress)),
                "rlt/progress_entropy": jnp.mean(prog_entropy),
            }

        wants_actions = "action" in self.rlt_objective or "behsim" in self.rlt_objective
        if wants_actions and actions is None:
            raise ValueError(
                f"rlt_objective={self.rlt_objective!r} needs the action chunk. It is only available "
                "from compute_loss — inference paths (extract_rl_token, annotation, eval) do not "
                "run these terms."
            )
        if "epadv" in self.rlt_objective and episode_index is None:
            raise ValueError(
                f"rlt_objective={self.rlt_objective!r} needs episode_index. Set "
                "include_episode_index=True on the data config."
            )

        if "action" in self.rlt_objective:
            tgt = actions[..., : self._rlt_act_dim]
            pred = self.rlt_action_out_proj(aux_feat).reshape(tgt.shape)
            act_loss = jnp.mean(jnp.square(pred - tgt), axis=(-1, -2))  # [b]
            rlt_loss = rlt_loss + self.rlt_action_loss_weight * act_loss
            aux |= {
                "rlt/loss_action": jnp.mean(act_loss),
                # Fraction of action variance left unexplained; 1.0 means the head learned the mean
                # action and nothing else, i.e. the token carries no behaviour.
                "rlt/action_nmse": jnp.mean(act_loss) / (jnp.mean(jnp.var(tgt, axis=0)) + 1e-6),
            }

        if "behsim" in self.rlt_objective:
            bs_loss, bs_aux = self._behsim_loss(z_rl, actions)
            rlt_loss = rlt_loss + self.rlt_behsim_weight * bs_loss
            aux |= {"rlt/loss_behsim": jnp.mean(bs_loss), **bs_aux}

        if "epadv" in self.rlt_objective:
            ea_loss, ea_aux = self._epadv_loss(z_rl, episode_index)
            rlt_loss = rlt_loss + self.rlt_epadv_weight * ea_loss
            aux |= {"rlt/loss_epadv": jnp.mean(ea_loss), **ea_aux}

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
        m = img_mask.astype(jnp.float32)

        def _recon_loss(token):
            recon, _ = self._decode(token, z, img_mask)
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

    def _denoise_from_cache(self, rng, observation, prefix_mask, kv_cache, *, num_samples, num_steps, noise_scale=1.0):
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

        # noise_scale > 1 widens the flow init, spreading the candidate set (diversity knob).
        # Scalar = uniform; per-sample array = mixed pool (e.g. safe core at 1.0 + diverse tail).
        scale = jnp.reshape(jnp.asarray(noise_scale, dtype=jnp.float32), (-1, 1, 1, 1))
        noises = scale * jax.random.normal(rng, (num_samples, b, self.action_horizon, self.action_dim))
        chunks = jax.vmap(denoise)(noises)  # [n, b, H, D]
        return jnp.transpose(chunks, (1, 0, 2, 3))  # [b, n, H, D]

    def extract_token_and_base_actions(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, num_samples: int, num_steps: int = 10, noise_scale: float = 1.0
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
            rng, observation, prefix_mask, kv_cache, num_samples=num_samples, num_steps=num_steps, noise_scale=noise_scale
        )
        return z_rl, base

    # ------------------------------------------------------------------
    # Latent BC probe: a flow-matching action head on the frozen RL token
    # ------------------------------------------------------------------

    def _probe_velocity(self, x_t, time, z_rl, state):
        """Flow-matching velocity over the REAL action dims, from (noisy chunk, time, z_rl, proprio).

        x_t is [b, H, probe_action_dim] and the output matches. z_rl is used as-is; callers pass a
        stop-gradient'd token during training so the probe never shapes it.
        """
        pw = self.rlt_probe_time.in_features
        t_emb = posemb_sincos(time, pw, min_period=4e-3, max_period=4.0)
        # One conditioning vector drives adaLN in every block: time + latent + proprio.
        cond = nnx.swish(self.rlt_probe_time(t_emb)) + self.rlt_probe_tok(z_rl) + self.rlt_probe_state(state)
        h = self.rlt_probe_in(x_t) + self.rlt_probe_pos.value  # [b, H, pw]
        for i in range(self._probe_depth):
            h = self.rlt_probe_blocks[f"blk_{i}"](h, cond)
        shift, scale = jnp.split(self.rlt_probe_out_ada(nnx.swish(cond))[:, None, :], 2, axis=-1)
        return self.rlt_probe_out(_layernorm(h) * (1 + scale) + shift)

    def _probe_loss(self, x_t, time, u_t, z_rl, state):
        # Only the real action dims: the padded ones are a constant whose target is exactly x_t/t,
        # an artificial regression problem that would otherwise dominate this loss.
        ad = self.probe_action_dim
        # proprio is a plain conditioning input, not part of the latent — no stop-gradient needed
        # (it does not touch z_rl), but the token itself is detached so the probe stays a pure probe.
        v = self._probe_velocity(x_t[..., :ad], time, jax.lax.stop_gradient(z_rl), state)
        return jnp.mean(jnp.square(v - u_t[..., :ad]), axis=(-1, -2))  # [b]

    def probe_sample_actions(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, num_steps: int = 10
    ) -> _model.Actions:
        """Roll-out action chunk from the probe head alone: extract z_rl, then flow-match on it."""
        z_rl = self.extract_rl_token(observation)  # preprocesses internally
        state = observation.state  # preprocess leaves state unchanged
        b = z_rl.shape[0]
        ad = self.probe_action_dim
        dt = -1.0 / num_steps

        def step(carry):
            x_t, t = carry
            v = self._probe_velocity(x_t, jnp.broadcast_to(t, b), z_rl, state)
            return x_t + dt * v, t + dt

        noise = jax.random.normal(rng, (b, self.action_horizon, ad))
        x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
        # Pad the modelled dims back out to action_dim so the output transform chain (which expects
        # the model's padded action space, then slices) is identical to the VLA's.
        if ad < self.action_dim:
            x_0 = jnp.pad(x_0, ((0, 0), (0, 0), (0, self.action_dim - ad)))
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
        rlt_loss, z_rl, z_tgt, rlt_aux = self._rlt_losses(
            z,
            img_mask,
            observation.state,
            observation.progress,
            actions=actions,
            episode_index=observation.episode_index,
            rng=jax.random.fold_in(rng, 7),
        )

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
