"""pi0.5 + AWR: advantage-weighted flow-BC fine-tune of the action expert.

Provenance:
  - Objective: Peng et al., "Advantage-Weighted Regression" (arXiv 1910.00177), Eq. 8 --
    supervised regression weighted by exp(A/beta). Weight mechanics follow the OFFICIAL code
    xbpeng/awr learning/awr_agent.py exactly: advantages are z-score normalized before
    exponentiation (awr_agent.py:403 `norm_adv = (adv - mean)/(std + eps)`), weights are
    `exp(norm_adv / temp)` (awr_agent.py:407) with temp=1.0 (awr_agent.py:43) and clipped at
    weight_clip=20 (awr_agent.py:41,409-410).
  - The regression loss for a flow policy is pi0.5's own flow-matching loss (pi0.py:189-214),
    weighted per sample -- AWR is agnostic to the regression family (paper Sec. 4: "any
    supervised regression procedure").

The split between this file and the data config follows where each piece is USED. The z-score is a
statistic of the whole dataset, computed once when the data config is built, and arrives on
`Observation.advantage`. Temperature and clipping are what an experiment SWEEPS, so they live here
on the model config and travel with the checkpoint.

This is a pure loss variant: the network is exactly pi0.5, so a trained checkpoint is served by the
BASE config unchanged (`--policy.config pi05_yam_lego_taxi`). Only `compute_loss` differs, which is
why there is no `sample_actions` override here -- unlike CFGRL, whose network really does differ.
"""

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils


@dataclasses.dataclass(frozen=True)
class Pi0AWRConfig(Pi0Config):
    """pi0.5 whose loss is advantage-weighted. Everything else is inherited from the base config."""

    # exp(norm_adv / temp). awr_agent.py:43 uses 1.0; lower = greedier.
    awr_temp: float = 1.0
    # Ceiling on the weight, so one high-advantage chunk cannot dominate a batch (awr_agent.py:41).
    awr_weight_clip: float = 20.0
    # Train the action expert only. A field rather than a launch flag so the checkpoint records
    # which budget the run had; see get_freeze_filter below.
    freeze_backbone: bool = True

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    def create(self, rng: at.KeyArrayLike) -> "Pi0AWR":
        return Pi0AWR(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train the action expert only; the backbone stays at its BC values.

        Extraction refines the head on top of a finished BC representation, and holding the
        backbone fixed is what keeps the arms comparable to each other -- same frozen features,
        same init, only the objective differs. Pass `--model.no-freeze-backbone`-style overrides by
        building the config with `freeze_backbone=False` if you want the BC budget instead.
        """
        if not self.freeze_backbone:
            return super().get_freeze_filter()
        return nnx.Not(
            nnx.Any(
                nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx_utils.PathRegex(".*(action_(in|out)_proj|time_mlp_(in|out)|state_proj).*"),
            )
        )


class Pi0AWR(Pi0):
    """pi0.5 with an advantage-weighted flow-matching loss."""

    def __init__(self, config: Pi0AWRConfig, *, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self._temp = float(config.awr_temp)
        self._clip = float(config.awr_weight_clip)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        """Per-sample AWR loss, plus the unweighted BC loss as a comparable diagnostic.

        Returns a PER-SAMPLE loss because that is what the trainer means: it takes the mean, so the
        weighted mean is what gets differentiated. `bc_loss` is the key every config emits so a
        weighted run and a plain BC run are directly comparable on one wandb chart -- the trainer
        pops it out of aux (see scripts/train.py).
        """
        if observation.advantage is None:
            raise ValueError(
                "Pi0AWR needs Observation.advantage, which the data config attaches. Use a data "
                "config built with advantage_dir=<annotate_advantage.py output> -- without it this "
                "would silently train plain BC, which is the failure CFGRL's config already had."
            )
        # pi0.5's own flow-matching loss is [b, ah] (pi0.py:214, mean over action dim)
        per_step = super().compute_loss(rng, observation, actions, train=train)
        per_sample = jnp.mean(per_step, axis=tuple(range(1, per_step.ndim)))

        # exp(norm_adv / temp), clipped (awr_agent.py:407,409-410). The advantage arrives already
        # z-scored over the dataset, so this is the official weight verbatim.
        weights = jnp.minimum(jnp.exp(observation.advantage / self._temp), self._clip)

        return weights * per_sample, {
            "bc_loss": jnp.mean(per_sample),
            "awr/weight_mean": jnp.mean(weights),
            "awr/weight_max": jnp.max(weights),
            "awr/frac_at_clip": jnp.mean((weights >= self._clip).astype(jnp.float32)),
        }
