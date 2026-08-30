"""Differentiable query interface to a trained patch critic, for policy-extraction methods.

Loading/preprocessing mirrors scripts/score_critic_cached.py:63-110 (the audited scorer): the
critic's own input_spec decides the preprocessing, PatchCriticEnsemble/PatchV are restored from
msgpack, and values are HL-Gauss expectations. The one addition is that everything is exposed as
pure jittable/differentiable functions of the NORMALIZED action chunk -- which per the input_spec
("pi05-normalized JOINT DELTA -- identical to the sampler's raw output") is exactly the space a
pi0.5 policy emits, so extraction losses can push gradients straight into policy outputs.

Used by: AWR (offline A annotation), CFGRL (labels), QAM/FlowDPG (grad_a Q), LPS/LPSD (Q of
steered actions). Ensemble reduction = mean, matching LPS agents/lps.py:190-199 and QAM
agents/qam.py:81-83 (both reduce the ensemble by mean inside the objective/gradient).
"""

import dataclasses
import json
import pathlib

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.patch_critic import preproc as critic_preproc
from openpi.patch_critic import spec as critic_spec
from openpi.patch_critic.critic import HLGauss
from openpi.patch_critic.critic import PatchCriticEnsemble
from openpi.patch_critic.critic import PatchV


@dataclasses.dataclass(frozen=True)
class CriticQ:
    """Frozen critic bundle. q/v operate on cached DINO features + NORMALIZED action chunks."""

    config: dict
    params: dict
    v_params: dict
    net: PatchCriticEnsemble
    v_net: PatchV
    hl: HLGauss
    proprio_idx: np.ndarray | None
    pre: critic_preproc.Pi05Preproc | None  # raw absolute chunk + raw state -> normalized delta

    def q_min(self, feats, chunk_norm, proprio):
        """Min-over-ensemble full-chunk Q — the twin-critic min of FlowDPG Eq. 5 (arXiv
        2606.22303 Sec. 3.2); our K=2 ensemble plays the twin pair."""
        logits = self.net.apply({"params": self.params}, feats, chunk_norm, proprio)
        return self.hl.from_logits(logits).min(axis=0)[..., -1]

    def q_mean(self, feats, chunk_norm, proprio):
        """Mean-over-ensemble, per-prefix-last (full-chunk) Q. Differentiable in chunk_norm.

        feats [B, P, E] fp32 patch features; chunk_norm [B, H, ad] normalized joint-delta;
        proprio [B, pd] the critic's proprio slice (RAW units per spec, pos-14).
        """
        logits = self.net.apply({"params": self.params}, feats, chunk_norm, proprio)  # [K,B,P,atoms]
        q = self.hl.from_logits(logits)  # [K, B, Pfx]
        return q.mean(axis=0)[..., -1]  # full-chunk prefix, ensemble mean

    def q_prefixes(self, feats, chunk_norm, proprio):
        logits = self.net.apply({"params": self.params}, feats, chunk_norm, proprio)
        return self.hl.from_logits(logits).mean(axis=0)  # [B, Pfx]

    def v(self, feats, proprio):
        logits = self.v_net.apply({"params": self.v_params}, feats, proprio)
        return self.hl.from_logits(logits)


def load(critic_dir) -> CriticQ:
    d = pathlib.Path(critic_dir)
    cc, _ = critic_spec.load(d)
    isp = cc.get("input_spec", {})
    if isp.get("normalization") != "pi05":
        raise ValueError(f"extraction needs a pi05-space critic, got {isp.get('normalization')!r}")
    ns = d / isp.get("norm_stats_file", "pi05_norm_stats.json")
    pre = critic_preproc.Pi05Preproc(
        ref=np.asarray(isp["joint_delta_reference"], np.int64),
        stats=critic_preproc.load_norm_stats(ns if ns.exists() else isp["norm_stats"]),
        use_quantiles=bool(isp["use_quantiles"]),
        delta=isp["delta_mode"] == "joint",
    )
    pidx = isp.get("proprio_indices")
    net = PatchCriticEnsemble(
        action_dim=cc["action_dim"],
        horizon=cc["horizon"],
        num_critics=cc["num_critics"],
        macro_group_size=cc["macro_group_size"],
        num_atoms=cc["num_atoms"],
    )

    def _unwrap(raw):
        # the msgpacks store the full variables dict {"params": ...} (score_critic_cached.py:43,55
        # passes them to apply() verbatim); CriticQ methods re-wrap, so strip the outer layer here
        return raw["params"] if set(raw.keys()) == {"params"} else raw

    return CriticQ(
        config=cc,
        params=_unwrap(flax.serialization.msgpack_restore((d / "params.msgpack").read_bytes())),
        v_params=_unwrap(flax.serialization.msgpack_restore((d / "v_params.msgpack").read_bytes())),
        net=net,
        v_net=PatchV(num_atoms=cc["num_atoms"]),
        hl=HLGauss(cc["v_min"], cc["v_max"], cc["num_atoms"]),
        proprio_idx=None if pidx is None else np.asarray(pidx, np.int64),
        pre=pre,
    )


class CacheView:
    """Row access to the DINO feature cache the critic was trained on (pc_cache/yam_s347)."""

    def __init__(self, cache_dir):
        c = pathlib.Path(cache_dir)
        meta = json.loads((c / "meta.json").read_text())
        n, p, e = meta["N"], meta["npatch"], meta["emb"]
        self.meta = meta
        self.feats = np.memmap(c / "features.dat", np.float16, "r", shape=(n, p, e))
        self.states = np.memmap(c / "state.dat", np.float32, "r", shape=(n, meta["sd"]))
        self.actions = np.memmap(c / "action.dat", np.float32, "r", shape=(n, meta["ad"]))

    def rows(self, idx, critic: CriticQ):
        f = np.asarray(self.feats[idx], np.float32)
        s = np.asarray(self.states[idx])
        pr = s if critic.proprio_idx is None else s[:, critic.proprio_idx]
        return f, s, pr


def grad_q_chunk(critic: CriticQ):
    """d(mean Q)/d(chunk_norm), clipped inside the call like QAM's clip_adj (qam agents/qam.py:80-81):
    actions are clipped to the normalized box before the critic sees them, so the gradient is
    exactly zero outside the box -- the hard saturation guard the official code uses."""

    def q_of(chunk, feats, proprio):
        return critic.q_mean(feats, jnp.clip(chunk, -1.0, 1.0), proprio).sum()

    return jax.grad(q_of, argnums=0)


def grad_qmin_chunk(critic: CriticQ):
    """d(min-ensemble Q)/d(chunk_norm) — FlowDPG Eq. 5 gradient (min over twins), with the
    QAM-style in-call clip guard."""

    def q_of(chunk, feats, proprio):
        return critic.q_min(feats, jnp.clip(chunk, -1.0, 1.0), proprio).sum()

    return jax.grad(q_of, argnums=0)
