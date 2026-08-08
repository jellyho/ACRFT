"""Critic architectures over precomputed RL tokens: QC (flat) and ARQ (prefix-conditioned).

Both score an action chunk from the compact RL token, and both come in a scalar or a distributional
(HL-Gauss) flavour. They differ in what they output:

**QC — Q-chunking.** One value for the whole chunk::

    Q(z, a_{1:H})

An MLP over ``concat(z, flatten(a))``. Simple and fast; the chunk length is fixed.

**ARQ — autoregressive / prefix-conditioned Q** (the ACSAC critic). A causal transformer over
``[z, m_1, ..., m_M]`` (one state token, then the chunk cut into ``M = H / macro_group_size`` macro
tokens) that returns, in a single forward, a value for *every prefix*::

    [ Q(z, a_{1:g}), Q(z, a_{1:2g}), ..., Q(z, a_{1:H}) ]

The causal mask is what makes this sound: output ``h`` attends only to ``(z, a_{1:h})``, never to the
suffix, so each head really is the value of committing to that prefix and re-planning. Grouping H
steps into macro tokens keeps the sequence short — with H=10 and g=2 there are 5 prefixes to choose
between rather than 10 nearly-identical ones. Because every head is trained against a return target
on the same discounted scale, the values are comparable across prefix lengths, which is what lets
deployment take a joint arg-max over (candidate, prefix length) and pick how far to commit.

Both are ensembled (K critics, min-aggregated) to blunt over-estimation.
"""

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp


def _mlp(x, dims, *, out_dim):
    for d in dims:
        x = nn.gelu(nn.Dense(d)(x))
    return nn.Dense(out_dim)(x)


class QCCritic(nn.Module):
    """Flat Q-chunking critic: one value per chunk."""

    action_dim: int
    horizon: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    num_atoms: int = 1  # 1 -> scalar Q; >1 -> HL-Gauss categorical logits
    layer_norm: bool = True
    # >0 = the last `proprio_dim` entries of obs are proprioception, to be given their own projection
    # instead of being 16 raw dims among 2048 token dims (~0.8% of the input, and swamped again by
    # the LayerNorm that follows).
    proprio_dim: int = 0
    proprio_embed: int = 128

    @nn.compact
    def __call__(self, obs, actions):
        # obs [..., D], actions [..., H, A] (or already flat [..., H*A])
        a = actions.reshape(*actions.shape[: -2 if actions.ndim > 2 else -1], -1)
        if self.proprio_dim:
            z, pro = obs[..., : -self.proprio_dim], obs[..., -self.proprio_dim :]
            # Widen proprio to a comparable number of features before it meets the token, so the
            # first Dense sees it as a peer rather than as rounding error.
            obs = jnp.concatenate([z, nn.gelu(nn.Dense(self.proprio_embed)(pro))], axis=-1)
        x = jnp.concatenate([obs, a], axis=-1)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        out = _mlp(x, self.hidden_dims, out_dim=self.num_atoms)
        return out if self.num_atoms > 1 else jnp.squeeze(out, -1)


class ARQCritic(nn.Module):
    """Prefix-conditioned causal-transformer critic: one value per prefix length."""

    action_dim: int
    horizon: int
    macro_group_size: int = 2
    num_layers: int = 3
    num_heads: int = 8
    head_dim: int = 48
    mlp_dim: int = 1024
    num_atoms: int = 1
    per_position_head: bool = True
    # >1 = ONE shared trunk with K independent per-position output heads, returned with a leading
    # ensemble axis [K, ..., mh(, atoms)] - the same contract Ensemble provides, at ~1/K the cost of
    # K full transformers. The members share features, so they are more correlated than true
    # ensembles (weaker per-member pessimism); the trade is that K can be large for free, which is
    # what the min/LCB over the DEPLOYMENT arg-max actually needs.
    head_ensemble: int = 1
    head_ensemble_hidden: int = 128
    # >0 = the last `proprio_dim` entries of obs are proprioception, given their OWN sequence
    # position instead of being concatenated into the state token. Concatenated, 16 proprio dims sit
    # beside 2048 token dims (0.8% of the input) and are then LayerNorm'd against the token's
    # statistics, so the single Dense that makes the state token can barely see them. As its own
    # token it gets its own projection and its own place in attention.
    proprio_dim: int = 0

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim

    # >0 = the LEADING history*history_token_dim entries of obs are K past rl_tokens (oldest first),
    # each becoming its own observation position (short-history conditioning: single frames are
    # ambiguous under occlusion and repeated motions - Robo-ValueRL ablation, 5 frames > none > 30).
    history: int = 0
    history_token_dim: int = 2048

    @property
    def macro_h(self) -> int:
        return self.horizon // self.macro_group_size

    @nn.compact
    def __call__(self, obs, actions):
        d, mh = self.n_embd, self.macro_h
        a = actions.reshape(*actions.shape[:-2], mh, self.macro_group_size * self.action_dim)

        hist_tok = None
        if self.history:
            hd = self.history * self.history_token_dim
            hist, obs = obs[..., :hd], obs[..., hd:]
            hist = hist.reshape(*hist.shape[:-1], self.history, self.history_token_dim)
            # ONE shared projection for every history slot; order is carried by the pos embedding.
            hist_tok = nn.Dense(d, name="hist_proj")(nn.LayerNorm()(hist))  # [..., K, d]

        if self.proprio_dim:
            z, pro = obs[..., : -self.proprio_dim], obs[..., -self.proprio_dim :]
            lead = jnp.concatenate(
                [nn.Dense(d)(nn.LayerNorm()(z))[..., None, :], nn.Dense(d)(nn.LayerNorm()(pro))[..., None, :]],
                axis=-2,
            )  # [..., 2, d]
        else:
            lead = nn.Dense(d)(nn.LayerNorm()(obs))[..., None, :]  # [..., 1, d]
        if hist_tok is not None:
            lead = jnp.concatenate([hist_tok, lead], axis=-2)
        nl = lead.shape[-2]
        act_tok = nn.Dense(d)(a)  # [..., mh, d]
        x = jnp.concatenate([lead, act_tok], axis=-2)  # [..., nl+mh, d]
        x = x + self.param("pos", nn.initializers.normal(0.02), (nl + mh, d))

        # Causal: prefix h must not see the suffix, otherwise its "value of committing to h steps"
        # would be contaminated by actions that would never be executed. The leading observation
        # tokens are visible to every action position.
        causal = jnp.tril(jnp.ones((nl + mh, nl + mh), dtype=bool))
        causal = causal.at[:, :nl].set(True)
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(h, h, mask=causal)
            h = nn.LayerNorm()(x)
            x = x + nn.Dense(d)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        x = nn.LayerNorm()(x)[..., nl:, :]  # drop the observation positions: [..., mh, d]

        if self.head_ensemble > 1:
            # Two-layer MLP per head, NOT a linear map: linear heads on a shared feature are K
            # reparametrisations of one function and diversify only through a single init matrix.
            ke, hh = self.head_ensemble, self.head_ensemble_hidden
            w1 = self.param("head_w1", nn.initializers.lecun_normal(), (ke, mh, d, hh))
            b1 = self.param("head_b1", nn.initializers.zeros, (ke, mh, hh))
            w2 = self.param("head_w2", nn.initializers.lecun_normal(), (ke, mh, hh, self.num_atoms))
            b2 = self.param("head_b2", nn.initializers.zeros, (ke, mh, self.num_atoms))
            pad = (slice(None),) + (None,) * (x.ndim - 2)
            h1 = nn.gelu(jnp.einsum("...hd,khdm->k...hm", x, w1) + b1[pad])
            out = jnp.einsum("k...hm,khma->k...ha", h1, w2) + b2[pad]
            return out if self.num_atoms > 1 else jnp.squeeze(out, -1)  # [K, ..., mh(, atoms)]
        if self.per_position_head:
            # A distinct head per prefix position (ACSAC Prop. G.7).
            kernel = self.param("head_k", nn.initializers.lecun_normal(), (mh, d, self.num_atoms))
            bias = self.param("head_b", nn.initializers.zeros, (mh, self.num_atoms))
            out = jnp.einsum("...hd,hda->...ha", x, kernel) + bias
        else:
            out = nn.Dense(self.num_atoms)(x)
        return out if self.num_atoms > 1 else jnp.squeeze(out, -1)  # [..., mh(, atoms)]


class ValueNet(nn.Module):
    """State value V(z) — no action input, one scalar per state.

    The IQL half of the critic. Its whole point is that it never sees an action, so learning it needs
    no candidate chunks and takes no max: the expectile regression in the trainer pushes V toward an
    upper quantile of Q(z, a) over the actions the DATA contains, which is what stands in for
    ``max_a Q``. That removes the arg-max over N candidates x P prefixes that the TD objective takes,
    and with it the upward bias of maxing over that many noisy estimates.

    Scalar even when Q is distributional: expectile regression is defined on a scalar residual, and a
    histogram head would have to be collapsed to one anyway before the asymmetric weight applies.
    """

    hidden_dims: tuple[int, ...] = (512, 512, 512)
    # (lo, hi) squashes the output onto the value support with a sigmoid. Plain IQL leaves this off:
    # V is anchored directly by its target (Q_tgt lives on the support). Dueling NEEDS it - Q = A + V
    # has a gauge freedom ((V+c, A-c) changes the loss only through the (gamma^h - 1)*c residue of
    # the bootstrap, ~0.02c here), and measured un-bounded, V drifted to +-200 within 1k steps while
    # A chased it. Bounding V is what pins the gauge.
    bound: tuple[float, float] | None = None
    # 1 = the classic scalar V. 1 + P = AQC layout: head 0 keeps the bootstrap semantics unchanged,
    # heads 1..P are per-prefix baselines b_h(z) ~ expectile_h of Q(z, a_demo, h) - each Q head gets
    # a baseline that shares its own systematic bias, so (Q_h - b_h) cancels it before the /gamma^h
    # rescale makes horizons comparable.
    out_dim: int = 1

    @nn.compact
    def __call__(self, obs):
        out = _mlp(nn.LayerNorm()(obs), self.hidden_dims, out_dim=self.out_dim)
        out = jnp.squeeze(out, -1) if self.out_dim == 1 else out  # [...] or [..., out_dim]
        if self.bound is not None:
            lo, hi = self.bound
            out = lo + (hi - lo) * nn.sigmoid(out)
        return out


class Ensemble(nn.Module):
    """K independent critics stacked on a leading axis (min-aggregated by the caller)."""

    make_critic: callable
    num_critics: int = 2

    @nn.compact
    def __call__(self, obs, actions):
        vmapped = nn.vmap(
            lambda mdl, o, a: mdl(o, a),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_critics,
        )
        return vmapped(self.make_critic(), obs, actions)


@dataclasses.dataclass(frozen=True)
class HLGauss:
    """Scalar <-> soft-histogram transforms for the distributional critic."""

    v_min: float
    v_max: float
    num_atoms: int
    sigma_frac: float = 0.75

    @property
    def edges(self):
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms + 1)

    @property
    def centers(self):
        e = self.edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def sigma(self):
        return self.sigma_frac * (self.v_max - self.v_min) / self.num_atoms

    def to_probs(self, y):
        cdf = jax.scipy.stats.norm.cdf(self.edges, loc=y[..., None], scale=self.sigma)
        p = cdf[..., 1:] - cdf[..., :-1]
        return p / (jnp.sum(p, axis=-1, keepdims=True) + 1e-8)

    def from_logits(self, logits):
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * self.centers, axis=-1)


def make_critic(kind: str, *, action_dim: int, horizon: int, num_atoms: int, **kw):
    """Build a QC or ARQ critic module. Both take (obs, actions) and return values."""
    if kind == "qc":
        return QCCritic(action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **kw)
    if kind == "arq":
        return ARQCritic(action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **kw)
    raise ValueError(f"critic kind must be 'qc' or 'arq', got {kind!r}")


def load_value_net(params_path):
    """Load the IQL state-value net saved beside a critic checkpoint, or None if the run has none.

    Returns ``v_apply(obs [..., D]) -> V [...]``. TD runs never write vparams, so a None here is the
    normal case, not an error - callers use it to decide whether a V trace exists at all.
    """
    import json
    import pathlib

    import flax.serialization

    params_path = pathlib.Path(params_path)
    run_dir = params_path.parent
    vpath = run_dir / params_path.name.replace("params", "vparams")
    if not vpath.exists():
        vpath = run_dir / "vparams.msgpack"
    if not vpath.exists():
        return None
    cfg = json.loads((run_dir / "config.json").read_text())
    out_dim = 1 + cfg.get("num_prefixes", 0) if cfg.get("aqc_baseline") else 1
    v_net = ValueNet(hidden_dims=tuple(cfg.get("hidden_dims", (512, 512, 512))), out_dim=out_dim)
    vparams = flax.serialization.msgpack_restore(vpath.read_bytes())
    if out_dim == 1:
        return lambda obs: v_net.apply(vparams, obs)
    return lambda obs: v_net.apply(vparams, obs)[..., 0]  # scalar V; baselines via load_prefix_baselines


def load_prefix_baselines(params_path):
    """b_h(z) [..., P] for AQC-style prefix selection, or None if the run trained no baselines."""
    import json
    import pathlib

    import flax.serialization

    run_dir = pathlib.Path(params_path).parent
    cfg = json.loads((run_dir / "config.json").read_text())
    if not cfg.get("aqc_baseline"):
        return None
    vpath = run_dir / pathlib.Path(params_path).name.replace("params", "vparams")
    if not vpath.exists():
        vpath = run_dir / "vparams.msgpack"
    if not vpath.exists():
        # td+aqcmax runs before the save fix trained baselines but never wrote them. Deployment
        # under the shared joint-argmax rule never reads b_h, so a missing file downgrades to None.
        import logging

        logging.getLogger(__name__).warning(f"aqc_baseline run without vparams at {run_dir}: baselines unavailable")
        return None
    v_net = ValueNet(hidden_dims=tuple(cfg.get("hidden_dims", (512, 512, 512))), out_dim=1 + cfg["num_prefixes"])
    vparams = flax.serialization.msgpack_restore(vpath.read_bytes())
    return lambda obs: v_net.apply(vparams, obs)[..., 1:]


def load_trained(params_path, *, action_dim: int, horizon: int):
    """Rebuild a trained critic from the params and the config train_rlt_critic.py saved beside them.

    The architecture has to match the checkpoint exactly, and re-declaring it at every call site is
    how that goes wrong silently. Returns ``(apply_fn, params, macro_group_size)`` where ``apply_fn``
    takes ``(obs [S, D], actions [S, M, H, A])`` and returns expected scalar values ``[K, S, M, P]``.
    """
    import json
    import pathlib

    import flax.serialization

    params_path = pathlib.Path(params_path)
    cfg = json.loads((params_path.parent / "config.json").read_text())
    g = cfg.get("macro_group_size", 2)
    num_atoms = cfg.get("num_atoms", 1)
    arch = (
        {
            "macro_group_size": g,
            "head_ensemble": cfg.get("head_ensemble", 1),
            "num_layers": cfg.get("num_layers", 3),
            "num_heads": cfg.get("num_heads", 8),
            "head_dim": cfg.get("head_dim", 48),
            "mlp_dim": cfg.get("mlp_dim", 1024),
            "history": cfg.get("history", 0),
            "history_token_dim": cfg.get("history_token_dim", 2048),
        }
        if cfg.get("kind", "arq") == "arq"
        else {"hidden_dims": tuple(cfg.get("hidden_dims", [512, 512, 512]))}
    )
    # Only proprio_mode='token' puts proprio inside the module (its own sequence position for ARQ, a
    # widened embedding for QC); 'concat' leaves it as ordinary input dims. Rebuilding without this
    # would silently construct a different network from the one the params came from.
    if cfg.get("use_proprio") and cfg.get("proprio_mode") == "token":
        import json as _json
        import pathlib as _pathlib

        arch["proprio_dim"] = _json.loads((_pathlib.Path(cfg["data"]) / "meta.json").read_text())["proprio_dim"]
    if cfg.get("head_ensemble", 1) > 1:
        # The K axis already leads from the head bank; wrapping in Ensemble would nest two
        # ensemble axes and break every consumer's [K, ...] contract.
        net = make_critic(cfg.get("kind", "arq"), action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **arch)
    else:
        net = Ensemble(
            make_critic=lambda: make_critic(
                cfg.get("kind", "arq"), action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **arch
            ),
            num_critics=cfg.get("num_critics", 2),
        )
    hl = HLGauss(v_min=cfg.get("v_min", 0.0), v_max=cfg.get("v_max", 1.0), num_atoms=max(num_atoms, 2))
    params = flax.serialization.msgpack_restore(params_path.read_bytes())
    a_norm = cfg.get("action_norm")
    if a_norm:
        # Training normalised actions per-dim (stats recorded in config.json); every caller keeps
        # passing RAW actions and the transform is applied here, once, for all of them.
        _mu = jnp.asarray(a_norm["mean"], dtype=jnp.float32)
        _sd = jnp.asarray(a_norm["std"], dtype=jnp.float32)

    v_add = None
    if cfg.get("dueling"):
        # The saved Q params are the ADVANTAGE head; deployment must add the value baseline back or
        # every consumer of "Q" (value traces, commit rules, diagnostics) reads a different quantity.
        # Within-state arg-max alone would survive without it - nothing else would.
        vp = params_path.parent / (
            "vparams.msgpack"
            if params_path.name == "params.msgpack"
            else params_path.name.replace("params_", "vparams_")
        )
        v_params = flax.serialization.msgpack_restore(vp.read_bytes())
        v_net = ValueNet(hidden_dims=tuple(cfg.get("hidden_dims", [512, 512, 512])))

        def v_add(obs):
            return v_net.apply(v_params, obs)

    if cfg.get("kind", "arq") == "qc":
        # QC has no prefix head, so the raw output is [K, S, M] and the promised trailing P axis is
        # missing - which is exactly the shape every caller then unpacks wrong (np.unravel_index over
        # a 1-tuple killed every QC rollout, silently, in three sweeps). A P=1 axis keeps the
        # contract; the matching "steps per prefix" is the whole horizon, because committing to QC's
        # one value IS committing to the full chunk.
        def apply_fn(obs, actions):
            if a_norm:
                actions = (actions - _mu) / _sd
            out = net.apply(params, obs, actions)
            out = (hl.from_logits(out) if num_atoms > 1 else out)[..., None]
            return out + v_add(obs)[..., None] if v_add is not None else out

        return apply_fn, params, horizon

    def apply_fn(obs, actions):
        if a_norm:
            actions = (actions - _mu) / _sd
        out = net.apply(params, obs, actions)
        out = hl.from_logits(out) if num_atoms > 1 else out
        if v_add is not None:
            out = out - out.mean(axis=-1, keepdims=True) + v_add(obs)[..., None]
        return out

    return apply_fn, params, g
