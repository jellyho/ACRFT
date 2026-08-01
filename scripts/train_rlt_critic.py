"""Stage 2: train a QC or ARQ critic on the precomputed RL tokens (see annotate_rlt.py).

The whole dataset is numeric and small (a few GB), so it is loaded ONCE onto the GPU and never
touched by a data loader again: batches are on-device gathers, and many update steps are fused into
a single dispatch with `lax.scan`. For a critic this matters more than the network itself — the
model is tiny, so Python/dispatch overhead and host->device traffic are what would otherwise
dominate the step time.

Targets, for a chunk of `h` steps starting at frame t:

    y_h = sum_{i<h} gamma^i r_{t+i}  +  gamma^h * (not terminal) * V(z_{t+h})
    V(s') = max over the VLA's candidate chunks (and, for ARQ, over prefix lengths) of
            min over the ensemble of Q_target(s', a')

`Q_target` is an EMA copy of the critic (Polyak), not the online network.

  * **QC** trains the single full-chunk value, so there is one target per transition.
  * **ARQ** trains every prefix head against its own horizon-matched target, which is what puts the
    per-prefix values on a common discounted scale and makes them comparable at deployment.

Usage:

    srun --gres=gpu:1 uv run scripts/train_rlt_critic.py \
        --data data/rlt_critic/PrepareCoffee --kind arq --steps 200000
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.rlt_critic import critic as _critic

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Data:
    """The annotated task, resident on the accelerator."""

    token: jax.Array  # [T, D]
    chunk: jax.Array  # [T, H, A]   executed demo chunk (the `a` in Q(s, a))
    cand: jax.Array  # [T, N, H, A] VLA candidates (the `a'` the bootstrap maximises over)
    reward: jax.Array  # [T]
    episode: jax.Array  # [T]
    mc_return: jax.Array  # [T]  discounted return the behaviour policy actually collected
    done: jax.Array  # [T]  1 at a terminal (task achieved, or the episode's last frame)
    done_cum: jax.Array  # [T]  running count of terminals, for "how many fall inside [t, t+h]"
    alive: jax.Array  # [T]  frame is at or before its episode's terminal (a real decision point)
    borrowed: jax.Array  # [T, N]  per-candidate return borrowed from the nearest executed chunk (or NaN)
    horizon: int
    action_dim: int
    num_samples: int


def _terminals(done, episode):
    """Precompute the two terminal lookups the target needs.

    A prefix that steps over a terminal must not bootstrap past it, and a frame that lies after its
    episode's terminal is not a decision point at all - with success made terminal in the annotation
    those frames still sit in the arrays, paying nothing, and training on them would teach the critic
    that the task's goal state is worth continuing from.
    """
    done = np.asarray(done, np.int64)
    alive = np.ones(len(done), np.float32)
    for e in np.unique(episode):
        w = np.flatnonzero(episode == e)
        fired = w[done[w] > 0]
        if len(fired):
            alive[fired[0] + 1 :][: w[-1] - fired[0]] = 0.0
    return {
        "done": jnp.asarray(done.astype(np.int32)),
        "done_cum": jnp.asarray(np.cumsum(done, dtype=np.int32)),
        "alive": jnp.asarray(alive),
    }


def load_data(path: pathlib.Path, *, max_frames: int = 0) -> Data:
    meta = json.loads((path / "meta.json").read_text())
    if meta["stride"] != 1:
        raise ValueError(
            f"stride={meta['stride']}: per-prefix reward sums need every frame. Re-run annotate_rlt.py with --stride 1."
        )
    T, D, H, A, N = (meta[k] for k in ("num_frames", "token_dim", "horizon", "action_dim", "num_samples"))
    if max_frames:
        T = min(T, max_frames)

    # Honour the dtype the annotation actually wrote; 16-bit storage is cast up on the way to the
    # device so the critic still computes in float32.
    import ml_dtypes

    store = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]

    def rd(name, shape, dtype=None):
        arr = np.memmap(path / f"{name}.dat", dtype=store if dtype is None else dtype, mode="r", shape=shape)[:T]
        return np.asarray(arr, dtype=np.float32) if dtype is None else np.asarray(arr)

    full = json.loads((path / "meta.json").read_text())["num_frames"]
    d = Data(
        token=jnp.asarray(rd("rl_token", (full, D))),
        chunk=jnp.asarray(rd("action_chunk", (full, H, A))),
        cand=jnp.asarray(rd("base_action", (full, N, H, A))),
        reward=jnp.asarray(rd("reward", (full,))),
        episode=jnp.asarray(rd("episode_index", (full,), np.int32)),
        mc_return=jnp.asarray(rd("mc_return", (full,), np.float32)),
        borrowed=jnp.asarray(
            rd("borrowed_return", (full, N), np.float32)
            if (path / "borrowed_return.dat").exists()
            else np.full((full, N), np.nan, np.float32)
        ),
        **_terminals(rd("done", (full,), np.int8), rd("episode_index", (full,), np.int32)),
        horizon=H,
        action_dim=A,
        num_samples=N,
    )
    gb = sum(x.size * x.itemsize for x in (d.token, d.chunk, d.cand)) / 1e9
    logger.info(f"loaded {T} frames onto {jax.devices()[0].platform} ({gb:.2f} GB): token {D}, chunk {H}x{A}, N={N}")
    return d


def make_update(data: Data, cfg, net, hl, act_scale, meta_support=(0.0, 1.0)):
    """One jitted critic update, written so it can be scanned over many steps."""
    T = data.token.shape[0]
    H, g = data.horizon, cfg.macro_group_size
    # Prefix lengths in real steps: ARQ trains all of them, QC only the full chunk.
    prefixes = jnp.arange(g, H + 1, g) if cfg.kind == "arq" else jnp.array([H])
    disc = cfg.discount ** jnp.arange(H, dtype=jnp.float32)  # [H]

    # Bound the target to the value support. Bootstrapped over-estimation is amplified here far more
    # than in ordinary actor-critic: the fixed point of a per-backup bias b under a prefix of h steps
    # is b*gamma^h/(1-gamma^h), which is 49x at the shortest prefix, so a bias too small to see in one
    # backup becomes many times the value range. Narrowing the arg-max cannot fix that - 128 to 32
    # candidates shrinks E[max of noise] by sqrt(ln128/ln32) = 1.18 - but the support caps the fixed
    # point directly. With success terminal and a 0/1 reward the support is [0, 1] whatever the task.
    sup = meta_support if cfg.v_clip == "auto" else (0.0, float(cfg.v_clip or 0.0))
    v_lo, v_hi = float(sup[0]), float(sup[1])
    logger.info(f"value support [{v_lo:.3f}, {v_hi:.3f}] ({cfg.v_clip})" if v_hi > v_lo else "value clip OFF")

    def targets(idx, tgt_params, rng):
        """Per-prefix (cum_reward, next_value, valid) for the sampled transitions."""
        ep = data.episode[idx]  # [B]
        # Rewards over the chunk, zeroed once the episode ends.
        off = jnp.arange(H)[None, :]
        r_idx = jnp.clip(idx[:, None] + off, 0, T - 1)
        same = data.episode[r_idx] == ep[:, None]
        rew = data.reward[r_idx] * same  # [B, H]
        cum_all = jnp.cumsum(rew * disc[None, :], axis=-1)  # [B, H]
        cum = cum_all[:, prefixes - 1]  # [B, P]

        nxt = jnp.clip(idx[:, None] + prefixes[None, :], 0, T - 1)  # [B, P]
        # Terminal handling follows the reference (vla_aqc.py): what matters is the state the prefix
        # LANDS on, not whether a terminal sits somewhere inside the window.
        #   terminals in [idx, idx+h] == 0                  -> ordinary transition, bootstrap
        #   == 1 and it is exactly the landing state        -> terminal transition, no bootstrap
        #   otherwise                                       -> the prefix runs past the terminal,
        #                                                      so the transition does not exist
        # Counting only up to idx+h-1 (the obvious reading) gets the third case right and the second
        # one wrong: landing exactly on the goal would bootstrap a value from the terminal state,
        # and every transition that reaches the goal is of that kind.
        crossed = data.done_cum[nxt] - jnp.where(idx > 0, data.done_cum[idx - 1], 0)[:, None]
        lands_on_term = (crossed == 1) & (data.done[nxt] > 0)
        boot = (crossed == 0) & (idx[:, None] + prefixes[None, :] < T)
        valid = (boot | lands_on_term) & (data.alive[idx][:, None] > 0)

        # V(s') aggregates the target Q over candidates (and prefixes, for ARQ). Both reductions are
        # configurable because a hard max over K*N*P noisy estimates is biased upward, and this method
        # maxes over far more items than ordinary actor-critic (N*H rather than one actor action).
        z_next = data.token[nxt]  # [B, P, D]
        a_next = data.cand[nxt]  # [B, P, N, H, A]
        B, P, N = a_next.shape[0], a_next.shape[1], a_next.shape[2]
        k_sub, k_noise = jax.random.split(rng)
        if 0 < cfg.bootstrap_candidates < N:
            # Bootstrap off a fresh random subset of the stored candidates each step rather than all
            # of them. This is the one knob that helps twice: the target forward is the dominant cost
            # of an update (it scores B*P*n chunks against the online loss's B), and the arg-max over
            # fewer items is less biased upward. Resampling every step still visits all N over
            # training, so nothing is discarded - only the per-step max is narrowed.
            n = cfg.bootstrap_candidates
            sel = jnp.argsort(jax.random.uniform(k_sub, (B, P, N)), axis=-1)[..., :n]  # without replacement
            a_next = jnp.take_along_axis(a_next, sel[..., None, None], axis=2)
            N = n
        if cfg.target_noise > 0:
            # Temporally COHERENT perturbation (constant offset + linear drift), scaled per action
            # dim. Per-step iid noise would make the chunk jitter into a trajectory the policy would
            # never emit; this keeps it a plausible neighbour of the candidate, which is the point -
            # it smooths Q locally so a lone spurious peak cannot win the arg-max.
            k1, k2 = jax.random.split(k_noise)
            ramp = jnp.linspace(-1.0, 1.0, a_next.shape[-2])[:, None]
            off = jax.random.normal(k1, (*a_next.shape[:-2], 1, a_next.shape[-1]))
            drift = jax.random.normal(k2, (*a_next.shape[:-2], 1, a_next.shape[-1])) * ramp
            eps = jnp.clip(off + drift, -cfg.target_noise_clip, cfg.target_noise_clip)
            a_next = a_next + cfg.target_noise * act_scale * eps
        q = net.apply(tgt_params, jnp.repeat(z_next[:, :, None], N, axis=2), a_next)
        q = hl.from_logits(q) if cfg.num_atoms > 1 else q  # [K, B, P, N(, prefix)]
        # Across the ensemble: `min` is the most pessimistic member; `lcb` is mean - beta*std, which
        # degrades gracefully as K grows and exposes the same uncertainty the online phase can use.
        q = jnp.mean(q, 0) - cfg.lcb_beta * jnp.std(q, 0) if cfg.ens_agg == "lcb" else jnp.min(q, 0)
        flat = q.reshape(B, P, -1)  # candidates (+ prefixes) flattened
        if cfg.v_agg == "topm":
            m = min(cfg.top_m, flat.shape[-1])
            v_next = jnp.mean(jax.lax.top_k(flat, m)[0], axis=-1)
        elif cfg.v_agg == "soft":
            w = jax.nn.softmax(flat / cfg.soft_tau, axis=-1)
            v_next = jnp.sum(w * flat, axis=-1)
        else:
            v_next = jnp.max(flat, axis=-1)

        gam = cfg.discount ** prefixes.astype(jnp.float32)  # [P]
        if v_hi > v_lo:
            v_next = jnp.clip(v_next, v_lo, v_hi)
        # At a terminal the critic has nothing to say, and the return from there is known exactly.
        if cfg.terminal_uses_mc:
            v_next = jnp.where(lands_on_term, data.mc_return[nxt], v_next)
            y = cum + gam[None, :] * (boot | lands_on_term) * v_next
        else:
            y = cum + gam[None, :] * boot * v_next
        floor_gap = jnp.maximum(data.mc_return[idx][:, None] - y, 0.0)
        if cfg.mc_lower_bound:
            # The behaviour policy demonstrably obtained mc_return from this state, so the optimal
            # value cannot be below it. Unlike the ceiling this rarely binds once the critic is
            # inflated; it matters early, and it stops a pessimistic aggregation from settling below
            # what the data proves is achievable.
            y = jnp.maximum(y, data.mc_return[idx][:, None])
        if v_hi > v_lo:
            y = jnp.clip(y, v_lo, v_hi)
        return y, valid, {"term_frac": lands_on_term, "floor_gap": floor_gap}

    def loss_fn(params, tgt_params, idx, rng):
        # `tgt_params` is whatever --bootstrap selected; stop_gradient makes the online choice safe.
        y, valid, tinfo = targets(idx, jax.lax.stop_gradient(tgt_params), rng)
        y = jax.lax.stop_gradient(y)
        pred = net.apply(params, data.token[idx], data.chunk[idx])  # [K, B(, P)(, atoms)]
        if cfg.kind == "qc":
            pred = pred[:, :, None] if cfg.num_atoms == 1 else pred[:, :, None, :]

        w = valid.astype(jnp.float32)[None]  # [1, B, P]
        if cfg.num_atoms > 1:
            probs = hl.to_probs(jnp.clip(y, hl.v_min, hl.v_max))[None]  # [1, B, P, atoms]
            per = -jnp.sum(probs * jax.nn.log_softmax(pred, axis=-1), axis=-1)
            q_mean = jnp.mean(hl.from_logits(pred))
        else:
            per = jnp.square(pred - y[None])
            q_mean = jnp.mean(pred)
        loss = jnp.sum(per * w) / (jnp.sum(w) * pred.shape[0] + 1e-8)
        vs = jnp.maximum(jnp.sum(valid), 1.0)

        # --- per-candidate borrowed-return regression -------------------------------------------
        # The TD loss above scores one action per state (the demo) and reads its target at the state
        # the demo reached, so it never says one candidate is worth more than another - which is why
        # a critic trained on it does not extract the within-state ranking signal that IS in the data
        # (signal_analysis.py: SNR ~ 11, but the critic's within-state Spearman against the true
        # outcome is 0.03). This term gives every candidate its OWN target: the return of the nearest
        # executed chunk, validated to recover the demo's known return at 0.86. Unlike CQL it does not
        # push the candidates down uniformly - it pulls each toward its estimated value, which is what
        # creates an ordering. The borrowed return is approximate, so this is a small auxiliary weight
        # and the TD term still carries the level.
        borrow = jnp.zeros(())
        if cfg.borrowed_weight > 0:
            bt = data.borrowed[idx]  # [B, N]
            ok = jnp.isfinite(bt).astype(jnp.float32)
            bt = jnp.where(ok > 0, bt, 0.0)
            qc = net.apply(params, jnp.repeat(data.token[idx][:, None], data.num_samples, axis=1), data.cand[idx])
            qc = hl.from_logits(qc) if cfg.num_atoms > 1 else qc  # [K, B, N(, P)]
            qc = qc[..., -1] if qc.ndim == 4 else qc  # full-chunk value [K, B, N]
            per_b = jnp.square(qc - jax.lax.stop_gradient(bt)[None])  # [K, B, N]
            borrow = jnp.sum(per_b * ok[None]) / (jnp.sum(ok) * qc.shape[0] + 1e-8)
            loss = loss + cfg.borrowed_weight * borrow

        return loss, {
            "loss": loss,
            "borrow_loss": borrow,
            "q_mean": q_mean,
            "target_mean": jnp.sum(y * valid) / vs,
            "valid": jnp.mean(w),
            # How often the transition lands on the goal, how often the MC floor lifts the target and
            # by how much, and whether the target left the support the histogram can represent.
            "term_frac": jnp.mean(tinfo["term_frac"]),
            "mc_floor_frac": jnp.sum((tinfo["floor_gap"] > 0) * valid) / vs,
            "mc_floor_gap": jnp.sum(tinfo["floor_gap"] * valid) / vs,
            "target_oob": jnp.sum(((y < v_lo) | (y > v_hi)) * valid) / vs,
        }

    tx = optax.adam(cfg.lr)

    def step(carry, rng):
        params, tgt_params, opt_state = carry
        k_idx, k_tgt = jax.random.split(rng)
        idx = jax.random.randint(k_idx, (cfg.batch_size,), 0, T)
        boot_params = params if cfg.bootstrap == "online" else tgt_params
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, boot_params, idx, k_tgt)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        tgt_params = optax.incremental_update(params, tgt_params, cfg.target_tau)
        return (params, tgt_params, opt_state), info

    return step, tx


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    ap.add_argument("--kind", choices=["qc", "arq"], default="arq")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--discount",
        type=float,
        default=None,
        help="Defaults to the discount the annotation's returns were accumulated with. Setting it by "
        "hand to something else makes the bootstrap and mc_return disagree about what a step costs.",
    )
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument(
        "--bootstrap",
        choices=["target", "online"],
        default="target",
        help="Which parameters score the bootstrap candidates. 'online' is the reference default "
        "(vla_aqc.py: the online critic under stop_gradient, no target network); 'target' uses the "
        "Polyak copy, which is the usual stabiliser but adds a second timescale to the fixed point.",
    )
    ap.add_argument("--num-critics", type=int, default=2)
    # --- bootstrap aggregation ---------------------------------------------------------------
    # The joint arg-max runs over N candidates x P prefixes, an order of magnitude more items than
    # ordinary actor-critic maxes over, so the upward bias of max-over-noisy-estimates is amplified
    # here by design. These knobs trade that bias against how sharply the target tracks the best.
    ap.add_argument("--ens-agg", choices=["min", "lcb"], default="min", help="Across ensemble members.")
    ap.add_argument("--lcb-beta", type=float, default=1.0, help="ens-agg=lcb: mean - beta*std.")
    ap.add_argument("--v-agg", choices=["max", "topm", "soft"], default="max", help="Across candidates/prefixes.")
    ap.add_argument(
        "--bootstrap-candidates",
        type=int,
        default=0,
        help="Bootstrap off this many of the N stored candidates, resampled every step (0 = all N). "
        "Scoring the candidates is the bulk of an update's cost, and a narrower arg-max is also "
        "less biased upward, so this trades speed and bias against how sharply the target tracks.",
    )
    ap.add_argument("--top-m", type=int, default=3, help="v-agg=topm: average the m best.")
    ap.add_argument("--soft-tau", type=float, default=0.1, help="v-agg=soft: softmax temperature.")
    ap.add_argument(
        "--target-noise",
        type=float,
        default=0.0,
        help="Target policy smoothing: perturb the bootstrap candidates by this many action-std "
        "(temporally coherent offset+drift, so the chunk stays plausible). 0 = off.",
    )
    ap.add_argument("--target-noise-clip", type=float, default=2.0)
    ap.add_argument(
        "--v-clip",
        default="auto",
        help="Ceiling on the bootstrap target and the backed-up value. 'auto' = the largest return "
        "present in the data (for a terminal success window that is also the largest obtainable); "
        "'off' disables it; a number sets it explicitly. This is a correctness constraint, not a "
        "tuning knob: without it the max-over-candidates fixed point sits far above any achievable "
        "return (measured: 26.5 against a ceiling of 15.7).",
    )
    ap.add_argument(
        "--terminal-uses-mc",
        action="store_true",
        help="At a transition that lands on a terminal, take the value from mc_return instead of "
        "dropping the bootstrap. The return from a terminal is known exactly, so this is the "
        "reference's choice (vla_aqc.py, terminal_uses_mc).",
    )
    ap.add_argument(
        "--borrowed-weight",
        type=float,
        default=0.0,
        help="Weight of a per-candidate regression toward the borrowed return (nearest executed "
        "chunk's mc_return, precomputed by compute_borrowed_returns.py). This is the only term that "
        "gives the candidates a target of their own; without it the critic learns V(s) and cannot "
        "rank actions. Needs borrowed_return.dat in the data dir. 0 = off.",
    )
    ap.add_argument(
        "--mc-lower-bound",
        action="store_true",
        help="Floor the target at the return the behaviour policy actually collected from this "
        "state. Sound by definition (that return is achievable), and free.",
    )
    ap.add_argument("--num-atoms", type=int, default=1, help="1 = scalar Q; >1 = HL-Gauss distributional")
    # The histogram's support has to be the value support, or the targets sit on its edge atoms and
    # the critic cannot represent them. Both default to whatever the re-labelling recorded.
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--v-max", type=float, default=None)
    ap.add_argument("--macro-group-size", type=int, default=2, help="ARQ: steps per macro token")
    # Capacity. Defaults follow the reference ACSAC critic (3 layers, 8 heads x 48 = 384 wide) and a
    # standard 3x512 MLP for QC. The input is already a learned 2048-d token, so the critic only has
    # to fit a fairly simple function on top of it — but sweep these rather than trust the default.
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512], help="QC: MLP widths")
    ap.add_argument("--num-layers", type=int, default=3, help="ARQ: transformer layers")
    ap.add_argument("--num-heads", type=int, default=8, help="ARQ: attention heads")
    ap.add_argument("--head-dim", type=int, default=48, help="ARQ: per-head dim (n_embd = heads*head_dim)")
    ap.add_argument("--mlp-dim", type=int, default=1024, help="ARQ: transformer MLP width")
    ap.add_argument("--steps-per-dispatch", type=int, default=100, help="update steps fused into one scan")
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also write params_<step>.msgpack every this many steps, so rollout success can later be "
        "measured against training progress rather than only at the end. 0 = final only.",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Compute the within-state diagnostics in-process every this many steps and log them (needs "
        "--wandb-project to be visible as a curve). Cheap: the candidates are already on the GPU.",
    )
    ap.add_argument(
        "--rollout-every",
        type=int,
        default=0,
        help="Run critic-guided rollouts in the RoboCasa sim every this many steps, exactly as VLA "
        "training evaluates the probe, and log the success rates. This is the actual objective; the "
        "offline diagnostics are only its proxy. 0 = off. Needs --vla-checkpoint.",
    )
    ap.add_argument("--vla-checkpoint", type=pathlib.Path, default=None, help="VLA checkpoint the tokens came from.")
    ap.add_argument("--vla-config", default=None, help="Registered config for the VLA (defaults to the annotation's).")
    ap.add_argument("--rollout-trials", type=int, default=20)
    ap.add_argument("--task", default="PrepareCoffee")
    ap.add_argument("--wandb-project", default=None, help="Log to this wandb project. Omit to log only to stdout.")
    ap.add_argument("--wandb-entity", default="RSS-PFT_RLLAB")
    ap.add_argument("--wandb-group", default=None, help="Bucket related critic runs (e.g. an ablation).")
    ap.add_argument("--wandb-name", default=None, help="Run name; defaults to the output dir's name.")
    cfg = ap.parse_args()

    cfg.v_clip = 0.0 if str(cfg.v_clip).lower() in ("off", "none", "0") else cfg.v_clip
    # The reward scheme decides the discount and the value support together (sparse terminal 0/1 with
    # gamma 0.99 lands in [0, 1]; the reference living-cost scheme with gamma 0.9999 lands in [-1, 0]),
    # and relabel_reward.py records both. Taking them from the data rather than from flags is what
    # keeps a re-labelled dataset from being trained against the previous scheme's constants.
    _meta = json.loads((cfg.data / "meta.json").read_text())
    meta_support = _meta.get("value_support", [0.0, 1.0])
    if cfg.discount is None:
        cfg.discount = _meta.get("discount", 0.99)
    if cfg.v_min is None:
        cfg.v_min = meta_support[0]
    if cfg.v_max is None:
        cfg.v_max = meta_support[1]
    logger.info(f"reward scheme {_meta.get('reward_scheme', 'raw')!r}: discount {cfg.discount}, support {meta_support}")
    data = load_data(cfg.data, max_frames=cfg.max_frames)
    if data.horizon % cfg.macro_group_size:
        raise ValueError(f"macro_group_size {cfg.macro_group_size} must divide horizon {data.horizon}")

    net = _critic.Ensemble(
        make_critic=lambda: _critic.make_critic(
            cfg.kind,
            action_dim=data.action_dim,
            horizon=data.horizon,
            num_atoms=cfg.num_atoms,
            **(
                {
                    "macro_group_size": cfg.macro_group_size,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "head_dim": cfg.head_dim,
                    "mlp_dim": cfg.mlp_dim,
                }
                if cfg.kind == "arq"
                else {"hidden_dims": tuple(cfg.hidden_dims)}
            ),
        ),
        num_critics=cfg.num_critics,
    )
    hl = _critic.HLGauss(v_min=cfg.v_min, v_max=cfg.v_max, num_atoms=max(cfg.num_atoms, 2))

    run = None
    if cfg.wandb_project:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            name=cfg.wandb_name or (cfg.out.name if cfg.out else f"critic_{cfg.kind}"),
            config=vars(cfg)
            | {"data": str(cfg.data), "reward_scheme": _meta.get("reward_scheme"), "frames": data.token.shape[0]},
        )

    rng = jax.random.key(cfg.seed)
    params = net.init(rng, data.token[:1], data.chunk[:1])
    n_param = sum(x.size for x in jax.tree.leaves(params))
    logger.info(f"{cfg.kind.upper()} critic: {n_param / 1e6:.2f}M params, {cfg.num_critics} ensemble members")

    # Noise scale from the data itself: one std per action dim, so the smoothing means the same
    # thing regardless of task or units, and no magic constant has to be guessed.
    act_scale = jnp.std(data.cand.reshape(-1, data.cand.shape[-1]), axis=0)[None, None, None, None, :]
    step_fn, tx = make_update(data, cfg, net, hl, act_scale, meta_support)
    opt_state = tx.init(params)
    tgt_params = params

    @jax.jit
    def run_chunk(carry, rng):
        return jax.lax.scan(step_fn, carry, jax.random.split(rng, cfg.steps_per_dispatch))

    # A cheap within-state readout computed from resident data, so it can run every --eval-every
    # steps and trace whether the ranking signal ever appears rather than only measuring it at the end.
    # These are the make-or-break metrics; the full eval_rlt_critic.py adds the slower ones after.
    diag_rng = np.random.default_rng(cfg.seed)
    diag_idx = jnp.asarray(
        np.sort(diag_rng.choice(data.token.shape[0], size=min(2048, data.token.shape[0]), replace=False))
    )

    @jax.jit
    def _diag(p):
        z = data.token[diag_idx]  # [S, D]
        qc = net.apply(p, jnp.repeat(z[:, None], data.num_samples, axis=1), data.cand[diag_idx])
        qc = hl.from_logits(qc) if cfg.num_atoms > 1 else qc
        qc = jnp.min(qc, axis=0)  # ensemble -> [S, N(, P)]
        qc = qc[..., -1] if qc.ndim == 3 else qc  # full-chunk value -> [S, N]
        qd = net.apply(p, z[:, None], data.chunk[diag_idx][:, None])
        qd = hl.from_logits(qd) if cfg.num_atoms > 1 else qd
        qd = (qd[..., -1] if qd.ndim == 4 else qd)[:, 0, 0] if qd.ndim >= 3 else qd[:, 0]
        within = jnp.mean(jnp.std(qc, axis=1))
        between = jnp.std(jnp.mean(qc, axis=1))
        return {
            "diag_step/within_state_std": within,
            "diag_step/between_state_std": between,
            "diag_step/action_sensitivity": jnp.mean(jnp.var(qc, axis=1)) / (between**2 + 1e-12),
            "diag_step/within_state_range": jnp.mean(jnp.max(qc, 1) - jnp.min(qc, 1)),
            "diag_step/range_over_std": jnp.mean(jnp.max(qc, 1) - jnp.min(qc, 1)) / (within + 1e-9),
            "diag_step/ranking_demo_vs_cand": jnp.mean(qd[:, None] > qc),
            "diag_step/q_demo_mean": jnp.mean(qd),
        }

    # In-process critic-guided rollout, built once so the 3B VLA stays resident and only the critic
    # params change between evaluations - the same pattern VLA training uses for its probe eval.
    vla = env = rollout_seed = None
    if cfg.rollout_every:
        if cfg.vla_checkpoint is None:
            raise ValueError("--rollout-every needs --vla-checkpoint (the VLA the tokens were annotated from)")
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples/robocasa"))
        import eval_critic as _ec
        import rollout as _ro

        vla_config = cfg.vla_config or _meta.get("config")
        vla = _ec.VLA(vla_config, cfg.vla_checkpoint, num_samples=data.num_samples, flow_steps=10, seed=cfg.seed)
        rollout_seed = cfg.seed
        env = _ro.make_env(cfg.task, camera_size=256, seed=rollout_seed)
        logger.info(f"in-process rollout ready: VLA {vla_config} @ {cfg.vla_checkpoint}, {cfg.rollout_trials} trials")

    def _rollout(p_live, step):
        """Critic-guided and plain-VLA success on the same scenes, using the live critic params."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples/robocasa"))
        import eval_critic as _ec
        import rollout as _ro

        macro = cfg.macro_group_size if cfg.kind == "arq" else data.horizon

        def score(obs, actions):  # live critic, current params
            out = net.apply(p_live, obs, actions)
            return hl.from_logits(out) if cfg.num_atoms > 1 else out

        score = jax.jit(score)
        rows = {}
        for mode in ("critic", "vla"):
            pol = _ec.make_policy_fn(vla, score, macro, mode=mode)
            r = _ro.run_trials(
                env, pol, task=cfg.task, num_trials=cfg.rollout_trials, seed=rollout_seed, replan_steps=vla.H
            )
            rows[mode] = r["success_rate"]
        logger.info(f"  [rollout @ {step}] critic {rows['critic']:.0%}  vla {rows['vla']:.0%}")
        return {f"rollout/{m}_success": v for m, v in rows.items()}

    carry = (params, tgt_params, opt_state)
    t0 = time.perf_counter()
    for s in range(0, cfg.steps, cfg.steps_per_dispatch):
        carry, infos = run_chunk(carry, jax.random.fold_in(rng, s))
        if s % cfg.log_interval == 0:
            info = jax.tree.map(lambda x: float(jnp.mean(x)), infos)
            step = s + cfg.steps_per_dispatch
            rate = step / max(time.perf_counter() - t0, 1e-6)
            logger.info(
                f"step {step}/{cfg.steps}  {rate:.0f} it/s  " + "  ".join(f"{k}={v:.4f}" for k, v in info.items())
            )
            if run is not None:
                run.log({f"train/{k}": v for k, v in info.items()} | {"it_per_s": rate}, step=step)

        step = s + cfg.steps_per_dispatch
        if cfg.eval_every and step % cfg.eval_every == 0:
            d = {k: float(v) for k, v in _diag(carry[0]).items()}
            logger.info(
                f"  [diag @ {step}] range/std {d['diag_step/range_over_std']:.2f}  rank {d['diag_step/ranking_demo_vs_cand']:.3f}"
            )
            if run is not None:
                run.log(d, step=step)
        if cfg.rollout_every and step % cfg.rollout_every == 0:
            r = _rollout(carry[0], step)
            if run is not None:
                run.log(r, step=step)
        if cfg.save_every and step % cfg.save_every == 0 and step < cfg.steps:
            out0 = cfg.out or (cfg.data / f"critic_{cfg.kind}")
            out0.mkdir(parents=True, exist_ok=True)
            import flax.serialization as _fser

            (out0 / f"params_{step}.msgpack").write_bytes(_fser.to_bytes(carry[0]))
            logger.info(f"  saved checkpoint params_{step}.msgpack")

    out = cfg.out or (cfg.data / f"critic_{cfg.kind}")
    out.mkdir(parents=True, exist_ok=True)
    import flax.serialization as fser

    (out / "params.msgpack").write_bytes(fser.to_bytes(carry[0]))
    cfg_out = vars(cfg) | {"data": str(cfg.data), "out": str(out)}
    if run is not None:
        cfg_out["wandb_id"] = run.id
    (out / "config.json").write_text(json.dumps(cfg_out, indent=2))
    logger.info(f"saved to {out} ({(time.perf_counter() - t0) / 60:.1f} min)")
    if run is not None:
        # eval_rlt_critic.py usually runs right after and writes diag.json next to the params; if it
        # is already there (e.g. on resume) surface it, otherwise the eval step logs it itself.
        diag = out / "diag.json"
        if diag.exists():
            run.summary.update(
                {f"diag/{k}": v for k, v in json.loads(diag.read_text()).items() if isinstance(v, int | float)}
            )
        run.finish()


if __name__ == "__main__":
    main()
