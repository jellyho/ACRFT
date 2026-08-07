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
import os as _os
import pathlib
import socket as _socket
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.rlt_critic import annotation as _annot
from openpi.rlt_critic import critic as _critic

logger = logging.getLogger(__name__)


def _where_it_ran() -> dict:
    """Host and device, recorded alongside what was run.

    Throughput and failures cluster by node on this cluster (one machine hands out GPUs jax cannot
    see), and without this the only way to tie a slow or dead run to its host is to go back to the
    slurm log. Written to BOTH the wandb config and config.json, so the correlation survives with or
    without wandb. Best-effort throughout: this must never be what fails a run.
    """
    try:
        d = jax.devices()[0]
        gpu = getattr(d, "device_kind", None) or str(d)
    except Exception:
        gpu = "unknown"
    return {
        "slurm_node": _os.environ.get("SLURMD_NODENAME") or _socket.gethostname(),
        "slurm_job": _os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": _os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": gpu,
    }


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


def load_data(path: pathlib.Path, *, max_frames: int = 0, use_proprio: bool = True) -> Data:
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

    obs = rd("rl_token", (full, D))
    if use_proprio:
        # The `noprop` bottleneck excludes proprio from the token on purpose, so the critic is meant
        # to be handed it separately (README: "critic must supply proprio"). extract_proprio.py joins
        # it back from the source dataset. z-scored per dim because the raw state mixes metres,
        # quaternions and gripper qpos, and the critic's first op is a LayerNorm over the whole
        # observation - unnormalised, 16 raw dims sitting beside 2048 token dims would be flattened by
        # the token's statistics. Constant dims (std 0) are passed through as zeros rather than NaN.
        pd_ = meta.get("proprio_dim")
        if not pd_ or not (path / "proprio.dat").exists():
            raise ValueError(f"no proprio.dat in {path}. Run slurm/extract_proprio.py --data {path}")
        pro = rd("proprio", (full, pd_), np.float32)
        mu, sd = pro.mean(0), pro.std(0)
        pro = np.where(sd > 1e-6, (pro - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0).astype(np.float32)
        obs = np.concatenate([obs, pro], axis=1)
        D = D + pd_
        logger.info(f"proprio: +{pd_} dims (z-scored, {int((sd <= 1e-6).sum())} constant dims zeroed) -> obs {D}")

    d = Data(
        token=jnp.asarray(obs),
        chunk=jnp.asarray(rd("action_chunk", (full, H, A))),
        cand=jnp.asarray(rd("base_action", (full, N, H, A))),
        reward=jnp.asarray(rd("reward", (full,))),
        episode=jnp.asarray(rd("episode_index", (full,), np.int32)),
        mc_return=jnp.asarray(rd("mc_return", (full,), np.float32)),
        **_terminals(rd("done", (full,), np.int8), rd("episode_index", (full,), np.int32)),
        horizon=H,
        action_dim=A,
        num_samples=N,
    )
    gb = sum(x.size * x.itemsize for x in (d.token, d.chunk, d.cand)) / 1e9
    logger.info(f"loaded {T} frames onto {jax.devices()[0].platform} ({gb:.2f} GB): token {D}, chunk {H}x{A}, N={N}")
    return d


def make_update(data: Data, cfg, net, hl, act_scale, meta_support=(0.0, 1.0), v_net=None):
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

    def targets(idx, tgt_params, rng, v_params=None):
        """Per-prefix (cum_reward, next_value, valid) for the sampled transitions."""
        ep = data.episode[idx]  # [B]
        # Rewards over the chunk, zeroed once the episode ends.
        off = jnp.arange(H)[None, :]
        r_idx = jnp.clip(idx[:, None] + off, 0, T - 1)
        same = data.episode[r_idx] == ep[:, None]
        rew = data.reward[r_idx] * same  # [B, H]
        cum_all = jnp.cumsum(rew * disc[None, :], axis=-1)  # [B, H]
        cum = cum_all[:, prefixes - 1]  # [B, P]

        # Where each prefix lands, and whether the episode is still running when it gets there.
        # `ended` = a terminal sits in [t, t+h-1]: the episode finished partway through the
        # commitment, so there is no successor and `cum` already holds the whole return (it sums
        # i < h and the terminal reward sits at i = d < h, giving exactly gamma^d).
        #
        # The window stops at h-1, not h. Including h would make a prefix that lands exactly ON the
        # goal look like it ran past it - and every transition that reaches the goal is of that kind,
        # so the success reward would enter no target and V=0 becomes the fixed point (measured:
        # corr(Q, mc_return) = -0.75).
        land = idx[:, None] + prefixes[None, :]  # [B, P]
        nxt = jnp.clip(land, 0, T - 1)
        ended = data.done_cum[jnp.clip(land - 1, 0, T - 1)] > jnp.where(idx > 0, data.done_cum[idx - 1], 0)[:, None]
        valid = (data.alive[idx][:, None] > 0) & (ended | (land < T))

        if cfg.objective == "iql":
            # No candidate array, no arg-max: the successor value is read straight off V. `ended`
            # still masks it, and a terminal successor still uses its own reward - those are facts
            # about the MDP, not about how V(s') is estimated.
            v_next = v_net.apply(v_params, data.token[nxt])  # [B, P]
            v_next = jnp.clip(v_next, v_lo, v_hi) if v_hi > v_lo else v_next
            v_next = jnp.where(data.done[nxt] > 0, data.reward[nxt], v_next)
            gam = cfg.discount ** prefixes.astype(jnp.float32)
            y = cum + gam[None, :] * ~ended * v_next
            floor_gap = jnp.maximum(data.mc_return[idx][:, None] - y, 0.0)
            if cfg.mc_lower_bound:
                y = jnp.maximum(y, data.mc_return[idx][:, None])
            if v_hi > v_lo:
                y = jnp.clip(y, v_lo, v_hi)
            return y, valid, {"floor_gap": floor_gap}

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
        # V is known in closed form at a terminal state: nothing follows it, so its value is its own
        # reward. Where that is true, use it instead of the network's guess. This is the same
        # V(s_{t+h}) slot as every other transition, not a separate branch.
        #
        # A pure TD target: reward plus the bootstrap, with mc_return nowhere in it. The precomputed
        # return is an auxiliary array for diagnostics and for the optional floor below - it has no
        # business standing in for a value the bootstrap states exactly.
        v_next = jnp.where(data.done[nxt] > 0, data.reward[nxt], v_next)
        y = cum + gam[None, :] * ~ended * v_next
        floor_gap = jnp.maximum(data.mc_return[idx][:, None] - y, 0.0)
        if cfg.mc_lower_bound:
            # The behaviour policy demonstrably obtained mc_return from this state, so the optimal
            # value cannot be below it. Unlike the ceiling this rarely binds once the critic is
            # inflated; it matters early, and it stops a pessimistic aggregation from settling below
            # what the data proves is achievable.
            y = jnp.maximum(y, data.mc_return[idx][:, None])
        if v_hi > v_lo:
            y = jnp.clip(y, v_lo, v_hi)
        return y, valid, {"floor_gap": floor_gap}

    def loss_fn(params, tgt_params, idx, rng, v_params=None):
        # `tgt_params` is whatever --bootstrap selected; stop_gradient makes the online choice safe.
        y, valid, tinfo = targets(idx, jax.lax.stop_gradient(tgt_params), rng, jax.lax.stop_gradient(v_params))
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

        extra = {}
        if cfg.objective == "iql":
            # Expectile regression pulls V toward an upper quantile of Q over the actions the data
            # contains. tau=0.5 is least squares and gives the MEAN; tau>0.5 penalises under-shooting
            # more than over-shooting, so V rises toward max_a Q without ever proposing an action.
            # Q_target is detached: this term trains V only, and the target above detaches V, so the
            # two halves cannot chase each other through the shared optimiser.
            qd = net.apply(jax.lax.stop_gradient(tgt_params), data.token[idx], data.chunk[idx])
            qd = hl.from_logits(qd) if cfg.num_atoms > 1 else qd  # [K, B(, P)]
            qd = jnp.min(qd, 0)  # ensemble min, matching how the deployed value is read
            qd = qd[:, None] if cfg.kind == "qc" else qd  # [B, P]
            v = v_net.apply(v_params, data.token[idx])[:, None]  # [B, 1] -> broadcast over prefixes
            u = jax.lax.stop_gradient(qd) - v
            wexp = jnp.abs(cfg.expectile - (u < 0).astype(jnp.float32))
            v_loss = jnp.sum(wexp * jnp.square(u) * valid) / jnp.maximum(jnp.sum(valid), 1.0)
            loss = loss + v_loss
            extra = {
                "v_loss": v_loss,
                "v_mean": jnp.sum(v * valid) / jnp.maximum(jnp.sum(valid), 1.0),
                # Positive means V sits BELOW the Q of the demonstrated action, i.e. the expectile is
                # not reaching the actions the data actually took.
                "v_minus_q": jnp.sum((v - qd) * valid) / jnp.maximum(jnp.sum(valid), 1.0),
            }

        if cfg.cql_alpha > 0:
            # Push down candidate-chunk Q relative to the demonstrated chunk, clamped at mc_return
            # (Cal-QL): the ONLY term in this file that gives Q an action-contrast on success-only
            # data. Candidates come from the stored per-frame policy samples.
            n = min(cfg.cql_candidates, data.cand.shape[1])
            k_cql = jax.random.fold_in(rng, 13)
            sel = jnp.argsort(jax.random.uniform(k_cql, (idx.shape[0], data.cand.shape[1])), axis=-1)[:, :n]
            a_c = jnp.take_along_axis(data.cand[idx], sel[:, :, None, None], axis=1)  # [B, n, H, A]
            z_c = jnp.repeat(data.token[idx][:, None], n, axis=1)  # [B, n, D]
            q_c = net.apply(params, z_c, a_c)  # [K, B, n(, P)(, atoms)]
            q_c = hl.from_logits(q_c) if cfg.num_atoms > 1 else q_c
            if cfg.kind == "qc":
                q_c = q_c[..., None]  # [K, B, n, 1]
            mc = data.mc_return[idx][None, :, None, None]
            q_c = jnp.maximum(q_c, mc)  # calibration clamp
            lse = jax.nn.logsumexp(q_c, axis=2) - jnp.log(n)  # [K, B, P]
            q_data = hl.from_logits(pred) if cfg.num_atoms > 1 else pred  # [K, B, P]
            cql_gap = (lse - q_data) * w
            cql_loss = jnp.sum(cql_gap) / (jnp.sum(w) * pred.shape[0] + 1e-8)
            loss = loss + cfg.cql_alpha * cql_loss
            extra = extra | {"cql_gap": jnp.sum(cql_gap) / (jnp.sum(w) * pred.shape[0] + 1e-8)}

        return loss, extra | {
            "loss": loss,
            "q_mean": q_mean,
            "target_mean": jnp.sum(y * valid) / vs,
            # What fraction of the batch is a usable transition at all. Like the terminal counts it
            # replaced, this is a property of the DATA and does not move during training - it is a
            # sanity check that reads wrong-if-it-changes, not a curve worth watching.
            "valid": jnp.mean(w),
            # How often the MC floor lifts the target and by how much, and whether the target left
            # the support the histogram can represent.
            "mc_floor_frac": jnp.sum((tinfo["floor_gap"] > 0) * valid) / vs,
            "mc_floor_gap": jnp.sum(tinfo["floor_gap"] * valid) / vs,
            "target_oob": jnp.sum(((y < v_lo) | (y > v_hi)) * valid) / vs,
        }

    tx = optax.adam(cfg.lr)
    tx_v = optax.adam(cfg.lr)

    def step(carry, rng):
        # v_params rides along even under --objective td, where it is an empty pytree and every
        # operation on it is a no-op. One carry shape means one jit signature for both objectives.
        params, tgt_params, opt_state, v_params, v_opt_state = carry
        k_idx, k_tgt = jax.random.split(rng)
        idx = jax.random.randint(k_idx, (cfg.batch_size,), 0, T)
        boot_params = params if cfg.bootstrap == "online" else tgt_params
        (_, info), (grads, v_grads) = jax.value_and_grad(loss_fn, argnums=(0, 4), has_aux=True)(
            params, boot_params, idx, k_tgt, v_params
        )
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        v_updates, v_opt_state = tx_v.update(v_grads, v_opt_state)
        v_params = optax.apply_updates(v_params, v_updates)
        tgt_params = optax.incremental_update(params, tgt_params, cfg.target_tau)
        return (params, tgt_params, opt_state, v_params, v_opt_state), info

    return step, tx, tx_v


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
        help="Defaults to the discount the annotation's returns were accumulated with. Setting it to "
        "anything else builds (or reuses) the matching annotation automatically - mc_return is "
        "re-accumulated at the new gamma, so the bootstrap and the return floor never disagree.",
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
    ap.add_argument(
        "--objective",
        choices=["td", "iql"],
        default="td",
        help="'td' bootstraps V(s') by maxing Q over the stored candidate chunks. 'iql' learns a "
        "separate state value V(z) by expectile regression on the DEMONSTRATED action only and "
        "bootstraps that instead - no candidate array, no arg-max, and therefore none of the upward "
        "bias of maxing over N*P noisy estimates (measured on the td runs: V over-estimated by "
        "+0.025 far from the goal, ~0 near it, which is what tilts the per-prefix targets).",
    )
    ap.add_argument(
        "--cql-alpha",
        type=float,
        default=0.0,
        help="CQL push-down: penalize logsumexp of Q over stored candidate chunks minus Q of the "
        "demonstrated chunk. THE action-contrast term: success-only data has no bad outcomes, so "
        "without it Q degenerates to a state-value (measured: candidate spread ~0.001, best-of-N "
        "a no-op) and the argmax exploits extrapolation noise (V-GPS Tab.10: IQL ranking dies at "
        "N>10, Cal-QL survives to 50). Calibrated: the push-down is clamped at the frame's "
        "mc_return so it never drives Q below what the data proved achievable (Cal-QL).",
    )
    ap.add_argument("--cql-candidates", type=int, default=4,
                    help="candidates per step in the CQL term (cost: extra forward x this).")
    ap.add_argument("--layernorm-v", action="store_true",
                    help="LayerNorm inside the V-net hidden layers (RLPD stabilizer).")
    ap.add_argument(
        "--expectile",
        type=float,
        default=0.7,
        help="tau for IQL's expectile regression. 0.5 is plain least squares (V -> mean Q); higher "
        "weights over-shoots more and approaches max_a Q. Ignored unless --objective iql.",
    )
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
        "--mc-lower-bound",
        action="store_true",
        help="Floor the target at the return the behaviour policy actually collected from this "
        "state. Sound by definition (that return is achievable), and free.",
    )
    ap.add_argument(
        "--proprio-mode",
        choices=["concat", "token"],
        default="concat",
        help="How proprioception reaches the network. 'concat' appends it to the RL token, where 16 "
        "dims sit among 2048 and are then LayerNorm'd against the token's statistics. 'token' gives "
        "it its own projection - a separate sequence position for ARQ, a widened embedding for QC - "
        "so it is not diluted.",
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

    # Not a flag. The `noprop` bottleneck leaves proprioception out of the RL token by design, so a
    # critic without it is asked to judge an action chunk without knowing where the arm is - which is
    # not an ablation anyone wants to run, it is a broken configuration. Still recorded in
    # config.json, because that is what tells every reader (eval, rollout, the probes) whether an
    # observation is token-only or token+proprio, and runs from before this was settled say False.
    cfg.use_proprio = True
    cfg.v_clip = 0.0 if str(cfg.v_clip).lower() in ("off", "none", "0") else cfg.v_clip
    # The reward scheme decides the discount and the value support together (sparse terminal 0/1 with
    # gamma 0.99 lands in [0, 1]; the reference living-cost scheme with gamma 0.9999 lands in [-1, 0]),
    # and relabel_reward.py records both. Taking them from the data rather than from flags is what
    # keeps a re-labelled dataset from being trained against the previous scheme's constants.
    # A discount the annotation was not accumulated at selects a DIFFERENT dataset, and this makes
    # it appear rather than requiring anyone to have prepared it: mc_return is re-derived from the
    # stored per-frame reward at the new gamma, the multi-GB arrays are shared by hardlink, and the
    # result is published by a single atomic rename so concurrent array tasks are safe. Without this
    # the flag trained the bootstrap at one gamma against returns from another - silently wrong
    # everywhere, and actively harmful under --mc-lower-bound, where mc_return is not a bystander
    # but the floor the target is clamped to.
    cfg.data = _annot.ensure_discount(cfg.data, cfg.discount)
    _meta = json.loads((cfg.data / "meta.json").read_text())
    meta_support = _meta.get("value_support", [0.0, 1.0])
    if cfg.discount is None:
        cfg.discount = _meta.get("discount", 0.99)
    if cfg.v_min is None:
        cfg.v_min = meta_support[0]
    if cfg.v_max is None:
        cfg.v_max = meta_support[1]
    logger.info(f"reward scheme {_meta.get('reward_scheme', 'raw')!r}: discount {cfg.discount}, support {meta_support}")
    data = load_data(cfg.data, max_frames=cfg.max_frames, use_proprio=cfg.use_proprio)
    if data.horizon % cfg.macro_group_size:
        raise ValueError(f"macro_group_size {cfg.macro_group_size} must divide horizon {data.horizon}")

    # Only the 'token' mode tells the module about proprio; 'concat' leaves it as plain input dims.
    _pro_dim = _meta.get("proprio_dim", 0) if (cfg.use_proprio and cfg.proprio_mode == "token") else 0
    net = _critic.Ensemble(
        make_critic=lambda: _critic.make_critic(
            cfg.kind,
            action_dim=data.action_dim,
            horizon=data.horizon,
            num_atoms=cfg.num_atoms,
            **({"proprio_dim": _pro_dim} if _pro_dim else {}),
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
            | {"data": str(cfg.data), "reward_scheme": _meta.get("reward_scheme"), "frames": data.token.shape[0]}
            # Where it ran, recorded alongside what it ran. Throughput and failures cluster by node on
            # this cluster (one machine hands out GPUs jax cannot see), and without this the only way
            # to correlate a slow or dead run with its host is to go back to the slurm log.
            | _where_it_ran(),
        )

    rng = jax.random.key(cfg.seed)
    params = net.init(rng, data.token[:1], data.chunk[:1])
    n_param = sum(x.size for x in jax.tree.leaves(params))
    logger.info(f"{cfg.kind.upper()} critic: {n_param / 1e6:.2f}M params, {cfg.num_critics} ensemble members")

    # Noise scale from the data itself: one std per action dim, so the smoothing means the same
    # thing regardless of task or units, and no magic constant has to be guessed.
    act_scale = jnp.std(data.cand.reshape(-1, data.cand.shape[-1]), axis=0)[None, None, None, None, :]
    # An empty pytree under --objective td: the V network is built only when something reads it, but
    # it still occupies its slot in the carry so both objectives compile to the same scan signature.
    v_net = _critic.ValueNet(hidden_dims=tuple(cfg.hidden_dims), use_ln=cfg.layernorm_v) if cfg.objective == "iql" else None
    v_params = v_net.init(jax.random.fold_in(rng, 1), data.token[:1]) if v_net is not None else {}
    if v_net is not None:
        logger.info(
            f"IQL: V head {sum(x.size for x in jax.tree.leaves(v_params)) / 1e6:.2f}M params, "
            f"expectile tau={cfg.expectile} (no candidate array, no arg-max in the target)"
        )
    step_fn, tx, tx_v = make_update(data, cfg, net, hl, act_scale, meta_support, v_net=v_net)
    opt_state = tx.init(params)
    v_opt_state = tx_v.init(v_params)
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
        qa = jnp.min(qc, axis=0)  # ensemble -> [S, N(, P)]
        qc = qa[..., -1] if qa.ndim == 3 else qa  # full-chunk value -> [S, N]
        qd = net.apply(p, z[:, None], data.chunk[diag_idx][:, None])
        qd = hl.from_logits(qd) if cfg.num_atoms > 1 else qd
        # Same reduction order as qc above: ensemble first, then the full-chunk prefix, then drop the
        # singleton candidate axis. Reducing in any other order collapses the STATE axis instead of
        # the ensemble axis and leaves qd with length K, which only shows up as a broadcast error
        # against qc once --eval-every actually fires.
        qd = jnp.min(qd, axis=0)  # ensemble -> [S, 1(, P)]
        qd = qd[..., -1] if qd.ndim == 3 else qd  # full-chunk value -> [S, 1]
        qd = qd[:, 0]  # [S]
        within = jnp.mean(jnp.std(qc, axis=1))
        between = jnp.std(jnp.mean(qc, axis=1))
        extra = {}
        if qa.ndim == 3:
            # The quantity deployment actually acts on: eval_critic.make_policy_fn takes a JOINT
            # arg-max over (candidate, prefix) and executes (p+1)*macro_group_size steps. Tracking it
            # during training shows directly whether the critic is collapsing onto short commitments,
            # rather than inferring it from rollout video afterwards.
            flat = qa.reshape(qa.shape[0], -1)
            p_idx = jnp.argmax(flat, axis=-1) % qa.shape[-1]
            extra = {
                "diag_step/avg_chosen_horizon": jnp.mean((p_idx + 1) * cfg.macro_group_size),
                "diag_step/frac_shortest_prefix": jnp.mean(p_idx == 0),
                "diag_step/frac_longest_prefix": jnp.mean(p_idx == qa.shape[-1] - 1),
            }
        return extra | {
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

    carry = (params, tgt_params, opt_state, v_params, v_opt_state)
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
                f"  [diag @ {step}] range/std {d['diag_step/range_over_std']:.2f}  "
                f"rank {d['diag_step/ranking_demo_vs_cand']:.3f}"
                + (
                    f"  horizon {d['diag_step/avg_chosen_horizon']:.2f}/{data.horizon}"
                    f" (shortest {d['diag_step/frac_shortest_prefix']:.0%})"
                    if "diag_step/avg_chosen_horizon" in d
                    else ""
                )
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
            # V goes in its own file so params_*.msgpack keeps the exact shape every existing reader
            # (load_trained, eval_rlt_critic, eval_critic) already expects.
            if cfg.objective == "iql":
                (out0 / f"vparams_{step}.msgpack").write_bytes(_fser.to_bytes(carry[3]))
            # Write the architecture alongside the FIRST intermediate checkpoint, not only at the
            # end. Everything that loads a checkpoint (load_trained, eval_rlt_critic) reads
            # config.json to rebuild the network, so without this an intermediate checkpoint is
            # unloadable until the run finishes - which defeats the point of saving it, and blocks
            # evaluating it concurrently on another GPU.
            _cfgf = out0 / "config.json"
            if not _cfgf.exists():
                _c = vars(cfg) | {"data": str(cfg.data), "out": str(out0)}
                if run is not None:
                    _c["wandb_id"] = run.id
                _cfgf.write_text(json.dumps(_c, indent=2, default=str))
            logger.info(f"  saved checkpoint params_{step}.msgpack")

    out = cfg.out or (cfg.data / f"critic_{cfg.kind}")
    out.mkdir(parents=True, exist_ok=True)
    import flax.serialization as fser

    (out / "params.msgpack").write_bytes(fser.to_bytes(carry[0]))
    if cfg.objective == "iql":
        (out / "vparams.msgpack").write_bytes(fser.to_bytes(carry[3]))
    cfg_out = vars(cfg) | {"data": str(cfg.data), "out": str(out)} | _where_it_ran()
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
