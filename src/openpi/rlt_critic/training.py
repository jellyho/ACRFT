"""The critic update (targets, losses, optimiser step) and the cheap resident diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.rlt_critic.data import Data


def make_update(data: Data, cfg, net, hl, support, v_net=None):
    """One jitted critic update, written to be scanned. Returns (step, tx, tx_v)."""
    T = data.token.shape[0]
    H = data.horizon
    prefixes = jnp.arange(cfg.macro_group_size, H + 1, cfg.macro_group_size) if cfg.kind == "arq" else jnp.array([H])
    step_discount = cfg.discount ** jnp.arange(H, dtype=jnp.float32)  # gamma^i, i < H
    gamma_h = cfg.discount ** prefixes.astype(jnp.float32)  # gamma^h per prefix
    lo, hi = float(support[0]), float(support[1])
    clamp = lambda x: jnp.clip(x, lo, hi)  # noqa: E731  - the value support bounds every estimate

    def targets(idx, tgt_params, v_params):
        """Per-prefix regression target y and its validity mask, for a batch of start frames."""
        # Rewards inside the window, cut at the episode boundary.
        window = jnp.clip(idx[:, None] + jnp.arange(H)[None, :], 0, T - 1)
        in_episode = data.episode[window] == data.episode[idx][:, None]
        reward_to_h = jnp.cumsum(data.reward[window] * in_episode * step_discount, axis=-1)[:, prefixes - 1]

        # Terminal handling, one rule: `ended` = a terminal inside [t, t+h-1]. Then the episode
        # finished during the commitment, there is no successor, and reward_to_h already holds the
        # whole return (the terminal reward sits at some i < h). The window must stop at h-1: a
        # prefix landing exactly ON the goal is a real transition whose successor value is known.
        landing = idx[:, None] + prefixes[None, :]  # [B, P]
        next_idx = jnp.clip(landing, 0, T - 1)
        terminals_before = jnp.where(idx > 0, data.done_cum[idx - 1], 0)[:, None]
        ended = data.done_cum[jnp.clip(landing - 1, 0, T - 1)] > terminals_before
        valid = (data.alive[idx][:, None] > 0) & (ended | (landing < T))

        if cfg.objective == "iql":
            v_out = v_net.apply(v_params, data.token[next_idx])
            v_next = clamp(v_out[..., 0] if getattr(cfg, "aqc_baseline", False) else v_out)  # [B, P]
        else:
            # TD: V(s') = max over the stored candidates (and prefixes) of the ensemble-min Q.
            q = net.apply(
                tgt_params, jnp.repeat(data.token[next_idx][:, :, None], data.num_samples, 2), data.cand[next_idx]
            )
            q = hl.from_logits(q) if cfg.num_atoms > 1 else q  # [K, B, P, N(, P')]
            q = jnp.min(q, axis=0)
            v_next = clamp(jnp.max(q.reshape(*landing.shape, -1), axis=-1))

        # A terminal state's value is exactly its own reward - substitute the known value.
        v_next = jnp.where(data.done[next_idx] > 0, data.reward[next_idx], v_next)
        y = reward_to_h + gamma_h[None, :] * ~ended * v_next
        if cfg.mc_lower_bound:
            # The behaviour policy provably collected mc_return from this state; V* cannot be below it.
            y = jnp.maximum(y, data.mc_return[idx][:, None])
        return clamp(y), valid

    def loss_fn(params, tgt_params, idx, v_params):
        y, valid = targets(idx, jax.lax.stop_gradient(tgt_params), jax.lax.stop_gradient(v_params))
        y = jax.lax.stop_gradient(y)
        w = valid.astype(jnp.float32)[None]  # [1, B, P]
        n_valid = jnp.maximum(jnp.sum(valid), 1.0)

        pred = net.apply(params, data.token[idx], data.chunk[idx])  # [K, B(, P)(, atoms)]
        if cfg.kind == "qc":
            pred = pred[:, :, None] if cfg.num_atoms == 1 else pred[:, :, None, :]
        if cfg.dueling:
            # Q = V + (A - mean_h A): the zero-mean advantage pins the (V+c, A-c) gauge, so the
            # target's absolute level has exactly one place to live (V). Scalar ARQ only.
            pred = pred - jnp.mean(pred, axis=-1, keepdims=True)
            v_out = v_net.apply(v_params, data.token[idx])
            pred = pred + (v_out[..., 0] if getattr(cfg, "aqc_baseline", False) else v_out)[None, :, None]

        if cfg.num_atoms > 1:
            probs = hl.to_probs(jnp.clip(y, hl.v_min, hl.v_max))[None]
            per = -jnp.sum(probs * jax.nn.log_softmax(pred, axis=-1), axis=-1)
            q_mean = jnp.mean(hl.from_logits(pred))
        else:
            per = jnp.square(pred - y[None])
            q_mean = jnp.mean(pred)
        loss = jnp.sum(per * w) / (n_valid * pred.shape[0] + 1e-8)

        info = {}
        if cfg.objective == "iql":
            # Expectile regression: V chases an upper quantile of Q(z, a_demo) - the stand-in for
            # max_a Q that never proposes an action and never takes an arg-max.
            q_demo = net.apply(jax.lax.stop_gradient(tgt_params), data.token[idx], data.chunk[idx])
            q_demo = hl.from_logits(q_demo) if cfg.num_atoms > 1 else q_demo
            q_demo = jnp.min(q_demo, axis=0)  # ensemble min, as deployment reads it
            q_demo = q_demo[:, None] if cfg.kind == "qc" else q_demo  # [B, P]
            v_all = v_net.apply(v_params, data.token[idx])
            v = (v_all[..., 0] if getattr(cfg, "aqc_baseline", False) else v_all)[:, None]
            if cfg.dueling:
                q_demo = q_demo - jnp.mean(q_demo, axis=-1, keepdims=True) + jax.lax.stop_gradient(v)
            u = jax.lax.stop_gradient(q_demo) - v
            weight = jnp.abs(cfg.expectile - (u < 0).astype(jnp.float32))
            v_loss = jnp.sum(weight * jnp.square(u) * valid) / n_valid
            loss = loss + v_loss
            info = {"v_loss": v_loss, "v_mean": jnp.sum(v * valid) / n_valid}
            if getattr(cfg, "aqc_baseline", False):
                # Per-prefix baselines: each b_h chases the kappa_b-expectile of ITS OWN Q head on the
                # demo action, elementwise over [B, P] - the paired zero-point (Q_h - b_h) reads.
                b = v_all[..., 1:]  # [B, P]
                u_b = jax.lax.stop_gradient(q_demo) - b
                w_b = jnp.abs(cfg.baseline_expectile - (u_b < 0).astype(jnp.float32))
                b_loss = jnp.sum(w_b * jnp.square(u_b) * valid) / n_valid
                loss = loss + b_loss
                info |= {"b_loss": b_loss, "b_mean": jnp.sum(b * valid) / n_valid}

        return loss, info | {
            "loss": loss,
            "q_mean": q_mean,
            "target_mean": jnp.sum(y * valid) / n_valid,
            "valid": jnp.mean(w),  # data property; a change here means a bug, not progress
            "mc_floor_frac": jnp.sum((jnp.abs(y - data.mc_return[idx][:, None]) < 1e-7) * valid) / n_valid
            if cfg.mc_lower_bound
            else jnp.float32(0.0),
        }

    tx = optax.adam(cfg.lr)
    tx_v = optax.adam(cfg.lr)

    def step(carry, rng):
        # v_params rides along under --objective td as an empty pytree: one carry shape, one jit.
        params, tgt_params, opt_state, v_params, v_opt_state = carry
        idx = jax.random.randint(rng, (cfg.batch_size,), 0, T)
        boot_params = params if cfg.bootstrap == "online" else tgt_params
        (_, info), (grads, v_grads) = jax.value_and_grad(loss_fn, argnums=(0, 3), has_aux=True)(
            params, boot_params, idx, v_params
        )
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        v_updates, v_opt_state = tx_v.update(v_grads, v_opt_state)
        v_params = optax.apply_updates(v_params, v_updates)
        tgt_params = optax.incremental_update(params, tgt_params, cfg.target_tau)
        return (params, tgt_params, opt_state, v_params, v_opt_state), info

    return step, tx, tx_v


def make_diag(data, cfg, net, hl):
    """Within-state diagnostics on resident data - the make-or-break metrics, cheap enough for every eval.

    Evaluated in SLICES of 256 states: the naive single call put all 2048 states x N candidates
    through the net in one jit program, whose compilation blew host memory alongside a large
    resident dataset and killed the job with no traceback (v12 mixed runs died at the first eval,
    step 5000, twice, before this was found). The slices reuse ONE compiled program of fixed shape.
    """
    rng = np.random.default_rng(cfg.seed)
    n_diag = min(2048, data.token.shape[0])
    n_diag -= n_diag % 256  # fixed slice shape -> single compilation
    diag_idx = jnp.asarray(np.sort(rng.choice(data.token.shape[0], size=max(n_diag, 256), replace=False)))

    @jax.jit
    def diag_slice(p, idx):
        z = data.token[idx]
        qa = net.apply(p, jnp.repeat(z[:, None], data.num_samples, axis=1), data.cand[idx])
        qa = hl.from_logits(qa) if cfg.num_atoms > 1 else qa
        qa = jnp.min(qa, axis=0)  # ensemble -> [S, N(, P)]
        q_demo = net.apply(p, z[:, None], data.chunk[idx][:, None])
        q_demo = hl.from_logits(q_demo) if cfg.num_atoms > 1 else q_demo
        q_demo = jnp.min(q_demo, axis=0)
        q_demo = (q_demo[..., -1] if q_demo.ndim == 3 else q_demo)[:, 0]
        return qa, q_demo

    def diag(p):
        qas, q_demos = [], []
        for i in range(0, len(diag_idx), 256):
            qa_s, qd_s = diag_slice(p, diag_idx[i : i + 256])
            qas.append(np.asarray(qa_s))
            q_demos.append(np.asarray(qd_s))
        qa = np.concatenate(qas)
        q_demo = np.concatenate(q_demos)
        q_cand = qa[..., -1] if qa.ndim == 3 else qa
        within = jnp.mean(jnp.std(q_cand, axis=1))
        between = jnp.std(jnp.mean(q_cand, axis=1))
        out = {
            "diag_step/within_state_std": within,
            "diag_step/between_state_std": between,
            "diag_step/action_sensitivity": jnp.mean(jnp.var(q_cand, axis=1)) / (between**2 + 1e-12),
            "diag_step/range_over_std": jnp.mean(jnp.max(q_cand, 1) - jnp.min(q_cand, 1)) / (within + 1e-9),
            "diag_step/ranking_demo_vs_cand": jnp.mean(q_demo[:, None] > q_cand),
            "diag_step/q_demo_mean": jnp.mean(q_demo),
        }
        if qa.ndim == 3:  # what the joint arg-max would commit to
            p_idx = jnp.argmax(qa.reshape(qa.shape[0], -1), axis=-1) % qa.shape[-1]
            out |= {
                "diag_step/avg_chosen_horizon": jnp.mean((p_idx + 1) * cfg.macro_group_size),
                "diag_step/frac_shortest_prefix": jnp.mean(p_idx == 0),
            }
        return out

    return diag
