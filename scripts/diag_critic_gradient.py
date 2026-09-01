"""Is this critic's GRADIENT usable, or only its ranking? An offline, robot-free falsification test.

Every arm in the extraction ring that moves the policy (FlowDPG, QAM, QPILOTS-U, and Q-VGM if we
were to add it) consumes the same object: grad_A Q at an action the critic has never been trained
on. Our real-robot evidence is ambiguous about whether that object carries signal -- BoN argmax over
N=8 policy samples tied with no selection at all (1.70 vs 1.70) while the stochastic expectile
lottery reached 2.70, and QPILOTS-U steering fell monotonically from 1.80 at alpha=0 to 0.30 at
alpha=0.1. Both are consistent with "ranking usable, gradient not", and both cost robot trials.
This script asks the same question on cached features for a few GPU-hours.

The candidates are REAL policy samples (scripts/sample_policy_chunks.py), not dataset chunks: the
distribution the arms actually score is the policy's, and a critic can rank demonstrated chunks
perfectly while being uninformative on the policy's own draws.

  action_sensitivity   within-state Q spread / between-state Q spread, over policy samples.
                       Definition and FAIL threshold from scripts/eval_rlt_critic.py:191-194
                       (a critic ignoring its action argument scores ~0; plain-IQL critics have
                       previously landed at ~1e-3 here).
  demo_beaten_frac     fraction of states where max_k Q(policy sample) > Q(demonstrated chunk).
                       The demonstrated chunk was actually executed, mostly in a successful episode.
                       A critic that routinely believes an unexecuted sample beats it is reporting
                       optimism about its own out-of-support inputs -- the OOD failure, measured.
  keep-best ladder     J norm-matched clipped ascent steps (the FlowDPG Eq. 6 step, scripts/
                       train_flowdpg.py:137-140, with the step scaled to the policy's OWN sampling
                       spread so alpha reads as "a fraction of the noise the policy already has").
                       Reports Q(j), the Q-VGM keep-best argmax j* and its FALLBACK RATE (j*==0) --
                       the statistic Q-VGM (arXiv 2606.08015v1) never reports and which decides
                       whether its safety valve does anything -- and ||A_j - A_demo||, because a
                       gradient that raises Q while walking AWAY from the executed action is
                       climbing the critic's error, not its value.
  bon_bias_curve       Q of the arg-max as a function of how many candidates it ranges over.
                       An unbiased critic saturates; max-over-noisy-estimates keeps climbing
                       (eval_rlt_critic.py's bias curve, on policy samples).
"""

# ruff: noqa: PLC0415

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--bank", type=pathlib.Path, default=R / ".scratch/extraction/policy_chunks_bc")
    ap.add_argument("--num-states", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ascent-steps", "-j", type=int, default=8)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="ascent step as a fraction of the policy's own per-state sampling spread",
    )
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/diag_critic_gradient.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.extraction import critic_q as cq

    critic = cq.load(a.critic)
    view = cq.CacheView(a.cache)
    bank_idx = np.load(a.bank / "idx.npy")
    bank = np.load(a.bank / "chunks.npy", mmap_mode="r")  # [N, K, H, ad]
    bmeta = json.loads((a.bank / "meta.json").read_text())
    n_all, K, H, ad = bank.shape
    print(f"critic={a.critic.name} bank={n_all}x{K}x{H}x{ad} from {bmeta['init_ckpt']}", flush=True)

    sel = np.linspace(0, n_all - 1, min(a.num_states, n_all)).astype(np.int64)
    grad_fn = jax.jit(cq.grad_q_chunk(critic))
    q_fn = jax.jit(critic.q_mean)

    def demo_chunk(rows):
        """The executed chunk at each state, assembled and pi05-normalized exactly as the critic's
        trainer does (train_patch_critic_cached.py:336-341): clamp-and-hold past the episode end,
        joint delta against the chunk's BASE frame, then quantile normalization."""
        eps = view.meta["episodes"]
        out = np.empty((len(rows), H, ad), np.float32)
        for i, g in enumerate(rows):
            for ep in eps.values():
                if ep["offset"] <= g < ep["offset"] + ep["full_len"]:
                    end = ep["offset"] + ep["full_len"] - 1
                    break
            gh = np.clip(g + np.arange(H), 0, end)
            out[i] = np.asarray(view.actions[gh])
        return out

    res = {"critic": str(a.critic), "bank": str(a.bank), "n_states": len(sel), "k": int(K)}
    q_pol, q_demo_all, ladder_q, ladder_d, fallback = [], [], [], [], []

    for s0 in range(0, len(sel), a.batch):
        b = sel[s0 : s0 + a.batch]
        rows = bank_idx[b]
        feats, raw_state, proprio = view.rows(rows, critic)
        cand = np.asarray(bank[b], np.float32)  # [B, K, H, ad]
        B = len(b)

        f_rep = jnp.asarray(np.repeat(feats, K, 0))
        p_rep = jnp.asarray(np.repeat(proprio, K, 0))
        qk = np.asarray(q_fn(f_rep, jnp.asarray(cand.reshape(B * K, H, ad)), p_rep)).reshape(B, K)
        q_pol.append(qk)

        dch = critic.pre.actions(demo_chunk(rows), raw_state)[..., :ad]
        q_demo_all.append(np.asarray(q_fn(jnp.asarray(feats), jnp.asarray(dch), jnp.asarray(proprio))))

        # --- keep-best ascent ladder, from the FIRST policy sample of each state ---------------
        # Step scale = the policy's own per-state spread, so alpha is dimensionless and comparable
        # across states: a step of alpha=1 moves as far as the sampler's own noise typically does.
        spread = np.sqrt(np.mean(np.var(cand, axis=1), axis=(1, 2))) + 1e-8  # [B]
        x = jnp.asarray(cand[:, 0])
        fj, pj = jnp.asarray(feats), jnp.asarray(proprio)
        qs = [np.asarray(q_fn(fj, x, pj))]
        ds = [np.linalg.norm(np.asarray(x) - dch, axis=(1, 2))]
        for _ in range(a.ascent_steps):
            g = grad_fn(x, fj, pj)
            gn = jnp.linalg.norm(g.reshape(B, -1), axis=-1)[:, None, None] + 1e-8
            x = jnp.clip(x + a.alpha * jnp.asarray(spread)[:, None, None] * np.sqrt(H * ad) * g / gn, -1.0, 1.0)
            qs.append(np.asarray(q_fn(fj, x, pj)))
            ds.append(np.linalg.norm(np.asarray(x) - dch, axis=(1, 2)))
        qs = np.stack(qs, 1)  # [B, J+1]
        ladder_q.append(qs)
        ladder_d.append(np.stack(ds, 1))
        fallback.append(qs.argmax(1) == 0)
        if s0 % (a.batch * 10) == 0:
            print(f"{s0}/{len(sel)}", flush=True)

    q_pol = np.concatenate(q_pol)  # [S, K]
    q_demo = np.concatenate(q_demo_all)  # [S]
    ladder_q = np.concatenate(ladder_q)  # [S, J+1]
    ladder_d = np.concatenate(ladder_d)
    fallback = np.concatenate(fallback)

    within = float(np.mean(np.var(q_pol, axis=1)))
    between = float(np.var(np.mean(q_pol, axis=1)))
    res["action_sensitivity"] = within / (between + 1e-12)
    res["within_state_std"] = float(np.mean(np.std(q_pol, axis=1)))
    res["between_state_std"] = float(np.sqrt(between))
    res["q_policy_mean"] = float(q_pol.mean())
    res["q_demo_mean"] = float(q_demo.mean())
    res["demo_beaten_frac"] = float(np.mean(q_pol.max(1) > q_demo))
    res["demo_beaten_margin"] = float(np.mean(q_pol.max(1) - q_demo))
    res["bon_bias_curve"] = [float(np.mean(q_pol[:, :n].max(1))) for n in range(1, K + 1)]
    res["ladder_q_mean"] = [float(v) for v in ladder_q.mean(0)]
    res["ladder_dist_to_demo_mean"] = [float(v) for v in ladder_d.mean(0)]
    res["keepbest_fallback_rate"] = float(fallback.mean())
    res["ladder_q_gain"] = float(np.mean(ladder_q[:, -1] - ladder_q[:, 0]))
    res["ladder_gain_vs_demo_gap"] = res["ladder_q_gain"] / (abs(res["q_demo_mean"] - res["q_policy_mean"]) + 1e-8)
    # Does climbing Q move toward the executed action, or away from it?
    res["ladder_dist_corr"] = float(
        np.corrcoef(
            (ladder_q[:, -1] - ladder_q[:, 0]),
            (ladder_d[:, -1] - ladder_d[:, 0]),
        )[0, 1]
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:28s} {v}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
