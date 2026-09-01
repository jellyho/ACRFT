"""Why best-of-N selection tied with no selection on the robot: the arg-max is selecting error.

Setup, self-contained. Our patch critic is trained by IQL expectile on 347 human demonstrations
(scripts/train_patch_critic_cached.py). IQL never evaluates an action outside the dataset, so
Q(s, A) is unconstrained for any chunk the policy actually proposes. At serving we then ask it to
arg-max over N such chunks. On the real robot (LEGOPROG) best-of-N at N=8 scored 1.70 mean progress
against 1.70 for no selection at all, while a stochastic expectile lottery over the same candidates
scored 2.70 -- soft selection worked, arg-max did not.

This measures the mechanism offline, on cached features, with no policy sampling and no robot time.

Probe. At each state we score (a) the chunk that was ACTUALLY EXECUTED there -- ground truth, drawn
from a mostly-successful episode -- and (b) N chunks executed at OTHER states, transplanted here.
(b) is the `shuffle` negative of --cql-negatives: on the action manifold, so the critic cannot
reject it on action statistics, only on whether it fits THIS state. We report

  bias curve   mean over states of  max_n Q(transplanted) - Q(executed),  as n grows.
               eval_rlt_critic.py's criterion: an unbiased critic SATURATES; a critic whose
               per-candidate error is noise keeps climbing like sigma*sqrt(2 ln n).
  lcb curve    the same, but selecting by CriticQ.q_lcb (mean - beta*std over the ensemble)
               instead of the mean. The gap between the two curves is how much of the winner's
               curse the ensemble can currently see.
  scales       per-state ensemble disagreement vs the within-state spread over candidates. If
               disagreement << spread, the ensemble is not measuring the uncertainty that the
               arg-max is exploiting, and no amount of LCB beta will fix it -- only more members.

Units. The reward is cost_to_goal at discount ~0.99964, so near the goal one value unit is
approximately one control step of time-to-goal; at 30 Hz, 100 units ~ 3.3 seconds.
"""

# ruff: noqa: PLC0415, ICN001

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
    ap.add_argument("--states", type=int, default=512)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--lcb-beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/diag_selection_bias.json")
    ap.add_argument("--fig", type=pathlib.Path, default=R / ".scratch/extraction/fig_selection_bias.png")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.extraction import critic_q as cq

    critic = cq.load(a.critic)
    view = cq.CacheView(a.cache)
    H, AD = critic.config["horizon"], critic.config["action_dim"]
    eps = list(view.meta["episodes"].values())
    rng = np.random.default_rng(a.seed)
    S = a.states
    rows = np.sort(rng.choice(view.meta["N"] - H - 1, S, replace=False))

    def executed(rs):
        """The chunk actually executed at each row, assembled exactly as the critic's trainer does
        (train_patch_critic_cached.py:336-341): clamp-and-hold at the episode end, then joint delta
        against the chunk's base frame + quantile normalization."""
        out = np.empty((len(rs), H, AD), np.float32)
        for i, g in enumerate(rs):
            ep = next(e for e in eps if e["offset"] <= g < e["offset"] + e["full_len"])
            end = ep["offset"] + ep["full_len"] - 1
            out[i] = np.asarray(view.actions[np.clip(g + np.arange(H), 0, end)])
        return out

    feats, raw, prop = view.rows(rows, critic)
    dch = critic.pre.actions(executed(rows), raw)[..., :AD]
    F, P = jnp.asarray(feats), jnp.asarray(prop)
    q_mean = jax.jit(critic.q_mean)
    q_lcb = jax.jit(lambda f, c, p: critic.q_lcb(f, c, p, a.lcb_beta))
    q_dis = jax.jit(critic.q_disagreement)

    q_exec = np.asarray(q_mean(F, jnp.asarray(dch), P))
    QN, QL = [], []
    for _ in range(a.candidates):
        p = rng.permutation(S)
        QN.append(np.asarray(q_mean(F, jnp.asarray(dch[p]), P)))
        QL.append(np.asarray(q_lcb(F, jnp.asarray(dch[p]), P)))
    QN, QL = np.stack(QN, 1), np.stack(QL, 1)  # [S, C]

    ns = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= a.candidates]
    res = {
        "critic": str(a.critic),
        "n_states": S,
        "n_candidates": a.candidates,
        "lcb_beta": a.lcb_beta,
        "n": ns,
        "q_executed_mean": float(q_exec.mean()),
        "per_candidate_margin_mean": float((QN - q_exec[:, None]).mean()),
        "per_candidate_margin_median": float(np.median(QN - q_exec[:, None])),
        "ranking_accuracy": float(np.mean(q_exec[:, None] > QN)),
        "ensemble_disagreement_std": float(np.mean(np.asarray(q_dis(F, jnp.asarray(dch), P)))),
        "within_state_candidate_std": float(np.mean(np.std(QN, 1))),
        "between_state_std": float(np.std(QN.mean(1))),
        "num_critics": int(critic.config["num_critics"]),
    }
    res["disagreement_to_spread_ratio"] = res["ensemble_disagreement_std"] / (res["within_state_candidate_std"] + 1e-9)
    argmax_m, lcb_m, argmax_f = [], [], []
    for n in ns:
        argmax_m.append(float((QN[:, :n].max(1) - q_exec).mean()))
        argmax_f.append(float(np.mean(QN[:, :n].max(1) > q_exec)))
        sel = QL[:, :n].argmax(1)
        lcb_m.append(float((QN[np.arange(S), sel] - q_exec).mean()))
    res["argmax_margin"] = argmax_m
    res["argmax_beats_executed_frac"] = argmax_f
    res["lcb_margin"] = lcb_m
    # sigma*sqrt(2 ln n) is the mean of the max of n zero-mean gaussian draws. sigma here is the
    # FULL within-state spread, which mixes real value differences with critic error, so this curve
    # is an UPPER ENVELOPE -- where the bias curve would sit if every bit of that spread were error.
    # It bounds the climb; it is not a sharp null, and the measured curve lying under it is expected.
    sig = res["within_state_candidate_std"]
    res["gaussian_null_margin"] = [
        float(
            res["per_candidate_margin_mean"] + sig * np.sqrt(2 * np.log(n))
            if n > 1
            else res["per_candidate_margin_mean"]
        )
        for n in ns
    ]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))

    import sys

    sys.path.insert(0, str(R / "slurm"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    PAL = plot_style.PALETTE
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.axhline(0, color="0.4", lw=1.0, ls="--", zorder=1)
    ax.plot(
        ns,
        res["gaussian_null_margin"],
        color="0.55",
        lw=1.4,
        ls=":",
        zorder=2,
        label="envelope if all spread were error",
    )
    ax.plot(ns, argmax_m, marker="o", ms=6, lw=1.8, color=PAL[3], zorder=4, label="arg-max Q (best-of-N)")
    ax.plot(
        ns,
        lcb_m,
        marker="s",
        ms=6,
        lw=1.8,
        color=PAL[0],
        zorder=3,
        label=f"arg-max LCB (β={a.lcb_beta:g}, K={res['num_critics']})",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("candidates ranked, N")
    ax.set_ylabel("Q(selected) − Q(executed)")
    ax.set_title("selection climbs above ground truth without saturating")
    ax.legend(loc="upper left")
    fig.tight_layout()
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=170)

    print(f"critic {a.critic.name}  K={res['num_critics']}  states={S}")
    print(
        f"per-candidate margin  mean {res['per_candidate_margin_mean']:.1f}  median {res['per_candidate_margin_median']:.1f}"
    )
    print(f"ranking accuracy (executed > transplanted): {res['ranking_accuracy']:.3f}   chance 0.5")
    print(
        f"ensemble disagreement {res['ensemble_disagreement_std']:.1f} vs within-state spread "
        f"{res['within_state_candidate_std']:.1f}  (ratio {res['disagreement_to_spread_ratio']:.3f})"
    )
    print("\n n    argmax margin  frac>0      LCB margin        envelope")
    for i, n in enumerate(ns):
        print(f"{n:<4d} {argmax_m[i]:13.1f} {argmax_f[i]:7.3f} {lcb_m[i]:15.1f} {res['gaussian_null_margin'][i]:15.1f}")
    print(f"\nwrote {a.out}\nwrote {a.fig}")


if __name__ == "__main__":
    main()
