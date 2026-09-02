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
  width sweep  the same bias, measured over candidate sets of CONTROLLED action-space width:
               chunks executed delta frames away in the same episode, delta from 1 (nearly the
               same action) up to a whole other state. This is what reconciles this probe with the
               hub entry q-landscape-ood, which measured the bias over the BC policy's OWN draws
               (sigma_BC ~ 0.009 per dim) and found it small and log(N)-bounded (+1.88 +- 0.62 at
               N=16). Both are true: the bias GROWS with how far the candidate set reaches, and
               best-of-N at serving reaches only as far as the sampler's own noise. Read together
               they say the serving question is not "does argmax select over-estimates" but
               "is there anything to select at that width at all".
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


# q-landscape-ood measured the SAME bias against the SAME anchor over the BC sampler's own draws.
# Its two numbers are the only external points on this axis, so they are plotted, not restated.
#
# GETTING THE UNITS RIGHT COST US A FACTOR OF 4.6. The probe reports sigma = 0.009, but that is
# `bc.std(axis=0).mean()` (probe_q_landscape.py:177) -- the MEAN over coordinates of the per-
# coordinate std. This axis is a per-dimension RMS, `norm(delta)/sqrt(H*AD)`, and mean-of-std is
# below RMS-of-std by Jensen whenever the coordinates are heterogeneous, which a 30-step chunk
# mixing arm joints with grippers certainly is. Recovering the RMS from the probe's own frozen
# output (slurm/probes/q_landscape.json.gz, origin/probe/q-landscape @ 0bc7bbf1):
#     pc_sigma    = [0.4565, 0.2205]   std along PC1, PC2 of the 420-dim draws
#     pc_var_frac = [0.544, 0.154]     share of the draw variance on each
#     total variance = 0.4565^2 / 0.544 = 0.383
#     per-coordinate RMS std = sqrt(0.383 / 420) = 0.0295
#     distance between two independent draws = sqrt(2) * 0.0295 = 0.0417
# So the BC cloud sits at 0.042 on this axis, not 0.009. At 0.042 the fitted power law gives +1.84,
# against the probe's independently measured +1.88 -- which also DISSOLVES the 14x residual this
# entry previously reported and hypothesised about ("BC draws are more adversarial per unit
# distance"). There was no residual; the marker was in the wrong place.
SIGMA_BC = 0.0417  # per-dim RMS distance between two BC draws, derived above
BON_BIAS = 1.88  # arg-max bias at N=16 over those draws
BON_CI = 0.62


def figure(res, path):
    """Regenerate the figure from the probe's own JSON. Split out so a report build can redraw
    without re-running the probe (the numbers still come from the source, never from prose)."""
    import sys

    sys.path.insert(0, str(R / "slurm"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    PAL = plot_style.PALETTE
    ns, W = res["n"], res["width_n"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

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
    ax.plot(ns, res["argmax_margin"], marker="o", ms=6, lw=1.8, color=PAL[3], zorder=4, label="arg-max Q (best-of-N)")
    ax.plot(
        ns,
        res["lcb_margin"],
        marker="s",
        ms=6,
        lw=1.8,
        color=PAL[0],
        zorder=3,
        label=f"arg-max LCB (β={res['lcb_beta']:g}, K={res['num_critics']})",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("candidates ranked, N")
    ax.set_ylabel("Q(selected) − Q(executed)")
    ax.set_title("transplanted candidates: no saturation")
    ax.legend(loc="upper left", fontsize=9)

    sp, bi = res["width_spread_rms"], res["width_argmax_bias"]
    k = np.polyfit(np.log(sp), np.log(bi), 1)[0]
    ax2.plot(sp, bi, marker="o", ms=6, lw=1.8, color=PAL[3], zorder=3, label=f"this probe  (slope {k:.1f} in log-log)")
    ax2.errorbar(
        [SIGMA_BC],
        [BON_BIAS],
        yerr=[BON_CI],
        marker="D",
        ms=8,
        lw=0,
        elinewidth=1.5,
        capsize=4,
        color=PAL[2],
        zorder=5,
        label="q-landscape-ood: the BC sampler's own draws",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("candidate spread from the executed chunk  (per-dim RMS)")
    ax2.set_ylabel(f"arg-max bias, N={W}  (control steps)")
    ax2.set_title("the bias is a function of candidate width")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    return k


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
    ap.add_argument(
        "--widths",
        type=int,
        nargs="*",
        default=[1, 2, 5, 10, 30, 100],
        help="candidate sets drawn from +-delta frames away in the same episode, one set per delta; "
        "the transplant-from-another-state set is appended as the widest point",
    )
    ap.add_argument("--width-n", type=int, default=16, help="candidates per width (matches q-landscape-ood's N=16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/diag_selection_bias.json")
    ap.add_argument("--fig", type=pathlib.Path, default=R / ".scratch/extraction/fig_selection_bias.png")
    ap.add_argument("--fig-only", action="store_true", help="redraw from the existing --out JSON, no recompute")
    a = ap.parse_args()

    if a.fig_only:
        k = figure(json.loads(a.out.read_text()), a.fig)
        print(f"redrew {a.fig}  (log-log slope {k:.2f})")
        return

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

    # Per-row episode bounds, built once. Doing this lookup inside the chunk assembly (a linear
    # scan over 347 episodes per row) made the width sweep O(rows * episodes * candidates) and
    # dominated the runtime.
    ep_start = np.empty(view.meta["N"], np.int64)
    ep_end = np.empty(view.meta["N"], np.int64)
    for e in eps:
        lo, hi = e["offset"], e["offset"] + e["full_len"]
        ep_start[lo:hi] = lo
        ep_end[lo:hi] = hi - 1
    ar_h = np.arange(H)

    def executed(rs):
        """The chunk actually executed at each row, assembled exactly as the critic's trainer does
        (train_patch_critic_cached.py:336-341): clamp-and-hold at the episode end, then joint delta
        against the chunk's base frame + quantile normalization (applied by the caller)."""
        g = np.clip(rs[:, None] + ar_h[None], ep_start[rs][:, None], ep_end[rs][:, None])
        return np.asarray(view.actions[g.reshape(-1)]).reshape(len(rs), H, AD)

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

    # ---- width sweep: how far does the candidate set have to reach before argmax bites? ---------
    # Candidates are chunks executed delta frames away in the SAME episode: real actions, at a
    # controlled action-space distance from the executed one. Distance is reported as per-dimension
    # RMS in normalized action units so it sits on the same axis as the BC sampler's own spread.
    starts, ends = ep_start[rows], ep_end[rows]
    W = a.width_n
    widths, wbias, wspread = [], [], []
    for delta in a.widths:
        off = rng.integers(-delta, delta + 1, size=(S, W))
        off[off == 0] = delta  # a zero offset would put the executed chunk in its own candidate set
        g2 = np.clip(rows[:, None] + off, starts[:, None], ends[:, None])
        qc = np.empty((S, W), np.float32)
        dist = np.empty((S, W), np.float32)
        for j in range(W):
            cj = critic.pre.actions(executed(g2[:, j]), raw)[..., :AD]
            qc[:, j] = np.asarray(q_mean(F, jnp.asarray(cj), P))
            dist[:, j] = np.linalg.norm((cj - dch).reshape(S, -1), axis=1) / np.sqrt(H * AD)
        widths.append(int(delta))
        wbias.append(float((qc.max(1) - q_exec).mean()))
        wspread.append(float(dist.mean()))
    # widest point: the transplant set already computed above
    tdist = np.mean([np.linalg.norm((dch[rng.permutation(S)] - dch).reshape(S, -1), axis=1) for _ in range(4)])
    widths.append(-1)  # -1 = another state entirely
    wbias.append(float((QN[:, :W].max(1) - q_exec).mean()))
    wspread.append(float(tdist / np.sqrt(H * AD)))
    res["width_deltas"] = widths
    res["width_spread_rms"] = wspread
    res["width_argmax_bias"] = wbias
    res["width_n"] = W

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))

    figure(res, a.fig)

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
