"""Could an oracle-retrieval policy work, and how much better than our critic is it?

Experiment A of the critic-vs-extraction split needs a selector that uses NO learned critic. The
data supplies one: for a demonstrated chunk we know its realized return exactly -- how long that
episode actually took from there. So "retrieve the k nearest demonstration states, execute the chunk
of whichever had the best realized outcome" is a selection policy whose scoring function is ground
truth. If it beats BC on the robot there IS headroom and our critic is what fails to find it; if it
ties BC, selection over real actions has nothing to give here and the critic is not the bottleneck.

That is a robot experiment, so this checks first, offline and free, whether it could be informative:

  mechanical sanity   a retrieved chunk is re-based as a delta from THIS state, so it cannot
                      teleport -- but it can still ask for motion the arm was not about to make.
                      Reported as the first-step delta and the fraction of the chunk that leaves the
                      normalized action box, both against what the demonstrator did here.
  oracle-vs-critic    on the SAME candidate set, how does our critic's ranking compare with ranking
                      by realized outcome? The gap is the effect size A and C are trying to detect.
                      A critic already at the oracle's ranking leaves A nothing to show.
  headroom, offline   among the retrieved candidates, how much better is the best realized outcome
                      than the one the demonstrator actually took here? An upper bound on what
                      perfect selection could buy, in the units the robot reports.
"""

# ruff: noqa: PLC0415

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--critic", type=pathlib.Path, default=R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k")
    ap.add_argument("--homing", type=pathlib.Path, default=R / ".scratch/yam_homing_onsets.json")
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--neighbours", "-k", type=int, default=16)
    ap.add_argument("--ref-stride", type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/retrieval_oracle.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from scipy import stats

    from openpi.extraction import critic_q as cq

    critic = cq.load(a.critic)
    view = cq.CacheView(a.cache)
    meta = view.meta
    n_all, hor, ad = meta["N"], critic.config["horizon"], critic.config["action_dim"]
    hom = json.loads(a.homing.read_text())
    items = list(meta["episodes"].items())
    epid = np.empty(n_all, np.int32)
    ep_end = np.empty(n_all, np.int64)
    eff_end = np.empty(n_all, np.int64)
    succ = np.zeros(len(items), bool)
    for i, (k, e) in enumerate(items):
        lo, hi = e["offset"], e["offset"] + e["full_len"]
        epid[lo:hi], ep_end[lo:hi], succ[i] = i, hi - 1, bool(e["success"])
        h = hom.get(k, hom.get(str(k)))
        # The onsets file stores {"len", "homing_onset", "task_frac"} per episode, not a bare int.
        # Reading it as a scalar made the fallback fire every time, so homing truncation has NEVER
        # applied here -- independent of which onsets file is supplied.
        onset = h.get("homing_onset") if isinstance(h, dict) else (int(h) if isinstance(h, int | float) else None)
        eff_end[lo:hi] = lo + (onset if onset is not None else e["full_len"]) - 1

    feats_pool = np.load(a.cache / "features_pooled_f32.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    # The bank a serving policy would carry: successful demonstrations only, strided.
    bank = np.arange(0, n_all, a.ref_stride)
    bank = bank[succ[epid[bank]]]
    bank = bank[bank < eff_end[bank] - hor]
    fb = np.asarray(feats_pool[bank], np.float32)
    fb /= np.linalg.norm(fb, axis=1, keepdims=True) + 1e-9
    ttg_bank = (eff_end[bank] - bank).astype(np.float64)

    q = np.sort(rng.choice(bank, min(a.queries, len(bank)), replace=False))
    fq = np.asarray(feats_pool[q], np.float32)
    fq /= np.linalg.norm(fq, axis=1, keepdims=True) + 1e-9
    sim = fq @ fb.T
    sim[epid[bank][None, :] == epid[q][:, None]] = -9  # never retrieve from the episode you are in
    order = np.argsort(-sim, axis=1)[:, : a.neighbours]
    nn = bank[order]

    def raw_chunks(rows):
        g = np.clip(rows[:, None] + np.arange(hor)[None], 0, ep_end[rows][:, None])
        return np.asarray(view.actions[g.reshape(-1)]).reshape(len(rows), hor, ad)

    fq_full, raws, props = view.rows(q, critic)
    own = critic.pre.actions(raw_chunks(q), raws)[..., :ad]
    ttg_q = (eff_end[q] - q).astype(np.float64)
    qm = jax.jit(critic.q_mean)

    first_step, oob, dist, rank_c, rank_o, gain, chosen_is_own_like = [], [], [], [], [], [], []
    dttg, dpro = [], []
    # proprio of every bank frame, so a retrieved neighbour can be scored on the key that is NOT
    # what it was retrieved by
    _, _, props_bank = view.rows(bank, critic)
    pos_of = {int(g): i for i, g in enumerate(bank)}
    nn_pos = np.array([[pos_of[int(x)] for x in row] for row in nn])
    props = props_bank[np.array([pos_of[int(x)] for x in q])]
    for i in range(len(q)):
        j = nn[i]
        # every candidate expressed as a delta from THIS state -- the space a policy proposal lives in
        cand = critic.pre.actions(raw_chunks(j), np.repeat(raws[i][None], len(j), 0))[..., :ad]
        mine = own[i]
        # mechanical sanity, against what the demonstrator did here
        first_step.append((float(np.median(np.abs(cand[:, 0]).mean(-1))), float(np.abs(mine[0]).mean())))
        oob.append(float(np.mean(np.abs(cand) > 1.0)))
        dist.append(float(np.median(np.linalg.norm((cand - mine).reshape(len(j), -1), axis=1) / np.sqrt(hor * ad))))
        # oracle ranking = realized remaining time of the episode each candidate came from
        truth = -ttg_bank[order[i]]
        fj = jnp.repeat(jnp.asarray(fq_full[i])[None], len(j), 0)
        pj = jnp.repeat(jnp.asarray(props[i])[None], len(j), 0)
        qs = np.asarray(qm(fj, jnp.asarray(cand), pj))
        if len(np.unique(truth)) > 2:
            rank_c.append(stats.spearmanr(qs, truth).statistic)
        rank_o.append(1.0)
        # headroom: best realized remaining time among candidates vs what was taken here
        gain.append(float(ttg_q[i] - ttg_bank[order[i]].min()))
        chosen_is_own_like.append(
            float(np.linalg.norm((cand[np.argmax(truth)] - mine).reshape(-1)) / np.sqrt(hor * ad))
        )
        # RETRIEVAL QUALITY -- the thing the whole experiment rests on. A neighbour that is not the
        # same situation makes the oracle an oracle over the wrong candidates, and A would then be
        # measuring the retriever rather than extraction.
        dttg.append(float(np.median(np.abs(ttg_bank[order[i]] - ttg_q[i]))))
        dpro.append(
            float(np.median(np.linalg.norm(props_bank[nn_pos[i]] - props[i], axis=1) / np.sqrt(props.shape[1])))
        )

    fs = np.array(first_step)
    rank_c = np.array([x for x in rank_c if np.isfinite(x)])
    res = {
        "queries": len(q),
        "bank_frames": len(bank),
        "k": a.neighbours,
        "mechanical": {
            "first_step_mean_abs_retrieved": float(np.median(fs[:, 0])),
            "first_step_mean_abs_demonstrated": float(np.median(fs[:, 1])),
            "ratio": float(np.median(fs[:, 0]) / (np.median(fs[:, 1]) + 1e-9)),
            "frac_out_of_box": float(np.mean(oob)),
            "chunk_distance_from_demonstrated_per_dim": float(np.median(dist)),
        },
        "ranking": {
            "critic_vs_realized_spearman_mean": float(rank_c.mean()),
            "critic_vs_realized_spearman_ci95": float(1.96 * rank_c.std(ddof=1) / np.sqrt(len(rank_c))),
            "oracle_by_construction": 1.0,
            "n_sets": len(rank_c),
        },
        "headroom_frames": {
            "median": float(np.median(gain)),
            "p25": float(np.percentile(gain, 25)),
            "p75": float(np.percentile(gain, 75)),
            "frac_positive": float(np.mean(np.asarray(gain) > 0)),
        },
        "oracle_pick_distance_from_demonstrated_per_dim": float(np.median(chosen_is_own_like)),
        "retrieval_quality": {
            "median_abs_delta_remaining_frames": float(np.median(dttg)),
            "median_proprio_distance_per_dim": float(np.median(dpro)),
            "note": "the neighbours were retrieved by pooled DINOv2; these score them on keys they "
            "were NOT retrieved by. Reference points measured elsewhere in this project: feature-NN "
            "gave |d remaining| 470 frames, while matching on proprio + phase gave 79.",
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    m, rk, hr = res["mechanical"], res["ranking"], res["headroom_frames"]
    print(f"bank {res['bank_frames']} frames, {res['queries']} queries, k={a.neighbours}\n")
    print("MECHANICAL SANITY (can this even be executed?)")
    print(
        f"  first-step |delta| retrieved {m['first_step_mean_abs_retrieved']:.4f} vs demonstrated "
        f"{m['first_step_mean_abs_demonstrated']:.4f}  ({m['ratio']:.2f}x)"
    )
    print(f"  fraction of retrieved chunk outside the action box: {m['frac_out_of_box'] * 100:.2f}%")
    print(f"  chunk distance from what was demonstrated here: {m['chunk_distance_from_demonstrated_per_dim']:.3f}/dim")
    print("\nORACLE vs OUR CRITIC on the same candidates")
    print(
        f"  critic ranks by realized outcome at {rk['critic_vs_realized_spearman_mean']:+.3f} "
        f"+- {rk['critic_vs_realized_spearman_ci95']:.3f}   (oracle = +1.000 by construction, n={rk['n_sets']})"
    )
    print("\nHEADROOM the oracle could capture (frames of remaining time, 30 fps)")
    print(
        f"  median {hr['median']:+.0f}  IQR [{hr['p25']:+.0f}, {hr['p75']:+.0f}]  "
        f"positive in {hr['frac_positive'] * 100:.0f}% of states"
    )
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
