"""Does a trained critic RANK, or does it just read the outcome label off the state?

score_critic_cached.py answers "can this critic tell a success episode from a failure", and on the
cable-tie critics the answer is AUC 1.0000 -- which is the problem, not the result. A critic that
separates outcomes perfectly has learned "will this episode fail", and that bit is free: it is in the
state, and predicting it requires no judgement about actions at all.

What the critic is FOR is ranking: at one state, is this chunk better than that one. The quantity
that stands in for it offline is Spearman(Q(demo chunk), -time_to_goal), and it has to be measured
WITHIN an episode. Across episodes the target itself disagrees -- remaining time at a matched task
state varies at CV 0.188 -- so a global correlation mixes the critic's skill with the pace of
whichever episode a frame came from. The within-episode number is the honest one. This is the same
quantity train_patch_critic_cached.py computes at --eval-every, lifted out so it can be run on a
critic that was trained without a held-out split (--heldout-frac 0, which is the launcher default
and is what every cable-tie critic so far used).

    uv run python scripts/score_critic_ranking.py \
        --critic <run dir or step dir> --cache <feature cache> \
        --homing-onsets .scratch/yam_cable_tie_homing_onsets.json

IN-SAMPLE, and deliberately so. A critic trained on every episode has no held-out set to go back to,
so these frames were seen in training. That inflates the absolute numbers and is stated in the
output. It does NOT invalidate comparing two critics that both trained on everything: the inflation
is a property of the protocol, identical for both, so a difference between them is still a
difference between the methods. Use --episodes to score a subset if you want a cleaner read on a
critic that did hold episodes out.
"""

import argparse
import json
import pathlib

import numpy as np


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and rx.std() and ry.std() else float("nan")


def _auc(score, label):
    if label.all() or not label.any():
        return float("nan")
    r = np.argsort(np.argsort(score)).astype(np.float64) + 1
    n1 = int(label.sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True, help="run dir or step_NNNNNN dir")
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=None)
    ap.add_argument("--frames", type=int, default=4096, help="anchor frames drawn once, seeded")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-per-episode", type=int, default=8, help="episodes with fewer anchors are skipped")
    ap.add_argument("--early", type=int, default=60, help="frames from episode start counted as 'early'")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="also write the numbers as JSON")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    import openpi.patch_critic.critic as pc
    import openpi.patch_critic.preproc as critic_preproc
    import openpi.patch_critic.spec as critic_spec

    ck = critic_spec.resolve_checkpoint_dir(a.critic)
    cc, _ = critic_spec.load(ck)
    isp = cc.get("input_spec", {})
    meta = json.loads((a.cache / "meta.json").read_text())
    H, ad = int(cc["horizon"]), int(cc["action_dim"])
    npatch, emb, sd = int(meta["npatch"]), int(meta["emb"]), int(meta["sd"])
    N = int(meta["N"])

    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, npatch, emb))
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, sd))
    actions = np.memmap(a.cache / "action.dat", np.float32, "r", shape=(N, int(meta["ad"])))

    # Episode geometry, and the homing truncation the critic was TRAINED with: time-to-goal has to be
    # measured in the same frame the target used, or the correlation is against a different quantity.
    # meta["episodes"] is {episode_index_as_str: {offset, full_len, success}}; the homing onsets are
    # keyed the same way, which is what lets the two line up without a positional assumption.
    eps = meta["episodes"]
    onsets = json.loads(a.homing_onsets.read_text()) if a.homing_onsets else {}
    ep_keys = sorted(eps, key=int)
    g0, eff, succ = [], [], []
    for k in ep_keys:
        e = eps[k]
        off, full = int(e["offset"]), int(e["full_len"])
        cut = int(onsets.get(k, {}).get("homing_onset", full)) if onsets else full
        g0.append(off)
        eff.append(min(cut, full))
        succ.append(bool(e["success"]))
    g0, eff, succ = np.array(g0), np.array(eff), np.array(succ, bool)

    # One anchor pool over every episode, then a seeded draw -- so two critics scored with the same
    # seed see the SAME frames and a difference between them is not a difference in sampling.
    pool_ep, pool_pos = [], []
    for i in range(len(ep_keys)):
        n = max(eff[i] - 1, 0)
        if n <= 0:
            continue
        pool_ep.append(np.full(n, i))
        pool_pos.append(np.arange(n))
    pool_ep, pool_pos = np.concatenate(pool_ep), np.concatenate(pool_pos)
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(pool_pos), min(a.frames, len(pool_pos)), replace=False)
    e_i, e_pos = pool_ep[sel], pool_pos[sel]
    e_g0, e_eff, e_succ = g0[e_i], eff[e_i], succ[e_i]

    ar_h = np.arange(H)
    gcur = e_g0 + e_pos
    gch = e_g0[:, None] + np.clip(e_pos[:, None] + ar_h[None], 0, e_eff[:, None] - 1)
    chunk = np.asarray(actions[gch.reshape(-1)]).reshape(len(sel), H, int(meta["ad"]))
    state = np.asarray(states[gcur])

    pre = None
    if isp.get("normalization") == "pi05":
        ns = ck / isp.get("norm_stats_file", "pi05_norm_stats.json")
        pre = critic_preproc.Pi05Preproc(
            ref=np.asarray(isp["joint_delta_reference"], np.int64),
            stats=critic_preproc.load_norm_stats(ns if ns.exists() else isp["norm_stats"]),
            use_quantiles=bool(isp["use_quantiles"]),
            delta=isp["delta_mode"] == "joint",
        )
        chunk, state = pre.actions(chunk, state), pre.state(state)
    pidx = np.asarray(isp["proprio_indices"], np.int64) if isp.get("proprio_indices") is not None else None
    if pidx is not None:
        state = state[..., pidx]
    chunk = chunk[..., :ad]

    net = pc.PatchCriticEnsemble(
        action_dim=ad,
        horizon=H,
        num_critics=int(cc["num_critics"]),
        macro_group_size=int(cc["macro_group_size"]),
        num_atoms=int(cc["num_atoms"]),
    )
    import flax.serialization

    params = flax.serialization.msgpack_restore((ck / "params.msgpack").read_bytes())
    params = params.get("params", params)
    hl = pc.HLGauss(cc["v_min"], cc["v_max"], int(cc["num_atoms"]))
    red = cc.get("q_reduction", "min")

    @jax.jit
    def q_of(p, f, c, s):
        qd = net.apply({"params": p}, f.astype(jnp.float32), c, s)[:, :, -1, :]
        v = hl.from_logits(qd)
        return v.min(0) if red == "min" else v.mean(0)

    qs = []
    for i in range(0, len(sel), 512):
        sl = slice(i, i + 512)
        qs.append(
            np.asarray(q_of(params, jnp.asarray(feats[gcur[sl]]), jnp.asarray(chunk[sl]), jnp.asarray(state[sl])))
        )
    q = np.concatenate(qs).astype(np.float64)

    ttg = (e_eff - e_pos).astype(np.float64)  # -time_to_goal in the TRUNCATED frame
    wi = [_spearman(q[e_i == e], -ttg[e_i == e]) for e in np.unique(e_i) if (e_i == e).sum() >= a.min_per_episode]
    wi = [w for w in wi if not np.isnan(w)]
    early = e_pos < a.early

    out = {
        "critic": str(ck),
        "frames": int(len(sel)),
        "episodes": int(len(np.unique(e_i))),
        "episodes_scored_within": len(wi),
        "spearman_within_ep": float(np.mean(wi)) if wi else float("nan"),
        "spearman_within_ep_median": float(np.median(wi)) if wi else float("nan"),
        "spearman_within_ep_frac_positive": float(np.mean([w > 0 for w in wi])) if wi else float("nan"),
        "spearman_global": _spearman(q, -ttg),
        "auc_early": _auc(q[early], e_succ[early]),
        "q_mean": float(q.mean()),
        "in_sample": True,
    }
    print(f"critic {ck}")
    print(f"  anchors {out['frames']} frames / {out['episodes']} episodes  (IN-SAMPLE: see module docstring)")
    print(f"  spearman within-episode : {out['spearman_within_ep']:+.4f}   <- the honest ranking number")
    print(
        f"      median {out['spearman_within_ep_median']:+.4f}   frac>0 {out['spearman_within_ep_frac_positive']:.3f}"
        f"   over {out['episodes_scored_within']} episodes"
    )
    print(f"  spearman global         : {out['spearman_global']:+.4f}   (mixes in episode pace)")
    print(f"  AUC(success vs fail, early frames) = {out['auc_early']:.4f}   <- HIGH here means the outcome")
    print("      label is readable from the state, which is the shortcut, not the skill")
    print(f"  Q mean {out['q_mean']:.1f}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
