"""What does INDEPENDENT training actually buy? Pairwise decorrelation at three levels.

The question this settles. A shared-trunk ensemble is cheap (one trunk + 8 MB per member) but the
objection is that members sharing a backbone are not independent, so their disagreement understates
the true error and any debiasing built on it under-corrects. Before redesigning around that, measure
what independence buys TODAY, because our members are already fully independent and the disagreement
is still tiny (6.3 against a within-state candidate spread of 134).

Three levels of separation, same probe:
  within      the two vmapped members inside ONE checkpoint. Separate parameters, same architecture,
              same data, same targets -- initialization is the only difference
              (patch_critic/critic.py nn.vmap: split_rngs={"params": True}, in_axes=None, and the
              trainer broadcasts one target with tgt[None]).
  same-recipe across two checkpoints that differ in one nuisance factor (augmentation, mc floor).
  cross-recipe across checkpoints that differ in macro_group_size or expectile.

Two quantities per pair, both computed WITHIN a state so a shared state-value offset cannot inflate
them:
  q_corr      correlation of Q over candidate chunks. High = the two agree on ranking.
  grad_cos    cosine similarity of grad_A Q at the demonstrated chunk. This is the one that matters:
              worker B's probe finds Q inflates by +32.8 along grad_A Q against -0.13 along random
              directions, and that it does so in all 9 critics. If grad_cos is high even ACROSS
              independently trained checkpoints, then independent training does not decorrelate the
              failure direction, the correlation is coming from the data and the objective rather
              than from shared weights, and sharing a trunk is not what is costing us the signal.

CORROBORATION FROM AN UNRELATED STATISTIC. ACRFT-WS's probe reports frac_outside_span on the same
nine critics, and it splits by MACRO GROUP and nothing else: the fixed family (macro 30) sits at
0.72-0.77, the g5 family (macro 5) at 0.83-0.88. That is the axis this script finds with gradient
cosines -- cross-recipe pairs, which differ in macro group, drop to 0.185, while pairs sharing the
recipe stay at 0.31-0.33 however independent their weights are. Two unrelated statistics picking out
macro_group_size is why the ensemble design varies head-side structure rather than adding more
identically-trained members.
"""

# ruff: noqa: PLC0415

import argparse
import itertools
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
SC = R / ".scratch"

DEFAULT_SETS = {
    "within": ["patch_critic_yam_s347_fixed_tau9_min_200k"],
    "same_recipe": [
        "patch_critic_yam_s347_fixed_tau9_min_200k",
        "patch_critic_yam_s347_fixed_tau9_noaug_200k",
        "patch_critic_yam_s347_fixed_200k",
        "patch_critic_yam_s347_fixed_nofloor_200k",
    ],
    "cross_recipe": [
        "patch_critic_yam_s347_fixed_tau9_min_200k",
        "patch_critic_yam_s347_g5_tau9_min",
        "patch_critic_yam_s347_g5_200k",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--states", type=int, default=128)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=SC / "extraction/diag_ensemble_independence.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.extraction import critic_q as cq

    names = sorted({n for v in DEFAULT_SETS.values() for n in v})
    critics = {}
    for n in names:
        d = SC / n
        if (d / "params.msgpack").exists():
            try:
                critics[n] = cq.load(d)
            except Exception as e:
                print(f"skip {n}: {e}")
    print(f"loaded {len(critics)} critics: {sorted(critics)}\n")

    ref = critics[DEFAULT_SETS["within"][0]]
    view = cq.CacheView(a.cache)
    H, AD = ref.config["horizon"], ref.config["action_dim"]
    eps = list(view.meta["episodes"].values())
    ep_start = np.empty(view.meta["N"], np.int64)
    ep_end = np.empty(view.meta["N"], np.int64)
    for e in eps:
        lo, hi = e["offset"], e["offset"] + e["full_len"]
        ep_start[lo:hi], ep_end[lo:hi] = lo, hi - 1
    ar = np.arange(H)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(view.meta["N"] - H - 1, a.states, replace=False))

    def chunks(rs):
        g = np.clip(rs[:, None] + ar[None], ep_start[rs][:, None], ep_end[rs][:, None])
        return np.asarray(view.actions[g.reshape(-1)]).reshape(len(rs), H, AD)

    feats, raw, prop = view.rows(rows, ref)
    F, P = jnp.asarray(feats), jnp.asarray(prop)
    demo = jnp.asarray(ref.pre.actions(chunks(rows), raw)[..., :AD])
    # candidate set for the ranking correlation: real chunks from other states, i.e. the same
    # `shuffle` family the CQL term uses and the width the selection-bias probe already characterised
    cand = jnp.stack([demo[rng.permutation(len(rows))] for _ in range(a.candidates)], 1)

    def member_fns(c):
        """Per-MEMBER Q and grad_A Q. critic_q's helpers reduce over the ensemble, which is exactly
        what has to be undone to compare members against each other."""

        def q_of(chunk, k):
            logits = c.net.apply({"params": c.params}, F, jnp.clip(chunk, -1.0, 1.0), P)
            return c.hl.from_logits(logits)[k][..., -1]

        return jax.jit(q_of, static_argnums=1), jax.jit(
            jax.grad(lambda ch, k: q_of(ch, k).sum(), argnums=0), static_argnums=1
        )

    # per (critic, member): Q over candidates [S, C], and grad at the demo chunk [S, H*AD]
    Q, G, tags = {}, {}, []
    for n, c in critics.items():
        qf, gf = member_fns(c)
        for k in range(c.config["num_critics"]):
            tag = f"{n}#m{k}"
            Q[tag] = np.stack([np.asarray(qf(cand[:, j], k)) for j in range(a.candidates)], 1)
            g = np.asarray(gf(demo, k)).reshape(len(rows), -1)
            G[tag] = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
            tags.append(tag)
        print(f"  scored {n}", flush=True)

    def pair_stats(t1, t2):
        q1, q2 = Q[t1], Q[t2]
        # correlation WITHIN each state across candidates, then averaged over states
        z1 = q1 - q1.mean(1, keepdims=True)
        z2 = q2 - q2.mean(1, keepdims=True)
        num = (z1 * z2).sum(1)
        den = np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1) + 1e-12
        return float(np.mean(num / den)), float(np.mean((G[t1] * G[t2]).sum(1)))

    res = {"states": len(rows), "candidates": a.candidates, "levels": {}}
    for level, group in DEFAULT_SETS.items():
        pairs = []
        if level == "within":
            for n in group:
                ks = [t for t in tags if t.startswith(n + "#")]
                pairs += list(itertools.combinations(ks, 2))
        else:
            avail = [n for n in group if n in critics]
            for n1, n2 in itertools.combinations(avail, 2):
                pairs.append((f"{n1}#m0", f"{n2}#m0"))
        qs, gs = [], []
        for t1, t2 in pairs:
            qc, gc = pair_stats(t1, t2)
            qs.append(qc)
            gs.append(gc)
        res["levels"][level] = {
            "n_pairs": len(pairs),
            "q_corr_mean": float(np.mean(qs)) if qs else None,
            "grad_cos_mean": float(np.mean(gs)) if gs else None,
            "grad_cos_min": float(np.min(gs)) if gs else None,
            "grad_cos_max": float(np.max(gs)) if gs else None,
            "pairs": [
                {"a": t1, "b": t2, "q_corr": q, "grad_cos": g} for (t1, t2), q, g in zip(pairs, qs, gs, strict=True)
            ],
        }
    # a random-direction floor: what cosine would two UNRELATED directions in this space give?
    d = H * AD
    res["random_direction_cos_scale"] = float(1.0 / np.sqrt(d))
    res["dim"] = int(d)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(f"\n{'level':14s} {'pairs':>5} {'Q corr (within-state)':>22} {'grad_A Q cosine':>18}")
    for level, v in res["levels"].items():
        if v["q_corr_mean"] is None:
            continue
        print(
            f"{level:14s} {v['n_pairs']:5d} {v['q_corr_mean']:22.3f} {v['grad_cos_mean']:18.3f}   [{v['grad_cos_min']:.3f}, {v['grad_cos_max']:.3f}]"
        )
    print(f"\nchance cosine for unrelated directions in {d} dims: ~{res['random_direction_cos_scale']:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
