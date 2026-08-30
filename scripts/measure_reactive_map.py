"""Reactive-map diagnostic: is a critic's k* split a real reactive-map, or the belief-shift leak?

Reports, per frame, the WINDOW-level leak of a trained (chunk-conditioned) patch-critic:

    Q_reg(o, a_{1:k}) = the critic's own per-prefix value              (chunk-conditioned -> leaky)
    Q_syn(o, a_{1:k}) = Σ γ^j r_j  +  γ^k V(o_{t+k})                    (analytic reward + boundary V)
    leak(t, k)        = Q_reg − Q_syn

This is the critic-side twin of paper 2's policy-side ΔL_val (Lazzati/Metelli/Levine 2608.02547): both
measure "how much does conditioning on the chunk change the value". By docs/reactive_commitment_value.md
Lemma 1 the leak equals I(s;chunk|o) cashed out in value units -- the state's non-Markovian content.

Honest scope: the boundary V here is the critic's OWN V, which is itself fit to the leaky Q, so this is
a WINDOW-level leak (the chunk-regression over the k-step window), not the full leak. The rigorous
version needs a chunk-free V_react trained by synthetic backup (docs/delayed_obs_critic.md). Reads the
critic's input_spec so it reproduces training preprocessing (pi05, proprio slice, homing).

    uv run python scripts/measure_reactive_map.py --critic <dir> --cache /data1/jellyho/pc_cache/yam_s347 \
        --outcomes .scratch/yam_outcomes_347.jsonl --homing-onsets .scratch/yam_homing_onsets.json \
        --out .scratch/reactive_map.json
"""

# ruff: noqa: SIM108, RUF003, PERF401  (readability in a probe script)
import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=None)
    ap.add_argument("--truncate-homing", choices=["all", "failure", "none"], default="all")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/reactive_map.json"))
    a = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp

    from openpi.patch_critic import preproc as critic_preproc
    from openpi.patch_critic import spec as critic_spec
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble
    from openpi.patch_critic.critic import PatchV

    cc, _ = critic_spec.load(a.critic)
    isp = cc.get("input_spec", {})
    H, gsz, atoms = cc["horizon"], cc["macro_group_size"], cc["num_atoms"]
    prefixes = list(range(gsz, H + 1, gsz))
    gamma, h_goal, v_min = cc["discount"], cc["h_goal"], cc["v_min"]

    pre = None
    if isp.get("normalization") == "pi05":
        ns = a.critic / isp.get("norm_stats_file", "pi05_norm_stats.json")
        pre = critic_preproc.Pi05Preproc(
            ref=np.asarray(isp["joint_delta_reference"], np.int64),
            stats=critic_preproc.load_norm_stats(ns if ns.exists() else isp["norm_stats"]),
            use_quantiles=bool(isp["use_quantiles"]),
            delta=isp["delta_mode"] == "joint",
        )
    pidx = isp.get("proprio_indices")
    pidx = None if pidx is None else np.asarray(pidx, np.int64)

    meta = json.loads((a.cache / "meta.json").read_text())
    N, npatch, emb, sd, ad = meta["N"], meta["npatch"], meta["emb"], meta["sd"], meta["ad"]
    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, npatch, emb))
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, sd))
    actions = np.memmap(a.cache / "action.dat", np.float32, "r", shape=(N, ad))
    outc = {
        int(json.loads(x)["episode"]): json.loads(x)["outcome"]
        for x in pathlib.Path(a.outcomes).read_text().splitlines()
        if x.strip()
    }
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets else None

    net = PatchCriticEnsemble(
        action_dim=ad, horizon=H, num_critics=cc["num_critics"], macro_group_size=gsz, num_atoms=atoms
    )
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    v_net = PatchV(num_atoms=atoms)
    v_params = flax.serialization.msgpack_restore((a.critic / "v_params.msgpack").read_bytes())
    centers = jnp.asarray(HLGauss(v_min, cc["v_max"], atoms).centers)
    pref = np.asarray(prefixes)

    @jax.jit
    def q_reg(p, chunk, s):  # per-prefix chunk-conditioned value  [B, P]
        out = net.apply(params, p.astype(jnp.float32), chunk, s)
        return jnp.mean(jnp.sum(jax.nn.softmax(out, -1) * centers, -1), 0)

    @jax.jit
    def v_of(p, s):  # boundary state value  [B]
        return jnp.sum(jax.nn.softmax(v_net.apply(v_params, p.astype(jnp.float32), s), -1) * centers, -1)

    def prep(chunk_raw, st_raw):
        ch = pre.actions(chunk_raw, st_raw) if pre is not None else chunk_raw
        stt = pre.state(st_raw) if pre is not None else st_raw
        if pidx is not None:
            stt = stt[..., pidx]
        return ch, stt

    # cost_to_goal analytic reward over a window from pos (success episodes have a goal region)
    def window_return(pos, eff, is_succ):
        # discounted sum of -1 per step up to each prefix boundary (0 inside the goal region of a success)
        out = np.zeros((len(pos), len(pref)), np.float32)
        goal_start = eff - h_goal
        for pi, k in enumerate(pref):
            steps = np.minimum(k, np.maximum(0, eff - pos))
            if is_succ:  # reward 0 once inside [goal_start, eff)
                paid = np.clip(np.minimum(pos + k, goal_start) - pos, 0, k)
            else:
                paid = steps
            # Σ_{j<paid} γ^j (-1)
            out[:, pi] = -(1 - gamma**paid) / (1 - gamma)
        return out

    recs = []
    ar = np.arange(H)
    for e_str, info in meta["episodes"].items():
        e = int(e_str)
        if e not in outc:
            continue
        full, off = info["full_len"], info["offset"]
        is_succ = outc[e] == "success"
        eff = full
        if (
            homing is not None
            and str(e) in homing
            and (a.truncate_homing == "all" or (a.truncate_homing == "failure" and not is_succ))
        ):
            eff = int(homing[str(e)]["homing_onset"])
        pos = np.arange(0, max(1, eff), a.stride)
        ef, es, ea = (
            np.asarray(feats[off : off + full]),
            np.asarray(states[off : off + full]),
            np.asarray(actions[off : off + full]),
        )
        for i in range(0, len(pos), a.batch):
            p = pos[i : i + a.batch]
            hp = np.clip(p[:, None] + ar[None], 0, eff - 1)
            chunk = ea[hp.reshape(-1)].reshape(len(p), H, ad)
            ch, st = prep(chunk, es[p])
            qr = np.asarray(q_reg(jnp.asarray(ef[p]), jnp.asarray(ch), jnp.asarray(st)))  # [b,P]
            # boundary V at pos+k (clamped), synthetic Q = window return + gamma^k V(boundary)
            wret = window_return(p, eff, is_succ)  # [b,P]
            qsyn = np.zeros_like(qr)
            for pi, k in enumerate(pref):
                bpos = np.minimum(p + k, eff - 1)
                _, stb = prep(ea[np.clip(bpos[:, None] + ar[None], 0, eff - 1)].reshape(len(p), H, ad), es[bpos])
                vb = np.asarray(v_of(jnp.asarray(ef[bpos]), jnp.asarray(stb)))
                reached = (p + k) >= (eff - h_goal) if is_succ else np.zeros(len(p), bool)
                qsyn[:, pi] = wret[:, pi] + np.where(reached, 0.0, gamma**k * vb)
            leak = qr - qsyn
            kstar = (np.argmax(qr, 1) + 1) * gsz
            for b in range(len(p)):
                recs.append(
                    {
                        "ep": e,
                        "outcome": "success" if is_succ else "failure",
                        "pos": int(p[b]),
                        "frac": float(p[b] / max(1, eff)),
                        "kstar": int(kstar[b]),
                        "leak_mean": float(leak[b].mean()),
                        "leak_full": float(leak[b, -1]),
                        "qreg_full": float(qr[b, -1]),
                        "qsyn_full": float(qsyn[b, -1]),
                    }
                )

    def col(rs, k):
        return np.array([r[k] for r in rs])

    succ = [r for r in recs if r["outcome"] == "success"]
    fail = [r for r in recs if r["outcome"] == "failure"]
    print(f"reactive-map: {a.critic.name}  |  {len(succ)} success / {len(fail)} failure frames")
    print("  (leak = Q_reg − Q_syn ; window-level proxy; > 0 means the chunk-conditioned value is optimistic)")
    for name, rs in (("success", succ), ("failure", fail)):
        if not rs:
            continue
        early = [r for r in rs if r["frac"] < 0.15]
        print(
            f"  {name:8s}: k*={col(rs, 'kstar').mean():5.1f}  leak(full)={col(rs, 'leak_full').mean():8.1f}  "
            f"leak@early={col(early, 'leak_full').mean() if early else float('nan'):8.1f}  "
            f"|corr(leak, k*)|={abs(np.corrcoef(col(rs, 'leak_full'), col(rs, 'kstar'))[0, 1]):.3f}"
        )
    # the decisive number: does the k* success/failure split track the leak?
    if succ and fail:
        print(
            f"\n  k* split (succ−fail) = {col(succ, 'kstar').mean() - col(fail, 'kstar').mean():+.2f}   "
            f"leak split (succ−fail) = {col(succ, 'leak_full').mean() - col(fail, 'leak_full').mean():+.1f}"
        )
        print(
            "  If the two splits move together, the k* divergence is contaminated by the leak (not a clean reactive-map)."
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "critic": str(a.critic),
                "config": {k: cc.get(k) for k in ("steps", "macro_group_size", "expectile", "git")},
                "n": len(recs),
                "frames": recs,
            },
            indent=1,
        )
    )
    print(f"\n  wrote {a.out}  ({len(recs)} frames)")


if __name__ == "__main__":
    main()
