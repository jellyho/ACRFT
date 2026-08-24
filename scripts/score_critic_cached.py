"""Score a patch-critic from the precomputed feature cache: success-vs-failure discrimination.

Uses the cache built by cache_patch_features.py, so no video decode and no DINOv2 forward -- scoring a
full 347-episode dataset takes seconds instead of an hour. Applies the same homing truncation the trainer
uses, so the frames scored are the frames the critic was trained on.

Reports, per episode, the ensemble-min full-horizon value along the episode, then the ROC-AUC of
success-vs-failure using two summaries (mean over the episode, and max = closest approach), plus the
value distribution's depth (how much mass the critic puts on the deep atoms) -- the diagnostic that
told us the critic was compressing everything into the top third of its support.
"""

import argparse
import json
import pathlib

import numpy as np


def roc_auc(pos, neg):
    """AUC = P(a random positive scores above a random negative), ties counted as 1/2."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=None)
    ap.add_argument(
        "--truncate-homing",
        choices=["all", "failure", "none"],
        default="all",
        help="must match how the critic was trained (see its config.json input_spec)",
    )
    ap.add_argument("--stride", type=int, default=10, help="frames to skip when scoring an episode")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="write per-episode records + aggregates to this JSON. The house rule is that report tables "
        "are recomputed from source JSON rather than transcribed, and stdout cannot be audited.",
    )
    a = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp

    from openpi.patch_critic import preproc as critic_preproc
    from openpi.patch_critic import spec as critic_spec
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    cc, _ = critic_spec.load(a.critic)
    # Scoring MUST reproduce the critic's training inputs. Reading them off its own input_spec is the
    # only way that stays true as the preprocessing changes: feeding a pi05-space critic raw absolute
    # actions produces confident, meaningless numbers rather than an error.
    isp = cc.get("input_spec", {})
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
    print(
        f"input space: {isp.get('normalization', 'raw')}  proprio={isp.get('proprio_dims', 'all')}"
        f"  homing={isp.get('truncate_homing', a.truncate_homing)}",
        flush=True,
    )
    meta = json.loads((a.cache / "meta.json").read_text())
    N, npatch, emb, sd, ad = meta["N"], meta["npatch"], meta["emb"], meta["sd"], meta["ad"]
    H, gsz, atoms = cc["horizon"], cc["macro_group_size"], cc["num_atoms"]

    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, npatch, emb))
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, sd))
    actions = np.memmap(a.cache / "action.dat", np.float32, "r", shape=(N, ad))

    outc = {}
    for line in pathlib.Path(a.outcomes).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outc[int(r["episode"])] = r["outcome"]
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets else None

    print(f"cache N={N} frames, {len(meta['episodes'])} episodes; scoring stride={a.stride}", flush=True)
    net = PatchCriticEnsemble(
        action_dim=ad, horizon=H, num_critics=cc["num_critics"], macro_group_size=gsz, num_atoms=atoms
    )
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(cc["v_min"], cc["v_max"], atoms)
    centers = jnp.asarray(hl.centers)

    from openpi.patch_critic.critic import PatchV

    v_net = PatchV(num_atoms=atoms)
    v_params = flax.serialization.msgpack_restore((a.critic / "v_params.msgpack").read_bytes())

    @jax.jit
    def value(p, chunk, s):
        pf = p.astype(jnp.float32)
        out = net.apply(params, pf, chunk, s)  # [K,B,P,atoms]
        prob = jax.nn.softmax(out, -1)
        qpref = jnp.mean(jnp.sum(prob * centers, -1), 0)  # [B,P] per-prefix value
        v = jnp.min(jnp.sum(prob[:, :, -1, :] * centers, -1), 0)  # ensemble-min at k=H
        deep = jnp.mean(jnp.sum(prob[:, :, -1, :] * (centers < 0.72 * cc["v_min"]), -1), 0)
        vs = jnp.sum(jax.nn.softmax(v_net.apply(v_params, pf, s), -1) * centers, -1)  # state value
        return v, deep, qpref, vs

    succ_stats, fail_stats = [], []
    ar = np.arange(H)
    import time as _t

    _t0 = _t.time()
    _done = 0
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
        # An episode is CONTIGUOUS in the cache, so read its block once (sequential) and subsample in
        # memory. Scattered memmap indexing issues one NFS read per frame and is ~50x slower.
        ep_feats = np.asarray(feats[off : off + full])
        ep_states = np.asarray(states[off : off + full])
        ep_actions = np.asarray(actions[off : off + full])
        vs, ds, qps, svs = [], [], [], []
        for i in range(0, len(pos), a.batch):
            p = pos[i : i + a.batch]
            hp = p[:, None] + ar[None]
            # Clamp to eff (not full) and HOLD the last valid action -- the training convention. No
            # zero-fill: in the normalized space the zero vector is not a "no motion" action.
            ch = ep_actions[np.clip(hp, 0, eff - 1).reshape(-1)].reshape(len(p), H, ad)
            st = ep_states[p]
            if pre is not None:
                ch = pre.actions(ch, st)
                st = pre.state(st)
            if pidx is not None:
                st = st[..., pidx]
            v, d, qp, sv = value(jnp.asarray(ep_feats[p]), jnp.asarray(ch), jnp.asarray(st))
            vs.append(np.asarray(v))
            ds.append(np.asarray(d))
            qps.append(np.asarray(qp))
            svs.append(np.asarray(sv))
        vs = np.concatenate(vs)
        ds = np.concatenate(ds)
        qps, svs = np.concatenate(qps), np.concatenate(svs)
        # Which commitment length the selector would pick, and how well the learned state value tracks
        # the TRUE cost_to_goal slope. All prefixes tie exactly under a correct V (Bellman consistency),
        # so a short-biased argmax is a symptom of V under-rising, not of the selector.
        kbest = float(np.mean((np.argmax(qps, axis=1) + 1) * gsz))
        # Only defined for SUCCESSES: cost_to_goal has no goal to count down to on a failure, whose
        # true value is the floor throughout, so the reference slope there is ~0 and the ratio is noise.
        slope_rec = float("nan")
        if is_succ:
            n_to_goal = np.maximum(0.0, (eff - cc["h_goal"]) - pos.astype(np.float64))
            v_true = -(1.0 - cc["discount"] ** n_to_goal) / (1.0 - cc["discount"])
            d_learn, d_true = np.diff(svs.astype(np.float64)), np.diff(v_true)
            ok = np.abs(d_true) > 1e-9
            if ok.any():
                slope_rec = float(np.mean(d_learn[ok] / d_true[ok]))
        rec = {
            "ep": e,
            "mean": float(vs.mean()),
            "max": float(vs.max()),
            "min": float(vs.min()),
            "first": float(vs[0]),
            "last": float(vs[-1]),
            "deep": float(ds.mean()),
            "n": len(vs),
            "outcome": "success" if is_succ else "failure",
            "kbest": kbest,
            "vfirst": float(svs[0]),
            "vmean": float(svs.mean()),
            "slope": slope_rec,
        }
        (succ_stats if is_succ else fail_stats).append(rec)
        _done += 1
        if _done % 10 == 0:
            print(f"  scored {_done} episodes  ({_t.time() - _t0:.0f}s)", flush=True)

    def col(rs, k):
        return np.array([r[k] for r in rs])

    print(f"critic {a.critic.name}  |  {len(succ_stats)} success / {len(fail_stats)} fail episodes")
    print(f"  AUC(mean value) = {roc_auc(col(succ_stats, 'mean'), col(fail_stats, 'mean')):.4f}")
    print(f"  AUC(max value)  = {roc_auc(col(succ_stats, 'max'), col(fail_stats, 'max')):.4f}")
    print(f"  AUC(last value) = {roc_auc(col(succ_stats, 'last'), col(fail_stats, 'last')):.4f}")
    for name, rs in (("success", succ_stats), ("fail", fail_stats)):
        if rs:
            slope = f"{np.nanmean(col(rs, 'slope')):5.2f}" if name == "success" else "  n/a"
            print(
                f"  {name:8s}: k*={col(rs, 'kbest').mean():5.1f}  Vslope={slope}  "
                f"V(s0)={col(rs, 'vfirst').mean():8.1f}  "
                f"first {col(rs, 'first').mean():9.1f}  last {col(rs, 'last').mean():9.1f} "
                f" mean {col(rs, 'mean').mean():9.1f}  min {col(rs, 'min').mean():9.1f} "
                f" deep-atom mass {col(rs, 'deep').mean():.3f}"
            )
    print(f"  (v_min = {cc['v_min']:.1f}; 'deep-atom mass' = probability below 0.72*v_min)")

    if a.out is not None:

        def agg(rs, k, *, nan=False):
            if not rs:
                return None
            v = col(rs, k)
            return float(np.nanmean(v) if nan else v.mean())

        # V(s0) is the sharp diagnostic: both classes start from visually identical frames, so a gap
        # here is hindsight leakage rather than prediction. Record the behaviour-policy value the two
        # SHOULD share, so a reader can size the gap without recomputing it.
        p_succ = len(succ_stats) / max(1, len(succ_stats) + len(fail_stats))
        out = {
            "critic": str(a.critic),
            "critic_name": a.critic.name,
            "cache": str(a.cache),
            "stride": a.stride,
            "truncate_homing": a.truncate_homing,
            "config": {
                k: cc.get(k)
                for k in (
                    "steps",
                    "saved_at_step",
                    "batch",
                    "macro_group_size",
                    "expectile",
                    "discount",
                    "v_min",
                    "lr",
                    "h_goal",
                    "mc_floor",
                    "num_critics",
                    "horizon",
                    "git",
                    "loader",
                )
            },
            "input_spec": isp,
            "counts": {"success": len(succ_stats), "failure": len(fail_stats)},
            "auc": {
                "mean": roc_auc(col(succ_stats, "mean"), col(fail_stats, "mean")),
                "max": roc_auc(col(succ_stats, "max"), col(fail_stats, "max")),
                "last": roc_auc(col(succ_stats, "last"), col(fail_stats, "last")),
            },
            "aggregates": {
                name: {
                    "kbest": agg(rs, "kbest"),
                    "vslope": agg(rs, "slope", nan=True),
                    "vfirst": agg(rs, "vfirst"),
                    "first": agg(rs, "first"),
                    "last": agg(rs, "last"),
                    "mean": agg(rs, "mean"),
                    "min": agg(rs, "min"),
                    "deep": agg(rs, "deep"),
                }
                for name, rs in (("success", succ_stats), ("failure", fail_stats))
            },
            "p_success": p_succ,
            "episodes": sorted(succ_stats + fail_stats, key=lambda r: r["ep"]),
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        print(f"  wrote {a.out}  ({len(out['episodes'])} episode records)")


if __name__ == "__main__":
    main()
