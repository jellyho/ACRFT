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
    ap.add_argument("--stride", type=int, default=10, help="frames to skip when scoring an episode")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp

    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    cc = json.loads((a.critic / "config.json").read_text())
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

    net = PatchCriticEnsemble(
        action_dim=ad, horizon=H, num_critics=cc["num_critics"], macro_group_size=gsz, num_atoms=atoms
    )
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(cc["v_min"], cc["v_max"], atoms)
    centers = jnp.asarray(hl.centers)

    @jax.jit
    def value(p, chunk, s):
        out = net.apply(params, p.astype(jnp.float32), chunk, s)  # [K,B,P,atoms]
        v = jnp.sum(jax.nn.softmax(out[:, :, -1, :], -1) * centers, -1)  # full-horizon prefix
        deep = jnp.sum(jax.nn.softmax(out[:, :, -1, :], -1) * (centers < 0.72 * cc["v_min"]), -1)
        return jnp.min(v, 0), jnp.mean(deep, 0)  # ensemble-min value, deep-atom mass

    succ_stats, fail_stats = [], []
    ar = np.arange(H)
    for e_str, info in meta["episodes"].items():
        e = int(e_str)
        if e not in outc:
            continue
        full, off = info["full_len"], info["offset"]
        is_succ = outc[e] == "success"
        eff = full
        if not is_succ and homing is not None and str(e) in homing:
            eff = int(homing[str(e)]["homing_onset"])
        pos = np.arange(0, max(1, eff), a.stride)
        vs, ds = [], []
        for i in range(0, len(pos), a.batch):
            p = pos[i : i + a.batch]
            g = off + p
            hp = p[:, None] + ar[None]
            gch = off + np.clip(hp, 0, full - 1)
            ch = np.asarray(actions[gch.reshape(-1)]).reshape(len(p), H, ad)
            ch[hp >= full] = 0.0
            v, d = value(jnp.asarray(np.asarray(feats[g])), jnp.asarray(ch), jnp.asarray(np.asarray(states[g])))
            vs.append(np.asarray(v))
            ds.append(np.asarray(d))
        vs = np.concatenate(vs)
        ds = np.concatenate(ds)
        rec = {
            "ep": e,
            "mean": float(vs.mean()),
            "max": float(vs.max()),
            "min": float(vs.min()),
            "first": float(vs[0]),
            "last": float(vs[-1]),
            "deep": float(ds.mean()),
            "n": len(vs),
        }
        (succ_stats if is_succ else fail_stats).append(rec)

    def col(rs, k):
        return np.array([r[k] for r in rs])

    print(f"critic {a.critic.name}  |  {len(succ_stats)} success / {len(fail_stats)} fail episodes")
    print(f"  AUC(mean value) = {roc_auc(col(succ_stats,'mean'), col(fail_stats,'mean')):.4f}")
    print(f"  AUC(max value)  = {roc_auc(col(succ_stats,'max'),  col(fail_stats,'max')):.4f}")
    print(f"  AUC(last value) = {roc_auc(col(succ_stats,'last'), col(fail_stats,'last')):.4f}")
    for name, rs in (("success", succ_stats), ("fail", fail_stats)):
        if rs:
            print(
                f"  {name:8s}: first {col(rs,'first').mean():9.1f}  last {col(rs,'last').mean():9.1f} "
                f" mean {col(rs,'mean').mean():9.1f}  min {col(rs,'min').mean():9.1f} "
                f" deep-atom mass {col(rs,'deep').mean():.3f}"
            )
    print(f"  (v_min = {cc['v_min']:.1f}; 'deep-atom mass' = probability below 0.72*v_min)")


if __name__ == "__main__":
    main()
