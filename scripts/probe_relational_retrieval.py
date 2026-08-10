"""Cross-episode retrieval probe: does an embedding bridge on RELATIONAL state or just stage?

Motivating example (user, 2026-08-10): two trajectories where background objects and absolute
block positions all differ, but the block<->gripper RELATIVE pose is similar at the pre-grasp
stage — stitching should ideally flow between them. We cannot read object poses from the
annotation cache, but the expert's action chunk is a witness of relational state: in a
near-Markov expert, similar relative geometry => similar commanded chunk. So:

  for query frames, retrieve kNN in the embedding restricted to OTHER episodes, and measure
  (a) demo-chunk agreement (cosine of z-scored flattened chunks)  -> relational-state proxy
  (b) |progress difference|                                       -> stage/phase proxy

against two baselines: random cross-episode pairs (chance) and progress-matched random
cross-episode pairs (what "bridging on stage alone" buys). An embedding whose neighbors beat
the progress-matched baseline on (a) is retrieving relational geometry beyond mere phase —
the property the user's example demands. If phi ~= progress-matched baseline, phi bridges
values across episodes but on the phase axis only, which is exactly compatible with
V being fine while candidate ranking stays blind.
"""

import argparse
import json
import pathlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_cache_PrepareCoffee"))
ap.add_argument("--phi-z", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout/z.npy"))
ap.add_argument("--refs", type=int, default=30000)
ap.add_argument("--queries", type=int, default=2000)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--progress-band", type=float, default=0.02)
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/probe_relational.json"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

meta = json.loads((a.annot / "meta.json").read_text())
n, H, A = meta["num_frames"], meta["horizon"], meta["action_dim"]
ep = np.array(np.memmap(a.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
ep_cache = np.load(a.cache / "episode_index.npy")
assert len(ep_cache) == n, "cache/annot length mismatch"
assert (ep_cache == ep).all(), "cache/annot row order mismatch"
prog = np.array(np.memmap(a.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))
chunk = np.array(np.memmap(a.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))).reshape(n, -1)
chunk = (chunk - chunk.mean(0)) / (chunk.std(0) + 1e-6)
chunk /= np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8

pm = np.asarray(meta["proprio_mean"], np.float32)
psd = np.asarray(meta["proprio_std"], np.float32)
prop = np.array(np.memmap(a.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, len(pm))))
prop = np.where(psd > 1e-6, (prop - pm) / np.where(psd > 1e-6, psd, 1.0), 0.0).astype(np.float32)

tok = np.load(a.cache / "features.npy").astype(np.float32)
tok = (tok - tok.mean(0)) / (tok.std(0) + 1e-6)
phi = np.load(a.phi_z).astype(np.float32)

ref = rng.choice(n, size=min(a.refs, n), replace=False)
qry = rng.choice(n, size=min(a.queries, n), replace=False)
ref_ep, qry_ep = ep[ref], ep[qry]
ref_chunk, qry_chunk = chunk[ref], chunk[qry]
ref_prog, qry_prog = prog[ref], prog[qry]


def knn_metrics(z, name):
    zr, zq = z[ref], z[qry]
    zr = zr / (np.linalg.norm(zr, axis=1, keepdims=True) + 1e-8)
    zq = zq / (np.linalg.norm(zq, axis=1, keepdims=True) + 1e-8)
    act_cos, dprog = [], []
    B = 256
    for i in range(0, len(zq), B):
        sim = zq[i : i + B] @ zr.T  # cosine
        sim[qry_ep[i : i + B, None] == ref_ep[None, :]] = -np.inf  # other episodes only
        top = np.argpartition(-sim, a.k, axis=1)[:, : a.k]
        act_cos.append((qry_chunk[i : i + B, None, :] * ref_chunk[top]).sum(-1).mean(1))
        dprog.append(np.abs(qry_prog[i : i + B, None] - ref_prog[top]).mean(1))
    act_cos, dprog = np.concatenate(act_cos), np.concatenate(dprog)
    r = {"act_cos": float(act_cos.mean()), "dprog": float(dprog.mean())}
    print(f"{name:>22}: neighbor act-cos {r['act_cos']:.3f}   |dprog| {r['dprog']:.3f}", flush=True)
    return r


res = {"spaces": {}}
res["spaces"]["raw_token_2048"] = knn_metrics(tok, "raw token 2048")
res["spaces"]["phi_128"] = knn_metrics(phi, "phi 128")
res["spaces"]["proprio_16"] = knn_metrics(prop, "proprio 16")
res["spaces"]["tok_plus_prop"] = knn_metrics(
    np.concatenate([tok, prop * np.sqrt(tok.shape[1] / prop.shape[1])], 1), "token+proprio"
)

# --- baselines: what does chance / stage-matching alone buy? ---
mask = qry_ep[:, None] != ref_ep[None, :]
rand_cos, band_cos, band_dprog, band_cnt = [], [], [], []
for i in range(len(qry)):
    ok = np.flatnonzero(mask[i])
    pick = rng.choice(ok, a.k, replace=False)
    rand_cos.append((qry_chunk[i] * ref_chunk[pick]).sum(-1).mean())
    near = ok[np.abs(ref_prog[ok] - qry_prog[i]) < a.progress_band]
    band_cnt.append(len(near))
    if len(near) >= a.k:
        pick = rng.choice(near, a.k, replace=False)
        band_cos.append((qry_chunk[i] * ref_chunk[pick]).sum(-1).mean())
        band_dprog.append(np.abs(ref_prog[pick] - qry_prog[i]).mean())
res["baselines"] = {
    "random_cross_ep": {"act_cos": float(np.mean(rand_cos))},
    "progress_matched": {
        "act_cos": float(np.mean(band_cos)),
        "dprog": float(np.mean(band_dprog)),
        "band": a.progress_band,
        "covered": len(band_cos),
    },
}
print(f"{'random cross-ep':>22}: act-cos {np.mean(rand_cos):.3f}")
print(
    f"{'progress-matched rand':>22}: act-cos {np.mean(band_cos):.3f}   |dprog| {np.mean(band_dprog):.3f}  (n={len(band_cos)})"
)

# same-progress + proprio-matched: the closest available surrogate for "same relative geometry"
prop_cos = []
for i in range(len(qry)):
    ok = np.flatnonzero(mask[i])
    near = ok[np.abs(ref_prog[ok] - qry_prog[i]) < a.progress_band]
    if len(near) < a.k:
        continue
    d = np.linalg.norm(prop[ref[near]] - prop[qry[i]], axis=1)
    pick = near[np.argsort(d)[: a.k]]
    prop_cos.append((qry_chunk[i] * ref_chunk[pick]).sum(-1).mean())
res["baselines"]["progress_and_proprio_matched"] = {"act_cos": float(np.mean(prop_cos)), "covered": len(prop_cos)}
print(f"{'prog+proprio matched':>22}: act-cos {np.mean(prop_cos):.3f}  (n={len(prop_cos)})")

res["cfg"] = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(a).items()}
a.out.write_text(json.dumps(res, indent=1))
print(f"-> {a.out}")
