"""Direct test of the user's multimodality hypothesis, WITHOUT the IQL scalar-backup bottleneck.

The idea: for a query state, retrieve its k nearest neighbours in an embedding from OTHER episodes,
at the same task stage (matched progress). Then look at those neighbours' ACTUAL eventual outcomes
(did their episode succeed). If the same embedding-state appears on both winning and losing
trajectories, the outcome distribution is bimodal -> the value target genuinely branches -> the
counterfactual / stitching the user is after EXISTS at that state. If neighbours are all-success or
all-failure, the embedding pins the outcome -> no branch.

Key contrast: does an embedding's neighbours mix outcomes MORE or LESS than a progress-matched-random
baseline (which mixes at the base rate)?
  - neighbours as mixed as random-stage  -> the embedding carries no outcome info beyond stage; the
    branch is real and unresolved (counterfactuals present, but the state doesn't determine outcome).
  - neighbours more outcome-consistent    -> the embedding predicts the outcome (control-relevant),
    so it resolves the ambiguity (less branch).

We run it for phi (episode-collapsed), raw token, PCA-128, proprio, vs the progress-matched-random
baseline. This measures the RETURN branch directly from data, not through a learned scalar critic.
"""

import argparse
import json
import pathlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
ap.add_argument("--phi-z", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout/z.npy"))
ap.add_argument("--refs", type=int, default=40000)
ap.add_argument("--queries", type=int, default=3000)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--progress-band", type=float, default=0.05)
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/probe_neighbor_return.json"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

meta = json.loads((a.annot / "meta.json").read_text())
n = meta["num_frames"]
ep = np.asarray(np.memmap(a.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
prog = np.asarray(np.memmap(a.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))
rew = np.asarray(np.memmap(a.annot / "reward.dat", dtype=np.float32, mode="r", shape=(n,)))

# eventual outcome per frame = did this frame's EPISODE end in success (terminal reward > 0)?
succ_ep = {}
for e in np.unique(ep):
    rows = np.flatnonzero(ep == e)
    succ_ep[int(e)] = float(rew[rows].max() > 0.5)
outcome = np.array([succ_ep[int(e)] for e in ep], np.float32)  # 1 = frame is on a successful episode
base_rate = float(outcome.mean())

# embeddings (all standardized)
raw = np.asarray(np.memmap(a.annot / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, meta["token_dim"])))
raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
phi = np.load(a.phi_z).astype(np.float32)
phi = (phi - phi.mean(0)) / (phi.std(0) + 1e-6)
sub = raw[rng.choice(n, min(60000, n), replace=False)]
_, _, vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
pca = raw @ vt[:128].T
pm = np.asarray(meta["proprio_mean"], np.float32)
psd = np.asarray(meta["proprio_std"], np.float32)
prop = np.asarray(np.memmap(a.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, len(pm))))
prop = np.where(psd > 1e-6, (prop - pm) / np.where(psd > 1e-6, psd, 1.0), 0.0).astype(np.float32)

ref = rng.choice(n, size=min(a.refs, n), replace=False)
qry = rng.choice(n, size=min(a.queries, n), replace=False)
ref_ep, qry_ep = ep[ref], ep[qry]
ref_prog, qry_prog = prog[ref], prog[qry]
ref_out = outcome[ref]


def neighbour_outcomes(z, name):
    """For each query, mean success among its k cross-episode NN; report outcome mixing."""
    zr = z[ref] / (np.linalg.norm(z[ref], axis=1, keepdims=True) + 1e-8)
    zq = z[qry] / (np.linalg.norm(z[qry], axis=1, keepdims=True) + 1e-8)
    nb_succ = []
    covered = 0
    for i in range(len(zq)):
        ok = np.flatnonzero((ref_ep != qry_ep[i]) & (np.abs(ref_prog - qry_prog[i]) < a.progress_band))
        if len(ok) < a.k:
            continue
        sim = zq[i] @ zr[ok].T
        top = ok[np.argpartition(-sim, a.k)[: a.k]]
        nb_succ.append(ref_out[top].mean())  # fraction of these NN on successful episodes
        covered += 1
    nb_succ = np.array(nb_succ)
    # outcome-mixing: 1 = perfectly mixed (0.5), 0 = pinned (all-succ or all-fail). Averaged.
    mix = float(np.mean(1.0 - np.abs(2 * nb_succ - 1.0)))
    # how often are a query's neighbours genuinely split (both outcomes present, 20-80%)?
    split = float(np.mean((nb_succ > 0.2) & (nb_succ < 0.8)))
    r = {"outcome_mix": mix, "split_frac": float(split), "neighbour_succ_std": float(nb_succ.std()), "covered": covered}
    print(
        f"{name:>16}: outcome-mix {mix:.3f}  split-frac {split:.3f}  nn-succ std {nb_succ.std():.3f}  (n={covered})",
        flush=True,
    )
    return r


# progress-matched-random baseline: neighbours picked at random within the same progress band
def random_baseline():
    nb = []
    for i in range(len(qry)):
        ok = np.flatnonzero((ref_ep != qry_ep[i]) & (np.abs(ref_prog - qry_prog[i]) < a.progress_band))
        if len(ok) < a.k:
            continue
        nb.append(ref_out[rng.choice(ok, a.k, replace=False)].mean())
    nb = np.array(nb)
    return {
        "outcome_mix": float(np.mean(1.0 - np.abs(2 * nb - 1.0))),
        "split_frac": float(np.mean((nb > 0.2) & (nb < 0.8))),
        "neighbour_succ_std": float(nb.std()),
        "covered": len(nb),
    }


res = {"base_success_rate": base_rate, "progress_band": a.progress_band, "k": a.k, "spaces": {}}
res["spaces"]["phi_128"] = neighbour_outcomes(phi, "phi 128")
res["spaces"]["raw_2048"] = neighbour_outcomes(raw, "raw 2048")
res["spaces"]["pca_128"] = neighbour_outcomes(pca, "pca 128")
res["spaces"]["proprio"] = neighbour_outcomes(prop, "proprio")
res["baseline_progress_random"] = random_baseline()
print(
    f"{'random@stage':>16}: outcome-mix {res['baseline_progress_random']['outcome_mix']:.3f}  "
    f"split-frac {res['baseline_progress_random']['split_frac']:.3f}  (base success rate {base_rate:.3f})"
)
a.out.write_text(json.dumps(res, indent=1))
print(f"-> {a.out}")
