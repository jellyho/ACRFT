"""Stitching probe: can a value/distance learned on HALF-trajectories answer cross-half queries?

Construction (from the survey's synthesis of OGBench-stitch 2410.20092 + Ghugare 2401.11237):
episodes are split into sets A and B; a distance head d(z_s, z_g) is trained ONLY on
within-episode pairs drawn from {A first-halves} ∪ {B second-halves} — so the model never sees
an A-late or B-early state, and never any cross-episode pair. Test queries:

  tier 1  within-episode held-out pairs                       (sanity — should be easy)
  tier 2  s from A-first-half, g from B-second-half           (THE stitch probe: state and goal
                                                               never co-observed, different demos)
  tier 3  cross-episode pairs at matched progress (|dp|~0)    (episode-identity leakage: d should
                                                               be ~0; large d = the bypass ratio
                                                               in goal-conditioned form)

Metrics: Spearman(d, |dp|) per tier (VOC bands: >=0.7 strong / 0.4 moderate / <0.3 fails to
stitch), the ratio rho_tier2/rho_tier1 (healthy >= ~0.8), and tier-3 mean d vs tier-2 mean d at
matched |dp|. Run for every representation on the SAME episode split so rows are comparable.

Everything runs on cached arrays (cheap-z z.npy and/or VLA rl_token.dat); CPU-fast per rep.
"""

import argparse
import json
import logging
import pathlib

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DistHead(nn.Module):
    """Small symmetric distance d(z_s, z_g) = ||phi(z_s) - phi(z_g)|| (HILP-style metric form)."""

    def __init__(self, zdim: int, out: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(zdim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, out))

    def forward(self, zs, zg):
        return torch.norm(self.phi(zs) - self.phi(zg), dim=-1)


def train_and_probe(z, ep, prog, seed=0, steps=8000, batch=1024, dev="cpu"):
    rng = np.random.default_rng(seed)
    eps_u = np.unique(ep)
    rng.shuffle(eps_u)
    A, _B = set(eps_u[: len(eps_u) // 2].tolist()), set(eps_u[len(eps_u) // 2 :].tolist())

    # partition rows: A-first-half / A-second / B-first / B-second, by within-episode progress
    half = prog < 0.5
    inA = np.isin(ep, list(A))
    a1 = np.flatnonzero(inA & half)
    b2 = np.flatnonzero(~inA & ~half)
    a2 = np.flatnonzero(inA & ~half)
    b1 = np.flatnonzero(~inA & half)
    train_rows = np.concatenate([a1, b2])

    # train pairs: within-episode only, both endpoints inside the allowed halves
    by_ep = {}
    for r in train_rows:
        by_ep.setdefault(int(ep[r]), []).append(r)
    by_ep = {e: np.array(v) for e, v in by_ep.items() if len(v) >= 8}
    train_eps = list(by_ep)

    zt = torch.from_numpy(z).float().to(dev)
    pg = torch.from_numpy(prog).float().to(dev)
    net = DistHead(z.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)

    for _step in range(steps):
        es = rng.choice(train_eps, size=batch)
        s_idx = np.empty(batch, dtype=np.int64)
        g_idx = np.empty(batch, dtype=np.int64)
        for i, e in enumerate(es):
            rows = by_ep[int(e)]
            s_idx[i], g_idx[i] = rng.choice(rows, size=2, replace=False)
        si = torch.from_numpy(s_idx).to(dev)
        gi = torch.from_numpy(g_idx).to(dev)
        d = net(zt[si], zt[gi])
        tgt = (pg[si] - pg[gi]).abs()
        loss = ((d - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def tier_pairs(src, dst, k=20000, match_dp=None):
        s = rng.choice(src, size=k)
        g = rng.choice(dst, size=k)
        if match_dp is not None:
            keep = np.abs(prog[s] - prog[g]) < match_dp
            s, g = s[keep], g[keep]
        return s, g

    def rho(s, g):
        with torch.no_grad():
            d = net(zt[s], zt[g]).cpu().numpy()
        dp = np.abs(prog[s] - prog[g])
        return float(spearmanr(d, dp).statistic), float(d.mean()), dp

    # tier 1: within-episode held-out pairs (from A2/B1 episodes' own halves — unseen states,
    # same-episode structure)
    t1_pool = np.concatenate([a2, b1])
    by_ep_t1 = {}
    for r in t1_pool:
        by_ep_t1.setdefault(int(ep[r]), []).append(r)
    s1, g1 = [], []
    for e, v in by_ep_t1.items():  # noqa: B007
        v = np.array(v)  # noqa: PLW2901
        if len(v) < 8:
            continue
        idx = rng.choice(v, size=(min(40, len(v) // 2), 2))
        s1.append(idx[:, 0])
        g1.append(idx[:, 1])
    s1, g1 = np.concatenate(s1), np.concatenate(g1)
    rho1, _, _ = rho(s1, g1)

    # tier 2: A-first-half -> B-second-half (never co-observed, cross-episode)
    s2, g2 = tier_pairs(a1, b2)
    rho2, d2_mean, dp2 = rho(s2, g2)

    # tier 3: cross-episode matched-progress (|dp|<0.05): d should be small
    s3, g3 = tier_pairs(np.concatenate([a1, b2]), np.concatenate([b2, a1]), match_dp=0.05)
    cross_ep = ep[s3] != ep[g3]
    s3, g3 = s3[cross_ep], g3[cross_ep]
    with torch.no_grad():
        d3 = net(zt[s3], zt[g3]).cpu().numpy()
    # matched-|dp| tier-2 reference: restrict tier2 to same dp band for a fair mean comparison
    band = dp2 < 0.05
    with torch.no_grad():
        d2_band = net(zt[s2[band]], zt[g2[band]]).cpu().numpy() if band.sum() else np.array([np.nan])

    return {
        "rho_tier1_within": rho1,
        "rho_tier2_stitch": rho2,
        "stitch_ratio": rho2 / max(rho1, 1e-9),
        "tier3_meand_matched_progress": float(d3.mean()),
        "tier2_meand_same_band": float(np.nanmean(d2_band)),
        "episode_leakage_ratio": float(d3.mean() / max(np.nanmean(d2_band), 1e-9)),
        "n_pairs": {"t1": len(s1), "t2": len(s2), "t3": len(s3)},
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reps",
        nargs="+",
        required=True,
        help="name=path pairs; path is a dir with z.npy OR an annot dir with rl_token.dat",
    )
    ap.add_argument(
        "--annot",
        type=pathlib.Path,
        default=pathlib.Path(".scratch/annot_noprop"),
        help="labels source (episode_index, progress)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/probe_stitching.json"))
    args = ap.parse_args()

    meta = json.loads((args.annot / "meta.json").read_text())
    n = meta["num_frames"]
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for spec in args.reps:
        name, _, p = spec.partition("=")
        p = pathlib.Path(p)
        if (p / "z.npy").exists():
            z = np.load(p / "z.npy")
        else:
            m2 = json.loads((p / "meta.json").read_text())
            z = np.array(np.memmap(p / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, m2["token_dim"])))
        logger.info(f"probing {name} ({z.shape[1]}d) on {dev}")
        results[name] = train_and_probe(z, ep, prog, seed=args.seed, dev=dev)
        logger.info(f"{name}: {results[name]}")

    args.out.write_text(json.dumps(results, indent=1))
    hdr = f"{'rep':14s} {'rho_within':>10s} {'rho_stitch':>10s} {'ratio':>6s} {'leakage':>8s}"
    print(hdr)
    for name, r in results.items():
        print(
            f"{name:14s} {r['rho_tier1_within']:10.3f} {r['rho_tier2_stitch']:10.3f} "
            f"{r['stitch_ratio']:6.2f} {r['episode_leakage_ratio']:8.2f}"
        )
    print("(bands: rho>=0.7 strong, 0.4 moderate, <0.3 fails; ratio>=0.8 healthy; leakage ~1 good, >>1 bad)")


if __name__ == "__main__":
    main()
