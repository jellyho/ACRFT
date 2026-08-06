"""v0 chunk-conditioned latent dynamics on cheap-z, with the calibration tripwires.

Trains an ensemble d_i(z_t, action-chunk-prefix) -> z_{t+p} on cached cheap-z, one-step
in CHUNK space (never composed: the licensed regime is single-jump prediction,
re-anchored on real observations — WCM 2607.29613 / CheckVLA 2607.26789; open-loop
composition is where narrow-data models break, MBPO's own analysis).

Prefix-conditioned: one model handles every macro prefix p in {2,4,...,H} by masking the
chunk embedding past p and feeding p explicitly. This is what both model-based ranking
(score = V(d(z, a, p))) and uncertainty-gated commit (disagreement as a function of p)
need — commit length IS the argmax over p, so p must be an input, not a fixed constant.

Reports the design doc's tripwires on held-out episodes:
  * per-prefix open-loop error ||d(z,a,p) - z_{t+p}|| (should grow smoothly with p)
  * disagreement-vs-error Spearman per prefix (calibration: must be > 0)
  * OOD sanity: disagreement on shuffled (z, chunk) pairs vs matched pairs (must be higher)

Runs entirely on cached arrays: minutes per run.
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Dyn(nn.Module):
    """One ensemble member: (z, chunk, prefix) -> delta-z prediction.

    Predicts the RESIDUAL z_{t+p} - z_t: at small p the identity is a strong prior, and
    residual parametrization stops the model wasting capacity re-encoding z. The chunk
    enters through a shared MLP over the flattened (masked) chunk; prefix p enters as a
    normalized scalar so one member serves the whole prefix grid.
    """

    def __init__(self, zdim: int, h: int, adim: int, hidden: int = 512):
        super().__init__()
        self.chunk_in = nn.Sequential(nn.Linear(h * adim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(zdim + hidden + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, zdim),
        )

    def forward(self, z, chunk_masked, p_frac):
        c = self.chunk_in(chunk_masked.flatten(-2))
        return z + self.net(torch.cat([z, c, p_frac[:, None]], dim=-1))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-dir", type=pathlib.Path, default=pathlib.Path(".scratch/cheap_z_v4b"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/cheapz_dyn_v0"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--heldout-frac", type=float, default=0.15, help="episodes held out for calibration")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.annot / "meta.json").read_text())
    n, H, A = meta["num_frames"], meta["horizon"], meta["action_dim"]
    z = np.load(args.z_dir / "z.npy")  # [n, zdim]
    zdim = z.shape[1]
    chunk = np.memmap(args.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))

    eps_u = np.unique(ep)
    rng.shuffle(eps_u)
    n_held = max(1, int(len(eps_u) * args.heldout_frac))
    held = set(eps_u[:n_held].tolist())
    logger.info(f"{n} frames, {len(eps_u)} episodes ({n_held} held out), z {zdim}d, chunk {H}x{A}")

    # Valid anchors: t and t+H inside the same episode (so every prefix target exists).
    ep_end = np.zeros(int(eps_u.max()) + 1, dtype=np.int64)
    for e in eps_u:
        ep_end[e] = np.flatnonzero(ep == e).max()
    valid = np.flatnonzero((np.arange(n) + H) <= ep_end[ep])
    train_rows = valid[~np.isin(ep[valid], list(held))]
    held_rows = valid[np.isin(ep[valid], list(held))]

    z_t = torch.from_numpy(z).to(dev)
    ch_t = torch.from_numpy(np.ascontiguousarray(chunk)).to(dev)
    prefixes = list(range(2, H + 1, 2))

    models = [Dyn(zdim, H, A).to(dev) for _ in range(args.members)]
    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    def batch_forward(m, rows, pfx):
        zt = z_t[rows]
        c = ch_t[rows].clone()
        mask = torch.zeros(len(rows), H, 1, device=dev)
        for i, p in enumerate(pfx):
            mask[i, :p] = 1.0
        return m(zt, c * mask, torch.tensor(pfx, device=dev, dtype=torch.float32) / H)

    for step in range(args.steps):
        rows = torch.from_numpy(rng.choice(train_rows, size=args.batch)).to(dev)
        pfx = rng.choice(prefixes, size=args.batch)
        tgt = z_t[rows + torch.from_numpy(pfx).to(dev)]
        loss = torch.zeros((), device=dev)
        for i, m in enumerate(models):
            # independent minibatch slices per member: cheap decorrelation (bootstrap-lite)
            sl = slice(i * args.batch // args.members, (i + 1) * args.batch // args.members)
            pred = batch_forward(m, rows[sl], pfx[sl])
            loss = loss + ((pred - tgt[sl]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            logger.info(f"step {step}: loss {loss.item() / args.members:.5f}")

    # ------------------------------------------------------------------ calibration report
    from scipy.stats import spearmanr

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({f"m{i}": m.state_dict() for i, m in enumerate(models)}, args.out / "ensemble.pt")

    report = {"per_prefix": {}}
    sub = rng.choice(held_rows, size=min(4000, len(held_rows)), replace=False)
    rows = torch.from_numpy(sub).to(dev)
    for p in prefixes:
        pf = np.full(len(sub), p)
        tgt = z_t[rows + p]
        with torch.no_grad():
            preds = torch.stack([batch_forward(m, rows, pf) for m in models])  # [M,B,z]
        mean_pred = preds.mean(0)
        err = torch.norm(mean_pred - tgt, dim=-1)
        disag = torch.norm(preds - preds.mean(0, keepdim=True), dim=-1).mean(0)
        # baseline: predicting "no change" — the model must beat it to be informative
        base = torch.norm(z_t[rows] - tgt, dim=-1)
        rho = float(spearmanr(disag.cpu().numpy(), err.cpu().numpy()).statistic)
        report["per_prefix"][p] = {
            "err": float(err.mean()),
            "err_baseline_identity": float(base.mean()),
            "disagreement": float(disag.mean()),
            "spearman_disag_err": rho,
        }

    # OOD sanity: shuffled chunks must raise disagreement above matched chunks.
    p = H
    pf = np.full(len(sub), p)
    perm = torch.from_numpy(rng.permutation(len(sub))).to(dev)
    with torch.no_grad():
        d_match = torch.stack([batch_forward(m, rows, pf) for m in models])
        # shuffle: pair each z with another frame's chunk
        zt = z_t[rows]
        c = ch_t[rows][perm]
        d_shuf = torch.stack([m(zt, c, torch.full((len(sub),), 1.0, device=dev)) for m in models])
    dis_m = torch.norm(d_match - d_match.mean(0, keepdim=True), dim=-1).mean().item()
    dis_s = torch.norm(d_shuf - d_shuf.mean(0, keepdim=True), dim=-1).mean().item()
    report["ood_disagreement_ratio"] = dis_s / max(dis_m, 1e-9)

    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'prefix':>6s} {'err':>8s} {'id-base':>8s} {'disag':>8s} {'rho(d,e)':>9s}")
    for p, r in report["per_prefix"].items():
        print(f"{p:>6} {r['err']:8.4f} {r['err_baseline_identity']:8.4f} {r['disagreement']:8.4f} {r['spearman_disag_err']:9.3f}")
    print(f"OOD disagreement ratio (shuffled/matched, want >1): {report['ood_disagreement_ratio']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
