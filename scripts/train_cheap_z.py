"""Train a cheap critic representation on cached frozen-DINOv2 features (no VLA, no video).

The candidate replacement for the VLA-extracted RL token: a small head over frozen DINOv2
patch-mean features, shaped by three terms whose combination is the only one our survey found
with component-wise evidence at (or near) a few-hundred-episode scale:

  L = L_VIP-I  +  1.0 * L_NCE  +  lambda_v * L_VICReg

  * L_VIP-I  (2210.00030) — the goal-conditioned value geometry: distance-to-goal in z shrinks
    along successful trajectories. IMPLEMENTATION NOTE: the paper's printed Eq.(6) has the
    distance signs flipped relative to the official facebookresearch/vip code; the printed form
    rewards collapse. This uses the code-true sign:
        (1-g) * ||z(o0) - z(goal)||  +  log mean_i exp( 1 + g*||z(o_k+1)-z(goal)|| - ||z(o_k)-z(goal)|| )
  * L_NCE — multi-positive InfoNCE where positives are (a) the SAME timestep seen from the other
    cameras (TCN 1704.06888, validated at ~133 sequences: buys view/appearance invariance) and
    (b) frames from OTHER episodes within +-0.05 of the anchor's progress. (b) is the term aimed
    square at our measured pathology (episode identity linearly decodable at ~100%, kNN
    neighbourhoods dominated by same-episode frames): it pulls "same task phase, different
    kitchen" together, which is what cross-trajectory stitching needs. Actions are deliberately
    NOT the pairing key (measured motor bias of PSE-style pairing).
  * L_VICReg (2105.04906, variance+covariance only) — the 1-D-collapse blocker: VIP geometry plus
    progress-matched positives jointly squeeze z toward the scalar progress axis (neural
    regression collapse, 2409.04180); the variance floor keeps the remaining dimensions alive.

Success criterion (decided in advance, from the probe baseline on the same frames):
  mc_return ridge R^2 >= 0.73 (parity with the VLA token)  AND  kNN same-episode purity << 0.42.

Runs entirely on the precomputed cache (.scratch/dino_cache_<task>): no video decode, no VLA.
~30k steps at batch 512 is minutes-to-an-hour on one GPU.
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Head(nn.Module):
    """Shared per-camera MLP -> per-cam embedding; frame embedding = mean over cameras.

    Per-camera embeddings exist so the view-consistency positives operate at the camera level;
    the frame-level z (what the critic would consume) averages them, which makes z itself
    view-ensembled for free.
    """

    def __init__(self, cam_dim: int, hidden: int = 512, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cam_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, feats):  # [..., C, cam_dim]
        zc = self.net(feats)  # [..., C, out]
        return zc, zc.mean(dim=-2)  # per-cam, frame-level


def vicreg(z, eps=1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    var = z.var(dim=0)
    v = F.relu(1.0 - torch.sqrt(var + eps)).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    c = (off**2).sum() / d
    return v + 0.1 * c


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(".scratch/dino_cache_PrepareCoffee"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"),
                    help="Annotation dir supplying per-frame progress/mc_return labels (indices align: both stride 1).")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/cheap_z_v1"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-eps", type=int, default=48, help="episodes per VIP batch")
    ap.add_argument("--nce-batch", type=int, default=256, help="anchor frames per NCE batch")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lambda-v", type=float, default=0.5)
    ap.add_argument("--prog-bin", type=float, default=0.05)
    ap.add_argument("--nce-x-weight", type=float, default=1.0, help="weight of the cross-episode term")
    ap.add_argument("--num-pos", type=int, default=1, help="cross-episode positives per anchor")
    ap.add_argument("--epadv-weight", type=float, default=0.0,
                    help="DANN-style episode adversary on z (gradient reversal); 0 = off")
    # HILP-style expectile-TD metric term (2402.15567): V(s,g) = -||phi(z_s)-phi(z_g)|| trained with
    # a TD backup. The stitching probe showed our MC/alignment losses do not compose across unseen
    # (s,g) pairs (rho_stitch 0.24 vs VLA-z 0.47) - exactly what the taxonomy (2401.11237) predicts:
    # only TD-bootstrapped objectives stitch. The 30% cross-episode goals are the stitching pressure.
    ap.add_argument("--hilp-weight", type=float, default=0.0, help="expectile-TD metric term; 0 = off")
    ap.add_argument("--hilp-tau", type=float, default=0.9, help="expectile")
    ap.add_argument("--hilp-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.cache / "meta.json").read_text())
    n, D, ncam = meta["num_frames"], meta["dim"], len(meta["cams"])
    cam_dim = D // ncam
    feats = np.load(args.cache / "features.npy", mmap_mode="r")  # [n, D] fp16
    ep = np.load(args.cache / "episode_index.npy")
    ameta = json.loads((args.annot / "meta.json").read_text())
    assert ameta["num_frames"] == n, "cache and annotation must cover the same frames"
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))

    # Episode row ranges (rows are dataset-ordered, episodes contiguous).
    eps_u, starts = np.unique(ep, return_index=True)
    ends = np.r_[starts[1:], n]
    ep_range = {int(e): (int(s), int(t)) for e, s, t in zip(eps_u, starts, ends)}
    logger.info(f"{n} frames, {len(eps_u)} episodes, {ncam} cams x {cam_dim}d")

    # Progress-bin index: bin -> row indices, for cross-episode positive lookup.
    nb = int(round(1.0 / args.prog_bin))
    bins = np.clip((prog / args.prog_bin).astype(np.int64), 0, nb - 1)
    bin_rows = [np.flatnonzero(bins == b) for b in range(nb)]

    # Whole cache on GPU if it fits (279k x 1152 fp16 = 640MB — easily).
    feats_t = torch.from_numpy(np.ascontiguousarray(feats)).to(dev)
    ep_t = torch.from_numpy(ep.astype(np.int64)).to(dev)

    head = Head(cam_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.hilp_weight > 0:
        # phi maps z -> metric space; gradient flows through the MAIN head too, so the TD geometry
        # shapes z itself (that is the point - stitching pressure on the representation).
        import copy
        head.hilp = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, args.hilp_dim)).to(dev)
        opt.add_param_group({"params": head.hilp.parameters()})
        tgt_head = copy.deepcopy(head)
        for q in tgt_head.parameters():
            q.requires_grad_(False)
        # episode start row per frame, for same-episode geometric-future goal sampling
        ep_start_of = np.zeros(n, dtype=np.int64)
        ep_end_of = np.zeros(n, dtype=np.int64)
        for e, (s0, t0) in ep_range.items():
            ep_start_of[s0:t0] = s0
            ep_end_of[s0:t0] = t0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def embed(rows_t):  # rows: LongTensor -> (per-cam [B,C,256], frame [B,256])
        f = feats_t[rows_t].float().view(-1, ncam, cam_dim)
        return head(f)

    for step in range(args.steps):
        # ---- VIP-I on (o0, ok, ok1, goal) from sampled episodes (all RoboCasa demos succeed).
        es = rng.choice(eps_u, size=args.batch_eps, replace=len(eps_u) < args.batch_eps)
        o0, okk, ok1, gg = [], [], [], []
        for e in es:
            s, t = ep_range[int(e)]
            k = int(rng.integers(s, t - 1))
            o0.append(s); okk.append(k); ok1.append(k + 1); gg.append(t - 1)
        rows = torch.tensor(o0 + okk + ok1 + gg, device=dev)
        _, z = embed(rows)
        B = args.batch_eps
        z0, zk, zk1, zg = z[:B], z[B:2*B], z[2*B:3*B], z[3*B:]
        d0 = torch.norm(z0 - zg, dim=-1)
        dk = torch.norm(zk - zg, dim=-1)
        dk1 = torch.norm(zk1 - zg, dim=-1)
        # code-true sign (see module docstring): NOT the paper's printed Eq. 6.
        l_vip = (1 - args.gamma) * d0.mean() + torch.logsumexp(1 + args.gamma * dk1 - dk, dim=0) - np.log(B)

        # ---- multi-positive InfoNCE: view positives + cross-episode progress-matched positives.
        a_rows = rng.integers(0, n, size=args.nce_batch)
        p_rows = np.empty((len(a_rows), args.num_pos), dtype=np.int64)
        for i, r in enumerate(a_rows):
            cand = bin_rows[bins[r]]
            other = cand[ep[cand] != ep[r]]
            pool = other if len(other) else cand
            p_rows[i] = rng.choice(pool, size=args.num_pos, replace=len(pool) < args.num_pos)
        ar = torch.from_numpy(a_rows).to(dev)
        pr = torch.from_numpy(p_rows.reshape(-1)).to(dev)
        zc_a, zf_a = embed(ar)   # [B,C,256], [B,256]
        _, zf_p = embed(pr)
        za = F.normalize(zf_a, dim=-1)
        zp = F.normalize(zf_p, dim=-1).view(len(a_rows), args.num_pos, -1)
        zc = F.normalize(zc_a, dim=-1)
        # multi-positive InfoNCE: mean over positives of -log p(pos | anchor).
        sim = za @ za.T / args.tau
        sim.fill_diagonal_(float("-inf"))
        pos = torch.einsum("bd,bpd->bp", za, zp) / args.tau  # [B, P]
        lse = torch.logsumexp(torch.cat([pos, sim], dim=1), dim=1, keepdim=True)
        l_nce_x = (lse - pos).mean()
        # view term: each camera embedding predicts its own frame among all frames in the batch.
        if ncam > 1:
            ci = int(rng.integers(0, ncam))
            view = zc[:, ci] @ za.T / args.tau
            l_nce_v = F.cross_entropy(view, torch.arange(len(ar), device=dev))
        else:  # single-source cache (e.g. the RLT token): no other view to be consistent with
            l_nce_v = torch.zeros((), device=dev)
        l_nce = args.nce_x_weight * l_nce_x + l_nce_v

        l_vc = vicreg(zf_a)
        loss = l_vip + l_nce + args.lambda_v * l_vc
        l_adv = torch.zeros((), device=dev)
        if args.epadv_weight > 0:
            # DANN gradient reversal: identity forward, -1x gradient into z (same construction as
            # the epadv term in pi0_rlt).  The adversary head is created lazily on first use.
            if not hasattr(head, "epadv"):
                head.epadv = nn.Sequential(
                    nn.Linear(256, 256), nn.GELU(), nn.Linear(256, int(eps_u.max()) + 1)
                ).to(dev)
                opt.add_param_group({"params": head.epadv.parameters()})
            z_rev = (1.0 + 1.0) * zf_a.detach() - 1.0 * zf_a
            logits_ep = head.epadv(z_rev)
            l_adv = F.cross_entropy(logits_ep, ep_t[ar])
            loss = loss + args.epadv_weight * l_adv

        l_hilp = torch.zeros((), device=dev)
        if args.hilp_weight > 0:
            B2 = 256
            srow = rng.integers(0, n - 1, size=B2)
            # keep s' inside the episode
            srow = np.where(srow + 1 < ep_end_of[srow], srow, srow - 1)
            nrow = srow + 1
            # goals: 20% g=s' / 50% geometric future in-episode / 30% random other episode
            u = rng.random(B2)
            grow = np.empty(B2, dtype=np.int64)
            fut = np.minimum(srow + rng.geometric(1 - args.gamma, size=B2), ep_end_of[srow] - 1)
            rnd = rng.integers(0, n, size=B2)
            grow = np.where(u < 0.2, nrow, np.where(u < 0.7, fut, rnd))
            si = torch.from_numpy(srow).to(dev); ni = torch.from_numpy(nrow).to(dev)
            gi = torch.from_numpy(grow).to(dev)
            _, zs = embed(si); _, zg = embed(gi)
            with torch.no_grad():
                f_n = tgt_head(feats_t[ni].float().view(-1, ncam, cam_dim))[1]
                f_g = tgt_head(feats_t[gi].float().view(-1, ncam, cam_dim))[1]
                v_next = -torch.norm(tgt_head.hilp(f_n) - tgt_head.hilp(f_g), dim=-1)
            v = -torch.norm(head.hilp(zs) - head.hilp(zg), dim=-1)
            not_goal = (si != gi).float()
            td = (-not_goal) + args.gamma * v_next * not_goal - v
            w = torch.abs(args.hilp_tau - (td < 0).float())
            l_hilp = (w * td ** 2).mean()
            loss = loss + args.hilp_weight * l_hilp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if args.hilp_weight > 0 and step % 5 == 0:
            with torch.no_grad():
                for q, p_ in zip(tgt_head.parameters(), head.parameters()):
                    q.mul_(0.995).add_(p_, alpha=0.005)

        if step % 1000 == 0:
            with torch.no_grad():
                s_ = torch.linalg.svdvals(zf_a - zf_a.mean(0))
                p_ = s_ / s_.sum()
                rankme = float(torch.exp(-(p_ * torch.log(p_ + 1e-9)).sum()))
            logger.info(
                f"step {step}: vip {l_vip.item():.3f} nce_x {l_nce_x.item():.3f} "
                f"nce_v {l_nce_v.item():.3f} vc {l_vc.item():.3f} adv {l_adv.item():.3f} hilp {l_hilp.item():.4f} rankme {rankme:.1f}"
            )

    # ---- encode everything, save, and probe with the SAME harness as the baseline.
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.out / "head.pt")
    zs = np.empty((n, 256), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, n, 4096):
            r = torch.arange(lo, min(lo + 4096, n), device=dev)
            zs[lo : lo + len(r)] = embed(r)[1].cpu().numpy()
    np.save(args.out / "z.npy", zs)

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from probe_cheap_z import probe

    rows = np.linspace(0, n - 1, 20000).astype(np.int64)
    mc = np.array(np.memmap(args.annot / "mc_return.dat", dtype=np.float32, mode="r", shape=(n,)))
    res = probe(zs[rows], ep[rows], mc[rows], prog[rows])
    (args.out / "probe.json").write_text(json.dumps(res, indent=1))
    logger.info(f"PROBE cheap_z: {res}")
    print(f"cheap_z  mc_R2 {res['mc_return_r2']:.3f}  prog_R2 {res['progress_r2']:.3f}  "
          f"ep_acc {res['episode_acc']:.3f}  purity {res['knn_purity']:.3f}")
    print("targets: mc_R2 >= 0.73 (VLA parity), purity << 0.42")


if __name__ == "__main__":
    main()
