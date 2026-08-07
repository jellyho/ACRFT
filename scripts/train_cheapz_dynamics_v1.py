"""Dynamics v1: the survey-settled recipe (macro-stride, causal transformer, NLL, rollout loss).

Upgrades over v0, each traced to a verified source (docs/overnight_report_2026-08-07.md §4.1):
  * macro-stride s=4: one model step = 4 control steps; a 16-chunk = 4 macro-steps with
    intermediate predictions at t+4/8/12 — required by the adaptive-commit rule, 4x less
    compounding than per-step, 4x more supervision than one-jump (DINO-WM frameskip precedent).
  * block-causal transformer over macro-steps, history H=3 (DINO-WM depth-6 ablation: removing
    the causal mask collapses 0.92->0.08; our 1 token/step makes it plain causal).
  * Gaussian NLL with softplus-bounded logvar (MBPO bnn.py trick) — per-member aleatoric head.
  * teacher-forcing + 2-step rollout loss at 1:1 (V-JEPA-2-AC's `loss = jloss + sloss`).
  * 5 fully independent members (init+shuffling only, no bootstrap), disagreement = variance of
    predicted MEANS (best-calibrated per 2110.04135); optional frozen random-prior nets
    (Osband 1806.03335) as the OOD-sharpening ablation.
  * NO VICReg/EMA/decoder: targets are frozen z — the collapse channel does not exist.

Eval per the instrument survey: open-loop error curve m=1..8 vs copy-forward (R^2), per-horizon
AUSE (sparsification; Spearman deprecated for calibration), OOD swap disagreement ratio.
"""

import argparse
import json
import logging
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DynV1(nn.Module):
    def __init__(self, zdim: int, act_seg: int, d: int = 288, depth: int = 4, heads: int = 8,
                 hist: int = 3, horizon_macro: int = 4, prior_scale: float = 0.0):
        super().__init__()
        self.hist, self.hm, self.zdim = hist, horizon_macro, zdim
        self.z_in = nn.Linear(zdim, d - 32)
        self.a_in = nn.Linear(act_seg, 32)
        L = hist + horizon_macro
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        self.head_mu = nn.Linear(d, zdim)
        self.head_lv = nn.Linear(d, zdim)
        self.max_lv, self.min_lv = nn.Parameter(torch.full((zdim,), 0.5)), nn.Parameter(torch.full((zdim,), -10.0))
        # frozen randomized prior (Osband): a fixed copy whose output is added to mu.
        self.prior_scale = prior_scale
        if prior_scale > 0:
            self.prior = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, zdim))
            for q in self.prior.parameters():
                q.requires_grad_(False)

    def forward(self, z_hist, a_seq):
        """z_hist [B, hist, zdim] (t-2s, t-s, t); a_seq [B, hm, act_seg] -> mu, lv [B, hm, zdim]."""
        B = z_hist.shape[0]
        # token m (m<hist): [z_hist_m | 0-action]; token hist+j: [z placeholder=last z | a_j]
        z_tok = self.z_in(z_hist)  # [B, hist, d-32]
        z_pad = z_tok[:, -1:].expand(B, self.hm, z_tok.shape[-1])
        a_tok = self.a_in(a_seq)  # [B, hm, 32]
        a_zero = torch.zeros(B, self.hist, 32, device=a_seq.device, dtype=a_tok.dtype)
        x = torch.cat([torch.cat([z_tok, a_zero], -1), torch.cat([z_pad, a_tok], -1)], dim=1)
        x = x + self.pos[None]
        h = self.tf(x, mask=self.mask.to(x.device))[:, self.hist:]  # [B, hm, d]
        mu = self.head_mu(h)
        if self.prior_scale > 0:
            mu = mu + self.prior_scale * self.prior(h)
        lv = self.max_lv - torch.nn.functional.softplus(self.max_lv - self.head_lv(h))
        lv = self.min_lv + torch.nn.functional.softplus(lv - self.min_lv)
        return mu, lv


def nll(mu, lv, tgt):
    return 0.5 * ((tgt - mu) ** 2 * torch.exp(-lv) + lv).mean()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-dir", type=pathlib.Path, default=pathlib.Path(".scratch/cheap_z_v4b"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/cheapz_dyn_v1"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--prior-scale", type=float, default=0.0)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.annot / "meta.json").read_text())
    n, H, A = meta["num_frames"], meta["horizon"], meta["action_dim"]
    s, hist = args.stride, args.hist
    hm = H // s  # macro steps per chunk
    z = np.load(args.z_dir / "z.npy").astype(np.float32)
    zmu, zsd = z.mean(0), z.std(0) + 1e-6
    z = (z - zmu) / zsd  # standardized (recipe)
    chunk = np.memmap(args.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))

    eps_u = np.unique(ep)
    rng.shuffle(eps_u)
    held = set(eps_u[: max(1, int(len(eps_u) * args.heldout_frac))].tolist())
    ep_start = np.zeros(n, dtype=np.int64); ep_end = np.zeros(n, dtype=np.int64)
    for e in eps_u:
        rows = np.flatnonzero(ep == e)
        ep_start[rows], ep_end[rows] = rows.min(), rows.max()
    # anchors need hist history at stride s behind AND a full chunk ahead
    ok = (np.arange(n) - (hist - 1) * s >= ep_start) & (np.arange(n) + H <= ep_end)
    train_rows = np.flatnonzero(ok & ~np.isin(ep, list(held)))
    held_rows = np.flatnonzero(ok & np.isin(ep, list(held)))
    logger.info(f"{n} frames | {len(train_rows)} train / {len(held_rows)} held anchors | hm={hm}")

    z_t = torch.from_numpy(z).to(dev)
    ch_t = torch.from_numpy(np.ascontiguousarray(chunk)).to(dev)

    def gather(rows):  # -> z_hist [B,hist,z], a_seq [B,hm,s*A], targets [B,hm,z]
        r = torch.as_tensor(rows, device=dev)
        hidx = r[:, None] + torch.arange(-(hist - 1) * s, 1, s, device=dev)[None]
        tidx = r[:, None] + torch.arange(s, H + 1, s, device=dev)[None]
        a = ch_t[r].view(len(rows), hm, s * A)
        return z_t[hidx], a, z_t[tidx]

    models = [DynV1(z.shape[1], s * A, hist=hist, horizon_macro=hm,
                    prior_scale=args.prior_scale).to(dev) for _ in range(args.members)]
    opt = torch.optim.AdamW([q for m in models for q in m.parameters() if q.requires_grad],
                            lr=args.lr, weight_decay=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    for step in range(args.steps):
        rows = rng.choice(train_rows, size=args.batch)
        zh, a, tgt = gather(rows)
        loss = torch.zeros((), device=dev)
        for i, m in enumerate(models):
            sl = slice(i * args.batch // args.members, (i + 1) * args.batch // args.members)
            mu, lv = m(zh[sl], a[sl])
            l_tf = nll(mu, lv, tgt[sl])
            # 2-step rollout (V-JEPA-2-AC): feed predicted z back as the newest history entry and
            # predict the SECOND macro-step again from it; one recurrent step differentiated.
            zh2 = torch.cat([zh[sl][:, 1:], mu[:, :1]], dim=1)
            mu2, lv2 = m(zh2, a[sl])
            l_ro = nll(mu2[:, :1], lv2[:, :1], tgt[sl][:, 1:2])
            loss = loss + l_tf + l_ro
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for m in models for q in m.parameters() if q.requires_grad], 20.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            logger.info(f"step {step}: loss {loss.item() / args.members:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"members": [m.state_dict() for m in models],
                "zmu": zmu, "zsd": zsd,
                "cfg": {k: getattr(args, k) for k in ("stride", "hist", "members", "prior_scale")}},
               args.out / "ensemble_v1.pt")

    # ---------------- eval: open-loop curve, AUSE, OOD swap ----------------
    sub = rng.choice(held_rows, size=min(3000, len(held_rows)), replace=False)
    zh, a, tgt = gather(sub)
    report = {"per_macro": {}}
    with torch.no_grad():
        mus = torch.stack([m(zh, a)[0] for m in models])  # [M, B, hm, z] teacher-forced multi-slot
    for j in range(hm):
        mp = mus[:, :, j]
        err = torch.norm(mp.mean(0) - tgt[:, j], dim=-1)
        base = torch.norm(zh[:, -1] - tgt[:, j], dim=-1)  # copy-forward
        dis = torch.var(mp, dim=0).sum(-1)
        # AUSE: sparsification — remove by disagreement vs by oracle error, compare mean-err curves.
        order_u = torch.argsort(dis, descending=True)
        order_e = torch.argsort(err, descending=True)
        fracs = torch.linspace(0, 0.9, 10)
        ause = 0.0
        for f in fracs:
            k = int(len(err) * (1 - f.item()))
            ause += (err[order_u[-k:]].mean() - err[order_e[-k:]].mean()).abs().item()
        ause /= (len(fracs) * err.mean().item())
        r2 = 1.0 - (err ** 2).mean().item() / (base ** 2).mean().item()
        report["per_macro"][int((j + 1) * s)] = {
            "err": err.mean().item(), "copyforward": base.mean().item(),
            "r2_vs_copyforward": r2, "ause": ause,
        }
    # OOD: swap chunks between anchors
    perm = torch.from_numpy(rng.permutation(len(sub))).to(dev)
    with torch.no_grad():
        mus_s = torch.stack([m(zh, a[perm])[0] for m in models])
    d_m = torch.var(mus, dim=0).sum(-1).mean().item()
    d_s = torch.var(mus_s, dim=0).sum(-1).mean().item()
    report["ood_swap_ratio"] = d_s / max(d_m, 1e-9)

    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'macro(+steps)':>13s} {'err':>7s} {'copyfwd':>8s} {'R2':>6s} {'AUSE':>6s}")
    for k, r in report["per_macro"].items():
        print(f"{k:>13} {r['err']:7.3f} {r['copyforward']:8.3f} {r['r2_vs_copyforward']:6.3f} {r['ause']:6.3f}")
    print(f"OOD swap disagreement ratio (want >1): {report['ood_swap_ratio']:.2f}")
    print("acceptance: R2(+4) > 0.6, AUSE <~ 0.2")


if __name__ == "__main__":
    main()
