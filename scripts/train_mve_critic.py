"""In-sample MAC critic ("MVE critic"): model-based value expansion without a policy.

The user's spec: no policy sampling, good per-candidate resolution, no overestimation.

    V:  chunk-level IQL on phi only - in-sample expectile, never queries an action the
        dataset didn't execute. phi-only so it can be evaluated at IMAGINED states.
    Q:  distilled from one-chunk model-based expansion targets
            y_demo = r_n + gamma^H * V(phi_{t+H})                     (real transition)
            y_i    = rhat(phi, a_i) + gamma^H * agg_m V(phihat_m(phi, a_i))   (stored candidates)
        agg = min over the 5 dynamics members (LEQ-flavoured conservatism, no penalty
        coefficient). Candidates are the VLA's own frozen samples, so no OOD action is
        ever queried (MAC / IQL-TD-MPC n_r=0 principle).

Sparse-reward closed form: mc_t = gamma^(T-t) => the chunk contains the success iff
mc_t > gamma^H, and then r_n = mc_t exactly (discounted success inside the window).

Everything lives in the dynamics model's standardized phi space.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_cheapz_dynamics_v1 import DynV1  # noqa: E402


def mlp(inp, out, hidden=512, depth=3):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--z", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout/z.npy"))
    ap.add_argument("--dyn", type=pathlib.Path, default=pathlib.Path(".scratch/phi_dyn_v1_h1/ensemble_v1.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/critic_mve"))
    ap.add_argument("--v-steps", type=int, default=50000)
    ap.add_argument("--q-steps", type=int, default=100000)
    ap.add_argument("--r-steps", type=int, default=30000)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--img-agg", choices=["min", "mean"], default="min")
    ap.add_argument("--no-reward-head", action="store_true",
                    help="Candidate targets use gamma^H*V(phihat) only - isolates the reward head's contribution.")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.annot / "meta.json").read_text())
    n, H, A, N = meta["num_frames"], meta["horizon"], meta["action_dim"], meta["num_samples"]
    g = meta["discount"]
    ck = torch.load(args.dyn, map_location="cpu", weights_only=False)
    zmu, zsd = np.asarray(ck["zmu"], np.float32), np.asarray(ck["zsd"], np.float32)
    cfg = ck["cfg"]
    s, hist = cfg["stride"], cfg["hist"]
    hm = H // s

    phi = (np.load(args.z).astype(np.float32) - zmu) / zsd
    D = phi.shape[1]
    mm = lambda name, shape, dt=np.float32: np.array(np.memmap(args.annot / name, dtype=dt, mode="r", shape=shape))
    demo = mm("action_chunk.dat", (n, H, A))
    cand = mm("base_action.dat", (n, N, H, A))
    ep = mm("episode_index.dat", (n,), np.int32)
    mc = mm("mc_return.dat", (n,))
    pro = mm("proprio.dat", (n, len(meta["proprio_mean"])))
    pmu, psd = np.asarray(meta["proprio_mean"], np.float32), np.asarray(meta["proprio_std"], np.float32)
    pro = np.where(psd > 1e-6, (pro - pmu) / np.where(psd > 1e-6, psd, 1.0), 0.0).astype(np.float32)
    P = pro.shape[1]

    eps_u = np.unique(ep)
    rng.shuffle(eps_u)
    held = set(eps_u[: max(1, int(len(eps_u) * args.heldout_frac))].tolist())
    ep_end = np.zeros(n, dtype=np.int64)
    for e in eps_u:
        rows = np.flatnonzero(ep == e)
        ep_end[rows] = rows.max()
    # valid anchors: room for a full chunk (next state) inside the episode OR terminal within it
    nxt = np.arange(n) + H
    term = mc > g ** H                          # success lands inside this chunk
    ok = (nxt <= ep_end) | term
    train_rows = np.flatnonzero(ok & ~np.isin(ep, list(held)))
    held_rows = np.flatnonzero(ok & np.isin(ep, list(held)))
    r_n = np.where(term, mc, 0.0).astype(np.float32)
    nxt = np.minimum(nxt, ep_end)
    print(f"{n} frames | {len(train_rows)} train / {len(held_rows)} held anchors | terminal-in-chunk {term.mean():.3f}")

    phi_t = torch.from_numpy(phi).to(dev)
    pro_t = torch.from_numpy(pro).to(dev)
    demo_t = torch.from_numpy(demo.reshape(n, H * A)).to(dev)
    rn_t = torch.from_numpy(r_n).to(dev)
    mc_t = torch.from_numpy(mc.astype(np.float32)).to(dev)
    nxt_t = torch.from_numpy(nxt).to(dev)
    term_t = torch.from_numpy(term).to(dev)
    tr = torch.from_numpy(train_rows).to(dev)

    # ---------------- stage 1: in-sample IQL (V phi-only, Qd for the expectile) ----------------
    V = mlp(D, 1).to(dev)
    Vt = mlp(D, 1).to(dev)
    Vt.load_state_dict(V.state_dict())
    Qd = mlp(D + H * A, 1).to(dev)
    opt = torch.optim.AdamW(list(V.parameters()) + list(Qd.parameters()), lr=3e-4)
    for step in range(args.v_steps):
        r = tr[torch.randint(len(tr), (args.batch,), device=dev)]
        q_tgt = rn_t[r] + (~term_t[r]) * (g ** H) * Vt(phi_t[nxt_t[r]]).squeeze(-1).detach()
        q = Qd(torch.cat([phi_t[r], demo_t[r]], -1)).squeeze(-1)
        lq = ((q - q_tgt) ** 2).mean()
        adv = Qd(torch.cat([phi_t[r], demo_t[r]], -1)).squeeze(-1).detach() - V(phi_t[r]).squeeze(-1)
        lv = (torch.abs(args.tau - (adv < 0).float()) * adv ** 2).mean()
        opt.zero_grad(set_to_none=True)
        (lq + lv).backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                for p_, q_ in zip(Vt.parameters(), V.parameters()):
                    p_.mul_(0.99).add_(q_, alpha=0.01)
        if step % 10000 == 0:
            print(f"[V] {step}: lq {lq.item():.5f} lv {lv.item():.5f}", flush=True)

    # ---------------- stage 1b: reward head on demo transitions ----------------
    Rh = mlp(D + H * A, 1, hidden=256, depth=2).to(dev)
    opt = torch.optim.AdamW(Rh.parameters(), lr=3e-4)
    for step in range(args.r_steps):
        r = tr[torch.randint(len(tr), (args.batch,), device=dev)]
        pred = Rh(torch.cat([phi_t[r], demo_t[r]], -1)).squeeze(-1)
        loss = ((pred - rn_t[r]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"[R] final loss {loss.item():.6f}", flush=True)

    # ---------------- stage 2: precompute expansion targets for ALL candidates ----------------
    members = []
    for sd in ck["members"]:
        m = DynV1(D, s * A, hist=hist, horizon_macro=hm, prior_scale=cfg["prior_scale"]).to(dev)
        m.load_state_dict(sd)
        m.eval()
        members.append(m)
    y = torch.zeros(n, N, device=dev)
    B = 4096
    rows_all = np.flatnonzero(ok)
    with torch.no_grad():
        for i0 in range(0, len(rows_all), B):
            r = torch.from_numpy(rows_all[i0:i0 + B]).to(dev)
            b = len(r)
            zh = phi_t[r][:, None, :].expand(b, hist, D).reshape(b, 1 * hist, D)
            a = torch.from_numpy(np.ascontiguousarray(cand[rows_all[i0:i0 + B]])).to(dev)  # [b,N,H,A]
            a_seq = a.reshape(b * N, hm, s * A)
            zh_rep = zh.repeat_interleave(N, dim=0)
            vs = []
            for m in members:
                mu, _ = m(zh_rep, a_seq)
                vs.append(V(mu[:, -1]).squeeze(-1))          # V at the imagined chunk end
            vs = torch.stack(vs)                              # [M, b*N]
            v_img = vs.min(0).values if args.img_agg == "min" else vs.mean(0)
            rh = Rh(torch.cat([phi_t[r].repeat_interleave(N, 0), a.reshape(b * N, H * A)], -1)).squeeze(-1)
            if args.no_reward_head:
                rh = torch.zeros_like(rh)
            y[r] = (rh.clamp(0.0, 1.0) + (g ** H) * v_img).clamp(0.0, 1.0).view(b, N)
            if i0 % (B * 16) == 0:
                print(f"[img] {i0}/{len(rows_all)}", flush=True)

    # ---------------- stage 3: final critic Q(phi + proprio, a) ----------------
    Q = mlp(D + P + H * A, 1).to(dev)
    opt = torch.optim.AdamW(Q.parameters(), lr=3e-4)
    for step in range(args.q_steps):
        r = tr[torch.randint(len(tr), (args.batch,), device=dev)]
        # half the batch on demo (real target), half on a random candidate (imagined target)
        half = args.batch // 2
        rd, rc = r[:half], r[half:]
        tgt_d = rn_t[rd] + (~term_t[rd]) * (g ** H) * V(phi_t[nxt_t[rd]]).squeeze(-1).detach()
        qd = Q(torch.cat([phi_t[rd], pro_t[rd], demo_t[rd]], -1)).squeeze(-1)
        ci = torch.randint(N, (len(rc),), device=dev)
        ac = torch.from_numpy(np.ascontiguousarray(cand[rc.cpu().numpy(), ci.cpu().numpy()])).to(dev)
        qc = Q(torch.cat([phi_t[rc], pro_t[rc], ac.reshape(len(rc), H * A)], -1)).squeeze(-1)
        loss = ((qd - tgt_d) ** 2).mean() + ((qc - y[rc, ci]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10000 == 0:
            print(f"[Q] {step}: loss {loss.item():.5f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"V": V.state_dict(), "Q": Q.state_dict(), "Rh": Rh.state_dict(),
                "zmu": zmu, "zsd": zsd, "meta": {"annot": str(args.annot), "z": str(args.z),
                "dyn": str(args.dyn), "img_agg": args.img_agg, "tau": args.tau}},
               args.out / "mve_critic.pt")

    # ---------------- diagnostics on held-out episodes ----------------
    hr = torch.from_numpy(held_rows[: min(4000, len(held_rows))]).to(dev)
    with torch.no_grad():
        obs = torch.cat([phi_t[hr], pro_t[hr]], -1)
        q_demo = Q(torch.cat([obs, demo_t[hr]], -1)).squeeze(-1)
        roll = torch.roll(hr, 1)
        q_other = Q(torch.cat([obs, demo_t[roll]], -1)).squeeze(-1)
        qc_all = []
        for i in range(N):
            a = torch.from_numpy(np.ascontiguousarray(cand[hr.cpu().numpy(), i])).to(dev)
            qc_all.append(Q(torch.cat([obs, a.reshape(len(hr), H * A)], -1)).squeeze(-1))
        qc_all = torch.stack(qc_all, 1)                       # [B, N]
    d = {}
    d["action_sensitivity"] = qc_all.std(1).mean().item()
    d["ranking_accuracy_demo_vs_other"] = (q_demo > q_other).float().mean().item()
    d["ranking_accuracy_demo_vs_candidate"] = (q_demo[:, None] > qc_all).float().mean().item()
    mch = mc_t[hr]
    d["q_demo_minus_mc_mean"] = (q_demo - mch).mean().item()
    d["frac_q_demo_above_support"] = (q_demo > 1.0).float().mean().item()
    qr = q_demo.cpu().numpy()
    mr = mch.cpu().numpy()
    from scipy.stats import spearmanr
    d["spearman_q_demo_vs_mc"] = float(spearmanr(qr, mr).statistic)
    d["within_state_q_range"] = (qc_all.max(1).values - qc_all.min(1).values).mean().item()
    print("\n=== MVE critic diagnostics (held-out) ===")
    for k, v in d.items():
        print(f"  {k:42s} {v:+.4f}")
    checks = [
        ("action sensitivity >= 0.03", d["action_sensitivity"] >= 0.03),
        ("binding (demo vs other-state demo) >= 0.9", d["ranking_accuracy_demo_vs_other"] >= 0.9),
        ("no overestimation: q_demo - mc <= 0.05", d["q_demo_minus_mc_mean"] <= 0.05),
        ("q respects support (frac > 1.0 == 0)", d["frac_q_demo_above_support"] < 0.01),
        ("value ordering: spearman >= 0.75", d["spearman_q_demo_vs_mc"] >= 0.75),
    ]
    ok_all = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok_all &= passed
    print("necessary conditions met; proceed to rollout evaluation" if ok_all else "some conditions FAILED")
    (args.out / "eval.json").write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
