"""MB-AC offline battery (E1-E3 of docs/reports/mbac_design_notes.md).

Runs against a trained DynV1 ensemble and answers, on held-out demo episodes:

  E1  cut-point value: does uncertainty-triggered commitment (cut when cumulative
      ensemble disagreement crosses tau) end commits at lower prediction error than
      fixed-k at the SAME average commitment length? Plus: where do cuts land on the
      task (progress histogram) and does error grow superlinearly past the cut?
  E2  binding through the model: demo chunk vs another state's demo chunk (the
      --cql-swap negative), scored purely model-based — terminal-state distance to
      the episode goal in phi space, and total disagreement. No CQL anywhere: does
      the dynamics model alone bind actions to states?
  E3  horizon of visibility: E2's ranking accuracy per macro-step — the shallowest
      commit depth at which model-based ranking is informative.

The heldout split replays the training script's episode shuffle (same seed), so no
anchor here was seen in training.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_cheapz_dynamics_v1 import DynV1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=pathlib.Path, default=pathlib.Path(".scratch/phi_dyn_v1/ensemble_v1.pt"))
    ap.add_argument("--z-dir", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/mbac_offline.json"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.annot / "meta.json").read_text())
    n, H, A = meta["num_frames"], meta["horizon"], meta["action_dim"]
    z_raw = np.load(args.z_dir / "z.npy").astype(np.float32)
    ck = torch.load(args.ensemble, map_location="cpu")
    zmu, zsd = ck["zmu"], ck["zsd"]
    cfg = ck["cfg"]
    s, hist, M = cfg["stride"], cfg["hist"], cfg["members"]
    hm = H // s
    z = (z_raw - zmu) / zsd

    chunk = np.memmap(args.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    prog = np.array(np.memmap(args.annot / "progress.dat", dtype=np.float32, mode="r", shape=(n,)))

    # replay the training split exactly (same rng consumption order as the trainer)
    eps_u = np.unique(ep)
    rng.shuffle(eps_u)
    held = set(eps_u[: max(1, int(len(eps_u) * args.heldout_frac))].tolist())
    ep_start = np.zeros(n, dtype=np.int64); ep_end = np.zeros(n, dtype=np.int64)
    for e in eps_u:
        rows = np.flatnonzero(ep == e)
        ep_start[rows], ep_end[rows] = rows.min(), rows.max()
    ok = (np.arange(n) - (hist - 1) * s >= ep_start) & (np.arange(n) + H <= ep_end)
    held_rows = np.flatnonzero(ok & np.isin(ep, list(held)))
    sub = rng.choice(held_rows, size=min(args.sample, len(held_rows)), replace=False)
    print(f"{len(held_rows)} held anchors, evaluating {len(sub)}")

    models = []
    for sd in ck["members"]:
        m = DynV1(z.shape[1], s * A, hist=hist, horizon_macro=hm, prior_scale=cfg["prior_scale"]).to(dev)
        m.load_state_dict(sd); m.eval(); models.append(m)

    z_t = torch.from_numpy(z).to(dev)
    r = torch.as_tensor(sub, device=dev)
    hidx = r[:, None] + torch.arange(-(hist - 1) * s, 1, s, device=dev)[None]
    tidx = r[:, None] + torch.arange(s, H + 1, s, device=dev)[None]
    zh = z_t[hidx]
    a_demo = torch.from_numpy(np.ascontiguousarray(chunk[sub])).to(dev).view(len(sub), hm, s * A)
    tgt = z_t[tidx]

    perm = torch.from_numpy(rng.permutation(len(sub))).to(dev)
    a_other = a_demo[perm]

    with torch.no_grad():
        mus_d = torch.stack([m(zh, a_demo)[0] for m in models])   # [M,B,hm,z]
        mus_o = torch.stack([m(zh, a_other)[0] for m in models])
    mean_d, mean_o = mus_d.mean(0), mus_o.mean(0)
    sig_d = torch.var(mus_d, dim=0).sum(-1)                        # [B,hm]
    sig_o = torch.var(mus_o, dim=0).sum(-1)
    err = torch.norm(mean_d - tgt, dim=-1)                         # [B,hm] true open-loop error (demo)

    report = {"n": int(len(sub))}

    # ---------------- E1: uncertainty-triggered commitment ----------------
    cum_sig = torch.cumsum(sig_d, dim=1)
    e1 = {}
    fixed_err = {int((j + 1) * s): err[:, j].mean().item() for j in range(hm)}
    e1["fixed_err_at_commit"] = fixed_err
    # taus swept so adaptive mean-k spans the fixed grid; cut = first j with cum sigma > tau
    for q in (0.3, 0.5, 0.7, 0.9):
        tau = torch.quantile(cum_sig[:, -1], q).item()
        over = (cum_sig > tau)
        first = torch.where(over.any(1), over.float().argmax(1), torch.full_like(r, hm - 1))
        kstar = (first + 1).clamp(1, hm)                            # macro steps committed
        e_at_cut = err.gather(1, (kstar - 1)[:, None]).squeeze(1)
        e1[f"adaptive_q{q}"] = {
            "mean_k_steps": (kstar.float().mean() * s).item(),
            "err_at_commit": e_at_cut.mean().item(),
            "frac_full_commit": (kstar == hm).float().mean().item(),
            "cut_progress_hist": np.histogram(prog[sub][(kstar < hm).cpu().numpy()],
                                              bins=5, range=(0, 1))[0].tolist(),
        }
    # superlinearity: error growth after the q=0.5 cut vs before, matched horizons
    tau = torch.quantile(cum_sig[:, -1], 0.5).item()
    over = (cum_sig > tau)
    first = torch.where(over.any(1), over.float().argmax(1), torch.full_like(r, hm - 1))
    early = first <= 1                                              # cut at macro 1-2
    if early.any() and (~early).any():
        e1["err_slope_cut_early"] = (err[early, -1] - err[early, 0]).mean().item()
        e1["err_slope_cut_late"] = (err[~early, -1] - err[~early, 0]).mean().item()
    report["E1"] = e1

    # ---------------- E2: binding without CQL ----------------
    goal_rows = torch.as_tensor(ep_end[sub], device=dev)
    goal = z_t[goal_rows]                                           # standardized phi of episode end
    d_d = torch.norm(mean_d[:, -1] - goal, dim=-1)
    d_o = torch.norm(mean_o[:, -1] - goal, dim=-1)
    report["E2"] = {
        "binding_by_goal_distance": (d_d < d_o).float().mean().item(),
        "binding_by_disagreement": (sig_d.sum(1) < sig_o.sum(1)).float().mean().item(),
        "sig_ratio_other_over_demo": (sig_o.sum(1).mean() / sig_d.sum(1).mean()).item(),
    }

    # ---------------- E3: horizon of visibility ----------------
    e3 = {}
    for j in range(hm):
        d_dj = torch.norm(mean_d[:, j] - goal, dim=-1)
        d_oj = torch.norm(mean_o[:, j] - goal, dim=-1)
        e3[int((j + 1) * s)] = {
            "binding_goal_dist": (d_dj < d_oj).float().mean().item(),
            "binding_disagreement": (sig_d[:, j] < sig_o[:, j]).float().mean().item(),
        }
    report["E3"] = e3

    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
