"""BC probe policy on a frozen embedding: how much CONTROL information does it carry?

Mirrors the rlt_bc_probe used during RLT training (a small actor reading only the token,
evaluated in sim), but standalone on annotation data so any embedding can be probed on
identical footing: raw RLT token (2048) vs HILP phi readout (128) vs anything with a z.npy.

Trains chunk regression a(z) -> demo chunk (raw action space) with an MLP; the sim rollout
side lives in eval_critic's `probe` mode, which reuses the exact paired-seed protocol.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument(
        "--z",
        type=pathlib.Path,
        default=None,
        help="Optional z.npy (e.g. phi space). Default: the annotation's raw rl_token.dat.",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="Project the (standardized) input to this many PCA dims before the MLP - the "
        "dimension-matched control for low-dim embeddings (is 128d enough, or did the "
        "objective discard information?). The projection is saved and replayed at rollout.",
    )
    ap.add_argument(
        "--use-proprio",
        action="store_true",
        help="Concatenate z-scored proprio (stats from annot meta) to the embedding before the MLP.",
    )
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.annot / "meta.json").read_text())
    n, H, A = meta["num_frames"], meta["horizon"], meta["action_dim"]
    if args.z is not None:
        z = np.load(args.z).astype(np.float32)
        src = str(args.z)
    else:
        z = np.array(np.memmap(args.annot / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, meta["token_dim"])))
        src = "rl_token"
    if args.use_proprio:
        pm = np.asarray(meta["proprio_mean"], np.float32)
        psd = np.asarray(meta["proprio_std"], np.float32)
        prop = np.array(np.memmap(args.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, len(pm))))
        prop = np.where(psd > 1e-6, (prop - pm) / np.where(psd > 1e-6, psd, 1.0), 0.0).astype(np.float32)
        z = np.concatenate([z, prop], axis=1)
        src += "+proprio"
    zmu, zsd = z.mean(0), z.std(0) + 1e-6
    z = (z - zmu) / zsd
    proj = None
    if args.pca_dim:
        sub = z[rng.choice(len(z), min(60000, len(z)), replace=False)]
        _, _, vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
        proj = vt[: args.pca_dim].T.astype(np.float32)  # [D, pca_dim]
        z = z @ proj
        src = f"{src}+pca{args.pca_dim}"
        print(f"PCA control: {proj.shape[0]}d -> {proj.shape[1]}d")
    chunk = np.array(np.memmap(args.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))).reshape(
        n, H * A
    )
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))

    eps_u = np.unique(ep)
    rng.shuffle(eps_u)
    held = set(eps_u[: max(1, int(len(eps_u) * args.heldout_frac))].tolist())
    tr = np.flatnonzero(~np.isin(ep, list(held)))
    ho = np.flatnonzero(np.isin(ep, list(held)))
    print(f"{n} frames, z dim {z.shape[1]} ({src}), train {len(tr)} / held {len(ho)}")

    z_t = torch.from_numpy(z).to(dev)
    c_t = torch.from_numpy(chunk).to(dev)
    net = nn.Sequential(
        nn.Linear(z.shape[1], args.hidden),
        nn.GELU(),
        nn.Linear(args.hidden, args.hidden),
        nn.GELU(),
        nn.Linear(args.hidden, args.hidden),
        nn.GELU(),
        nn.Linear(args.hidden, H * A),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    for step in range(args.steps):
        idx = torch.from_numpy(rng.choice(tr, args.batch)).to(dev)
        loss = ((net(z_t[idx]) - c_t[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 10000 == 0:
            with torch.no_grad():
                hi = torch.from_numpy(rng.choice(ho, 2048)).to(dev)
                hl = ((net(z_t[hi]) - c_t[hi]) ** 2).mean().item()
            print(f"step {step}: train {loss.item():.5f} held {hl:.5f}", flush=True)

    with torch.no_grad():
        hi = torch.from_numpy(ho).to(dev)
        pred = net(z_t[hi])
        mse = ((pred - c_t[hi]) ** 2).mean().item()
        # per-dim R^2 against the held-out chunk variance
        var = c_t[hi].var(0).mean().item()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": net.state_dict(),
            "zmu": zmu,
            "zsd": zsd,
            "proj": proj,
            "cfg": {
                "in_dim": int(z.shape[1]),
                "hidden": args.hidden,
                "H": H,
                "A": A,
                "z_src": src,
                "use_proprio": args.use_proprio,
                "annot": str(args.annot),
            },
        },
        args.out / "probe.pt",
    )
    r2 = 1 - mse / (var + 1e-12)
    (args.out / "eval.json").write_text(json.dumps({"held_mse": mse, "held_r2": r2}, indent=1))
    print(f"held-out chunk MSE {mse:.5f}  R2 {r2:.3f}  -> {args.out}")


if __name__ == "__main__":
    main()
