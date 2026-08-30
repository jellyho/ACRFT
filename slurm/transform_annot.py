"""Create an annotation variant whose rl_token is an embedding of the original (PCA / HILP phi).

Non-token streams are hard-linked (identical bytes); meta.json gets the new token_dim plus a
`token_transform` record so evaluation can apply the SAME map to live tokens.

    uv run slurm/transform_annot.py SRC OUT pca --dim 128
    uv run slurm/transform_annot.py SRC OUT phi --phi-run $CACHE_DIR/critic_runs/phi/phi_mixed
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("kind", choices=["pca", "phi"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--phi-run", type=pathlib.Path, default=None)
    ap.add_argument("--use-proprio", action="store_true", help="phi variants trained on token+proprio input")
    cfg = ap.parse_args()

    meta = json.loads((cfg.src / "meta.json").read_text())
    T, D = meta["num_frames"], meta["token_dim"]
    tok = np.memmap(cfg.src / "rl_token.dat", dtype=np.float32, mode="r", shape=(T, D))
    cfg.out.mkdir(parents=True, exist_ok=True)

    if cfg.kind == "pca":
        rng = np.random.default_rng(0)
        sub = np.asarray(tok[np.sort(rng.choice(T, min(20000, T), replace=False))])
        mu = sub.mean(0)
        _, _, vt = np.linalg.svd(sub - mu, full_matrices=False)
        comps = vt[: cfg.dim]
        emb = np.empty((T, cfg.dim), np.float32)
        for i in range(0, T, 50000):
            emb[i : i + 50000] = (np.asarray(tok[i : i + 50000]) - mu) @ comps.T
        np.savez(cfg.out / "transform.npz", mean=mu, components=comps)
        transform = {"kind": "pca", "file": "transform.npz", "dim": cfg.dim}
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
        import flax.serialization as fser
        from train_hilp_readout import Phi

        params = fser.msgpack_restore((cfg.phi_run / "phi.msgpack").read_bytes())
        net = Phi(dim=cfg.dim)
        inp_extra = 0
        if cfg.use_proprio:
            pd_ = meta["proprio_dim"]
            pro = np.memmap(cfg.src / "proprio.dat", dtype=np.float32, mode="r", shape=(T, pd_))
            mu_p, sd_p = np.asarray(pro).mean(0), np.asarray(pro).std(0)
            inp_extra = pd_
        emb = np.empty((T, cfg.dim), np.float32)
        for i in range(0, T, 20000):
            x = np.asarray(tok[i : i + 20000])
            if cfg.use_proprio:
                pz = np.where(sd_p > 1e-6, (np.asarray(pro[i : i + 20000]) - mu_p) / np.where(sd_p > 1e-6, sd_p, 1), 0)
                x = np.concatenate([x, pz.astype(np.float32)], axis=1)
            emb[i : i + 20000] = np.asarray(net.apply(params, x))
        (cfg.out / "phi.msgpack").write_bytes((cfg.phi_run / "phi.msgpack").read_bytes())
        transform = {
            "kind": "phi",
            "file": "phi.msgpack",
            "dim": cfg.dim,
            "use_proprio": cfg.use_proprio,
            "input_dim": D + inp_extra,
        }

    emb.tofile(cfg.out / "rl_token.dat")
    for f in cfg.src.glob("*.dat"):
        if f.name == "rl_token.dat":
            continue
        dst = cfg.out / f.name
        if not dst.exists():
            os.link(f, dst)
    meta["token_dim"] = cfg.dim
    meta["token_transform"] = transform
    meta["source_annot"] = str(cfg.src)
    (cfg.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{cfg.out}: token {D} -> {cfg.dim} ({cfg.kind}), {T:,} frames")


if __name__ == "__main__":
    main()
