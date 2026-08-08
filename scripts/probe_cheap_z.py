"""Ridge-probe harness: is a frozen generic encoder good enough to replace the VLA's z?

Motivation. Extracting the critic observation z from the 3B VLA is expensive (a full-dataset
annotation pass per checkpoint) and inherits the VLA's appearance biases. The probing literature
(2605.28527) reports frozen DINOv2 lands within ~0.04 R² of the best VLA hidden states for value
prediction — but on LIBERO, not on our data. This measures the same quantities HERE, so the
"cheap standalone encoder" decision rests on our own numbers rather than a transferred claim.

For every representation (each --annot dir's rl_token, plus frozen DINOv2-small features computed
on the same frames) it reports, with episodes held out:
  mc_return R²   — the quantity a critic must actually regress (the decision-relevant probe)
  progress  R²   — task-phase structure
  episode  acc   — linear decodability of "which demo" (the measured RLT pathology; lower better)
  knn purity     — same-episode fraction of each frame's k nearest neighbours (geometry; lower better)

Frames are subsampled evenly across the dataset; all representations are probed on the IDENTICAL
frame set with the IDENTICAL episode split, so rows are directly comparable.

Run under slurm (one GPU decodes video + runs the 22M encoder comfortably):
  sbatch ... uv run scripts/probe_cheap_z.py \
      --annot rlt5=.scratch/annot_noprop mae05=.scratch/annot_mae05 \
      --repo-id jellyho/robocasa365-PrepareCoffee --num-frames 20000 --out .scratch/probe_cheap_z.json
"""

import argparse
import json
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def load_annot(path: pathlib.Path):
    meta = json.loads((path / "meta.json").read_text())
    n, d = meta["num_frames"], meta["token_dim"]
    return {
        "z": np.memmap(path / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, d)),
        "ep": np.memmap(path / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)),
        "frame": np.memmap(path / "frame_index.dat", dtype=np.int32, mode="r", shape=(n,)),
        "mc": np.memmap(path / "mc_return.dat", dtype=np.float32, mode="r", shape=(n,)),
        "prog": np.memmap(path / "progress.dat", dtype=np.float32, mode="r", shape=(n,)),
        "n": n,
    }


def dino_features(repo_id: str, rows: np.ndarray, batch: int = 64) -> np.ndarray:
    """Frozen DINOv2-small mean-pooled patch features for the 3 cameras, concatenated.

    Mean over PATCH tokens (not CLS): every comparison that isolates this (DINO-WM's encoder
    ablation, Theia) finds the spatial tokens carry the manipulation-relevant signal.
    """
    from lerobot.datasets import lerobot_dataset
    import torch
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    ds = lerobot_dataset.LeRobotDataset(repo_id)
    cams = list(ds.meta.video_keys)
    logger.info(f"cameras: {cams}")

    feats = np.empty((len(rows), 384 * len(cams)), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(rows), batch):
            idxs = rows[lo : lo + batch]
            per_cam = {c: [] for c in cams}
            for i in idxs:
                item = ds[int(i)]
                for c in cams:
                    per_cam[c].append(item[c])  # [3,H,W] float in [0,1]
            outs = []
            for c in cams:
                x = torch.stack(per_cam[c]).to(dev)
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                x = (x - mean) / std
                h = model(pixel_values=x).last_hidden_state  # [b, 1+256, 384]
                outs.append(h[:, 1:].mean(1))  # mean over patch tokens
            feats[lo : lo + len(idxs)] = torch.cat(outs, dim=-1).float().cpu().numpy()
            if (lo // batch) % 20 == 0:
                logger.info(f"dino {lo + len(idxs)}/{len(rows)}")
    return feats


def probe(z: np.ndarray, ep: np.ndarray, mc: np.ndarray, prog: np.ndarray, seed: int = 0, k: int = 10) -> dict:
    """Held-out-by-episode ridge probes + episode decodability + kNN purity."""
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    rng.shuffle(eps)
    folds = np.array_split(eps, 4)

    z = StandardScaler().fit_transform(z)

    def heldout_r2(target):
        num = den = 0.0
        for f in folds:
            te = np.isin(ep, f)
            tr = ~te
            m = Ridge(alpha=100.0).fit(z[tr], target[tr])
            pred = m.predict(z[te])
            num += float(np.sum((target[te] - pred) ** 2))
            den += float(np.sum((target[te] - np.mean(target[tr])) ** 2))
        return 1.0 - num / max(den, 1e-9)

    # Episode identity: alternate frames within each episode (identity is what's probed, so the
    # split must hold out FRAMES, not episodes).
    order = np.lexsort((np.arange(len(ep)), ep))
    alt = np.zeros(len(ep), dtype=bool)
    alt[order[::2]] = True
    clf = RidgeClassifier(alpha=100.0).fit(z[alt], ep[alt])
    ep_acc = float(np.mean(clf.predict(z[~alt]) == ep[~alt]))

    # kNN same-episode purity (cosine).
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sub = rng.choice(len(zn), size=min(4000, len(zn)), replace=False)
    sims = zn[sub] @ zn.T
    sims[np.arange(len(sub)), sub] = -np.inf
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    purity = float(np.mean(ep[nn] == ep[sub][:, None]))
    chance = float(np.mean([np.mean(ep == e) for e in np.unique(ep)]))

    return {
        "mc_return_r2": heldout_r2(mc),
        "progress_r2": heldout_r2(prog),
        "episode_acc": ep_acc,
        "episode_acc_chance": 1.0 / len(eps),
        "knn_purity": purity,
        "knn_purity_chance": chance,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", nargs="+", required=True, help="name=path pairs of annotation dirs.")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--num-frames", type=int, default=20000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/probe_cheap_z.json"))
    args = ap.parse_args()

    annots = {}
    for spec in args.annot:
        name, _, p = spec.partition("=")
        annots[name] = load_annot(pathlib.Path(p))

    ref = next(iter(annots.values()))
    rows = np.linspace(0, ref["n"] - 1, args.num_frames).astype(np.int64)
    ep = np.asarray(ref["ep"][rows])
    mc = np.asarray(ref["mc"][rows])
    prog = np.asarray(ref["prog"][rows])

    results = {}
    for name, a in annots.items():
        if a["n"] != ref["n"]:
            raise ValueError(f"{name}: frame count {a['n']} != {ref['n']} (different stride?)")
        if not np.array_equal(np.asarray(a["ep"][rows]), ep):
            raise ValueError(f"{name}: episode indices disagree with the reference annotation")
        logger.info(f"probing {name} (VLA z, {a['z'].shape[1]}d)")
        results[name] = probe(np.asarray(a["z"][rows]), ep, mc, prog)

    logger.info("computing frozen DINOv2-small features")
    dz = dino_features(args.repo_id, rows)
    logger.info("probing dinov2_small")
    results["dinov2_small"] = probe(dz, ep, mc, prog)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    hdr = f"{'repr':16s} {'mc_R2':>7s} {'prog_R2':>8s} {'ep_acc':>7s} {'purity':>7s}"
    print(hdr)
    for name, r in results.items():
        print(
            f"{name:16s} {r['mc_return_r2']:7.3f} {r['progress_r2']:8.3f} "
            f"{r['episode_acc']:7.3f} {r['knn_purity']:7.3f}"
        )
    print(f"(chance: ep_acc {1.0 / len(np.unique(ep)):.3f}, purity {results['dinov2_small']['knn_purity_chance']:.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
