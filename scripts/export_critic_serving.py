"""Package a trained critic for serving: copy its proprio statistics out of the annotation.

The critic's config.json names the annotation directory it was trained on, but that directory
is not a serving artifact - a robot host has only the critic directory. This writes
``<critic_dir>/proprio_stats.json`` so CriticSelectPolicy can normalize proprioception the
same way training did.

    uv run python scripts/export_critic_serving.py --critic .scratch/critic_yam_s200_iql
"""

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--proprio-key", default="observation/state")
    args = ap.parse_args()

    cfg = json.loads((args.critic / "config.json").read_text())
    if not cfg.get("use_proprio"):
        print("critic was trained without proprio; nothing to export")
        return
    annot = pathlib.Path(cfg["data"])
    meta = json.loads((annot / "meta.json").read_text())
    if "proprio_mean" in meta:
        mean, std = meta["proprio_mean"], meta["proprio_std"]
    else:
        # Older annotations carry the raw column but not its statistics - recompute them the
        # way the trainer z-scores: per-dim over every frame.
        import numpy as np

        pdim = meta["proprio_dim"]
        pr = np.memmap(annot / "proprio.dat", dtype=np.float32, mode="r", shape=(meta["num_frames"], pdim))
        sub = np.asarray(pr[:: max(1, meta["num_frames"] // 200000)])
        mean, std = sub.mean(0).tolist(), sub.std(0).tolist()
    out = {"mean": mean, "std": std, "key": args.proprio_key}
    (args.critic / "proprio_stats.json").write_text(json.dumps(out))
    print(f"wrote {args.critic / 'proprio_stats.json'} ({len(out['mean'])}-d)")


if __name__ == "__main__":
    main()
