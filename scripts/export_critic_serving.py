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
    meta = json.loads((pathlib.Path(cfg["data"]) / "meta.json").read_text())
    out = {"mean": meta["proprio_mean"], "std": meta["proprio_std"], "key": args.proprio_key}
    (args.critic / "proprio_stats.json").write_text(json.dumps(out))
    print(f"wrote {args.critic / 'proprio_stats.json'} ({len(out['mean'])}-d)")


if __name__ == "__main__":
    main()
