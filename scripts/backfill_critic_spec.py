"""Add the input contract to a critic checkpoint that predates it.

Checkpoints saved before the contract existed carry no record of what input space they were trained
in, so the serving wrapper cannot tell a raw-units critic from a pi05-space one and has to guess. This
stamps the contract onto an existing directory (config.json gains ``input_spec``; ``norm_stats.json``
is written alongside) so that guess becomes a check.

Every checkpoint that predates the contract is by definition RAW-units -- pi05-space training did not
exist yet -- so that is what is stamped, with the cache the run read supplying the dimensions.

    uv run --no-sync python scripts/backfill_critic_spec.py \
        --critic /data5/jellyho/critics/yam/g5_s347 --cache /data1/jellyho/pc_cache/yam_s347
"""

import argparse
import json
import pathlib

from openpi.patch_critic import spec as critic_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True, nargs="+", help="checkpoint dir(s)")
    ap.add_argument("--cache", type=pathlib.Path, required=True, help="the feature cache the run trained on")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = json.loads((a.cache / "meta.json").read_text())
    stats = critic_spec.norm_stats(a.cache, meta)

    for d in a.critic:
        cfg = json.loads((d / "config.json").read_text())
        if "input_spec" in cfg:
            print(f"{d}: already has input_spec (normalization={cfg['input_spec'].get('normalization')}) -- skipped")
            continue
        spec = critic_spec.input_spec(meta, horizon=int(cfg["horizon"]))
        spec["cache"] = str(a.cache)
        spec["n_episodes"] = len(meta["episodes"])
        spec["backfilled"] = True  # the contract was inferred from the cache, not recorded by the run
        cfg["input_spec"] = spec
        if a.dry_run:
            print(f"{d}: would write input_spec + norm_stats.json\n{json.dumps(spec, indent=2)}")
            continue
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        (d / "norm_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"{d}: wrote input_spec (normalization=raw) + norm_stats.json")


if __name__ == "__main__":
    main()
