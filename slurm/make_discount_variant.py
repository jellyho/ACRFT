"""Build a discount-swept copy of an annotation ahead of time.

This is a convenience, not a prerequisite: `train_rlt_critic.py --discount 0.999` builds exactly the
same directory by itself, through the same `openpi.rlt_critic.annotation.ensure_discount` this calls.
Use it to pay the one-off cost once, visibly, before launching a sweep whose arms would otherwise all
race to build it on their first step - or to see what a gamma does to the returns with --dry-run.

Only the per-frame return has to change. `rl_token` / `action_chunk` / `base_action` are ~99.9% of
the bytes and are identical across gammas, so they are hardlinked; the columns any re-labelling
rewrites in place are copied for real, so writing to the variant cannot reach back into the source.

    uv run slurm/make_discount_variant.py --data /scratch/jellyho/acrft/annot/noprop --discount 0.999
    uv run slurm/make_discount_variant.py --data ... --discount 0.9995 --dry-run
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from openpi.rlt_critic import annotation as _annot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="source annotate_rlt.py output dir")
    ap.add_argument("--discount", required=True, type=float, help="gamma for the new copy")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="default: <data>_g<digits of gamma>")
    ap.add_argument("--dry-run", action="store_true", help="report what the new returns look like, write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    meta = json.loads((args.data / "meta.json").read_text())
    T = meta["num_frames"]
    out = args.out or args.data.parent / f"{args.data.name}_{_annot.discount_tag(args.discount)}"
    print(
        f"source    {args.data}  ({T} frames, scheme {meta.get('reward_scheme', 'sparse')}, "
        f"gamma {meta.get('discount', 0.99)})"
    )
    print(f"target    {out}  (gamma {args.discount})")

    if args.dry_run:
        rd = lambda n, dt: np.asarray(np.memmap(args.data / f"{n}.dat", dt, "r", shape=(T,)))  # noqa: E731
        mc = _annot.mc_return_at(
            rd("reward", np.float32), rd("done", np.int8), rd("episode_index", np.int32), args.discount
        )
        print(f"mc_return min {mc.min():.6f}  max {mc.max():.6f}  mean {mc.mean():.6f}")
        print("dry run: nothing written")
        return

    built = _annot.ensure_discount(args.data, args.discount)
    print(f"ready: {built}")


if __name__ == "__main__":
    main()
