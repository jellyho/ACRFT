"""Recover the proprioception the critic is supposed to read, without re-annotating.

The `noprop` RLT token deliberately excludes proprio — that is the paper-faithful bottleneck, and it
is why README says "critic must supply proprio". But nothing supplies it: `train_rlt_critic` calls
`net.apply(params, token, action_chunk)` and the annotation stores no state array, so the critic
judges an action chunk without knowing where the arm is.

Re-running annotate_rlt.py would cost a full VLA pass over every frame. It is unnecessary: the
annotation was written at stride 1 straight down the dataset and keeps `episode_index` /
`frame_index`, and the source LeRobot parquet carries `observation.state`. So the state can simply be
joined back on. This checks that alignment explicitly rather than trusting it, then writes
`proprio.dat` [T, S] next to the annotation.

    uv run slurm/extract_proprio.py --data /scratch/jellyho/acrft/annot/noprop
"""

import argparse
import glob
import json
import pathlib

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    ap.add_argument("--lerobot-root", type=pathlib.Path, default=None, help="default: $HF_LEROBOT_HOME/<repo_id>")
    ap.add_argument("--key", default="observation.state")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import os

    import pyarrow.parquet as pq

    meta = json.loads((args.data / "meta.json").read_text())
    T, stride = meta["num_frames"], meta.get("stride", 1)
    root = (
        args.lerobot_root
        or pathlib.Path(os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")))
        / meta["repo_id"]
    )

    files = sorted(glob.glob(str(root / "data/chunk-*/file-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {root}/data — run slurm/fetch_data.sh --lerobot")

    cols = [args.key, "episode_index", "frame_index"]
    tabs = [pq.read_table(f, columns=cols) for f in files]
    state = np.concatenate([np.stack(t.column(args.key).to_numpy(zero_copy_only=False)) for t in tabs]).astype(
        np.float32
    )
    ep_src = np.concatenate([t.column("episode_index").to_numpy() for t in tabs]).astype(np.int32)
    fr_src = np.concatenate([t.column("frame_index").to_numpy() for t in tabs]).astype(np.int32)
    print(f"source   : {root}\n           {len(state)} rows, {args.key} is {state.shape[1]}-d")

    keep = np.arange(0, len(state), stride)[:T]
    if len(keep) != T:
        raise SystemExit(f"dataset has {len(state)} rows; annotation wants {T} at stride {stride}")

    # Verify the join rather than assume it: the annotation stores the same two index columns, so a
    # mismatch here means the rows do not correspond and the proprio would be silently misaligned.
    for name, src in (("episode_index", ep_src), ("frame_index", fr_src)):
        f = args.data / f"{name}.dat"
        if not f.exists():
            print(f"  (no {name}.dat to check against)")
            continue
        got = np.asarray(np.memmap(f, np.int32, "r", shape=(T,)))
        bad = int((got != src[keep]).sum())
        if bad:
            raise SystemExit(f"ALIGNMENT FAILED: {name} differs on {bad}/{T} rows — do not use this join")
        print(f"  {name}: matches on all {T} rows")

    out = state[keep]
    mu, sd = out.mean(0), out.std(0)
    print(f"\nper-dim std: min {sd.min():.4g}  max {sd.max():.4g}  (dims with std<1e-6: {int((sd < 1e-6).sum())})")
    print(f"writing {out.shape} to {args.data / 'proprio.dat'}")
    if args.dry_run:
        print("dry run: nothing written")
        return

    np.memmap(args.data / "proprio.dat", np.float32, "w+", shape=out.shape)[:] = out
    meta |= {
        "proprio_dim": int(out.shape[1]),
        "proprio_key": args.key,
        # Stored raw; the trainer z-scores on load. Kept here so the normalisation is inspectable and
        # identical across the discount variants, which share this file by hardlink.
        "proprio_mean": [float(x) for x in mu],
        "proprio_std": [float(x) for x in sd],
    }
    (args.data / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"updated {args.data / 'meta.json'} (proprio_dim={out.shape[1]})")


if __name__ == "__main__":
    main()
