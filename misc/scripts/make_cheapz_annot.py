"""Package a trained cheap-z representation as an annotate_rlt.py-compatible directory.

The whole point of the cheap-z line is to swap the critic's observation WITHOUT touching the
critic stack: train_rlt_critic.py reads an annotation dir, so we synthesize one whose
``rl_token.dat`` holds the cheap encoder's z and whose every other array (labels, action chunks,
candidate actions, proprio) is HARDLINKED from a reference VLA annotation of the same dataset.
Hardlinks, not copies: base_action.dat alone is 3.4 GB and the bytes are identical by
construction — both annotations describe the same frames in the same order (both stride 1).

The critic comparison this enables is then exactly controlled: same labels, same candidates,
same proprio, same training code and seed — the ONLY difference is which z sits in rl_token.
"""

import argparse
import json
import os
import pathlib

import numpy as np

SHARED = [
    "action_chunk.dat",
    "base_action.dat",
    "base_action_heldout.dat",
    "reward.dat",
    "done.dat",
    "episode_index.dat",
    "frame_index.dat",
    "mc_return.dat",
    "progress.dat",
    "proprio.dat",
    "borrowed_return.dat",
    "_progress.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=pathlib.Path, required=True, help="cheap-z run dir containing z.npy")
    ap.add_argument("--ref", type=pathlib.Path, required=True, help="reference VLA annotation dir")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    z = np.load(args.z / "z.npy")  # [n, d] float32
    meta = json.loads((args.ref / "meta.json").read_text())
    if z.shape[0] != meta["num_frames"]:
        raise ValueError(f"z has {z.shape[0]} frames, reference {meta['num_frames']}")

    args.out.mkdir(parents=True, exist_ok=True)
    tok = np.memmap(args.out / "rl_token.dat", dtype=np.float32, mode="w+", shape=z.shape)
    tok[:] = z.astype(np.float32)
    tok.flush()

    for name in SHARED:
        src, dst = args.ref / name, args.out / name
        if not src.exists():
            continue
        if dst.exists():
            dst.unlink()
        os.link(src, dst)

    meta["token_dim"] = int(z.shape[1])
    meta["dtype"] = "float32"
    meta["cheap_z_source"] = str(args.z)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out}: token_dim={z.shape[1]}, {z.shape[0]} frames, labels hardlinked from {args.ref}")


if __name__ == "__main__":
    main()
