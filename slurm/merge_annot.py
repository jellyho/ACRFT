"""Merge annotated memmap datasets (demos + rollout collections) into one training dir.

Concatenates every shared .dat stream and rewrites episode_index with a running offset so
episode ids stay globally unique — the terminal indexing in data.load_data depends on it.

    uv run --no-sync python slurm/merge_annot.py OUT PART [PART ...]
    e.g.  merge_annot.py /scratch/.../annot/mixed_v14 .../annot/noprop .../annot/kroll
"""

import json
import pathlib
import sys

import numpy as np

DTYPES = {"episode_index": np.int32, "done": np.int8, "reward": np.float32, "mc_return": np.float32}


def main():
    out, parts = pathlib.Path(sys.argv[1]), [pathlib.Path(p) for p in sys.argv[2:]]
    metas = [json.loads((p / "meta.json").read_text()) for p in parts]
    defaults = {"stride": 1, "dtype": "float32"}  # annotate_rollouts writes a lean meta; these are its actual values
    for k in ("horizon", "action_dim", "num_samples", "token_dim", "proprio_dim", "stride", "dtype"):
        vals = [m.get(k) if m.get(k) is not None else defaults.get(k) for m in metas]
        assert len(set(map(str, vals))) == 1, f"meta mismatch on {k}: {vals}"
    out.mkdir(parents=True, exist_ok=True)

    streams = set.intersection(*(set(f.stem for f in p.glob("*.dat")) for p in parts))
    assert {"rl_token", "action_chunk", "base_action", "reward", "episode_index", "done", "proprio", "mc_return"} <= streams, streams

    for name in sorted(streams):
        if name == "episode_index":
            continue
        with open(out / f"{name}.dat", "wb") as w:
            for p in parts:
                w.write((p / f"{name}.dat").read_bytes())

    ep_off, chunks = 0, []
    for p, m in zip(parts, metas):
        ep = np.fromfile(p / "episode_index.dat", dtype=np.int32)
        assert len(ep) == m["num_frames"], (p, len(ep), m["num_frames"])
        chunks.append(ep + ep_off)
        ep_off += int(ep.max()) + 1
    np.concatenate(chunks).tofile(out / "episode_index.dat")

    meta = dict(metas[0])
    meta["num_frames"] = sum(m["num_frames"] for m in metas)
    meta["source"] = " + ".join(str(p) for p in parts)
    meta["mixture"] = {str(p): m["num_frames"] for p, m in zip(parts, metas)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: {meta['num_frames']:,} frames, {ep_off:,} episodes, streams={sorted(streams)}")


if __name__ == "__main__":
    main()
