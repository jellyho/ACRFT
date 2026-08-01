"""Precompute, for every frame and every candidate, the return of the nearest executed chunk.

signal_analysis.py established that a candidate's outcome can be approximated by the mc_return of the
most similar demonstrated chunk in the same token-space neighbourhood (validated at 0.86 correlation
against the demo's own known return), and that this borrowed return carries a within-state ranking
signal the trained critic does not extract. To hand the critic that signal as a per-candidate target,
the borrowing has to be available at training time - and doing the token-space search inside the
critic loop would dominate it, so it is computed once here and written next to the annotation as
`borrowed_return.dat` [T, N].

The neighbourhood excludes the frame itself and its immediate temporal neighbours, so a candidate
borrows from a genuinely different transition rather than reading back the current state's own value.

Usage:
    uv run scripts/compute_borrowed_returns.py --data .scratch/annot_noprop
"""

import argparse
import json
import pathlib
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--knn", type=int, default=64, help="Token-space neighbours searched per frame.")
    ap.add_argument("--proj-dim", type=int, default=128, help="Random projection for the neighbour search.")
    ap.add_argument("--exclude", type=int, default=8, help="Skip frames within this many steps of the query.")
    ap.add_argument("--block", type=int, default=2048, help="Query frames per matmul block.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = json.loads((args.data / "meta.json").read_text())
    T, N, H, A, D = m["num_frames"], m["num_samples"], m["horizon"], m["action_dim"], m["token_dim"]
    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[m.get("dtype", "float32")]

    def rd(name, shape, d=None):
        return np.asarray(np.memmap(args.data / f"{name}.dat", dtype=d or dt, mode="r", shape=shape))

    tok = rd("rl_token", (T, D)).astype(np.float32)
    chunk = rd("action_chunk", (T, H, A)).astype(np.float32).reshape(T, -1)
    cand = rd("base_action", (T, N, H, A)).astype(np.float32).reshape(T, N, -1)
    mc = rd("mc_return", (T,), np.float32)
    ep = rd("episode_index", (T,), np.int32)
    frame = rd("frame_index", (T,), np.int32)

    rng = np.random.default_rng(args.seed)
    proj = (rng.standard_normal((D, args.proj_dim)) / np.sqrt(args.proj_dim)).astype(np.float32)
    tp = tok @ proj
    tp /= np.linalg.norm(tp, axis=1, keepdims=True) + 1e-9

    out = np.zeros((T, N), np.float32)
    t0 = time.perf_counter()
    for b0 in range(0, T, args.block):
        b1 = min(b0 + args.block, T)
        sims = tp[b0:b1] @ tp.T  # [block, T]
        for r, i in enumerate(range(b0, b1)):
            nbr = np.argpartition(-sims[r], args.knn + 2 * args.exclude)[: args.knn + 2 * args.exclude]
            keep = (ep[nbr] == ep[i]) & (np.abs(frame[nbr].astype(np.int64) - int(frame[i])) > args.exclude)
            nbr = nbr[keep]
            if len(nbr) == 0:
                out[i] = mc[i]  # no neighbour: fall back to the current value, spread 0
                continue
            d = np.linalg.norm(cand[i][:, None, :] - chunk[nbr][None, :, :], axis=-1)  # [N, K]
            out[i] = mc[nbr][np.argmin(d, axis=1)]
        if b0 % (args.block * 10) == 0:
            el = time.perf_counter() - t0
            print(f"  {b1}/{T}  ({el:.0f}s, eta {el / max(b1, 1) * (T - b1):.0f}s)", flush=True)

    (args.data / "borrowed_return.dat").write_bytes(out.astype(np.float32).tobytes())
    spread = out.max(1) - out.min(1)
    m["has_borrowed_return"] = True
    (args.data / "meta.json").write_text(json.dumps(m, indent=2))
    print(f"\nwrote borrowed_return.dat [{T}, {N}]  ({(time.perf_counter() - t0) / 60:.1f} min)")
    print(
        f"per-frame spread: median {np.median(spread):.4f}  p90 {np.percentile(spread, 90):.4f}  >0.05 in {(spread > 0.05).mean():.1%}"
    )


if __name__ == "__main__":
    main()
