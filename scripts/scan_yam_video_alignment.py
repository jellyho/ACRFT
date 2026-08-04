"""Measure the video/parquet frame offset at every episode boundary, for every camera and file.

The episode joins inside each concatenated mp4 are hard visual cuts, so the frame at which the cut
actually lands, compared to where `from_timestamp` says it should, gives the accumulated frame
offset of that stream at that point in the file. Running it everywhere -- not just on the files whose
frame count is visibly wrong -- also catches the nastier case where a stream both dropped and
duplicated a frame, which leaves the total count correct while everything in between is misaligned.

Writes .scratch/boundary_offsets.json.
"""

import glob
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

D = "/data5/jellyho/.cache/huggingface/lerobot/jellyho/yam_lego_taxi"
FPS = 30
KEYS = ["observation.images.wrist_left", "observation.images.wrist_right", "observation.images.agentview"]
RADIUS = 10


def episodes() -> pd.DataFrame:
    parts = [pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{D}/meta/episodes/chunk-*/file-*.parquet"))]
    return pd.concat(parts).sort_values("episode_index").reset_index(drop=True)


def cut_offset(dec: VideoDecoder, expected: int) -> tuple[int, float]:
    start = max(0, expected - RADIUS)
    stop = min(dec.metadata.num_frames, expected + RADIUS + 1)
    frames = dec.get_frames_in_range(start, stop).data[:, :, ::4, ::4].float().numpy()
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
    peak = int(np.argmax(diff))
    sharpness = float(diff[peak] / (np.delete(diff, peak).mean() + 1e-9))
    return (start + peak + 1) - expected, sharpness


def main() -> None:
    ep = episodes()
    out = {}
    for key in KEYS:
        fi_col, ts_col = f"videos/{key}/file_index", f"videos/{key}/from_timestamp"
        for file_index in sorted(ep[fi_col].unique()):
            sub = ep[ep[fi_col] == file_index].reset_index(drop=True)
            path = f"{D}/videos/{key}/chunk-000/file-{int(file_index):03d}.mp4"
            dec = VideoDecoder(path)
            n_actual, n_expected = dec.metadata.num_frames, int(sub.length.sum())
            rows = []
            for _, e in sub.iloc[1:].iterrows():
                expected = round(e[ts_col] * FPS)
                off, sharp = cut_offset(dec, expected)
                rows.append(
                    {"episode": int(e.episode_index), "expected": expected, "offset": off, "sharpness": round(sharp, 1)}
                )
            out[f"{key}/file-{int(file_index):03d}"] = {
                "n_actual": n_actual,
                "n_expected": n_expected,
                "shortfall": n_expected - n_actual,
                "boundaries": rows,
            }
            bad = [r for r in rows if r["offset"] != 0]
            weak = [r for r in rows if r["sharpness"] <= 3]
            print(
                f"{key.split('.')[-1]:>12} file-{int(file_index):03d}: {n_actual}/{n_expected} "
                f"({n_actual - n_expected:+d})  nonzero-offset boundaries: {len(bad)}/{len(rows)}"
                f"{'  weak-cut: ' + str(len(weak)) if weak else ''}",
                flush=True,
            )
            for r in bad:
                print(f"      ep {r['episode']} expected {r['expected']} -> offset {r['offset']:+d} "
                      f"(sharpness {r['sharpness']})", flush=True)

    with open("/data5/jellyho/ACRFT/openpi/.scratch/boundary_offsets.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote .scratch/boundary_offsets.json")


if __name__ == "__main__":
    main()
