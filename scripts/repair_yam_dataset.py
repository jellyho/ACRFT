"""Repair the two export bugs in `jellyho/yam_lego_taxi` in place, without re-encoding video.

Bug 1: the agentview stream dropped frames *mid-file*, not at the tail, so every episode after a
drop is silently misaligned with its states and actions, and the last episode of the file overshoots
the end of the stream and crashes the loader. Where the drops landed is recovered from the episode
joins -- each is a hard visual cut, so the frame at which it actually appears against the frame
`from_timestamp` predicts gives the stream's accumulated offset. Correcting `from_timestamp` by that
offset realigns every episode; no video is touched, so there is no re-encode and no generation loss.

Bug 2: `timestamp` is float32, whose resolution at the ~2000 s reached by concatenated files is
coarser than LeRobot's frame-matching tolerance, so lookups fail on precision alone. Rewritten as
float64 on the exact `frame_index / fps` grid.

Run `scripts/scan_yam_video_alignment.py` first to produce the offsets this consumes. `--dry-run`
reports what would change without writing.
"""

import argparse
import glob
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D = pathlib.Path("/data5/jellyho/.cache/huggingface/lerobot/jellyho/yam_lego_taxi")
OFFSETS = pathlib.Path(__file__).resolve().parents[1] / ".scratch/boundary_offsets.json"
FPS = 30


def episode_end_offsets(scan: dict) -> tuple[dict[tuple[str, int], int], list[dict]]:
    """Per-episode frame offset in effect at the *end* of the episode, plus episodes with a splice.

    Shifting an episode by the offset at its end guarantees its last frame lands inside the stream.
    An episode whose start and end offsets differ has a drop somewhere inside it: no single shift can
    align both halves, so one side keeps a residual of a frame or two. Those are reported, not hidden.

    Two guards on the measurement. A stream whose frame count already matches the parquet is left
    alone outright -- its cut positions are ground truth by construction. And within a short stream,
    only non-positive, non-increasing offsets are believed: frames can only go missing, never appear,
    so a positive reading is the probe latching onto a fast camera motion just after the cut rather
    than the cut itself.
    """
    end_off: dict[tuple[str, int], int] = {}
    spliced: list[dict] = []
    for name, rec in scan.items():
        key, file_tag = name.rsplit("/", 1)
        file_index = int(file_tag.split("-")[1])
        if rec["shortfall"] == 0:
            continue
        bounds = rec["boundaries"]
        episodes = [bounds[0]["episode"] - 1] + [b["episode"] for b in bounds]
        # boundaries[i] is the cut *into* episode i+1, i.e. the offset in effect from its first frame.
        raw = [0] + [min(0, b["offset"]) for b in bounds]
        start_off = list(np.minimum.accumulate(raw))
        # The last episode has no following cut; the file's total shortfall pins its end offset.
        ends = start_off[1:] + [-rec["shortfall"]]
        for ep, s, e in zip(episodes, start_off, ends, strict=True):
            end_off[(key, ep)] = e
            if s != e:
                spliced.append({"key": key, "file": file_index, "episode": ep, "start_offset": s, "end_offset": e})
    return end_off, spliced


def fix_episode_metadata(end_off: dict[tuple[str, int], int], *, dry_run: bool) -> None:
    for path in sorted(glob.glob(str(D / "meta/episodes/chunk-*/file-*.parquet"))):
        table = pq.read_table(path)
        episode_index = table.column("episode_index").to_numpy()
        changed = 0
        # Edit the two timestamp columns in place on the arrow table. Round-tripping through pandas
        # would rewrite the deeply nested per-camera stats columns as well, for no reason.
        for key in {k for k, _ in end_off}:
            shift = np.array([end_off.get((key, int(e)), 0) for e in episode_index], dtype=np.float64) / FPS
            changed += int((shift != 0).sum())
            for col_name in (f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"):
                i = table.schema.get_field_index(col_name)
                shifted = table.column(col_name).to_numpy(zero_copy_only=False) + shift
                table = table.set_column(i, table.schema.field(i), pa.array(shifted, type=pa.float64()))
        print(f"  {pathlib.Path(path).name}: shifted {changed} (episode, camera) pairs")
        if not dry_run:
            _backup(path)
            pq.write_table(table, path)


def fix_timestamps(*, dry_run: bool) -> None:
    for path in sorted(glob.glob(str(D / "data/chunk-*/file-*.parquet"))):
        table = pq.read_table(path)
        frame_index = table.column("frame_index").to_numpy()
        exact = frame_index.astype(np.float64) / FPS
        old = table.column("timestamp").to_numpy(zero_copy_only=False).astype(np.float64)
        drift = np.abs(old - exact).max()
        col = table.schema.get_field_index("timestamp")
        table = table.set_column(col, pa.field("timestamp", pa.float64()), pa.array(exact, type=pa.float64()))
        print(f"  {pathlib.Path(path).name}: timestamp -> float64, max shift from stored value {drift:.2e} s")
        if not dry_run:
            _backup(path)
            pq.write_table(table, path)


def fix_info(*, dry_run: bool) -> None:
    path = D / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["timestamp"]["dtype"] = "float64"
    print("  info.json: timestamp dtype -> float64")
    if not dry_run:
        _backup(str(path))
        path.write_text(json.dumps(info, indent=4))


def _backup(path: str) -> None:
    """Keep one pristine copy so a re-run repairs the original rather than compounding the shift."""
    orig = pathlib.Path(str(path) + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scan = json.loads(OFFSETS.read_text())
    end_off, spliced = episode_end_offsets(scan)

    nonzero = {k: v for k, v in end_off.items() if v != 0}
    print(f"episodes needing a shift: {len(nonzero)}")
    for (key, ep), off in sorted(nonzero.items(), key=lambda kv: kv[0][1]):
        print(f"  ep {ep:>3} {key.split('.')[-1]:>12}: shift {off:+d} frames")

    print(
        f"\nepisodes with a drop *inside* them (residual <=|{max([abs(s['end_offset'] - s['start_offset']) for s in spliced], default=0)}| frame): {len(spliced)}"
    )
    for s in spliced:
        print(
            f"  ep {s['episode']:>3} {s['key'].split('.')[-1]:>12} file-{s['file']:03d}: "
            f"offset {s['start_offset']:+d} at start, {s['end_offset']:+d} at end"
        )

    print("\nepisode metadata:")
    fix_episode_metadata(end_off, dry_run=args.dry_run)
    print("\nframe timestamps:")
    fix_timestamps(dry_run=args.dry_run)
    print("\ndataset info:")
    fix_info(dry_run=args.dry_run)
    print("\nDRY RUN - nothing written" if args.dry_run else "\nwritten (originals kept alongside as *.orig)")


if __name__ == "__main__":
    main()
