"""Extract per-task preview media from the converted RoboCasa 365 datasets.

For each task it writes:
  - examples/robocasa/site/videos/<Task>.mp4  — the FULL episode-0 trajectory (agent-view left),
    cut at the exact episode boundary from the LeRobot v3.0 episode metadata.
  - examples/robocasa/assets/<Task>.jpg        — a poster frame (used as the video poster and as
    the thumbnail in the embedded artifact dashboard).

Run inside the openpi venv (needs pyarrow to read metadata; ffmpeg must be on PATH):

    uv run examples/robocasa/make_previews.py --output-dir /data5/jellyho/robocasa365

The full-trajectory videos are large and are git-ignored; regenerate them locally. Pair this
with `gen_dashboard.py --mode site` to build a page that plays them.
"""

import argparse
import glob
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_CAM = "observation.images.robot0_agentview_left"


def episode0(task_dir: Path):
    """Return (video_path, from_ts, to_ts) for episode 0's agent-view-left video, or None."""
    ep_files = sorted(glob.glob(str(task_dir / "meta/episodes/chunk-*/*.parquet")))
    if not ep_files:
        return None
    # Episode 0 lives in the first metadata file.
    d = pq.read_table(ep_files[0]).to_pydict()
    ci = d[f"videos/{_CAM}/chunk_index"][0]
    fi = d[f"videos/{_CAM}/file_index"][0]
    frm = float(d[f"videos/{_CAM}/from_timestamp"][0])
    to = float(d[f"videos/{_CAM}/to_timestamp"][0])
    video = task_dir / "videos" / _CAM / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
    if not video.exists():
        return None
    return video, frm, to


def run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def write_instruction_counts(output_dir: Path, out_json: Path) -> None:
    """Count each task's distinct language instructions and dump {task: count} to JSON."""
    import json

    counts: dict[str, int] = {}
    for task_dir in sorted(p for p in output_dir.iterdir() if (p / "meta").is_dir()):
        seen = set()
        for f in sorted((task_dir / "data").glob("chunk-*/*.parquet")):
            col = pq.read_table(f, columns=["annotation.human.task_description"]).to_pydict()
            seen.update(v[0] if isinstance(v, list) else v for v in col["annotation.human.task_description"])
        counts[task_dir.name] = len(seen)
        print(f"[{task_dir.name}] {len(seen)} distinct instruction(s)")
    out_json.write_text(json.dumps(counts, indent=1, sort_keys=True))
    n_single = sum(1 for c in counts.values() if c <= 1)
    print(f"Wrote {out_json}: {len(counts)} tasks ({n_single} single, {len(counts) - n_single} multi).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("/data5/jellyho/robocasa365"))
    ap.add_argument("--only", type=str, nargs="+", default=None, help="Only these task names.")
    ap.add_argument("--overwrite", action="store_true", help="Re-extract even if outputs exist.")
    ap.add_argument(
        "--instruction-counts",
        action="store_true",
        help="Instead of extracting media, scan each task's distinct language instructions "
        "and write assets/instruction_counts.json (used by the dashboard).",
    )
    args = ap.parse_args()

    _HERE.joinpath("assets").mkdir(parents=True, exist_ok=True)
    if args.instruction_counts:
        write_instruction_counts(args.output_dir, _HERE / "assets" / "instruction_counts.json")
        return

    videos_dir = _HERE / "site" / "videos"
    assets_dir = _HERE / "assets"
    videos_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(p.name for p in args.output_dir.iterdir() if (p / "meta").is_dir())
    if args.only:
        tasks = [t for t in tasks if t in args.only]

    n_vid = n_img = 0
    for task in tasks:
        info = episode0(args.output_dir / task)
        if info is None:
            print(f"[{task}] no episode-0 video found, skipping")
            continue
        video, frm, to = info
        dur = max(0.1, to - frm)

        out_mp4 = videos_dir / f"{task}.mp4"
        if args.overwrite or not out_mp4.exists():
            ok = run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{frm}",
                    "-t",
                    f"{dur}",
                    "-i",
                    str(video),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_mp4),
                ]
            )
            n_vid += ok
            print(f"[{task}] video {'ok' if ok else 'FAILED'} ({dur:.1f}s)")

        out_jpg = assets_dir / f"{task}.jpg"
        if args.overwrite or not out_jpg.exists():
            # Poster from ~35% into the episode (usually mid-manipulation).
            ok = run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{frm + 0.35 * dur}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=400:-1",
                    "-q:v",
                    "4",
                    str(out_jpg),
                ]
            )
            n_img += ok

    print(f"Done. {n_vid} videos, {n_img} posters -> {videos_dir} , {assets_dir}")


if __name__ == "__main__":
    main()
