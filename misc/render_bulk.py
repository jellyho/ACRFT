"""Render every episode of a dataset, then zip the folder.

The per-episode renderer is the unit of work; this is the loop around it. It exists because
rendering a run one episode at a time means sitting through forty invocations and remembering
which ones are done.

Two properties matter more than speed here:

  - a failing episode does not end the batch. Episode 17 having no candidates recorded is not a
    reason to lose episodes 18..40, so failures are collected and reported at the end.
  - a killed batch resumes. Renders already on disk are skipped unless --overwrite, so re-running
    after a Ctrl-C picks up where it stopped rather than starting over.

Usage:
    misc/yam-misc render-bulk --repo-id lerobot_rollout/my_run --root ~/lerobot_rollout
"""

import argparse
import pathlib
import sys
import traceback
import zipfile


def parse_episodes(spec: str, available: int) -> list:
    """"all" | "3" | "0-9" | "0,3,5-7" -> a sorted list of episode indices.

    Out-of-range indices are an error rather than a silent trim: asking for 0-49 of a 20-episode
    dataset means one of the two numbers is wrong, and quietly rendering 20 hides which.
    """
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return list(range(available))
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    bad = sorted(e for e in out if not 0 <= e < available)
    if bad:
        raise SystemExit(f"episode(s) {bad} out of range -- the dataset has {available} (0..{available - 1})")
    return sorted(out)


def zip_folder(folder: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Zip the rendered folder, STORED rather than deflated.

    The payload is h264 in mp4: already compressed, so deflate spends CPU on every byte to save
    roughly none. Stored keeps a large batch quick to write and quick to open.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder))
    return dest


def main() -> None:
    from misc.dataset_reader import DatasetReader
    from misc.render_deploy_samples import build_parser
    from misc.render_deploy_samples import render

    p = build_parser()
    p.description = __doc__
    # --episode/--out are per-episode; bulk decides both. Kept in the parser (harmless) so the
    # option surface stays identical to the single-episode renderer.
    p.add_argument("--episodes", default="all", help='which to render: "all", "3", "0-9", "0,3,5-7"')
    p.add_argument("--out-dir", default=None, help="folder for the mp4s (default ~/<dataset>_renders)")
    p.add_argument("--zip", dest="zip_to", default=None, help="zip path (default <out-dir>.zip)")
    p.add_argument("--no-zip", action="store_true", help="leave the folder unzipped")
    p.add_argument("--overwrite", action="store_true", help="re-render episodes whose mp4 already exists")
    args = p.parse_args()

    name = args.repo_id.rstrip("/").split("/")[-1]
    out_dir = pathlib.Path(args.out_dir).expanduser() if args.out_dir else pathlib.Path.home() / f"{name}_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = DatasetReader(args.repo_id, args.root)
    reader.load()
    episodes = parse_episodes(args.episodes, reader.num_episodes)
    del reader  # the renderer opens its own; holding this one just pins the metadata

    print(f"{len(episodes)} episode(s) -> {out_dir}")
    done, skipped, failed = [], [], []
    for i, ep in enumerate(episodes, 1):
        out = out_dir / f"{name}_ep{ep:03d}.mp4"
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(episodes)}] ep{ep}: exists, skipping")
            skipped.append(ep)
            continue
        args.episode, args.out = ep, str(out)
        print(f"[{i}/{len(episodes)}] ep{ep} -> {out.name}")
        try:
            render(args)
            done.append(ep)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - one bad episode must not end the batch
            failed.append((ep, str(e) or type(e).__name__))
            print(f"[{i}/{len(episodes)}] ep{ep}: FAILED -- {e}", file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)

    print(f"\nrendered {len(done)}, skipped {len(skipped)}, failed {len(failed)}")
    for ep, why in failed:
        print(f"  ep{ep}: {why}", file=sys.stderr)

    if not args.no_zip and (done or skipped):
        dest = pathlib.Path(args.zip_to).expanduser() if args.zip_to else out_dir.with_suffix(".zip")
        zip_folder(out_dir, dest)
        print(f"zipped -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
