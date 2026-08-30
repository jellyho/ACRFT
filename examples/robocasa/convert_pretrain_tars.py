"""Convert the RoboCasa 365 *pretrain*-split LeRobot tars (already downloaded locally) to v3.0.

The pretrain split of ``nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`` holds 300 tasks
(65 atomic + 235 composite), each a pre-converted LeRobot **v2.1** dataset shipped as
``pretrain/{atomic,composite}/<Task>/<date>/lerobot.tar``. This mirrors the target-split pipeline in
``prepare_robocasa365.py`` but (a) works off already-downloaded local tars (no registry filter — the
pretrain split is not in the RoboCasa target registry) and (b) is disk-frugal: after each task is
upgraded to v3.0 it deletes the extracted v2.1 backup and (optionally) the source tar.

    # after the tars are snapshot_download-ed to --tar-root:
    uv run examples/robocasa/convert_pretrain_tars.py \
        --tar-root /data5/jellyho/robocasa365_pretrain \
        --out-dir  /data5/jellyho/robocasa365_pretrain_v3 --rm-tar

Resumable: a task already at v3.0 in --out-dir is skipped. Run it repeatedly (even while the
download is still going) to process tars as they land.
"""

import argparse
import logging
import pathlib
import shutil
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_pretrain_tars")


def codebase_version(dataset_dir: pathlib.Path) -> str | None:
    import json

    info = dataset_dir / "meta" / "info.json"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text()).get("codebase_version")
    except (json.JSONDecodeError, OSError):
        return None


def extract_tar(task: str, tar: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    dest = out_dir / task
    if dest.exists():
        shutil.rmtree(dest)
    tmp = out_dir / f"{task}__extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r") as t:
        t.extractall(path=tmp)
    lerobot_dir = tmp / "lerobot"
    if not lerobot_dir.is_dir():
        cands = list(tmp.rglob("meta/info.json"))
        if not cands:
            raise RuntimeError(f"[{task}] no LeRobot dataset inside {tar}")
        lerobot_dir = cands[0].parent.parent
    shutil.move(str(lerobot_dir), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def upgrade(task: str, out_dir: pathlib.Path) -> None:
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    convert_dataset(repo_id=task, root=str(out_dir), push_to_hub=False, force_conversion=True)
    backup = out_dir / f"{task}_old"
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-root", type=pathlib.Path, required=True, help="root the pretrain tars were downloaded to")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="where the v3.0 per-task datasets go")
    ap.add_argument("--rm-tar", action="store_true", help="delete each source tar after a successful convert")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    tars = sorted((a.tar_root / "pretrain").rglob("lerobot.tar"))
    if a.limit:
        tars = tars[: a.limit]
    logger.info("found %d pretrain tar(s) under %s", len(tars), a.tar_root)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    failures: list[tuple[str, str]] = []
    for i, tar in enumerate(tars, 1):
        # .../pretrain/<cat>/<Task>/<date>/lerobot.tar  -> task name is 2 levels up
        task = tar.parent.parent.name
        if codebase_version(a.out_dir / task) == "v3.0":
            skipped += 1
            continue
        logger.info("=== (%d/%d) %s ===", i, len(tars), task)
        try:
            extract_tar(task, tar, a.out_dir)
            upgrade(task, a.out_dir)
            if a.rm_tar:
                tar.unlink(missing_ok=True)
            done += 1
        except Exception as e:
            logger.exception("[%s] FAILED: %s", task, e)
            failures.append((task, str(e)))

    logger.info("done: %d converted, %d already-v3 skipped, %d failed", done, skipped, len(failures))
    if failures:
        logger.error("failures:\n  " + "\n  ".join(f"{t}: {e}" for t, e in failures))


if __name__ == "__main__":
    main()
