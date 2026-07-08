"""Download RoboCasa 365 *target*-split human demonstrations and convert them to
LeRobot dataset format **v3.0**.

Pipeline per task:

1. Download the pre-converted LeRobot dataset tar (``.../lerobot.tar``) for the task's
   ``target`` split from the (public) Hugging Face dataset
   ``nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos``. These tars already contain a
   LeRobot dataset, but in the **v2.1** format (RoboCasa's converter pins ``lerobot==0.3.3``),
   so no MuJoCo sim replay is required here.
2. Extract it to ``<output_dir>/<Task>``.
3. Upgrade it in place to LeRobot format **v3.0** using lerobot's official
   ``convert_dataset_v21_to_v30`` (lerobot 0.4.4, CODEBASE_VERSION "v3.0").

The set of target tasks is parsed from the RoboCasa registry in the ``third_party/robocasa``
submodule; the actual tar paths (task/date) are discovered directly from the HF repo, so the
script is robust to date drift between the registry and the published dataset. All target
tasks that exist in both are processed by default.

NOTE: RoboCasa's own ``download_datasets.py`` points at ``nvidia/PhysicalAI-Robotics-Kitchen-
Sim-Demos``, which is not published (HF returns RepoNotFound). The live, public mirror is
``nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`` and is used here. It also drops the
``v1.0/`` path prefix that the registry uses.

Requirements (already provided by the main openpi environment after ``uv sync``):
    - huggingface_hub  (download)
    - lerobot==0.4.4   (v2.1 -> v3.0 conversion)

The dataset is public, so no Hugging Face login is required. (If you have a token configured
it will simply be used.)

Example usage (run inside the openpi venv):
    uv run examples/robocasa/prepare_robocasa365.py --output-dir /data5/jellyho/robocasa365
    uv run examples/robocasa/prepare_robocasa365.py --output-dir /data/rc365 --tasks OpenDrawer CloseFridge
    uv run examples/robocasa/prepare_robocasa365.py --output-dir /data/rc365 --limit 1
"""

import argparse
import json
import logging
from pathlib import Path
import re
import shutil
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_robocasa365")

# Public HF dataset that hosts the pre-converted (v2.1) LeRobot tars for RoboCasa 365.
HF_REPO_ID = "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos"

# Path to the RoboCasa dataset registry inside the vendored submodule.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "third_party" / "robocasa" / "robocasa" / "utils" / "dataset_registry.py"


def codebase_version(dataset_dir: Path) -> str | None:
    """Return the LeRobot ``codebase_version`` of a local dataset, or None if absent/unreadable."""
    info = dataset_dir / "meta" / "info.json"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text()).get("codebase_version")
    except (json.JSONDecodeError, OSError):
        return None


def parse_target_tasks() -> set[str]:
    """Return the set of task names that expose a ``target`` split in the RoboCasa registry.

    Parses ``dataset_registry.py`` textually so we don't need to import RoboCasa (which pins a
    conflicting ``lerobot==0.3.3``).
    """
    if not _REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"RoboCasa registry not found at {_REGISTRY_PATH}. "
            "Initialize the submodule: `git submodule update --init third_party/robocasa`."
        )
    txt = _REGISTRY_PATH.read_text()
    tasks: set[str] = set()
    for m in re.finditer(r"\n    (\w+)=dict\(", txt):
        name = m.group(1)
        start = m.end()
        nxt = re.search(r"\n    \w+=dict\(", txt[start:])
        block = txt[start : start + (nxt.start() if nxt else len(txt))]
        if re.search(r"target=dict\(", block):
            tasks.add(name)
    return tasks


def discover_tar_paths() -> dict[str, str]:
    """Return ``{task_name: repo_relative_tar_path}`` for the HF dataset's target split.

    Walks only ``target/{atomic,composite}/<task>/<date>`` (a handful of API calls) instead of
    listing the entire repo tree, which holds hundreds of thousands of episode files.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    api = HfApi()
    out: dict[str, str] = {}
    for cat in ("atomic", "composite"):
        base = f"target/{cat}"
        try:
            task_entries = list(api.list_repo_tree(HF_REPO_ID, path_in_repo=base, repo_type="dataset", recursive=False))
        except Exception as e:  # noqa: BLE001 - a missing category is not fatal
            logger.warning("could not list %s: %s", base, e)
            continue
        task_dirs = [e.path for e in task_entries if isinstance(e, RepoFolder)]
        logger.info("scanning %d %s target task dir(s) ...", len(task_dirs), cat)
        for tpath in task_dirs:
            task = tpath.split("/")[-1]
            date_entries = api.list_repo_tree(HF_REPO_ID, path_in_repo=tpath, repo_type="dataset", recursive=False)
            for d in date_entries:
                if isinstance(d, RepoFolder):
                    # Layout is <task>/<date>/lerobot.tar (verified). Take the first date dir.
                    out[task] = f"{d.path}/lerobot.tar"
                    break
    return out


def download_and_extract(task: str, tar_path: str, output_dir: Path, *, overwrite: bool) -> Path:
    """Download ``tar_path`` from HF and extract its LeRobot dataset to ``<output_dir>/<task>``."""
    from huggingface_hub import hf_hub_download

    dest = output_dir / task
    if dest.exists() and not overwrite:
        logger.info("[%s] dataset already present at %s (use --overwrite to redownload)", task, dest)
        return dest
    if dest.exists():
        shutil.rmtree(dest)

    logger.info("[%s] downloading %s ...", task, tar_path)
    local_tar = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=tar_path, revision="main")

    logger.info("[%s] extracting ...", task)
    tmp = output_dir / f"{task}__extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r") as tar:
        tar.extractall(path=tmp)  # noqa: S202 - trusted first-party archive

    # The tar root is a single `lerobot/` directory holding meta/, data/, videos/.
    lerobot_dir = tmp / "lerobot"
    if not lerobot_dir.is_dir():
        candidates = list(tmp.rglob("meta/info.json"))
        if not candidates:
            raise RuntimeError(f"[{task}] could not locate a LeRobot dataset inside {local_tar}")
        lerobot_dir = candidates[0].parent.parent

    shutil.move(str(lerobot_dir), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    logger.info("[%s] extracted to %s", task, dest)
    return dest


def upgrade_to_v30(task: str, output_dir: Path, *, keep_backup: bool) -> None:
    """Upgrade ``<output_dir>/<task>`` from LeRobot v2.1 to v3.0 in place."""
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    dataset_dir = output_dir / task
    version = codebase_version(dataset_dir)
    if version is None:
        raise FileNotFoundError(f"[{task}] no readable meta/info.json at {dataset_dir}; download/extract first.")
    if version == "v3.0":
        logger.info("[%s] already v3.0 — skipping conversion.", task)
        return

    logger.info("[%s] converting %s -> v3.0 ...", task, version)
    # The converter resolves the dataset at ``<root>/<repo_id>`` and writes the v3.0 result back
    # there, leaving the original at ``<root>/<repo_id>_old``.
    convert_dataset(repo_id=task, root=str(output_dir), push_to_hub=False, force_conversion=True)

    backup = output_dir / f"{task}_old"
    if backup.exists() and not keep_backup:
        shutil.rmtree(backup, ignore_errors=True)
    logger.info("[%s] done -> %s (v3.0)", task, dataset_dir)


def push_task_to_hub(task: str, output_dir: Path, repo_id: str, *, private: bool) -> str:
    """Upload the local v3.0 dataset ``<output_dir>/<task>`` to ``repo_id`` on the HF Hub.

    Uses HfApi directly (create_repo + upload_folder + version tag) rather than
    ``LeRobotDataset.push_to_hub`` to avoid re-loading the dataset from disk, then generates the
    LeRobot dataset card the same way ``push_to_hub`` would. Result: a dataset repo whose files are
    the v3.0 dataset, tagged ``v3.0``, with a proper LeRobot dataset card.
    """
    from huggingface_hub import HfApi

    dataset_dir = output_dir / task
    if codebase_version(dataset_dir) != "v3.0":
        raise RuntimeError(f"[{task}] {dataset_dir} is not a v3.0 dataset; convert it first.")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info("[%s] uploading -> https://huggingface.co/datasets/%s ...", task, repo_id)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        ignore_patterns=["images/"],
        commit_message=f"Upload RoboCasa 365 target/human '{task}' (LeRobot v3.0)",
    )
    # Tag the codebase version so `LeRobotDataset(repo_id)` resolves the v3.0 revision.
    try:
        api.create_tag(repo_id=repo_id, tag="v3.0", repo_type="dataset", exist_ok=True)
    except Exception as e:  # noqa: BLE001 - a missing tag is non-fatal
        logger.warning("[%s] could not create v3.0 tag: %s", task, e)
    push_dataset_card(task, output_dir, repo_id)
    logger.info("[%s] pushed.", task)
    return repo_id


def push_dataset_card(task: str, output_dir: Path, repo_id: str) -> None:
    """Generate and upload the LeRobot dataset card (README) for an already-uploaded repo.

    Reads ``meta/info.json`` and renders the same card ``LeRobotDataset.push_to_hub`` would, without
    loading the full dataset. Safe to call on its own to backfill cards on existing repos.
    """
    import json

    from lerobot.datasets.utils import create_lerobot_dataset_card

    info_path = output_dir / task / "meta" / "info.json"
    if not info_path.exists():
        logger.warning("[%s] no meta/info.json; skipping dataset card.", task)
        return
    info = json.loads(info_path.read_text())
    card = create_lerobot_dataset_card(
        tags=["robocasa", "robocasa365", "robosuite", "kitchen", "manipulation"],
        dataset_info=info,
        license="apache-2.0",
        repo_id=repo_id,
    )
    card.push_to_hub(repo_id=repo_id, repo_type="dataset")
    logger.info("[%s] dataset card pushed.", task)


def build_collection(title: str, namespace: str, repo_ids: list[str], *, private: bool) -> str | None:
    """Create (or reuse) a HF collection and add each dataset repo to it. Returns the slug."""
    from huggingface_hub import HfApi

    api = HfApi()
    coll = api.create_collection(title=title, namespace=namespace, private=private, exists_ok=True)
    for rid in repo_ids:
        try:
            api.add_collection_item(coll.slug, item_id=rid, item_type="dataset", exists_ok=True)
        except Exception as e:  # noqa: BLE001 - keep adding the rest
            logger.warning("could not add %s to collection: %s", rid, e)
    logger.info("Collection ready: https://huggingface.co/collections/%s", coll.slug)
    return coll.slug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store the converted datasets.")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None, help="Subset of task names. Defaults to all target tasks."
    )
    parser.add_argument("--download-only", action="store_true", help="Download + extract but skip the v3.0 upgrade.")
    parser.add_argument("--convert-only", action="store_true", help="Skip download; only run the v3.0 upgrade.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload/extract even if the dataset exists.")
    parser.add_argument("--keep-backup", action="store_true", help="Keep the pre-upgrade v2.1 copy (<task>_old).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tasks (for testing).")
    # --- Hugging Face Hub upload (runs after download/convert) ---
    parser.add_argument(
        "--push-to-hub", action="store_true", help="After converting, upload each v3.0 dataset to the HF Hub."
    )
    parser.add_argument(
        "--push-cards-only", action="store_true",
        help="Backfill: only (re)generate and push the LeRobot dataset card to already-uploaded "
             "repos. Skips download/convert/upload.",
    )
    parser.add_argument(
        "--hf-user", type=str, default=None, help="HF username/org for repos. Defaults to the logged-in user."
    )
    parser.add_argument(
        "--repo-prefix", type=str, default="robocasa365", help="Repo name prefix: <hf-user>/<prefix>-<Task>."
    )
    parser.add_argument("--private", action="store_true", help="Create the HF repos/collection as private.")
    parser.add_argument(
        "--collection-title",
        type=str,
        default="RoboCasa 365 Target (Human) — LeRobot v3.0",
        help="Title of the HF collection to create. Use --no-collection to skip.",
    )
    parser.add_argument("--no-collection", action="store_true", help="Do not create a HF collection when pushing.")
    args = parser.parse_args()

    registry_tasks = parse_target_tasks()
    logger.info("RoboCasa registry lists %d target tasks.", len(registry_tasks))

    if args.convert_only or args.push_cards_only:
        # Work from what's already on disk.
        selected_tasks = sorted(p.name for p in args.output_dir.iterdir() if (p / "meta" / "info.json").exists())
        tar_paths: dict[str, str] = {}
    else:
        available = discover_tar_paths()
        logger.info("HF dataset %s exposes %d target-task tars.", HF_REPO_ID, len(available))
        missing = sorted(registry_tasks - set(available))
        if missing:
            logger.warning("%d registry target task(s) have no tar in the HF repo: %s", len(missing), missing)
        extra = sorted(set(available) - registry_tasks)
        if extra:
            logger.info("%d task(s) in the HF repo are not registry target tasks (ignored): %s", len(extra), extra)
        tar_paths = {t: p for t, p in available.items() if t in registry_tasks}
        selected_tasks = sorted(tar_paths)

    if args.tasks:
        unknown = [t for t in args.tasks if t not in selected_tasks]
        if unknown:
            raise SystemExit(
                f"Requested task(s) not available: {unknown}\nAvailable:\n  " + "\n  ".join(sorted(selected_tasks))
            )
        selected_tasks = [t for t in selected_tasks if t in args.tasks]
    if args.limit is not None:
        selected_tasks = selected_tasks[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %d task(s) -> %s", len(selected_tasks), args.output_dir)

    failures: list[tuple[str, str]] = []
    skipped = 0
    if not args.push_cards_only:
        for i, task in enumerate(selected_tasks, 1):
            logger.info("=== (%d/%d) %s ===", i, len(selected_tasks), task)
            # Resume support: a fully converted (v3.0) task is skipped entirely (no redownload,
            # no reconversion) unless --overwrite is given.
            want_convert = not args.download_only
            if not args.overwrite and codebase_version(args.output_dir / task) == "v3.0" and want_convert:
                logger.info("[%s] already v3.0 — skipping.", task)
                skipped += 1
                continue
            try:
                if not args.convert_only:
                    download_and_extract(task, tar_paths[task], args.output_dir, overwrite=args.overwrite)
                if want_convert:
                    upgrade_to_v30(task, args.output_dir, keep_backup=args.keep_backup)
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                logger.exception("[%s] FAILED: %s", task, e)
                failures.append((task, str(e)))

        done = len(selected_tasks) - len(failures) - skipped
        logger.info("Finished. %d converted, %d already-done skipped, %d failed.", done, skipped, len(failures))

    push_failures: list[tuple[str, str]] = []
    if args.push_to_hub or args.push_cards_only:
        from huggingface_hub import HfApi

        user = args.hf_user or HfApi().whoami()["name"]
        cards_only = args.push_cards_only
        verb = "cards" if cards_only else "datasets"
        logger.info("Pushing %s to HF Hub under '%s' (%s) ...", verb, user, "private" if args.private else "public")
        pushed: list[str] = []
        # Only push tasks that are actually v3.0 on disk (skips any that failed to convert).
        pushable = [t for t in selected_tasks if codebase_version(args.output_dir / t) == "v3.0"]
        not_ready = sorted(set(selected_tasks) - set(pushable))
        if not_ready:
            logger.warning("%d task(s) not v3.0 on disk, skipping: %s", len(not_ready), not_ready)
        for i, task in enumerate(pushable, 1):
            repo_id = f"{user}/{args.repo_prefix}-{task}"
            logger.info("=== push %s (%d/%d) %s ===", verb, i, len(pushable), task)
            try:
                if cards_only:
                    push_dataset_card(task, args.output_dir, repo_id)
                else:
                    push_task_to_hub(task, args.output_dir, repo_id, private=args.private)
                pushed.append(repo_id)
            except Exception as e:  # noqa: BLE001 - keep pushing the rest
                logger.exception("[%s] PUSH FAILED: %s", task, e)
                push_failures.append((task, str(e)))

        if pushed and not args.no_collection:
            try:
                build_collection(args.collection_title, user, pushed, private=args.private)
            except Exception as e:  # noqa: BLE001 - non-fatal
                logger.exception("Collection creation FAILED: %s", e)
        logger.info("Push finished. %d uploaded, %d failed.", len(pushed), len(push_failures))

    if failures or push_failures:
        if failures:
            logger.error("Failed (convert):\n  " + "\n  ".join(f"{t}: {e}" for t, e in failures))
        if push_failures:
            logger.error("Failed (push):\n  " + "\n  ".join(f"{t}: {e}" for t, e in push_failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
