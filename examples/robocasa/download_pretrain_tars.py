"""Download the RoboCasa365 pretrain-split lerobot.tar for every task, robustly.

snapshot_download on this repo is unreliable: it has ~270k files (300 tasks x ~900 each), so the
single repo-wide file listing truncates and only a fraction of the tars come down. Instead we resolve
each task's tar path explicitly (list only that task dir, non-recursive, to find its <date> subdir)
and hf_hub_download it one by one -- no repo-wide listing. Skips tars already present locally.

    uv run examples/robocasa/download_pretrain_tars.py --out /data5/jellyho/robocasa365_pretrain
"""

import argparse
import concurrent.futures as cf
import pathlib

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/data5/jellyho/robocasa365_pretrain"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    api = HfApi()

    # task dirs (non-recursive listing per split; fast)
    tasks = []
    for split in ("atomic", "composite"):
        # each child dir is a task, e.g. pretrain/atomic/AdjustToasterOvenTemperature
        tasks.extend(
            it.path
            for it in api.list_repo_tree(REPO, path_in_repo=f"pretrain/{split}", repo_type="dataset", recursive=False)
        )
    print(f"{len(tasks)} pretrain tasks", flush=True)

    def resolve_tar(task_path):
        # each task dir contains one <date> subdir; the tar is <task>/<date>/lerobot.tar
        for it in api.list_repo_tree(REPO, path_in_repo=task_path, repo_type="dataset", recursive=False):
            name = it.path.split("/")[-1]
            if name.isdigit():  # a date dir like 20250829
                return f"{task_path}/{name}/lerobot.tar"
        return None

    def get(task_path):
        rel = resolve_tar(task_path)
        if rel is None:
            return (task_path, "NO_DATE")
        local = a.out / rel
        if local.exists():
            return (task_path, "have")
        try:
            hf_hub_download(REPO, rel, repo_type="dataset", local_dir=str(a.out))
            return (task_path, "downloaded")
        except Exception as e:
            return (task_path, f"ERR {type(e).__name__}")

    done = have = err = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (task, status) in enumerate(ex.map(get, tasks)):
            if status == "downloaded":
                done += 1
            elif status == "have":
                have += 1
            elif status.startswith("ERR") or status == "NO_DATE":
                err += 1
                print(f"  ! {task}: {status}", flush=True)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)}  (new {done} / have {have} / err {err})", flush=True)
    print(f"DONE: {done} downloaded, {have} already present, {err} errors", flush=True)


if __name__ == "__main__":
    main()
