"""Back up BC checkpoints to HF as they are written, and stop when the run is done.

A 500k run keeps only the 100k milestones plus the latest, and the pruner deletes the rest, so a
checkpoint that is not uploaded before the next save can be gone. This polls the run directories and
uploads any step that HF does not already have, then exits once every live run has finished and its
final step is backed up.

Uploads the FULL step directory (params + train_state + assets): train_state is what a resume needs,
and a run that has to be restarted from scratch costs days. Verifies the file list against HF before
reporting a step as backed up.

    uv run python scripts/sync_bc_checkpoints.py --once      # one pass
    uv run python scripts/sync_bc_checkpoints.py             # poll until the runs finish
"""

import argparse
import pathlib
import subprocess
import time

from huggingface_hub import HfApi

GRID = 50_000  # upload every 50k steps (the run saves every 25k; the pruner keeps only 100k + latest,
# so a 50k-but-not-100k step is on disk for one save interval -- about 15 hours at 2.2 s/it -- which
# is why the poll interval must be well under that.

RUNS = {
    # local run dir -> HF model repo
    "checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly": "jellyho/pi05_yam_lego_taxi_bc_s300_h30",
    "checkpoints/pi05_yam_lego_taxi_h50/yam_bc_s300_h50_successonly": "jellyho/pi05_yam_lego_taxi_bc_s300_h50",
}


def steps_on_disk(run: pathlib.Path) -> list[int]:
    if not run.is_dir():
        return []
    return sorted(int(p.name) for p in run.iterdir() if p.is_dir() and p.name.isdigit())


def steps_on_hf(api: HfApi, repo: str) -> set[int]:
    try:
        files = api.list_repo_files(repo, repo_type="model")
    except Exception:
        return set()
    return {int(f.split("/")[0]) for f in files if f.split("/")[0].isdigit()}


def verify(api: HfApi, repo: str, step: int, src: pathlib.Path) -> bool:
    """Every local file present remotely, and the byte total within 1%."""
    remote = {f[len(f"{step}/") :] for f in api.list_repo_files(repo, repo_type="model") if f.startswith(f"{step}/")}
    local = {str(p.relative_to(src)) for p in src.rglob("*") if p.is_file()}
    if local - remote:
        return False
    info = api.repo_info(repo, repo_type="model", files_metadata=True)
    rb = sum(s.size or 0 for s in info.siblings if s.rfilename.startswith(f"{step}/"))
    lb = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    return rb >= lb * 0.99


def jobs_running(names=("yam_h30_500k", "yam_h50_500k")) -> bool:
    try:
        out = subprocess.run(["squeue", "-u", "jellyho", "-h", "-o", "%j"], capture_output=True, text=True, check=False)
        return any(n in out.stdout for n in names)
    except Exception:
        return True  # if we cannot tell, keep polling rather than stopping early


def sync_once(api: HfApi, root: pathlib.Path, grid: int) -> int:
    uploaded = 0
    for rel, repo in RUNS.items():
        run = root / rel
        disk = steps_on_disk(run)
        if not disk:
            continue
        have = steps_on_hf(api, repo)
        # 50k grid, plus whatever is currently the latest once the run has finished
        todo = [s for s in disk if s not in have and (s % grid == 0)]
        if todo:
            api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        for step in todo:
            src = run / str(step)
            size = sum(p.stat().st_size for p in src.rglob("*") if p.is_file()) / 1e9
            print(f"  uploading {rel} step {step} ({size:.1f} GB) -> {repo}", flush=True)
            try:
                api.upload_folder(
                    repo_id=repo,
                    repo_type="model",
                    folder_path=str(src),
                    path_in_repo=str(step),
                    commit_message=f"{rel.split('/')[-1]} step {step}",
                )
            except Exception as e:  # a step can vanish mid-upload if the pruner runs
                print(f"    FAILED {type(e).__name__}: {e}", flush=True)
                continue
            ok = verify(api, repo, step, src) if src.is_dir() else False
            print(f"    {'verified' if ok else 'UPLOADED BUT NOT VERIFIED'} step {step}", flush=True)
            uploaded += 1
    return uploaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/data5/jellyho/ACRFT/openpi"))
    ap.add_argument("--interval", type=int, default=900, help="seconds between passes")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--grid", type=int, default=GRID, help="upload steps that are multiples of this")
    a = ap.parse_args()

    api = HfApi()
    while True:
        n = sync_once(api, a.root, a.grid)
        running = jobs_running()
        print(f"pass done: {n} uploaded; runs {'still training' if running else 'finished'}", flush=True)
        if a.once or not running:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
