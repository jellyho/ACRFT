"""Watch a training run's checkpoint dir and upload each step to HF as soon as it's written.

Orbax writes a step dir atomically-ish; we treat a step as ready once its `params/` subdir exists
and the dir's mtime has been stable for a settle window (so we never upload a half-written ckpt).
Each ready step is uploaded once to a per-step path in the HF model repo.

    uv run python scripts/watch_upload_ckpts.py \
        --ckpt-dir checkpoints/pi05_yam_lego_taxi_rlt/yam_lego_taxi_rlt_s300_successonly \
        --repo jellyho/pi05_yam_lego_taxi_rlt_s300 --steps 50000 100000 150000 200000
"""

import argparse
import pathlib
import time

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=pathlib.Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--settle", type=int, default=90, help="seconds the step dir must be unmodified before upload")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=24.0)
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    done = set()
    t0 = time.monotonic()
    print(f"watching {a.ckpt_dir} -> {a.repo} for steps {a.steps}", flush=True)
    while len(done) < len(a.steps) and (time.monotonic() - t0) < a.max_hours * 3600:
        for step in a.steps:
            if step in done:
                continue
            d = a.ckpt_dir / str(step)
            params = d / "params"
            if not params.exists() or not any(params.iterdir()):
                continue
            # settle: newest mtime under the step dir must be older than `settle`
            newest = max((p.stat().st_mtime for p in d.rglob("*")), default=0)
            if time.time() - newest < a.settle:
                continue
            try:
                api.upload_folder(
                    folder_path=str(d),
                    repo_id=a.repo,
                    repo_type="model",
                    path_in_repo=str(step),
                    commit_message=f"checkpoint step {step} (s300, live upload during training)",
                )
                done.add(step)
                print(f"UPLOADED step {step} -> {a.repo}/{step}", flush=True)
            except Exception as e:
                print(f"step {step} upload failed ({type(e).__name__}: {e}); will retry", flush=True)
        time.sleep(a.poll)
    print(f"watcher done: uploaded {sorted(done)}", flush=True)


if __name__ == "__main__":
    main()
