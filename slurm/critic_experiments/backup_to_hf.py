"""Back up everything on this machine that git does not hold, before the server is returned.

Order is by what cannot be recreated. The BC checkpoints are 2.5 GPU-days each and the critics ~3.4
hours each; the 35 GB feature cache is 12 minutes from a dataset that already lives on the Hub, so it
is deliberately NOT uploaded -- spending the remaining time on it would risk the things that matter.
"""
import sys, pathlib, time
from huggingface_hub import HfApi

api = HfApi()
JOBS = [
    # (local path, repo_id, repo_type, path_in_repo)
    ("/NHNHOME/jellyho/jellyho/critics", "jellyho/acrft-cable-tie-critics", "model", "."),
    ("/NHNHOME/jellyho/jellyho/ACRFT/checkpoints/pi05_yam_cable_tie/yam_cable_tie",
     "jellyho/pi05_yam_cable_tie_bc", "model", "withhoming_200k"),
    ("/NHNHOME/jellyho/jellyho/ACRFT/checkpoints/pi05_yam_cable_tie_success/yam_cable_tie_success",
     "jellyho/pi05_yam_cable_tie_bc", "model", "success_200k"),
]
for src, repo, rtype, dest in JOBS:
    p = pathlib.Path(src)
    if not p.exists():
        print(f"SKIP (missing): {src}", flush=True); continue
    print(f"[{time.strftime('%T')}] -> {repo}/{dest}  from {src}", flush=True)
    try:
        api.create_repo(repo, repo_type=rtype, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(p), repo_id=repo, repo_type=rtype, path_in_repo=dest)
        print(f"[{time.strftime('%T')}] DONE {repo}/{dest}", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%T')}] FAILED {repo}/{dest}: {type(e).__name__}: {str(e)[:300]}", flush=True)
print("backup script finished", flush=True)
