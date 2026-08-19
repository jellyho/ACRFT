"""Push the serving half of checkpoint 50000 to a private HF model repo.

Only `params/` and `assets/` go up. `policy_config.py:57` reads params and `:64`
reads norm stats from the checkpoint's own assets dir, so those two are
self-sufficient for serving. `train_state/` (30 GB) exists to resume training and
is deliberately left behind.

    python push_checkpoint_to_hf.py                     # step 50000 (default)
    python push_checkpoint_to_hf.py --step 30000        # the other candidate
    python push_checkpoint_to_hf.py --step 30000 --dry-run
"""

import argparse
import pathlib
import tempfile

from huggingface_hub import HfApi

CKPT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "checkpoints/pi05_yam_cable_tie/cable_tie_bc"
PATTERNS = ["params/**", "assets/**", "_CHECKPOINT_METADATA"]

# One repo per checkpoint rather than step subfolders in a shared repo: each repo
# then downloads straight into a usable --policy.dir, and uploading a second
# checkpoint cannot disturb an upload already in flight for the first.
def repo_for(step: int) -> str:
    return f"Gwanwoo/pi05_yam_cable_tie_bc_{step // 1000}k"

CARD = """---
license: apache-2.0
tags: [robotics, openpi, pi05, yam, behaviour-cloning]
---

# pi05_yam_cable_tie — BC finetune, step {step}

Serving half of an ACRFT checkpoint. `train_state/` is intentionally absent, so
this can run a policy server but cannot resume training.

- config: `pi05_yam_cable_tie` — must match exactly at serve time
- data: 100 episodes / 157,023 frames, 30 fps, YAM bimanual teleoperation
- action: 14-d — relative joint displacement per arm (6 joints), grippers absolute
- chunk: 30 steps = 1.0 s at 30 fps
- BC loss at step {step}: {loss}

## Serve

```bash
huggingface-cli download {repo} --local-dir ckpt_{step}
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \\
    --policy.config pi05_yam_cable_tie --policy.dir ckpt_{step}
```

Observation keys the server expects: `observation/image` (agentview),
`observation/wrist_image` (left wrist), `observation/image_right` (right wrist),
`observation/state` (42-d), `prompt`.

Returns `(30, 14)` **absolute** joint targets. Do not add the current state back.
"""


# Training-batch BC loss at each saved step, for the model card.
LOSS = {10000: "0.0088", 20000: "0.0062", 30000: "0.0048", 40000: "0.0042", 50000: "0.0041"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=50000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    SRC = CKPT_ROOT / str(args.step)
    REPO = repo_for(args.step)
    if not (SRC / "params").is_dir():
        avail = sorted(d.name for d in CKPT_ROOT.iterdir() if (d / "params").is_dir())
        raise SystemExit(f"no params/ under {SRC}; available steps: {avail}")

    files = [p for pat in ("params", "assets") for p in (SRC / pat).rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"source : {SRC}")
    print(f"upload : {len(files)} files, {total / 2**30:.1f} GiB  (train_state excluded)")
    print(f"repo   : {REPO}  (private)")
    if args.dry_run:
        return

    api = HfApi()
    print("whoami :", api.whoami()["name"])
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        card = pathlib.Path(td) / "README.md"
        card.write_text(CARD.format(step=args.step, repo=REPO, loss=LOSS.get(args.step, "see notes.md")))
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=REPO, repo_type="model")

    api.upload_folder(
        repo_id=REPO,
        repo_type="model",
        folder_path=str(SRC),
        allow_patterns=PATTERNS,
        commit_message="pi05_yam_cable_tie BC finetune, step 50000 (serving weights only)",
    )
    print(f"done: https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
