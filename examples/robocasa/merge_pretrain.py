"""Merge the converted RoboCasa365 pretrain v3 tasks into ONE LeRobot dataset.

The Human300 pretrain split is 300 per-task v3 datasets (from convert_pretrain_tars.py). Rather than a
multi-dataset loader, aggregate them into a single LeRobot dataset with LeRobot's aggregate_datasets;
training then uses the standard single-repo path. The merged dataset's `task` per frame is preserved,
so PromptFromLeRobotTask works off the unified tasks table.

Point HF_LEROBOT_HOME at --out's PARENT and use --name as repo_id to train locally, e.g.
    HF_LEROBOT_HOME=/data5/jellyho/robocasa365_pretrain_merged \
    ... scripts/train.py pi05_robocasa_pretrain ...

    uv run examples/robocasa/merge_pretrain.py \
        --v3-dir /data5/jellyho/robocasa365_pretrain_v3 \
        --out    /data5/jellyho/robocasa365_pretrain_merged/robocasa365_pretrain_human300
"""

import argparse
import pathlib

from lerobot.datasets.aggregate import aggregate_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-dir", type=pathlib.Path, default=pathlib.Path("/data5/jellyho/robocasa365_pretrain_v3"))
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/robocasa365_pretrain_merged/robocasa365_pretrain_human300"),
    )
    ap.add_argument("--name", default="robocasa365_pretrain_human300")
    a = ap.parse_args()

    tasks = sorted(p.name for p in a.v3_dir.iterdir() if p.is_dir() and not p.name.endswith("_v30"))
    roots = [a.v3_dir / t for t in tasks]
    print(f"merging {len(tasks)} pretrain tasks -> {a.out}", flush=True)
    aggregate_datasets(repo_ids=tasks, aggr_repo_id=a.name, roots=roots, aggr_root=a.out)
    print(f"DONE: merged dataset at {a.out}", flush=True)


if __name__ == "__main__":
    main()
