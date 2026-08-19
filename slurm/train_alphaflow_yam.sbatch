#!/usr/bin/env bash
# alpha-Flow pi05 YAM main run (200k, B200) — pinned WORKTREE launcher.
# Runs from /data5/jellyho/ACRFT/openpi-alphaflow (branch fql-one-step-actor), NOT the shared
# checkout: peer sessions switch that checkout's branch mid-flight, which already killed one
# submission (integration has no pi0_alphaflow). Shares the main repo's venv read-only (--no-sync).
#SBATCH --job-name=af_yam_200k
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/af_yam_200k.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
# the venv's openpi is an editable install of the SHARED checkout; put this worktree's src first
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
exec uv run --no-sync python scripts/train.py pi05_yam_lego_taxi_alphaflow \
  --exp-name=yam_alphaflow_200k --project-name yam-rlt --wandb-entity RSS-PFT_RLLAB \
  --num-workers 12 --data.assets.asset-id jellyho/yam_lego_taxi
