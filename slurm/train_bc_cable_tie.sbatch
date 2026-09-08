#!/usr/bin/env bash
# BC finetune of pi0.5 on the YAM cable-tie teleop set (jellyho/yam_cable_tie: 160 episodes,
# 251,080 frames, 30 fps, three cameras -- the same rig as the lego set).
#
#   sbatch                                   slurm/train_bc_cable_tie.sbatch   # all 160 episodes
#   sbatch --export=ALL,SUCCESS=1,NAME=...   slurm/train_bc_cable_tie.sbatch   # success-only
#
# SUCCESS=1 resolves 150 of the 160 episodes. It did not until the dataset's v3.0 TAG was moved onto
# main on 2026-09-07: the verdicts were pushed on 09-02, but lerobot resolves the tag for every read
# (CODEBASE_VERSION, lerobot_dataset.py:83) and the tag still pointed before that commit, so
# next.success / next.done were invisible to training while the Hub page showed them.
#SBATCH --job-name=bc_cable
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/bc_cable_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src:/data5/jellyho/ACRFT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
SUCC_ARGS=""
if [ -n "${SUCCESS:-}" ]; then SUCC_ARGS="--data.success-only"; fi
echo "[$(date +%T)] bc_cable starting on $(hostname) (success_only=${SUCCESS:-0})"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train.py pi05_yam_cable_tie \
  --exp-name="${NAME:-yam_bc_cable_tie_all160}" --project-name yam-rlt --wandb-entity jellyho_ \
  --checkpoint-base-dir /data1/jellyho/acrft_ckpts --overwrite \
  --num-train-steps "${STEPS:-100000}" --save-interval "${SAVE:-25000}" ${SUCC_ARGS} \
  --num-workers 12
