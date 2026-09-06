#!/usr/bin/env bash
# BC finetune of pi0.5 on the YAM cable-tie teleop set (jellyho/yam_cable_tie: 160 episodes,
# 251,080 frames, 30 fps, three cameras -- the same rig as the lego set).
#
#   sbatch                                   slurm/train_bc_cable_tie.sbatch   # all 160 episodes
#   sbatch --export=ALL,SUCCESS=1,NAME=...   slurm/train_bc_cable_tie.sbatch   # success-only
#
# SUCCESS=1 does not work yet, and fails loudly rather than quietly: this dataset carries no episode
# verdicts -- it has neither `next.success` nor `next.done` among its frame features, unlike
# jellyho/yam_lego_taxi which has both and resolves to 300 of 347. Verdicts have to be migrated into
# the dataset first (the recorder's `workstation/yam-data migrate-outcomes`); the flag is wired here
# so the run is one word away once they are.
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
