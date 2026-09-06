#!/usr/bin/env bash
# alpha-Flow on the 100k schedule. A SCHEDULE-ONLY diff against yam_alphaflow_s300_200k: same data
# (success-only, 300 eps), same norm-stats asset, same official curriculum -- only the budget changes.
#
# The curriculum rescales correctly and that is not an assumption: pi0_alphaflow.py takes
# `progress = step / num_train_steps` through train.py's wants_progress hook (pi0_alphaflow.py:44,
# 83, 221), so the sigmoid anneal completes inside whatever budget is passed. The config's own
# `steps = 200_000` only sets the LR decay horizon, and since peak_lr == decay_lr == 5e-5 that
# schedule is constant anyway -- nothing to keep in sync.
#SBATCH --job-name=af_100k
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/af_100k_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src:/data5/jellyho/ACRFT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
echo "[$(date +%T)] af_100k starting on $(hostname)"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train.py pi05_yam_lego_taxi_alphaflow \
  --exp-name="${NAME:-yam_alphaflow_s300_100k}" --project-name yam-rlt --wandb-entity jellyho_ \
  --checkpoint-base-dir /data1/jellyho/acrft_ckpts --overwrite \
  --num-train-steps "${STEPS:-100000}" --save-interval "${SAVE:-25000}" \
  --data.assets.asset-id jellyho/yam_lego_taxi_s300h30 --data.success-only \
  --num-workers 12
