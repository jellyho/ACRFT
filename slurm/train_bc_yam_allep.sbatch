#!/usr/bin/env bash
# YAM lego BC on ALL 347 episodes -- failures included.
#
# The deployed policy (yam_bc_s300_h30_successonly) was trained with --data.success-only, i.e. on the
# 300 successes only; the 47 failures were dropped. This is the same config WITHOUT that flag, which
# is the whole method-only-diff: same model, same horizon, same schedule, same steps.
#
# Norm stats are recomputed here rather than borrowed. --data.assets.asset-id is deliberately NOT
# passed: the s300h30 asset's quantiles were computed on the 300 successes, and reusing them on a
# 347-episode run would silently normalize the new data by the old data's spread. That does mean the
# comparison against the s300 policy carries two differences (episodes AND stats), which is inherent
# -- the statistics ARE a property of the data -- and is worth stating when the two are compared.
#SBATCH --job-name=bc_allep
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/bc_allep_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src:/data5/jellyho/ACRFT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
echo "[$(date +%T)] bc_allep starting on $(hostname)"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train.py pi05_yam_lego_taxi \
  --exp-name="${NAME:-yam_bc_s347_h30_allep}" --project-name yam-rlt --wandb-entity jellyho_ \
  --checkpoint-base-dir /data1/jellyho/acrft_ckpts --overwrite \
  --num-train-steps "${STEPS:-200000}" --save-interval "${SAVE:-50000}" \
  --num-workers 12
