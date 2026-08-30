#!/usr/bin/env bash
# 12-step GPU verify of the --train-flow-bc path (3rd optimizer, flow FM loss, donation) before run2.
#SBATCH --job-name=fql_verify_flowbc
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/fql_verify_flowbc.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/scripts
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
exec uv run --no-sync python scripts/train_fql.py \
  --init-base /data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/100000/params \
  --action-dim 32 --action-horizon 30 --discount 0.99964 --v-min -2777.78 --v-max 0.0 \
  --train-flow-bc --flow-lr 5e-5 \
  --batch 8 --steps 12 --critic-warmup-steps 6 --num-workers 4 --save-every 999999
