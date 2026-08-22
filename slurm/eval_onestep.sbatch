#!/usr/bin/env bash
# Offline 1-step BC-quality eval: alpha-Flow {1,2,10}-step vs BC baseline 10-step, robot-space MSE.
#SBATCH --job-name=af_eval1step
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/af_eval1step.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export LEROBOT_VIDEO_BACKEND=pyav
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec uv run --no-sync python -u /data5/jellyho/ACRFT/openpi-alphaflow/scripts/eval_onestep_bc.py \
  --af-ckpt "${AF_CKPT:-/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/160000}" \
  --bc-ckpt /data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/70000 \
  --out "${OUT:-.scratch/eval_onestep_160k}"
