#!/usr/bin/env bash
# Pi0FQL GPU smoke: real backbone (paligemma 2b + flow/actor 300m + critic 150m).
# 1) structural smoke (three forwards, shapes)  2) 10-step synthetic training smoke.
#SBATCH --job-name=fql_smoke2
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/fql_smoke2.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
set -e
uv run --no-sync python -u scripts/fql_smoke.py
uv run --no-sync python -u scripts/train_fql.py --synthetic --steps 10 --batch 2 --action-horizon 8 --flow-ode-steps 2
