#!/usr/bin/env bash
# alpha-Flow JVP explosion stress (bf16 or f32 via DT env) — pinned-worktree launcher.
#SBATCH --job-name=af_jvp3
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec uv run --no-sync python -u scripts/alphaflow_jvp_stress.py --dtype "${DT:-bfloat16}" --steps 60
