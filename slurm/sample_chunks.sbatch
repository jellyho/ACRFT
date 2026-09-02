#!/usr/bin/env bash
# Frozen-BC policy chunk bank: the action distribution our critic has never been trained on.
# Feeds scripts/diag_critic_gradient.py and --cql-negatives bank.
#SBATCH --job-name=chunk_bank
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/chunk_bank_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
exec uv run --no-sync python -u scripts/sample_policy_chunks.py \
  --num-samples "${NK:-8}" --num-states "${NS:-100000}" --batch "${BS:-8}" \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/extraction/policy_chunks_bc"
