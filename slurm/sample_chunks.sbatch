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
echo "[$(date +%T)] chunk_bank starting on $(hostname)"
# NOTE: we call the venv interpreter directly instead of `uv run --no-sync`. On node100, `uv run`
# hangs before it ever starts Python -- jobs 37235-37239 sat there for two hours writing a zero-byte
# log, and a staged diagnostic (logs/diag100_37263.log) showed the bash preamble printing and then
# both a 120s and a 180s `timeout uv run` expiring with no output, while the identical script
# finished in under a second on node200. With UV_PROJECT_ENVIRONMENT already pointing at the venv,
# `uv run --no-sync` only means "use this venv's python", so calling it directly is equivalent and
# removes the thing that wedges. The echo below exists so a silent log is diagnosable next time.
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/sample_policy_chunks.py \
  --num-samples "${NK:-8}" --num-states "${NS:-100000}" --batch "${BS:-8}" \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/extraction/policy_chunks_bc"
