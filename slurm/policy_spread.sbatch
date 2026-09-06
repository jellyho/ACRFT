#!/usr/bin/env bash
# Has BC collapsed its per-state action distribution? Compares the policy's sampling spread at each
# checkpoint step against the spread the DEMONSTRATORS showed at matched task states.
#SBATCH --job-name=pol_spread
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/policy_spread_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
echo "[$(date +%T)] policy_spread starting on $(hostname)"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/diag_policy_spread.py \
  --states "${STATES:-64}" --samples "${SAMPLES:-16}" --steps ${STEPS:-100000 200000}
