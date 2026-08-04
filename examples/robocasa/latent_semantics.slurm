#!/usr/bin/env bash
# Measure what the RL token encodes (episode identity vs task phase) across a run's checkpoints.
#
#   sbatch --export=ALL,CONFIG=pi05_robocasa_PrepareCoffee_rlt,\
# ROOT=checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt7_pardec_noprop_mae0.5,\
# OUT=.scratch/semantics_mae05.png -J latsem -o latent_semantics.slurm.log examples/robocasa/latent_semantics.slurm
#
# Read-only: it loads saved params and runs forward passes, so it is safe to run beside training. One
# L40S is enough - batch 8 of a bf16 3B prefix, no optimizer state.
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -uo pipefail
cd /data5/jellyho/ACRFT/openpi
export LD_LIBRARY_PATH="/data5/jellyho/ffmpeg6/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
exec uv run scripts/latent_semantics.py \
  --config "$CONFIG" \
  --checkpoint-root "$ROOT" \
  --out "${OUT:-.scratch/latent_semantics.png}" \
  ${EXTRA:-}
