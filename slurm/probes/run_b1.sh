#!/bin/bash
#SBATCH -J b1sft
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jellyho/acrft/probes/b1_%j.out
# Baseline ladder rung B1: convert one task's collection under the requested filter, then continue
# training the OFFICIAL RoboCasa-365 pi05 on it. Conversion runs inside the job so the dataset and
# the run that consumes it can never drift apart.
#
#   sbatch run_b1.sh <Task> [FILTER] [STEPS]
#     FILTER: success (B1) | weighted (B2) | all (the unfiltered control that separates
#             "more data" from "better data")
set -uo pipefail
TASK=${1:?usage: sbatch run_b1.sh Task [filter] [steps]}
FILTER=${2:-success}
STEPS=${3:-10000}
unset LD_LIBRARY_PATH
cd /home/jellyho/ACRFT/ACRFT
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline
export PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/robocasa:${PYTHONPATH:-}
ROLLOUT_ROOT=/scratch/jellyho/acrft/rollout_v3
export HF_LEROBOT_HOME=$ROLLOUT_ROOT
REPO="jellyho/rc_${FILTER}_${TASK}"

echo "=== [1/2] build dataset $REPO (filter=$FILTER) ==="
.venv/bin/python examples/robocasa/convert_rollouts.py \
    --task "$TASK" --filter "$FILTER" --repo-id "$REPO" --root "$ROLLOUT_ROOT" 2>&1 | tail -12 \
    || { echo "CONVERT FAILED $TASK"; exit 1; }

echo "=== [2/2] fine-tune from the released checkpoint ==="
EXP="b1_${FILTER}_${TASK}"
uv run scripts/train.py pi05_robocasa_b1 \
    --exp-name "$EXP" \
    --data.repo-id "$REPO" \
    --num-train-steps "$STEPS" \
    --checkpoint-base-dir /scratch/jellyho/acrft/checkpoints/b1 2>&1 | tail -20

echo "B1_DONE ${TASK} ${FILTER}"
