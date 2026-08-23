#!/bin/bash
#SBATCH -J b1sft
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:4
#SBATCH -c 12
#SBATCH --mem=160G
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
BATCH=${4:-8}   # activations scale with batch: 32 needs a 14.5 GB single allocation, which OOMs
                 # on a 48 GB A6000 even with the parameters sharded. Both arms use the same batch,
                 # so the success-vs-all comparison stays a method-only difference.
unset LD_LIBRARY_PATH
cd /home/jellyho/ACRFT/ACRFT
# Memory ladder actually walked: one GPU died at train-state init (0.5 GB short), two GPUs with
# batch 32 died on a 14.5 GB activation allocation, two GPUs with batch 16 died on 9.3 GB. Four-way
# sharding cuts parameters and optimizer to about a quarter each, and batch 8 quarters the
# activations relative to the released recipe. Both arms share these settings, so success-vs-all
# stays a method-only difference; the deviation from the released batch is reported with results.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 WANDB_MODE=offline
# torchcodec needs system FFmpeg, which these nodes lack once LD_LIBRARY_PATH is unset (and it must
# be unset, or miniconda's libcrypto breaks the eval python). pyav ships its own FFmpeg, so it
# decodes the rollout videos everywhere -- the same fix openpi already applies in the critic loader.
export LEROBOT_VIDEO_BACKEND=pyav
export PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/robocasa:${PYTHONPATH:-}
ROLLOUT_ROOT=/scratch/jellyho/acrft/rollout_v3
export HF_LEROBOT_HOME=$ROLLOUT_ROOT
REPO="jellyho/rc_${FILTER}_${TASK}"

if [ -f "$ROLLOUT_ROOT/$REPO/conversion_summary.json" ]; then
  echo "=== [1/2] dataset $REPO already built, reusing ==="
else
  echo "=== [1/2] build dataset $REPO (filter=$FILTER) ==="
  rm -rf "$ROLLOUT_ROOT/$REPO"   # drop any half-written dataset before rebuilding
  .venv/bin/python examples/robocasa/convert_rollouts.py \
      --task "$TASK" --filter "$FILTER" --repo-id "$REPO" --root "$ROLLOUT_ROOT" 2>&1 | tail -12 \
      || { echo "CONVERT FAILED $TASK"; exit 1; }
fi

echo "=== [2/2] fine-tune from the released checkpoint ==="
EXP="b1_${FILTER}_${TASK}"
uv run scripts/train.py pi05_robocasa_b1 \
    --exp-name "$EXP" \
    --data.repo-id "$REPO" \
    --num-train-steps "$STEPS" \
    --fsdp-devices 4 \
    --batch-size "$BATCH" \
    --overwrite \
    --checkpoint-base-dir /scratch/jellyho/acrft/checkpoints/b1 2>&1 | tail -20

echo "B1_DONE ${TASK} ${FILTER}"
