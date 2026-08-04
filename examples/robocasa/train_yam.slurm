#!/usr/bin/env bash
# Train Pi0RLT on the YAM lego-taxi dataset on a B200. Submit with CONFIG/EXP in the environment:
#
#   sbatch --export=ALL,CONFIG=pi05_yam_lego_taxi_rlt      -J yam_reljoint -o train_yam_reljoint.slurm.log examples/robocasa/train_yam.slurm
#   sbatch --export=ALL,CONFIG=pi05_yam_lego_taxi_none_rlt -J yam_none     -o train_yam_none.slurm.log     examples/robocasa/train_yam.slurm
#
# node200 is the only B200 node; pi05 is a full finetune (no freeze) at batch 32 with three cameras,
# which does not fit the 48GB L40S.
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1

set -uo pipefail
cd /data5/jellyho/ACRFT/openpi
export LD_LIBRARY_PATH="/data5/jellyho/ffmpeg6/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Three video streams decoded per sample makes the loader the bottleneck; leave a few cores for the
# main process and wandb rather than handing every allocated CPU to the workers.
export NUM_WORKERS="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} - 8 ))}"
exec bash examples/robocasa/run_train_yam.sh
