#!/usr/bin/env bash
# alpha-Flow pi05 YAM main run (200k, B200 required: full-FT Adam+EMA does not fit an L40S).
# Uses examples/robocasa/run_train_yam.sh (norm stats check + wandb) with the alphaflow config.
#SBATCH --job-name=af_yam_200k
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00
#SBATCH --output=af_yam_200k.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
CONFIG=pi05_yam_lego_taxi_alphaflow EXP=yam_alphaflow_200k PROJECT=yam-rlt NUM_WORKERS=12 \
  bash examples/robocasa/run_train_yam.sh
