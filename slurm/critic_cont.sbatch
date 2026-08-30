#!/usr/bin/env bash
# Continue the g5_pi05 patch-critic: it stopped at 20k steps with q_loss still falling
# (2.98 -> 2.78 over the last 4k) -- clearly unconverged; the clip-trainer default is 120k.
# Warm-start from the 20k checkpoint, +100k steps, NEW out dir (the 20k ckpt stays as-is
# for before/after comparison -- the value videos rendered from it are the "before").
#SBATCH --job-name=critic_g5_pi05_cont
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/critic_g5_pi05_cont.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow:/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
exec uv run --no-sync python -u scripts/train_patch_critic_cached.py \
  --cache /data1/jellyho/pc_cache/yam_s347 \
  --outcomes /data5/jellyho/yam_v2/lerobot/outcomes.jsonl \
  --input-mode pi05 \
  --norm-stats /data5/jellyho/ACRFT/openpi/assets/pi05_yam_lego_taxi_rlt/jellyho/yam_lego_taxi_s300/norm_stats.json \
  --init-params /data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_g5_pi05 \
  --steps 100000 --preload \
  --wandb --wandb-name g5_pi05_s347_cont100k \
  --out .scratch/patch_critic_yam_s347_g5_pi05_cont
