#!/usr/bin/env bash
# mc_floor ablation: retrain the canonical fixed/adaptive IQL pair with --no-mc-floor,
# everything else identical to fixed_200k / g5_200k (lr 1e-4, h_goal 30, 200k, K=2, tau 0.7,
# pi05 spec, homing=all, q-reduction min). Completes the 2x2 (chunking x floor); the
# floor-off arms are also one step closer to faithful DEAS (which has no MC floor).
# Usage: sbatch --export=ALL,GSZ=30,NAME=fixed_nofloor_200k slurm/train_critic_nofloor.sbatch
#        sbatch --export=ALL,GSZ=5,NAME=g5_nofloor_200k    slurm/train_critic_nofloor.sbatch
#SBATCH --job-name=critic_nofloor
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/critic_nofloor_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow:/data5/jellyho/ACRFT/openpi-alphaflow/src
exec uv run --no-sync python -u scripts/train_patch_critic_cached.py \
  --cache /data1/jellyho/pc_cache/yam_s347 \
  --outcomes /data5/jellyho/ACRFT/openpi/.scratch/yam_outcomes_347.jsonl \
  --homing-onsets /data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json \
  --norm-stats /data5/jellyho/ACRFT/openpi/assets/pi05_yam_lego_taxi_rlt/jellyho/yam_lego_taxi_s300/norm_stats.json \
  --macro-group-size "${GSZ:?set GSZ=30 or 5}" \
  --no-mc-floor \
  --h-goal 30 --lr 1e-4 --steps 200000 --preload \
  --wandb --wandb-name "${NAME:?set NAME}" \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_${NAME}"
