#!/usr/bin/env bash
# tau/aug ablation launcher, parametrized: everything follows the canonical 200k recipe
# (h_goal 30, lr 1e-4, K=2, min reduction, homing=all, mc_floor on) except what's exported.
# Purpose: the tau9+aug arm cut the V(s0) leakage gap 986 -> 342, but tau and aug moved
# together; these arms span tau alone and aug alone, on BOTH chunking modes:
#   sbatch --export=ALL,GSZ=5,TAU=0.9,NAME=g5_tau9_noaug_200k       train_critic_tau9_noaug.sbatch
#   sbatch --export=ALL,GSZ=30,TAU=0.9,NAME=fixed_tau9_noaug_200k   train_critic_tau9_noaug.sbatch
#   sbatch --export=ALL,GSZ=30,TAU=0.9,AUG=1,NAME=fixed_tau9_min_200k  train_critic_tau9_noaug.sbatch
#SBATCH --job-name=critic_tau_ablate
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/critic_tau_ablate_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow:/data5/jellyho/ACRFT/openpi-alphaflow/src
AUG_ARGS=""
if [ "${AUG:-0}" = "1" ]; then AUG_ARGS="--feat-dropout 0.1 --feat-noise 0.05"; fi
exec uv run --no-sync python -u scripts/train_patch_critic_cached.py \
  --cache /data1/jellyho/pc_cache/yam_s347 \
  --outcomes /data5/jellyho/ACRFT/openpi/.scratch/yam_outcomes_347.jsonl \
  --homing-onsets /data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json \
  --norm-stats /data5/jellyho/ACRFT/openpi/assets/pi05_yam_lego_taxi_rlt/jellyho/yam_lego_taxi_s300/norm_stats.json \
  --macro-group-size "${GSZ:?}" --expectile "${TAU:?}" ${AUG_ARGS} \
  --h-goal 30 --lr 1e-4 --steps 200000 --preload \
  --wandb --wandb-name "${NAME:?}" \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_${NAME}"
