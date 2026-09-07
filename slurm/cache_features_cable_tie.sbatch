#!/usr/bin/env bash
# Frozen DINOv2 patch-feature cache for the YAM cable-tie set (jellyho/yam_cable_tie: 160 episodes,
# 251,080 frames, 3 cameras at 480x640). This is the artifact every RL step downstream addresses by
# row index -- cache row == LeRobot global frame index == torch-dataset index -- so it is built once
# over ALL 160 episodes. Success-only filtering is a BC-side concept (--data.success-only); the
# critic needs the 10 failures, they are its only negatives.
#
#   sbatch slurm/cache_features_cable_tie.sbatch
#
# Expected: features.dat 251080 x 192 x 384 float16 = 37.0 GB, state.dat [251080,42] f32 = 42 MB,
# action.dat [251080,14] f32 = 14 MB. At the lego build's measured 117 f/s that is ~36 min; the 12 h
# request is headroom for a slow node. /data1 has ~1.6 T free.
#
# THREE THINGS THAT MUST STAY TRUE OF THIS FILE:
#
#  1. NO --root. The flag means two incompatible things at once -- LeRobotDataset takes it as THE
#     dataset directory, outcomes.dataset_root() reads it as <root>/<repo_id> -- and the lego default
#     satisfied both only because /data5/jellyho/yam_v2/lerobot happens to be a lego dataset dir that
#     also contains jellyho/yam_lego_taxi/. No single value works for cable tie. Omitting it lets
#     both sides resolve through HF_LEROBOT_HOME, which is the only way they agree.
#
#  2. NO --outcomes. Cable tie carries next.success / next.done in its own schema (150 success,
#     10 fail, 0 unknown), so the verdicts come from the dataset. The lego sidecar
#     .scratch/yam_outcomes_347.jsonl covers episodes 0..346 and cable tie's are 0..159, so a copied
#     --outcomes line would be PREFERRED over the schema and would silently label cable-tie episodes
#     with lego verdicts -- no shape error, no index error, just a WARNING that every lego log prints
#     too. Grep any new cable-tie launcher for 'yam_outcomes' before submitting it.
#
#  3. NO `uv run`. On node100 it hangs before Python starts -- jobs 37235-37239 wrote zero-byte logs
#     for two hours. UV_PROJECT_ENVIRONMENT already names the venv, so calling its interpreter
#     directly is equivalent and removes the thing that wedges.
#SBATCH --job-name=pc_cache_ct
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/pc_cache_ct_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME=/data5/jellyho/.cache/huggingface
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
echo "[$(date +%F\ %T)] pc_cache_ct starting on $(hostname)"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/cache_patch_features.py \
  --repo-id jellyho/yam_cable_tie \
  --out /data1/jellyho/pc_cache/yam_cable_tie_s160 \
  --backbone small --dino-dtype bfloat16 --img-size 224 \
  --horizon 30 --clip-len 120 --num-workers 16 --encode-batch 8
