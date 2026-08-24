#!/usr/bin/env bash
# Score the canonical fixed-vs-adaptive IQL critic pair on YAM (first eval of this pair --
# fixed_200k had never been scored at all; provenance from the patch-critic session 2026-08-24).
# Metrics of record: V(s0) success/failure gap, k* (selected horizon), Vslope -- NOT AUC,
# which saturates at 1.0 under hindsight leakage and cannot separate the arms.
#SBATCH --job-name=score_iql_pair
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/score_iql_pair.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow:/data5/jellyho/ACRFT/openpi-alphaflow/src
OUT=/data5/jellyho/ACRFT/openpi-alphaflow/.scratch/iql_pair_eval
mkdir -p "$OUT"
for arm in fixed_200k g5_200k; do
  echo "==== scoring $arm ===="
  uv run --no-sync python -u scripts/score_critic_cached.py \
    --critic /data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_${arm} \
    --cache /data1/jellyho/pc_cache/yam_s347 \
    --outcomes /data5/jellyho/ACRFT/openpi/.scratch/yam_outcomes_347.jsonl \
    --homing-onsets /data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json \
    --truncate-homing all --stride 20 \
    2>&1 | tee "$OUT/${arm}.log"
done
echo "==== done ===="
