#!/usr/bin/env bash
# OOD-defence ablation on the patch critic. Everything follows the canonical 200k recipe of
# .scratch/patch_critic_yam_s347_fixed_tau9_min_200k (macro_group_size 30, expectile 0.9, h_goal 30,
# lr 1e-4, K=2, min reduction, aug on, mc_floor on) EXCEPT the one factor named by NAME, so the
# existing checkpoint is the control:
#
#   k10_min   sbatch --export=ALL,NAME=k10_min,K=10                              slurm/critic_ood.sbatch
#   k10_mean  sbatch --export=ALL,NAME=k10_mean,K=10,RED=mean                    slurm/critic_ood.sbatch
#   cql       sbatch --export=ALL,NAME=cql_shuffle,ACQL=1.0                      slurm/critic_ood.sbatch
#   cql+bank  sbatch --export=ALL,NAME=cql_bank,ACQL=1.0,NEG=bank,BANK=<dir>     slurm/critic_ood.sbatch
#
# Why the two K=10 arms and not one: turning K up while reducing by `min` confounds "more members"
# (a better-estimated disagreement) with "more pessimism" (min-of-10 < min-of-2 by construction).
# k10_mean moves the pessimism out of training and into the serving read (CriticQ.q_lcb), so the
# two effects can be told apart.
#SBATCH --job-name=critic_ood
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/critic_ood_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
CQL_ARGS=""
if [ -n "${ACQL:-}" ]; then
  CQL_ARGS="--alpha-cql ${ACQL} --cql-negatives ${NEG:-shuffle} --cql-n ${CQLN:-8} --cql-batch ${CQLB:-64}"
  if [ -n "${BANK:-}" ]; then CQL_ARGS="${CQL_ARGS} --cql-bank ${BANK}"; fi
fi
exec uv run --no-sync python -u scripts/train_patch_critic_cached.py \
  --cache /data1/jellyho/pc_cache/yam_s347 \
  --outcomes /data5/jellyho/ACRFT/openpi/.scratch/yam_outcomes_347.jsonl \
  --homing-onsets /data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json \
  --norm-stats /data5/jellyho/ACRFT/openpi/assets/pi05_yam_lego_taxi_rlt/jellyho/yam_lego_taxi_s300/norm_stats.json \
  --macro-group-size 30 --expectile 0.9 --feat-dropout 0.1 --feat-noise 0.05 \
  --h-goal 30 --lr 1e-4 --steps "${STEPS:-200000}" \
  --num-critics "${K:-2}" --q-reduction "${RED:-min}" ${CQL_ARGS} \
  --wandb --wandb-name "${NAME:?}" --wandb-group patch-critic-ood \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_${NAME}"
