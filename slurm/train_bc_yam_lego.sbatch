#!/usr/bin/env bash
# YAM lego BC. One launcher, both episode conditions, chosen by an env var rather than by which file
# you happened to open.
#
#   sbatch --export=ALL,SUCCESS=1,ASSET_ID=jellyho/yam_lego_taxi_s300,NAME=yam_bc_s300_h30_true_successonly \
#          slurm/train_bc_yam_lego.sbatch          # 300 successes  <- the arm that does not exist yet
#   sbatch --export=ALL,NAME=yam_bc_s347_h30_allep slurm/train_bc_yam_lego.sbatch   # all 347
#
# WHY THIS FILE REPLACES train_bc_yam_allep.sbatch. That launcher's header said the deployed policy
# yam_bc_s300_h30_successonly "was trained with --data.success-only, i.e. on the 300 successes only",
# and that running the same config WITHOUT the flag "is the whole method-only-diff". Both halves are
# false. wandb kept the launch argv, and it reads:
#
#     run-20260820_104806-v3ffoek0: pi05_yam_lego_taxi --exp-name yam_bc_s300_h30_successonly --overwrite
#
# -- no --data.success-only, no --data.assets.asset-id. The run logged success_only=False, resolved
# episodes=None, and baked the repo-id-fallback ALL-347 statistics (md5 70418210e195f595). Its h50
# sibling, typed fifteen minutes later, is the same. So the deployed baseline already IS the
# all-episodes arm, job 37348 was reproducing it rather than contrasting with it, and no genuinely
# success-only YAM BC policy has ever existed. The header's "two differences (episodes AND
# statistics) ... up to 2.5% in action q99" described a gap between two assets that neither run used.
#
# Only those two hand-typed runs are affected. Every YAM run launched through a script passed its
# flags correctly, the s100/s200/s300 scaling study included.
#
# ASSET_ID pins the statistics file. Pass it for the success-only arm --
# jellyho/yam_lego_taxi_s300 (md5 2d87bbfe63daf02b) is the genuine success-only h30 asset and it
# already exists on disk. Omit it for the all-347 arm, where the repo-id fallback IS the right file.
# A named-but-missing asset now raises rather than training unnormalized, so a typo fails loudly.
#SBATCH --job-name=bc_lego
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/bc_lego_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME=/data5/jellyho/.cache/huggingface
# Node-local arrow cache: flock does not return on the NFS mounts (measured on node100, /data1 and
# /data5 both time out while local ext4 /tmp returns instantly), and HuggingFace datasets takes one
# per cache entry. Sweep stale ones on the way in, since this script execs and cannot clean up after.
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID:-$$}
find /tmp -maxdepth 1 -name 'hf_datasets_cache_*' -mtime +1 -exec rm -rf {} + 2>/dev/null
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src:/data5/jellyho/ACRFT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
SUCC_ARGS=""
if [ -n "${SUCCESS:-}" ]; then SUCC_ARGS="--data.success-only"; fi
ASSET_ARGS=""
if [ -n "${ASSET_ID:-}" ]; then ASSET_ARGS="--data.assets.asset-id ${ASSET_ID}"; fi
echo "[$(date +%F\ %T)] bc_lego ${NAME:?} on $(hostname) (success_only=${SUCCESS:-0} asset=${ASSET_ID:-<repo-id fallback>})"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train.py pi05_yam_lego_taxi \
  --exp-name="${NAME:?}" --project-name yam-rlt --wandb-entity jellyho_ \
  --checkpoint-base-dir /data1/jellyho/acrft_ckpts --overwrite \
  --num-train-steps "${STEPS:-200000}" --save-interval "${SAVE:-50000}" ${SUCC_ARGS} ${ASSET_ARGS} \
  --num-workers 12
