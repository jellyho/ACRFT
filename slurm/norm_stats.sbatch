#!/usr/bin/env bash
# Norm stats for any config/subset, as a job rather than a login-node command.
#
#   sbatch --export=ALL,CONFIG=pi05_yam_cable_tie,SUCCESS=1,ASSET_ID=jellyho/yam_cable_tie_s150 \
#          slurm/norm_stats.sbatch
#
# CPU only (JAX_PLATFORMS=cpu) and no video is decoded -- compute_norm_stats.py reads the
# low-dimensional state/action fields with skip_videos. It is still a full pass over every frame
# (251k for cable tie, 938k for lego), which is why it belongs in a job: run from a login shell it
# competes with everyone else's interactive work and dies without leaving a log behind.
#
# ASSET_ID names the output directory, which is the exact path a training run then LOADS from. Name
# it whenever the policy and the critic must share ONE statistics file rather than two that are
# merely very likely equal -- that "very likely" is what the alpha-Flow 08-23 mis-normalization was.
#SBATCH --job-name=norm_stats
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/norm_stats_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME=/data5/jellyho/.cache/huggingface
# A node-LOCAL arrow cache, and this is not a tidiness preference: file locking is BROKEN on the
# NFS mounts. Measured on node100 -- `flock` on /data1 and /data5 (both nfs4 from 172.20.1.99) times
# out, while the same flock on the node's local ext4 /tmp returns instantly. HuggingFace `datasets`
# takes a filelock per cache entry, so with the cache on NFS the job sits in D state forever: job
# 37395 logged 1 CPU tick per 45 s for 20 minutes and wrote nothing. This is the same failure that
# left chunk-bank jobs 37235-37239 with zero-byte logs for two hours, and it is why `uv run` wedges
# on node100 too -- uv takes file locks of its own. Nothing in any log says "lock"; a blocked job and
# a slow one look identical.
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID:-$$}
# The cache is node-local and the job execs, so nothing can clean up on exit; sweep stale ones on the
# way in instead. A day is well past any job that could still be using one.
find /tmp -maxdepth 1 -name 'hf_datasets_cache_*' -mtime +1 -exec rm -rf {} + 2>/dev/null
export JAX_PLATFORMS=cpu
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
SUCC_ARGS=""
if [ -n "${SUCCESS:-}" ]; then SUCC_ARGS="--success-only"; fi
ASSET_ARGS=""
if [ -n "${ASSET_ID:-}" ]; then ASSET_ARGS="--asset-id ${ASSET_ID}"; fi
echo "[$(date +%F\ %T)] norm_stats ${CONFIG:?} on $(hostname) (success_only=${SUCCESS:-0} asset=${ASSET_ID:-<repo id>})"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/compute_norm_stats.py \
  --config-name "${CONFIG:?}" ${SUCC_ARGS} ${ASSET_ARGS}
