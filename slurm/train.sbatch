#!/usr/bin/env bash
# One launcher for every training run. The CONFIG names the experiment; nothing else here does.
#
#   sbatch --export=ALL,CONFIG=pi05_yam_cable_tie_success  slurm/train.sbatch
#   sbatch --export=ALL,CONFIG=pi05_yam_cable_tie_withhoming slurm/train.sbatch
#   sbatch --export=ALL,CONFIG=pi05_yam_cable_tie          slurm/train.sbatch
#
# WHAT IS AND IS NOT ON THE COMMAND LINE. Everything that defines the experiment -- episode subset,
# whether failure give-up frames are cut, steps, save interval, batch, LR schedule, project, entity,
# norm stats -- lives in the config, so `--config-name` reproduces the run and nothing else has to be
# remembered. Only the machine's own facts stay here: where checkpoints go and how many loader
# workers this node can feed. That split is what makes the same config runnable on another server.
#
# This exists because the alternative already cost us. The deployed BC policy is named
# `yam_bc_s300_h30_successonly` and trained on ALL 347 episodes: wandb kept the argv --
# `pi05_yam_lego_taxi --exp-name yam_bc_s300_h30_successonly --overwrite` -- with no
# --data.success-only on it, and its h50 sibling was typed the same way fifteen minutes later. An
# --exp-name is a label and cannot enforce anything; a config name is the experiment.
#
# NAME defaults to the config name, so a run is self-identifying unless you deliberately relabel it.
#SBATCH --job-name=train
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/train_%x_%j.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME=/data5/jellyho/.cache/huggingface
# Node-local arrow cache: flock does not return on this cluster's NFS mounts, and HuggingFace
# datasets takes one per cache entry, so a job with its cache on /data1 or /data5 sits in D state
# forever. Sweep stale ones on the way in, since this script execs and cannot clean up after itself.
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID:-$$}
find /tmp -maxdepth 1 -name 'hf_datasets_cache_*' -mtime +1 -exec rm -rf {} + 2>/dev/null
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src:/data5/jellyho/ACRFT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
echo "[$(date +%F\ %T)] ${CONFIG:?set CONFIG to a registered config name} on $(hostname)"
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train.py "${CONFIG:?}" \
  --exp-name="${NAME:-${CONFIG}}" --overwrite \
  --checkpoint-base-dir "${CKPT_DIR:-/data1/jellyho/acrft_ckpts}" \
  --num-workers "${WORKERS:-12}"
