#!/usr/bin/env bash
# Real training run for one extraction arm: sbatch --job-name=ex_<arm> slurm/extraction_train.sbatch <arm>
# Arms: awr cfgrl flowdpg qam lps lpsd fqlx flowdagger. Test-time arms (qpilots/idql/bon) need no
# training — they run through scripts/eval_extraction.py directly.
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
# override at submit time for the B200 node:  sbatch --gres=gpu:B200:1 -w node200 ...
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/extraction_%x.slurm.log
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
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
# Direct venv interpreter, never `uv run`: on node100 it wedges before Python starts, and
# UV_PROJECT_ENVIRONMENT already names the venv so the two are equivalent when it works.
arm=${1:?usage: sbatch slurm/extraction_train.sbatch <arm> [extra args forwarded to the trainer]}
shift  # forward everything after the arm; without this a --batch/--lr/--out is dropped SILENTLY
case "$arm" in
  awr)        exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_awr.py      --wandb "$@" ;;
  cfgrl)      exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_cfgrl.py    --wandb "$@" ;;
  flowdpg)    exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_flowdpg.py  --wandb "$@" ;;
  qam)        exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_qam.py      --wandb "$@" ;;
  lps)        exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_lps.py      --mode lps  --wandb "$@" ;;
  lpsd)       exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_lps.py      --mode lpsd --wandb "$@" ;;
  fqlx)       exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_fqlx.py     --wandb "$@" ;;
  flowdagger) exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_flowdagger.py --phase all --wandb "$@" ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
