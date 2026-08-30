#!/usr/bin/env bash
# Real training run for one extraction arm: sbatch --job-name=ex_<arm> slurm/extraction_train.sbatch <arm>
# Arms: awr cfgrl flowdpg qam lps lpsd dql fqlx flowdagger. Test-time arms (qpilots/idql/bon) need no
# training — they run through scripts/eval_extraction.py directly.
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/extraction_%x.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
arm=${1:?usage: sbatch slurm/extraction_train.sbatch <arm> [extra args passed to the trainer]}
shift  # anything after the arm is forwarded, so a batch size can be set per submission
case "$arm" in
  awr)        exec uv run --no-sync python -u scripts/train_awr.py      --wandb "$@" ;;
  cfgrl)      exec uv run --no-sync python -u scripts/train_cfgrl.py    --wandb "$@" ;;
  flowdpg)    exec uv run --no-sync python -u scripts/train_flowdpg.py  --wandb "$@" ;;
  qam)        exec uv run --no-sync python -u scripts/train_qam.py      --wandb "$@" ;;
  lps)        exec uv run --no-sync python -u scripts/train_lps.py      --mode lps  --wandb "$@" ;;
  lpsd)       exec uv run --no-sync python -u scripts/train_lps.py      --mode lpsd --wandb "$@" ;;
  dql)        exec uv run --no-sync python -u scripts/train_dql.py      --wandb "$@" ;;
  fqlx)       exec uv run --no-sync python -u scripts/train_fqlx.py     --wandb "$@" ;;
  flowdagger) exec uv run --no-sync python -u scripts/train_flowdagger.py --phase all --wandb "$@" ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
