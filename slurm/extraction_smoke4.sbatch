#!/usr/bin/env bash
# Sequential few-step smoke of the 5 arms fixed after round 1 (cfgrl import; lps/lpsd/qam tracer; flowdagger constant OOM) on one GPU. Continue on failure;
# per-method verdict lines are the artifact.
#SBATCH --job-name=extract_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/extraction_smoke4.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
run() {
  name=$1; shift
  echo "==== SMOKE $name ===="
  if timeout 3000 uv run --no-sync python -u "$@"; then echo "VERDICT $name PASS"; else echo "VERDICT $name FAIL($?)"; fi
}
run qam       scripts/train_qam.py      --steps 3  --batch 2 --save-every 999999 --out /tmp/ex_qam
echo "==== SMOKE DONE ===="
