#!/usr/bin/env bash
# Sequential few-step smoke of all 7 extraction trainers on one GPU. Continue on failure;
# per-method verdict lines are the artifact.
#SBATCH --job-name=extract_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/extraction_smoke.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
run() {
  name=$1; shift
  echo "==== SMOKE $name ===="
  if timeout 3000 uv run --no-sync python -u "$@"; then echo "VERDICT $name PASS"; else echo "VERDICT $name FAIL($?)"; fi
}
run awr       scripts/train_awr.py      --steps 10 --batch 8 --save-every 999999 --out /tmp/ex_awr
run cfgrl     scripts/train_cfgrl.py    --steps 10 --batch 4 --save-every 999999 --out /tmp/ex_cfgrl
run flowdpg   scripts/train_flowdpg.py  --steps 10 --batch 4 --save-every 999999 --out /tmp/ex_fdpg
run lps       scripts/train_lps.py      --mode lps  --steps 10 --batch 4 --save-every 999999 --out /tmp/ex_lps
run lpsd      scripts/train_lps.py      --mode lpsd --steps 10 --batch 4 --save-every 999999 --out /tmp/ex_lpsd
run qam       scripts/train_qam.py      --steps 3  --batch 2 --save-every 999999 --out /tmp/ex_qam
run flowdagger scripts/train_flowdagger.py --phase all --stride 50000 --collect-batch 4 --bc-steps 100 --out /tmp/ex_fdag
echo "==== SMOKE DONE ===="
