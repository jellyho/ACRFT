#!/usr/bin/env bash
# Offline eval, wave 1: arms that need no trained checkpoint (bc / flowdagger / qpilots / idql).
#SBATCH --job-name=ex_eval1
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/extraction_eval1.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
run() { echo "==== EVAL $1 ===="; if uv run --no-sync python -u scripts/eval_extraction.py "${@:2}"; then echo "EVAL $1 OK"; else echo "EVAL $1 FAIL($?)"; fi }
run bc         --arm bc --label bc_base
run flowdagger --arm flowdagger --label flowdagger --steering-head /data1/jellyho/acrft_ckpts/extraction/flowdagger_run1
run qpilots01  --arm qpilots --alpha 0.1 --label qpilots_a01
run qpilots02  --arm qpilots --alpha 0.2 --label qpilots_a02
run qpilots03  --arm qpilots --alpha 0.3 --label qpilots_a03
run idql       --arm idql --n-samples 64 --label idql_n64
run bon8       --arm bon  --n-samples 8  --label bon_n8
echo "==== EVAL1 DONE ===="
