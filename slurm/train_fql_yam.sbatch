#!/usr/bin/env bash
# QC-FQL on YAM (347 eps, success+failure) -- first real run.
#
# Frozen base = BC s300 h30 checkpoint (its own norm stats travel with it through the loader).
# Staged recipe: 10k critic warmup (pure MC regression, actor distill-only), then actor-critic
# with the gamma^H SMDP backup + MC floor. Critic gradient never enters the backbone (run 1:
# the conservative arm; 'warmup'/'always' are the ablation knobs). Value scale is the house
# patch-critic one: gamma 0.99964, support [-2778, 0], failure homing truncation + v_min anchor.
#SBATCH --job-name=fql_yam_100k
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/fql_yam_100k.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/scripts
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
exec uv run --no-sync python scripts/train_fql.py \
  --init-base /data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/200000/params \
  --action-dim 32 --action-horizon 30 \
  --discount 0.99964 --v-min -2777.78 --v-max 0.0 --num-atoms 101 \
  --batch 32 --steps 100000 --critic-warmup-steps 10000 --critic-backbone never \
  --train-flow-bc --flow-lr 5e-5 \
  --num-workers 12 --save-every 10000 \
  --out /data1/jellyho/acrft_ckpts/fql_yam/run2 \
  --wandb --wandb-name fql_yam_100k_run2_flowbc
