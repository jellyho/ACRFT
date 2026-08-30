#!/usr/bin/env bash
# alpha-Flow v2: the s300-convention rerun. v1 (yam_alphaflow_200k) trained on ALL 347 episodes
# (47 failures cloned -- success_only was never set) with 08-03 norm stats whose action q99 is
# ~37% narrower than this data; both deviations break the method-only-diff claim vs BC-h30.
# v2 fixes both: success-only (300 eps via outcomes.jsonl) + the BC-h30 checkpoint's EXACT norm
# stats (asset jellyho/yam_lego_taxi_s300h30, provenance-stamped; the loader now verifies it).
# Same 200k official curriculum as v1; the JVP polish can follow as a continuation, as before.
#SBATCH --job-name=af_yam_s300
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/af_yam_s300.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
exec uv run --no-sync python scripts/train.py pi05_yam_lego_taxi_alphaflow \
  --exp-name=yam_alphaflow_s300_200k --project-name yam-rlt --wandb-entity RSS-PFT_RLLAB \
  --checkpoint-base-dir /data1/jellyho/acrft_ckpts --overwrite \
  --num-train-steps 200000 --save-interval 50000 \
  --data.assets.asset-id jellyho/yam_lego_taxi_s300h30 --data.success-only \
  --num-workers 12
