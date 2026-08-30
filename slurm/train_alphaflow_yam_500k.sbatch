#!/usr/bin/env bash
# alpha-Flow pi05 YAM continuation: resume 200k -> 500k in PURE MeanFlow (JVP) mode.
#
# The 200k run finished the official curriculum and sat on the discrete alpha=5e-3 floor from
# ~142k. This continuation is the "JVP polish" the config docstring reserved as a followup:
# alpha pinned to 0 (exact MeanFlow limit, JVP branch) for the whole 200k->500k stretch. The
# JVP stress test (6/6 stable, incl. bf16) is the evidence this is safe. Same exp dir + --resume,
# so optimizer state and the wandb run (c4vy84yy) continue; checkpoints every 50k.
# lr is effectively constant 5e-5 (peak_lr == decay_lr), so extending steps changes no schedule.
#SBATCH --job-name=af_yam_500k
#SBATCH --partition=debug
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi-alphaflow
#SBATCH --output=/data5/jellyho/ACRFT/openpi-alphaflow/af_yam_500k.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
# the venv's openpi is an editable install of the SHARED checkout; put this worktree's src first
export PYTHONPATH=/data5/jellyho/ACRFT/openpi-alphaflow/src:/data5/jellyho/ACRFT/openpi-alphaflow/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
# alpha-floor must be re-zeroed explicitly: the registered config instance has already resolved
# its floor to 5e-3, and meanflow_jvp=True rejects a nonzero floor at validation.
exec uv run --no-sync python scripts/train.py pi05_yam_lego_taxi_alphaflow \
  --exp-name=yam_alphaflow_200k --project-name yam-rlt --wandb-entity RSS-PFT_RLLAB \
  --checkpoint-base-dir /data1/jellyho/acrft_ckpts --resume \
  --num-train-steps 500000 --save-interval 50000 \
  --model.alpha-init 0.0 --model.alpha-final 0.0 --model.meanflow-jvp --model.alpha-floor 0.0 \
  --num-workers 12 --data.assets.asset-id jellyho/yam_lego_taxi
