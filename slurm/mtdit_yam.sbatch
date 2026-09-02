#!/usr/bin/env bash
# LeRobot Multitask DiT on yam_lego_taxi -- a test run of the policy and the pipeline.
#
# Multitask DiT is a ~450M diffusion transformer conditioned on CLIP vision AND language embeddings
# (the TRI Large Behavior Models recipe). It landed in LeRobot v0.6.0 and does NOT exist in the
# 0.4.4 this repo pins, so it runs from an ISOLATED venv at /data5/jellyho/ACRFT/mtdit/.venv
# (python 3.12 -- lerobot>=0.6 requires it, openpi is on 3.11). Nothing here touches openpi's venv.
#
# DATA: the SAME dataset the BC checkpoint was trained on --
# /data5/jellyho/.cache/huggingface/lerobot/jellyho/yam_lego_taxi, v3.0, 347 episodes /
# 937,993 frames, which is the N the patch-critic cache was built over. (Not the 235-episode copy
# under /data1/zach, which is a different subset.) It has exactly ONE task, "assemble lego blocks
# to make yellow taxi", so the language pathway is constant: this tests the architecture and the
# pipeline, not the multi-task claim.
#
# OBJECTIVE defaults to flow_matching, not the doc's diffusion. Its timestep sampling is
# beta(alpha=1.5, beta=1.0) scaled by 0.999 -- identical to pi0.5's own
# (`jax.random.beta(rng, 1.5, 1) * 0.999`, src/openpi/models/pi0.py), so a flow-matching run is the
# apples-to-apples comparison against our BC policy. `OBJ=diffusion` switches to the doc's recipe.
#
# Everything else follows the official 30 Hz recipe in docs/source/multi_task_dit.mdx (horizon 32,
# n_action_steps 24) at the documented MINIMUM batch of 64 rather than its recommended 256-320,
# which does not fit an L40S. B200 is deliberately avoided so the extraction-arm queue is not
# delayed. num_integration_steps stays at its default 100; it only affects inference and can be
# overridden when the checkpoint is served (pi0.5 serves at 10 Euler steps).
#SBATCH --job-name=mtdit_yam
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/mtdit_yam_%j.slurm.log
echo "[$(date +%T)] mtdit_yam starting on $(hostname)"
VENV=/data5/jellyho/ACRFT/mtdit/.venv
# Fail loudly and immediately if this torch build cannot see the GPU, instead of after the dataset
# scan. torch here is 2.11+cu130 while openpi's is 2.7.1+cu126, so the driver check is not free.
$VENV/bin/python -c "import torch;assert torch.cuda.is_available(),'no CUDA';print('[cuda]',torch.cuda.get_device_name(0),torch.__version__)" || exit 1
exec $VENV/bin/lerobot-train \
  --dataset.repo_id=jellyho/yam_lego_taxi \
  --dataset.root=/data5/jellyho/.cache/huggingface/lerobot/jellyho/yam_lego_taxi \
  --dataset.video_backend=pyav \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective="${OBJ:-flow_matching}" \
  --batch_size="${BS:-64}" \
  --steps="${STEPS:-30000}" \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=8 \
  --output_dir="/data1/jellyho/acrft_ckpts/mtdit_yam_${OBJ:-flow_matching}" \
  --job_name=mtdit_yam_${OBJ:-flow_matching} \
  --wandb.enable=true \
  --wandb.project=yam-mtdit
