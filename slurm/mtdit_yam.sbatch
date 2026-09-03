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
# SUCCESS-ONLY, like BC. The 347 episodes are 300 success + 47 fail, and the BC checkpoint is
# yam_bc_s300_h30_successonly -- openpi resolves that subset at startup via
# LeRobotYAMDataConfig.success_only (config.py:573-580). LeRobot has no such flag, so the episode
# list is derived HERE from the same outcomes file openpi uses and passed as --dataset.episodes.
# Deriving it at launch rather than pasting 300 indices keeps it in sync with the outcomes file and
# makes the count assertion below the thing that fails if they ever diverge.
#
# OBJECTIVE defaults to flow_matching, not the doc's diffusion. Its timestep sampling is
# beta(alpha=1.5, beta=1.0) scaled by 0.999 -- identical to pi0.5's own
# (`jax.random.beta(rng, 1.5, 1) * 0.999`, src/openpi/models/pi0.py), so a flow-matching run is the
# apples-to-apples comparison against our BC policy. `OBJ=diffusion` switches to the doc's recipe.
#
# TORCH BUILD IS PINNED TO cu126 ON PURPOSE. `pip install lerobot[multi-task-dit]` pulls torch
# 2.11+cu130 and the compute nodes' driver is CUDA 12.2, so job 37273 died at the check below with
# "The NVIDIA driver on your system is too old (found version 12020)". lerobot 0.6.1 allows
# torch>=2.7,<2.12, so the venv is held at torch 2.8.0+cu126 / torchvision 0.23.0+cu126 from the
# pytorch cu126 index; re-run that install if the venv is ever rebuilt. torchcodec then cannot load
# its native libs (libavutil.so.56 absent -- the compute nodes have no ffmpeg), which is exactly why
# the video backend below is pyav: pyav ships its own.
#
# AND THE GPU TYPE IS PINNED TO L40S FOR THE SAME REASON, FROM THE OTHER SIDE. Job 37276 landed on
# node200 and warned "Found GPU0 NVIDIA B200 which is of cuda capability 10.0. Minimum and Maximum
# cuda capability supported by this version of PyTorch is (5.0) - (9.0)". That was a warning and not
# what actually killed the job -- a missing `accelerate` did, and the venv now carries
# lerobot[training] -- but it is a real incompatibility that would have failed at the first kernel.
# The two node types
# want incompatible builds: node100's L40S sits behind a CUDA 12.2 driver, so torch must be cu126 or
# older, while the B200 needs sm_100 support that only arrives with cu128, which that driver cannot
# run. One venv cannot serve both, so this job takes the L40S where cu126 works -- which is also
# where it belongs, since B200 time is what the extraction arms are queued for.
#
# Everything else follows the official 30 Hz recipe in docs/source/multi_task_dit.mdx (horizon 32,
# n_action_steps 24) at the documented MINIMUM batch of 64 rather than its recommended 256-320,
# which does not fit an L40S. B200 is deliberately avoided so the extraction-arm queue is not
# delayed. num_integration_steps stays at its default 100; it only affects inference and can be
# overridden when the checkpoint is served (pi0.5 serves at 10 Euler steps).
#SBATCH --job-name=mtdit_yam
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/mtdit_yam_%j.slurm.log
echo "[$(date +%T)] mtdit_yam starting on $(hostname)"
VENV=/data5/jellyho/ACRFT/mtdit/.venv
DATA_ROOT=/data5/jellyho/.cache/huggingface/lerobot/jellyho/yam_lego_taxi
EPISODES=$(/data5/jellyho/ACRFT/openpi/.venv/bin/python - "$DATA_ROOT" <<'PYEOF'
import pathlib, sys

# The verdict lives in the LeRobot schema itself -- per-frame next.success / next.done, aggregated
# per episode in meta/episodes -- and openpi.training.outcomes is the one reader for it (ACRFT
# master f218e82). This launcher used to inline that rule because the module was not on master yet.
# Three properties we rely on and no longer have to re-implement: "unknown" (kept but never judged)
# is a state distinct from "fail" and is never counted as success; the source is logged; and a
# dataset with no verdict features returns None rather than a silently empty list.
sys.path.insert(0, "/data5/jellyho/ACRFT/openpi/src")
from openpi.training import outcomes  # noqa: E402

root = pathlib.Path(sys.argv[1])
verdicts = outcomes.episode_outcomes(root)
if verdicts is None:
    raise SystemExit(f"{root} carries no next.success/next.done; migrate it with `yam-data migrate-outcomes`")
succ = sorted(e for e, v in verdicts.items() if v == "success")
fail = sorted(e for e, v in verdicts.items() if v == "fail")
unknown = sorted(e for e, v in verdicts.items() if v == "unknown")
# Kept deliberately even though both YAM sets are 0 unknown today: eval-rollout datasets record
# unjudged episodes as unknown BY DESIGN, and that is exactly where treating unknown as fail would
# throw away good data without saying so.
if unknown:
    raise SystemExit(f"{len(unknown)} episodes have no verdict: {unknown[:10]}. Decide explicitly, do not guess.")
if len(succ) != 300 or len(fail) != 47:
    raise SystemExit(f"expected 300 success / 47 fail, got {len(succ)}/{len(fail)}")
print("[" + ",".join(map(str, succ)) + "]")
print(f"verdicts from the dataset schema: {len(succ)} success / {len(fail)} fail / 0 unknown", file=sys.stderr)
PYEOF
) || exit 1
echo "[$(date +%T)] training on ${OBJ:-flow_matching}, success-only: 300 of 347 episodes (47 failures excluded)"
# Fail loudly and immediately if this torch build cannot see the GPU, instead of after the dataset
# scan. torch here is 2.11+cu130 while openpi's is 2.7.1+cu126, so the driver check is not free.
$VENV/bin/python -c "import torch;assert torch.cuda.is_available(),'no CUDA';print('[cuda]',torch.cuda.get_device_name(0),torch.__version__)" || exit 1
exec $VENV/bin/lerobot-train \
  --dataset.repo_id=jellyho/yam_lego_taxi \
  --dataset.root=/data5/jellyho/.cache/huggingface/lerobot/jellyho/yam_lego_taxi \
  --dataset.video_backend=pyav \
  --dataset.episodes="$EPISODES" \
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
