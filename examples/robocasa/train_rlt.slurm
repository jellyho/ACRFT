#!/usr/bin/env bash
# Train pi05 + RLT on a RoboCasa task. Pass the launcher's options through ARGS:
#
#   sbatch --export=ALL,ARGS="PrepareCoffee --parallel-decoder --no-proprio --mask-ratio 0.5 \
# --objective reconstruction+action --exp-suffix rlt8" -J rlt8_act -o train_rlt8_act.slurm.log \
# examples/robocasa/train_rlt.slurm
#
# node200 (B200) because every previous RoboCasa RLT run used it: pi05 is a full finetune at batch
# 32, and keeping the hardware fixed is what makes step times and loss curves comparable across
# ablations.
#
# The CPU request is NOT incidental. RoboCasa frames come out of encoded video, so the loader is the
# bottleneck: the currently-running rlt7_mae075 was submitted with 2 CPUs and sits at 2.1 s/it,
# ~4x slower than the same model with a full worker pool.
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1

set -uo pipefail
cd /data5/jellyho/ACRFT/openpi
export LD_LIBRARY_PATH="/data5/jellyho/ffmpeg6/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Leave cores for the main process, wandb, and the periodic probe rollouts rather than handing
# every allocated CPU to the dataloader workers.
export NUM_WORKERS="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} - 8 ))}"
exec bash examples/robocasa/run_train_rlt.sh ${ARGS}
