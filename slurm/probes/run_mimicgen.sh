#!/bin/bash
#SBATCH -J mgtrain
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jellyho/acrft/probes/mgtrain_%j.out
# One MimicGen ladder arm, trained then evaluated at several commitment lengths.
#
# Training and evaluation live in different interpreters on purpose: MimicGen's environments target
# robosuite 1.4.x while the main environment carries 1.5.2 for RoboCasa. The checkpoint is the
# handoff, so nothing about the model depends on which simulator version is installed.
#
#   sbatch run_mimicgen.sh <task> <arm> [seed] [steps]
set -uo pipefail
TASK=${1:?usage: sbatch run_mimicgen.sh task arm [seed] [steps]}
ARM=${2:?need arm}
SEED=${3:-0}
STEPS=${4:-30000}
unset LD_LIBRARY_PATH   # miniconda libcrypto poisons system python (_hashlib OPENSSL mismatch)
cd /home/jellyho/ACRFT/ACRFT
export MUJOCO_GL=osmesa
RUN=/scratch/jellyho/acrft/mimicgen/runs/${TASK}_${ARM}_s${SEED}

echo "=== train ${TASK} / ${ARM} / seed ${SEED} ==="
.venv/bin/python slurm/probes/mimicgen_train.py --task "$TASK" --arm "$ARM" --seed "$SEED" \
    --steps "$STEPS" --critic-steps 15000 2>&1 | tail -20 || { echo "TRAIN FAILED"; exit 1; }

echo "=== evaluate ==="
# every arm is measured at the same fixed lengths, and the ones with a critic also at their own
# choice, so an adaptive number is never compared against an unmeasured constant
for K in 1 2 4 8 16; do
  PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/mimicgen .venv-mimicgen/bin/python \
      slurm/probes/mimicgen_eval.py --ckpt "$RUN/ckpt.pt" --commit "$K" --episodes 50 --max-steps 500 \
      2>&1 | grep -E '"success_rate"|"mean_steps"' | tr '\n' ' ' | sed "s/^/  k=$K /" ; echo
done
if [ "$ARM" = "awr" ] || [ "$ARM" = "cfac" ]; then
  PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/mimicgen .venv-mimicgen/bin/python \
      slurm/probes/mimicgen_eval.py --ckpt "$RUN/ckpt.pt" --commit 0 --episodes 50 --max-steps 500 \
      2>&1 | grep -E '"success_rate"|"mean_steps"|"mean_k"' | tr '\n' ' ' | sed 's/^/  critic /' ; echo
fi
echo "MGTRAIN_DONE ${TASK} ${ARM} ${SEED}"
