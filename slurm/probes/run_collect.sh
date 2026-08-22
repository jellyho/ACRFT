#!/bin/bash
#SBATCH -J collect
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -t 20:00:00
#SBATCH -o /scratch/jellyho/acrft/probes/collect_%j.out
# B1 data collection: roll out the official RoboCasa-365 pi05 on ONE task and record every episode
# (successes AND failures) as npz. The success filter is applied later at dataset-build time, so the
# same collection also feeds B2 (advantage-weighted) and B3 (Q-filtered) without re-running the GPU.
#
# The execution length is the task's best fixed k from the M4 sweep (hub entry m4-ksweep), not the
# default full chunk: at the same GPU cost it yields more successful episodes, and M4 established
# best-fixed-k as the honest operating point.
#
#   sbatch run_collect.sh <Task> <K> [N_TRIALS] [SEED]
set -uo pipefail
TASK=${1:?usage: sbatch run_collect.sh Task K [N] [SEED]}
K=${2:?need K}
N=${3:-150}
SEED=${4:-7000}   # disjoint from the eval seed (3000) so collection scenes are not the eval scenes
unset LD_LIBRARY_PATH   # miniconda libcrypto poisons system python (_hashlib OPENSSL mismatch)
cd /home/jellyho/ACRFT/ACRFT
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=offline
export HF_LEROBOT_HOME=/scratch/jellyho/acrft/lerobot
export PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/robocasa:${PYTHONPATH:-}
CKPT=/scratch/jellyho/acrft/checkpoints/robocasa365_official/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_robocasa_pretrained
PORT=$((8400 + K + RANDOM % 50))
OUT=/scratch/jellyho/acrft/collect/${TASK}
mkdir -p "$OUT"
EVAL_PYTHON=.venv/bin/python

echo "=== collect $TASK k=$K n=$N seed=$SEED (port $PORT) ==="
uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config "$CONFIG" --policy.dir "$CKPT" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

"$EVAL_PYTHON" - "$PORT" <<'PY' || { echo "SERVER WARMUP FAILED"; exit 1; }
import sys, time
from openpi_client import websocket_client_policy as W
from openpi.policies.robocasa_policy import make_robocasa_example
port = int(sys.argv[1]); ex = make_robocasa_example()
for _ in range(200):
    try:
        W.WebsocketClientPolicy("127.0.0.1", port).infer(ex)
        print("warmup ok"); break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit(1)
PY

"$EVAL_PYTHON" examples/robocasa/main.py --task "$TASK" --host 127.0.0.1 --port "$PORT" \
    --num-trials "$N" --num-videos 0 --seed "$SEED" \
    --env-action-order --replan-steps "$K" \
    --traj-dir "$OUT/traj" --output-json "$OUT/results.json" 2>&1 | tail -4

"$EVAL_PYTHON" - "$OUT" <<'PY' || true
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
r = json.load(open(out / "results.json"))
n = len(list((out / "traj").glob("*.npz")))
print(f"COLLECT {out.name}: {r['successes']}/{r['num_trials']} = {r['success_rate']} | {n} episodes recorded")
PY
echo "COLLECT_DONE ${TASK}"
