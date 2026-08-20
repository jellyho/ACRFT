#!/bin/bash
#SBATCH -J taskscan
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -t 16:00:00
#SBATCH -o /scratch/jellyho/acrft/probes/taskscan_%j.out
# Multi-task SR scan: the official RoboCasa-365 pi05 (pretrain_human300/75000, worker A's serve fix)
# evaluated on 14 atomic/pick-place tasks to find the 30-60% band for the baseline ladder.
# One server (task-agnostic, prompt_from_task) + client loop. 20 trials/task, fixed seed -> same
# scene set convention as run_eval.sh. --env-action-order is REQUIRED for this checkpoint.
set -uo pipefail
unset LD_LIBRARY_PATH   # miniconda libcrypto poisons system python (_hashlib OPENSSL mismatch)
cd /home/jellyho/ACRFT/ACRFT
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=offline
export HF_LEROBOT_HOME=/scratch/jellyho/acrft/lerobot
export PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/robocasa:${PYTHONPATH:-}   # robocasa is vendored, not installed
CKPT=/scratch/jellyho/acrft/checkpoints/robocasa365_official/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_robocasa_pretrained
PORT=8321
OUT=/scratch/jellyho/acrft/gr1_eval/task_scan
mkdir -p "$OUT"
EVAL_PYTHON=.venv/bin/python

TASKS=(OpenDrawer CloseDrawer OpenDoor CloseDoor \
       CoffeeSetupMug CoffeeServeMug StartCoffeeMachine \
       TurnOnStove TurnOffStove TurnOnSinkFaucet \
       PickPlaceCounterToSink PickPlaceSinkToCounter \
       PickPlaceCounterToMicrowave PickPlaceMicrowaveToCounter)

echo "=== starting policy server (config=$CONFIG) ==="
uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config "$CONFIG" --policy.dir "$CKPT" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

# wait for server + warm up (JIT)
"$EVAL_PYTHON" - "$PORT" <<'PY' || { echo "SERVER WARMUP FAILED"; exit 1; }
import sys, time
from openpi_client import websocket_client_policy as W
from openpi.policies.robocasa_policy import make_robocasa_example
port = int(sys.argv[1]); ex = make_robocasa_example()
for i in range(200):
    try:
        W.WebsocketClientPolicy("127.0.0.1", port).infer(ex)
        print("warmup ok"); break
    except Exception as e:
        time.sleep(5)
else:
    raise SystemExit(1)
PY

for TASK in "${TASKS[@]}"; do
  echo "================ $TASK ================"
  "$EVAL_PYTHON" examples/robocasa/main.py --task "$TASK" --host 127.0.0.1 --port "$PORT" \
      --num-trials 20 --num-videos 2 --seed 3000 \
      --env-action-order \
      --video-dir "$OUT/$TASK/videos" --output-json "$OUT/$TASK/results.json" \
      2>&1 | tail -5 || echo "!! $TASK failed"
  # print running summary line
  "$EVAL_PYTHON" - "$OUT/$TASK/results.json" <<'PY' 2>/dev/null || true
import json,sys
r=json.load(open(sys.argv[1])); print(f"SCAN {sys.argv[1].split('/')[-2]}: {r.get('successes')}/{r.get('num_trials')} = {r.get('success_rate')}")
PY
done
echo "SCAN_ALL_DONE"