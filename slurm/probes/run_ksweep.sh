#!/bin/bash
#SBATCH -J ksweep
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -t 16:00:00
#SBATCH -o /scratch/jellyho/acrft/probes/ksweep_%j.out
# M4 fixed-k sweep (pre-registered in hub entry theory-preexp): official RoboCasa-365 pi05
# (pretrain_human300/75000) on the 5 selected tasks, executing only the first K actions of each
# 16-action chunk (--replan-steps K). One job per K in {1,2,4,8,12,16}; K=16 reruns the full-chunk
# control in-job. Same seed/scene convention as the task scan (seed 3000, 20 trials).
# Registered prediction (Thm floor / P4): best fixed k varies by task, k=1 not globally optimal,
# SR(k) non-monotone. Rejected if k=1 is globally optimal.
set -uo pipefail
K=${1:?usage: sbatch run_ksweep.sh K}
unset LD_LIBRARY_PATH   # miniconda libcrypto poisons system python (_hashlib OPENSSL mismatch)
cd /home/jellyho/ACRFT/ACRFT
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=offline
export HF_LEROBOT_HOME=/scratch/jellyho/acrft/lerobot
export PYTHONPATH=/home/jellyho/ACRFT/ACRFT/third_party/robocasa:${PYTHONPATH:-}   # robocasa is vendored, not installed
CKPT=/scratch/jellyho/acrft/checkpoints/robocasa365_official/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_robocasa_pretrained
PORT=$((8300 + K))   # unique per K so two jobs sharing a node never collide
OUT=/scratch/jellyho/acrft/gr1_eval/ksweep/k${K}
mkdir -p "$OUT"
EVAL_PYTHON=.venv/bin/python

TASKS=(OpenDrawer CoffeeServeMug TurnOnStove PickPlaceSinkToCounter PickPlaceCounterToMicrowave)

echo "=== ksweep K=$K: starting policy server (config=$CONFIG, port=$PORT) ==="
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
    except Exception:
        time.sleep(5)
else:
    raise SystemExit(1)
PY

for TASK in "${TASKS[@]}"; do
  echo "================ K=$K $TASK ================"
  "$EVAL_PYTHON" examples/robocasa/main.py --task "$TASK" --host 127.0.0.1 --port "$PORT" \
      --num-trials 20 --num-videos 1 --seed 3000 \
      --env-action-order --replan-steps "$K" \
      --video-dir "$OUT/$TASK/videos" --output-json "$OUT/$TASK/results.json" \
      2>&1 | tail -5 || echo "!! K=$K $TASK failed"
  "$EVAL_PYTHON" - "$OUT/$TASK/results.json" "$K" <<'PY' 2>/dev/null || true
import json, sys
r = json.load(open(sys.argv[1]))
print(f"KSWEEP k={sys.argv[2]} {sys.argv[1].split('/')[-2]}: {r.get('successes')}/{r.get('num_trials')} = {r.get('success_rate')}")
PY
done
echo "KSWEEP_K${K}_DONE"
