#!/usr/bin/env bash
# Evaluate EVERY checkpoint of a trained RoboCasa pi05 model: N rollouts each, saving per-step
# success rates (+ a summary) and a few sample rollout videos.
#
# Uses openpi's server/client split, so TWO Python envs are involved:
#   - policy server  -> `uv run` (the openpi / LeRobot v3.0 training venv)
#   - rollout client -> $EVAL_PYTHON (a venv with robosuite + robocasa + openpi-client;
#                       robosuite pins numpy<2, so keep it separate from the training venv)
# For each checkpoint the script starts a server, waits for it, runs the rollouts, then stops it.
#
# Usage:
#   examples/robocasa/run_eval.sh PrepareCoffee
#   EVAL_PYTHON=/path/to/robocasa-venv/bin/python examples/robocasa/run_eval.sh PrepareCoffee
#
# Env overrides:
#   EXP_SUFFIX=run     exp name suffix -> checkpoints/pi05_robocasa_<Task>/<Task>_<EXP_SUFFIX>
#   NUM_TRIALS=50      rollouts per checkpoint
#   NUM_VIDEOS=5       sample videos saved per checkpoint
#   PORT=8000          policy server port
#   EVAL_PYTHON=python client interpreter (must have robosuite/robocasa/openpi-client)
#   OUT_DIR=...        output root (default eval/<config>/<exp>)
#   STEPS="10000 20000"  only these checkpoints (default: all)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${1:-}"
[ -z "$TASK" ] && { echo "Usage: $0 <Task>"; exit 1; }
EXP_SUFFIX="${EXP_SUFFIX:-run}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_VIDEOS="${NUM_VIDEOS:-5}"
PORT="${PORT:-8000}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
CONFIG="pi05_robocasa_${TASK}"
EXP="${TASK}_${EXP_SUFFIX}"
CKPT_BASE="$REPO_ROOT/checkpoints/$CONFIG/$EXP"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval/$CONFIG/$EXP}"

[ -d "$CKPT_BASE" ] || { echo "No checkpoints at $CKPT_BASE"; exit 1; }
if [ -n "${STEPS:-}" ]; then
  read -r -a STEP_LIST <<< "$STEPS"
else
  mapfile -t STEP_LIST < <(find "$CKPT_BASE" -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' | sort -n)
fi
[ "${#STEP_LIST[@]}" -eq 0 ] && { echo "No numeric checkpoint dirs in $CKPT_BASE"; exit 1; }
echo "Config: $CONFIG | ${#STEP_LIST[@]} checkpoints: ${STEP_LIST[*]}"
echo "Output: $OUT_DIR"
mkdir -p "$OUT_DIR"

SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  pkill -f "scripts/serve_policy.py" 2>/dev/null
  SERVER_PID=""
  # wait for the port to be released
  for _ in $(seq 1 30); do
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && { exec 3>&- 3<&-; sleep 1; } || break
  done
}
trap 'stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

wait_for_port() {  # $1 tries (x2s)
  for ((i=0; i<${1:-600}; i++)); do
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
    sleep 2
  done
  return 1
}

for STEP in "${STEP_LIST[@]}"; do
  CKPT="$CKPT_BASE/$STEP"
  STEP_OUT="$OUT_DIR/$STEP"
  mkdir -p "$STEP_OUT"
  echo "================ checkpoint $STEP ================"

  echo "  starting policy server (config=$CONFIG dir=$CKPT) ..."
  uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config "$CONFIG" --policy.dir "$CKPT" --port "$PORT" \
      > "$STEP_OUT/server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_port 600; then
    echo "  !! server did not come up for $STEP (see $STEP_OUT/server.log)"
    stop_server; continue
  fi

  echo "  running $NUM_TRIALS rollouts (saving up to $NUM_VIDEOS videos) ..."
  "$EVAL_PYTHON" examples/robocasa/main.py --task "$TASK" --host 127.0.0.1 --port "$PORT" \
      --num-trials "$NUM_TRIALS" --num-videos "$NUM_VIDEOS" \
      --video-dir "$STEP_OUT/videos" --output-json "$STEP_OUT/results.json" \
      2>&1 | tee "$STEP_OUT/client.log" || echo "  !! rollouts failed for $STEP"

  stop_server
done

# Aggregate per-checkpoint success rates into one CSV.
SUMMARY="$OUT_DIR/summary.csv"
echo "step,successes,num_trials,success_rate" > "$SUMMARY"
for STEP in "${STEP_LIST[@]}"; do
  R="$OUT_DIR/$STEP/results.json"
  [ -f "$R" ] || continue
  python3 -c "import json;d=json.load(open('$R'));print(f\"$STEP,{d['successes']},{d['num_trials']},{d['success_rate']:.4f}\")" >> "$SUMMARY"
done
echo "================ summary ================"
cat "$SUMMARY"
echo "Wrote $SUMMARY (+ per-step results.json and videos under $OUT_DIR)."
