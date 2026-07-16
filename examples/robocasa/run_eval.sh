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
#   SEED=0             base seed; trial i uses seed+i, fixed across checkpoints so every
#                      checkpoint is evaluated on the identical set of scenes (fair comparison)
#   PORT=8000          policy server port (auto-bumped to the next free port if busy, so several
#                      eval runs can share a node; server shutdown is scoped to this run's port)
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
SEED="${SEED:-0}"          # fixed across checkpoints -> identical scenes for a fair comparison
PORT="${PORT:-8000}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
CONFIG="pi05_robocasa_${TASK}"
EXP="${TASK}_${EXP_SUFFIX}"
CKPT_BASE="$REPO_ROOT/checkpoints/$CONFIG/$EXP"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval/$CONFIG/$EXP}"

port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; }; return 1; }
# Pick the first free port at/after $PORT so several eval runs can share a node without clashing.
_req_port="$PORT"
for _ in $(seq 1 200); do port_in_use "$PORT" || break; PORT=$((PORT + 1)); done
[ "$PORT" != "$_req_port" ] && echo "port $_req_port busy -> using $PORT"

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

# The rollout client (main.py) needs `robocasa` and `openpi_client`, both of which live in this
# repo — add them to the client's import path so EVAL_PYTHON only needs the heavy sim deps
# (numpy==2.2.5, robosuite, mujoco, websockets, imageio).
export PYTHONPATH="$REPO_ROOT/third_party/robocasa:$REPO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

# Pre-flight: verify the client env before spinning up any server, so failures are immediate and clear.
if ! _err="$("$EVAL_PYTHON" -c "import numpy, robosuite, robocasa, openpi_client.websocket_client_policy, imageio" 2>&1)"; then
  echo "ERROR: EVAL_PYTHON ($EVAL_PYTHON) cannot import the eval client stack:"
  echo "$_err" | tail -3 | sed 's/^/    /'
  echo
  echo "Point EVAL_PYTHON at a Python env with: numpy==2.2.5, robosuite, mujoco, websockets, imageio."
  echo "(robocasa + openpi_client come from this repo via PYTHONPATH — you don't install those.)"
  echo "See examples/robocasa/README.md for a one-time env setup. Example:"
  echo "    EVAL_PYTHON=/home/jellyho/miniconda3/envs/robocasa_eval/bin/python $0 $TASK"
  exit 1
fi

SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  # Only kill the server for THIS run's port (matched in the cmdline), so concurrent eval runs
  # on the same node don't kill each other's servers.
  pkill -f "serve_policy.py.*--port ${PORT}([^0-9]|\$)" 2>/dev/null
  SERVER_PID=""
  # wait for the port to be released
  for _ in $(seq 1 30); do
    port_in_use "$PORT" && sleep 1 || break
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
  # Top-level options (--port) go BEFORE the `policy:checkpoint` subcommand; subcommand options
  # (--policy.*) go after (tyro applies args to the directly preceding subcommand).
  uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
      --policy.config "$CONFIG" --policy.dir "$CKPT" \
      > "$STEP_OUT/server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_port 600; then
    echo "  !! server did not come up for $STEP (see $STEP_OUT/server.log)"
    stop_server; continue
  fi

  echo "  running $NUM_TRIALS rollouts (saving up to $NUM_VIDEOS videos) ..."
  "$EVAL_PYTHON" examples/robocasa/main.py --task "$TASK" --host 127.0.0.1 --port "$PORT" \
      --num-trials "$NUM_TRIALS" --num-videos "$NUM_VIDEOS" --seed "$SEED" \
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
