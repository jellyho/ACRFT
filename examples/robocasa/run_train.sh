#!/usr/bin/env bash
# Train pi05 on one or more RoboCasa 365 target tasks.
#
# Each task has a registered config `pi05_robocasa_<Task>` (repo_id jellyho/robocasa365-<Task>,
# pi05, constant 5e-5, batch 32, 100k steps, checkpoint every 10k). This script runs the two
# steps for each task: compute norm stats, then train.
#
# Usage:
#   examples/robocasa/run_train.sh PrepareCoffee
#   examples/robocasa/run_train.sh PrepareCoffee OpenDrawer TurnOnMicrowave   # sequential
#
# Env overrides:
#   EXP_SUFFIX=run           exp name is "<Task>_<EXP_SUFFIX>" (checkpoints/<config>/<exp>)
#   SKIP_NORM_STATS=1        skip norm-stats (e.g. already computed)
#   ROBOCASA_LOCAL_DIR=...   local converted datasets to symlink into the HF cache (avoids
#                            re-downloading from the Hub). Default /data5/jellyho/robocasa365.
#   HF_USER=jellyho          Hub owner used in the repo id / cache path.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXP_SUFFIX="${EXP_SUFFIX:-run}"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-0}"
HF_USER="${HF_USER:-jellyho}"
ROBOCASA_LOCAL_DIR="${ROBOCASA_LOCAL_DIR:-/data5/jellyho/robocasa365}"
# Match lerobot's cache resolution: HF_LEROBOT_HOME, else $HF_HOME/lerobot, else ~/.cache/...
LEROBOT_CACHE="${HF_LEROBOT_HOME:-${HF_HOME:+$HF_HOME/lerobot}}"
LEROBOT_CACHE="${LEROBOT_CACHE:-$HOME/.cache/huggingface/lerobot}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <Task> [<Task> ...]"
  echo "Example: $0 PrepareCoffee"
  exit 1
fi

# Symlink a local converted dataset into the HF cache so the Hub repo id resolves to it
# (no multi-hundred-MB re-download). No-op if already present or the local copy is missing.
link_local() {
  local task="$1"
  local src="$ROBOCASA_LOCAL_DIR/$task"
  local dst="$LEROBOT_CACHE/$HF_USER/robocasa365-$task"
  if [ -d "$src/meta" ] && [ ! -e "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst" && echo "  [+] linked local dataset -> $dst"
  fi
}

failures=()
for TASK in "$@"; do
  CONFIG="pi05_robocasa_${TASK}"
  EXP="${TASK}_${EXP_SUFFIX}"
  echo "================================================================"
  echo " Task: $TASK   Config: $CONFIG   Exp: $EXP"
  echo "================================================================"

  link_local "$TASK"

  if [ "$SKIP_NORM_STATS" != "1" ]; then
    echo "[1/2] compute norm stats ($CONFIG)"
    if ! uv run scripts/compute_norm_stats.py --config-name="$CONFIG"; then
      echo "  !! norm stats failed for $TASK"; failures+=("$TASK (norm-stats)"); continue
    fi
  else
    echo "[1/2] skipping norm stats (SKIP_NORM_STATS=1)"
  fi

  echo "[2/2] train ($CONFIG --exp-name=$EXP)"
  if ! uv run scripts/train.py "$CONFIG" --exp-name="$EXP"; then
    echo "  !! training failed for $TASK"; failures+=("$TASK (train)"); continue
  fi
  echo "  done: checkpoints/$CONFIG/$EXP"
done

echo "================================================================"
if [ "${#failures[@]}" -eq 0 ]; then
  echo "All tasks finished."
else
  echo "Failures:"; printf '  - %s\n' "${failures[@]}"; exit 1
fi
