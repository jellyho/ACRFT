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
# Norm stats are computed once per task and saved to assets/<config>/<repo_id>/norm_stats.json;
# subsequent runs detect the file and skip recomputation automatically.
#
# Env overrides:
#   EXP_SUFFIX=run           exp name is "<Task>_<EXP_SUFFIX>" (checkpoints/<config>/<exp>)
#   SKIP_NORM_STATS=1        always skip norm-stats
#   FORCE_NORM_STATS=1       recompute norm-stats even if the file already exists
#   RESUME=1                 resume training from the last checkpoint
#   OVERWRITE=1              wipe the checkpoint dir and start fresh
#     (default: a dir with real checkpoints is protected; a stale dir with no saved
#      steps is auto-overwritten)
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

# True if the exp dir holds an actual saved checkpoint (a numeric step subdir), vs. just stale
# metadata (e.g. wandb_id.txt) left by a run that failed before the first save.
has_checkpoint() {
  [ -d "$1" ] && find "$1" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | grep -q .
}

failures=()
for TASK in "$@"; do
  CONFIG="pi05_robocasa_${TASK}"
  EXP="${TASK}_${EXP_SUFFIX}"
  echo "================================================================"
  echo " Task: $TASK   Config: $CONFIG   Exp: $EXP"
  echo "================================================================"

  link_local "$TASK"

  # Norm stats are computed once and persisted here; training reloads them (no need to recompute).
  NORM_STATS_FILE="$REPO_ROOT/assets/$CONFIG/$HF_USER/robocasa365-$TASK/norm_stats.json"
  if [ "$SKIP_NORM_STATS" = "1" ]; then
    echo "[1/2] skipping norm stats (SKIP_NORM_STATS=1)"
  elif [ -f "$NORM_STATS_FILE" ] && [ "${FORCE_NORM_STATS:-0}" != "1" ]; then
    echo "[1/2] norm stats already exist, skipping ($NORM_STATS_FILE)"
    echo "      set FORCE_NORM_STATS=1 to recompute."
  else
    echo "[1/2] compute norm stats ($CONFIG)"
    if ! uv run scripts/compute_norm_stats.py --config-name="$CONFIG"; then
      echo "  !! norm stats failed for $TASK"; failures+=("$TASK (norm-stats)"); continue
    fi
  fi

  # Decide how to handle an existing checkpoint dir.
  CKPT_DIR="$REPO_ROOT/checkpoints/$CONFIG/$EXP"
  TRAIN_FLAGS=()
  if [ "${OVERWRITE:-0}" = "1" ]; then
    TRAIN_FLAGS+=(--overwrite)
  elif [ "${RESUME:-0}" = "1" ]; then
    TRAIN_FLAGS+=(--resume)
  elif [ -d "$CKPT_DIR" ]; then
    if has_checkpoint "$CKPT_DIR"; then
      echo "  !! $CKPT_DIR has saved checkpoints; set RESUME=1 to continue or OVERWRITE=1 to restart."
      failures+=("$TASK (checkpoint exists)"); continue
    fi
    echo "  stale checkpoint dir (no saved steps) -> auto --overwrite"
    TRAIN_FLAGS+=(--overwrite)
  fi

  echo "[2/2] train ($CONFIG --exp-name=$EXP ${TRAIN_FLAGS[*]})"
  if ! uv run scripts/train.py "$CONFIG" --exp-name="$EXP" "${TRAIN_FLAGS[@]}"; then
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
