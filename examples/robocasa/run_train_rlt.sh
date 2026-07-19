#!/usr/bin/env bash
# Train pi05 + RLT ("RL Token") on one or more RoboCasa 365 target tasks.
#
# Same recipe as run_train.sh, except the model is Pi0RLT: the RL-token bottleneck is learned
# JOINTLY with the BC finetune (one backbone forward computes both losses). Each task has a
# registered config `pi05_robocasa_<Task>_rlt`.
#
# Usage:
#   examples/robocasa/run_train_rlt.sh PrepareCoffee
#   examples/robocasa/run_train_rlt.sh PrepareCoffee OpenDrawer          # sequential
#
# Because RLT trains together with BC, this script runs the SAME two steps as run_train.sh:
# compute norm stats (if missing), then train. RLT does not change normalization, so it reuses the
# BC config's stats at assets/pi05_robocasa_<Task>/... — shared with `run_train.sh`, computed once.
#
# Env overrides (same as run_train.sh):
#   EXP_SUFFIX=rlt           exp name is "<Task>_<EXP_SUFFIX>" (checkpoints/<config>/<exp>)
#   SKIP_NORM_STATS=1        always skip norm-stats
#   FORCE_NORM_STATS=1       recompute norm-stats even if the file already exists
#   RESUME=1 / OVERWRITE=1   resume from / wipe the checkpoint dir
#   ROBOCASA_LOCAL_DIR=...   local converted datasets to symlink into the HF cache
#   HF_USER=jellyho          Hub owner used in the repo id / cache path
#
# RLT variant switches (each gets its own exp-name tag so variants never share a checkpoint dir):
#   BACKBONE_GRAD=1          let the RLT loss reshape the VLM backbone (default: stop-gradient,
#                            i.e. the RL token is a pure readout head and BC alone shapes the VLM)
#   NO_TARGET_SG=1           do not stop-gradient the reconstruction target (collapse-prone)
#   RLT_LOSS_WEIGHT=1.0      weight of the RLT loss relative to the BC loss
#   MONITOR_INTERVAL=1000    steps between RLT embedding diagnostics (0 disables)
#   VIS_INTERVAL=5000        steps between RLT embedding trajectory visualizations (0 disables)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXP_SUFFIX="${EXP_SUFFIX:-rlt}"
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

# Build the RLT variant flags, and tag the exp name so different variants can't collide.
RLT_FLAGS=()
VARIANT_TAG=""
if [ "${BACKBONE_GRAD:-0}" = "1" ]; then
  RLT_FLAGS+=(--model.rlt-backbone-gradient); VARIANT_TAG="${VARIANT_TAG}_bbgrad"
fi
if [ "${NO_TARGET_SG:-0}" = "1" ]; then
  RLT_FLAGS+=(--model.no-rlt-target-stop-gradient); VARIANT_TAG="${VARIANT_TAG}_notgtsg"
fi
if [ -n "${RLT_LOSS_WEIGHT:-}" ]; then
  RLT_FLAGS+=(--model.rlt-loss-weight "$RLT_LOSS_WEIGHT"); VARIANT_TAG="${VARIANT_TAG}_w${RLT_LOSS_WEIGHT}"
fi
[ -n "${MONITOR_INTERVAL:-}" ] && RLT_FLAGS+=(--rlt-monitor-interval "$MONITOR_INTERVAL")
[ -n "${VIS_INTERVAL:-}" ] && RLT_FLAGS+=(--rlt-vis-interval "$VIS_INTERVAL")
[ -n "$VARIANT_TAG" ] && echo "RLT variant flags: ${RLT_FLAGS[*]}  (exp-name tag: $VARIANT_TAG)"

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
  CONFIG="pi05_robocasa_${TASK}_rlt"
  EXP="${TASK}_${EXP_SUFFIX}${VARIANT_TAG}"
  echo "================================================================"
  echo " Task: $TASK   Config: $CONFIG   Exp: $EXP"
  echo "================================================================"

  link_local "$TASK"

  # Norm stats: the RLT configs read the SHARED stats checked into the repo, so this normally just
  # confirms the file is present and skips — a fresh clone can train without downloading a dataset to
  # recompute them. If it is ever missing, we fall back to the same computation run_train.sh uses.
  NORM_STATS_FILE="$REPO_ROOT/examples/robocasa/norm_stats/robocasa365_shared/norm_stats.json"
  if [ "$SKIP_NORM_STATS" = "1" ]; then
    echo "[1/2] skipping norm stats (SKIP_NORM_STATS=1)"
  elif [ -f "$NORM_STATS_FILE" ] && [ "${FORCE_NORM_STATS:-0}" != "1" ]; then
    echo "[1/2] norm stats already exist, skipping ($NORM_STATS_FILE)"
    echo "      set FORCE_NORM_STATS=1 to recompute."
  else
    # Shared normalization across ALL tasks (writes every task's norm_stats.json in one pass).
    # Per-task stats are ill-conditioned for near-stationary tasks (base/control ~constant -> a
    # ~0 range that blows up the loss); a shared range is well-conditioned. Runs once: after the
    # first task all others hit the auto-skip above.
    echo "[1/2] compute SHARED norm stats (all tasks)"
    if ! uv run examples/robocasa/compute_shared_norm_stats.py --output-dir "$ROBOCASA_LOCAL_DIR" --hf-user "$HF_USER"; then
      echo "  !! norm stats failed for $TASK"; failures+=("$TASK (norm-stats)"); continue
    fi
    # That script writes per-task files under assets/pi05_robocasa_<Task>/...; publish one copy at
    # the shared path the RLT configs read (all per-task files are byte-identical by construction).
    SRC="$REPO_ROOT/assets/pi05_robocasa_${TASK}/$HF_USER/robocasa365-$TASK/norm_stats.json"
    if [ -f "$SRC" ]; then
      mkdir -p "$(dirname "$NORM_STATS_FILE")" && cp "$SRC" "$NORM_STATS_FILE"
      echo "  [+] published shared norm stats -> $NORM_STATS_FILE"
    else
      echo "  !! expected $SRC after computing norm stats"; failures+=("$TASK (norm-stats)"); continue
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

  echo "[2/2] train ($CONFIG --exp-name=$EXP ${TRAIN_FLAGS[*]} ${RLT_FLAGS[*]})"
  if ! uv run scripts/train.py "$CONFIG" --exp-name="$EXP" "${TRAIN_FLAGS[@]}" "${RLT_FLAGS[@]}"; then
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
