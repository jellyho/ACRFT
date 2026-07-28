#!/usr/bin/env bash
# Train pi05 + RLT ("RL Token") on one or more RoboCasa 365 target tasks.
#
# Same recipe as run_train.sh, except the model is Pi0RLT: the RL-token bottleneck is learned
# JOINTLY with the BC finetune (one backbone forward computes both losses). Each task has a
# registered config `pi05_robocasa_<Task>_rlt`.
#
# Usage:
#   examples/robocasa/run_train_rlt.sh <Task> [<Task> ...] [options]
#
#   examples/robocasa/run_train_rlt.sh PrepareCoffee
#   examples/robocasa/run_train_rlt.sh PrepareCoffee --backbone-grad
#   examples/robocasa/run_train_rlt.sh PrepareCoffee OpenDrawer --loss-weight 0.5 --vis-interval 2000
#
# Options and tasks may be given in any order. Every option also has an env-var form (shown in
# parentheses); the flag wins if both are given.
#
#   RLT ablation switches — each one tags the exp name, so variants never share a checkpoint dir:
#     --backbone-grad          let the RLT loss reshape the VLM backbone      (BACKBONE_GRAD=1)
#                              default: stop-gradient, i.e. the RL token is a pure readout head
#                              and BC alone shapes the VLM
#     --no-backbone-grad       force the default off                          (BACKBONE_GRAD=0)
#     --no-target-sg           do not stop-gradient the reconstruction target (NO_TARGET_SG=1)
#                              (collapse-prone; try --backbone-grad alone first)
#     --parallel-decoder       decode every token from z_rl alone, no teacher forcing, so the
#                              bottleneck cannot be bypassed via neighbouring-token context.
#                              Deviates from the paper's Eq. 2 — check rlt/bypass_ratio first.
#     --no-proprio             build the token from image+language only: no proprio token into the
#                              encoder and no proprio reconstruction term. This is the paper's own
#                              setup, where the critic takes (z_rl, s^p) and gets proprio directly,
#                              so whatever consumes z_rl must supply proprio itself. (NO_PROPRIO=1)
#     --objective STR          reconstruction | progress | reconstruction+progress (RLT_OBJECTIVE=)
#                              anything with "progress" regresses time-to-success, derived from the
#                              sparse success reward (see training/progress.py)
#     --scalar-head            progress head = MSE regression instead of the default HL-Gauss
#                              histogram + cross-entropy                     (SCALAR_HEAD=1)
#     --progress-bins INT      histogram bins for the distributional head    (PROGRESS_BINS=)
#     --loss-weight FLOAT      weight of the RLT loss vs the BC loss          (RLT_LOSS_WEIGHT=)
#     --token-dim INT          bottleneck size of the RL token (default 2048)  (TOKEN_DIM=)
#                              smaller = cheaper critic queries; tags the exp name _d<N>
#
#   Monitoring:
#     --monitor-interval INT   steps between RLT embedding diagnostics, 0 off (MONITOR_INTERVAL=)
#     --vis-interval INT       steps between trajectory visualizations, 0 off (VIS_INTERVAL=)
#
#   Run control:
#     --exp-suffix STR         exp name is "<Task>_<suffix>" (default: rlt)   (EXP_SUFFIX=)
#     --project STR            wandb project name (default: acrft)            (PROJECT=)
#     --entity STR             wandb entity / team (default: your wandb user)  (ENTITY=)
#     --group STR              wandb group: bucket related runs (e.g. a sweep) (GROUP=)
#     --resume                 resume from the last checkpoint                (RESUME=1)
#     --overwrite              wipe the checkpoint dir and start fresh        (OVERWRITE=1)
#     --skip-norm-stats        never compute norm stats                       (SKIP_NORM_STATS=1)
#     --force-norm-stats       recompute even if they already exist           (FORCE_NORM_STATS=1)
#
# Because RLT trains together with BC, this runs the SAME two steps as run_train.sh: check/compute
# norm stats, then train. The RLT configs read the SHARED stats checked into the repo, so this
# normally just confirms the file is present — a fresh clone can train with no dataset download.
#
# Other env: HF_USER=jellyho, ROBOCASA_LOCAL_DIR=/data5/jellyho/robocasa365
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

# Env-var defaults; the matching flags below override them.
EXP_SUFFIX="${EXP_SUFFIX:-rlt}"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-0}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
BACKBONE_GRAD="${BACKBONE_GRAD:-0}"
NO_TARGET_SG="${NO_TARGET_SG:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
PROJECT="${PROJECT:-robocasa-pi05}"
ENTITY="${ENTITY:-RSS-PFT_RLLAB}"
GROUP="${GROUP:-RLT_DEBUG}"
HF_USER="${HF_USER:-jellyho}"
ROBOCASA_LOCAL_DIR="${ROBOCASA_LOCAL_DIR:-/data5/jellyho/robocasa365}"
# Match lerobot's cache resolution: HF_LEROBOT_HOME, else $HF_HOME/lerobot, else ~/.cache/...
LEROBOT_CACHE="${HF_LEROBOT_HOME:-${HF_HOME:+$HF_HOME/lerobot}}"
LEROBOT_CACHE="${LEROBOT_CACHE:-$HOME/.cache/huggingface/lerobot}"

TASKS=()
while [ "$#" -gt 0 ]; do
  # Accept both "--opt value" and "--opt=value" by splitting the latter into the former.
  case "$1" in --*=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;; esac
  case "$1" in
    --backbone-grad)     BACKBONE_GRAD=1 ;;
    --no-backbone-grad)  BACKBONE_GRAD=0 ;;
    --no-target-sg)      NO_TARGET_SG=1 ;;
    --parallel-decoder)  PARALLEL_DECODER=1 ;;
    --no-proprio)        NO_PROPRIO=1 ;;
    --objective)         RLT_OBJECTIVE="$2"; shift ;;
    --scalar-head)       SCALAR_HEAD=1 ;;
    --progress-bins)     PROGRESS_BINS="$2"; shift ;;
    --loss-weight)       RLT_LOSS_WEIGHT="$2"; shift ;;
    --token-dim)         TOKEN_DIM="$2"; shift ;;
    --monitor-interval)  MONITOR_INTERVAL="$2"; shift ;;
    --vis-interval)      VIS_INTERVAL="$2"; shift ;;
    --exp-suffix)        EXP_SUFFIX="$2"; shift ;;
    --project)           PROJECT="$2"; shift ;;
    --entity)            ENTITY="$2"; shift ;;
    --group)             GROUP="$2"; shift ;;
    --resume)            RESUME=1 ;;
    --overwrite)         OVERWRITE=1 ;;
    --skip-norm-stats)   SKIP_NORM_STATS=1 ;;
    --force-norm-stats)  FORCE_NORM_STATS=1 ;;
    -h|--help)           usage; exit 0 ;;
    -*)                  echo "Unknown option: $1"; echo; usage; exit 1 ;;
    *)                   TASKS+=("$1") ;;
  esac
  shift
done

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "Error: no task given."; echo; usage; exit 1
fi

# Translate the switches into train.py flags, and tag the exp name so ablations can't collide.
RLT_FLAGS=()
VARIANT_TAG=""
if [ "$BACKBONE_GRAD" = "1" ]; then
  RLT_FLAGS+=(--model.rlt-backbone-gradient); VARIANT_TAG="${VARIANT_TAG}_bbgrad"
fi
if [ "$NO_TARGET_SG" = "1" ]; then
  RLT_FLAGS+=(--model.no-rlt-target-stop-gradient); VARIANT_TAG="${VARIANT_TAG}_notgtsg"
fi
if [ "${PARALLEL_DECODER:-0}" = "1" ]; then
  RLT_FLAGS+=(--model.rlt-decoder-mode parallel); VARIANT_TAG="${VARIANT_TAG}_pardec"
fi
if [ "${NO_PROPRIO:-0}" = "1" ]; then
  RLT_FLAGS+=(--model.no-rlt-include-proprio); VARIANT_TAG="${VARIANT_TAG}_noprop"
fi
if [ -n "${RLT_OBJECTIVE:-}" ]; then
  # Explicit tags: truncating the objective string would collide "reconstruction" with
  # "reconstruction+progress" and silently point two ablations at the same checkpoint dir.
  case "$RLT_OBJECTIVE" in
    reconstruction)          OBJ_TAG="_recon" ;;
    progress)                OBJ_TAG="_prog" ;;
    reconstruction+progress) OBJ_TAG="_reconprog" ;;
    *) echo "Unknown --objective: $RLT_OBJECTIVE (reconstruction | progress | reconstruction+progress)"; exit 1 ;;
  esac
  RLT_FLAGS+=(--model.rlt-objective "$RLT_OBJECTIVE"); VARIANT_TAG="${VARIANT_TAG}${OBJ_TAG}"
fi
if [ "${SCALAR_HEAD:-0}" = "1" ]; then
  RLT_FLAGS+=(--model.rlt-progress-head regression); VARIANT_TAG="${VARIANT_TAG}_scalar"
fi
if [ -n "${PROGRESS_BINS:-}" ]; then
  RLT_FLAGS+=(--model.rlt-progress-bins "$PROGRESS_BINS")
fi
if [ -n "${TOKEN_DIM:-}" ]; then
  RLT_FLAGS+=(--model.rlt-token-dim "$TOKEN_DIM"); VARIANT_TAG="${VARIANT_TAG}_d${TOKEN_DIM}"
fi
if [ -n "${RLT_LOSS_WEIGHT:-}" ]; then
  RLT_FLAGS+=(--model.rlt-loss-weight "$RLT_LOSS_WEIGHT"); VARIANT_TAG="${VARIANT_TAG}_w${RLT_LOSS_WEIGHT}"
fi
RLT_FLAGS+=(--project-name "$PROJECT")
RLT_FLAGS+=(--wandb-entity "$ENTITY")
[ -n "$GROUP" ] && RLT_FLAGS+=(--wandb-group "$GROUP")
[ -n "${MONITOR_INTERVAL:-}" ] && RLT_FLAGS+=(--rlt-monitor-interval "$MONITOR_INTERVAL")
[ -n "${VIS_INTERVAL:-}" ] && RLT_FLAGS+=(--rlt-vis-interval "$VIS_INTERVAL")
[ -n "$VARIANT_TAG" ] && echo "RLT variant: ${RLT_FLAGS[*]}   (exp-name tag: $VARIANT_TAG)"

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
for TASK in "${TASKS[@]}"; do
  CONFIG="pi05_robocasa_${TASK}_rlt"
  EXP="${TASK}_${EXP_SUFFIX}${VARIANT_TAG}"
  echo "================================================================"
  echo " Task: $TASK   Config: $CONFIG   Exp: $EXP"
  echo "================================================================"

  link_local "$TASK"

  # Norm stats: the RLT configs read the SHARED stats checked into the repo, so this normally just
  # confirms the file is present and skips. If it is ever missing, we fall back to the same
  # computation run_train.sh uses.
  NORM_STATS_FILE="$REPO_ROOT/examples/robocasa/norm_stats/robocasa365_shared/norm_stats.json"
  if [ "$SKIP_NORM_STATS" = "1" ]; then
    echo "[1/2] skipping norm stats (--skip-norm-stats)"
  elif [ -f "$NORM_STATS_FILE" ] && [ "$FORCE_NORM_STATS" != "1" ]; then
    echo "[1/2] norm stats already exist, skipping ($NORM_STATS_FILE)"
    echo "      pass --force-norm-stats to recompute."
  else
    # Shared normalization across ALL tasks (writes every task's norm_stats.json in one pass).
    # Per-task stats are ill-conditioned for near-stationary tasks (base/control ~constant -> a
    # ~0 range that blows up the loss); a shared range is well-conditioned.
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
  if [ "$OVERWRITE" = "1" ]; then
    TRAIN_FLAGS+=(--overwrite)
  elif [ "$RESUME" = "1" ]; then
    TRAIN_FLAGS+=(--resume)
  elif [ -d "$CKPT_DIR" ]; then
    if has_checkpoint "$CKPT_DIR"; then
      echo "  !! $CKPT_DIR has saved checkpoints; pass --resume to continue or --overwrite to restart."
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
