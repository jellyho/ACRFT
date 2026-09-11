#!/usr/bin/env bash
# Start a training run on the machine you are sitting at -- no slurm, no sbatch.
#
# The launchers in slurm/ hard-code one cluster's paths (/data5/jellyho/ACRFT/openpi, a B200
# partition, an #SBATCH log path). This one derives everything from where the repo actually is, so
# it runs on a workstation, inside an interactive salloc, in a container, or over ssh. It sets the
# same environment those jobs set, because two of those variables are correctness, not taste --
# see LEROBOT_VIDEO_BACKEND and HF_DATASETS_CACHE below.
#
#   scripts/train_local.sh pi05_yam_lego_taxi_success
#   scripts/train_local.sh --list                                  # what config names exist
#   NAME=cable_h30 STEPS=100000 scripts/train_local.sh pi05_yam_cable_tie
#   BG=1 scripts/train_local.sh pi05_yam_lego_taxi                 # detach, keep running after logout
#   scripts/train_local.sh pi05_yam_lego_taxi --batch-size 16      # unknown flags pass through to train.py
#
# Environment (every one optional; flags on the command line beat all of them):
#
#   NAME / EXP     run name -> checkpoints/<config>/<NAME>   (default: the config name minus "pi05_")
#   STEPS          --num-train-steps          (default: the config's own)
#   SAVE           --save-interval            (default: the config's own)
#   BATCH          --batch-size               (default: the config's own)
#   WORKERS        --num-workers              (default: half the cores, capped at 12; see below)
#   CKPT_DIR       --checkpoint-base-dir      (default: ./checkpoints)
#   PROJECT        --project-name             (default: the config's own)
#   ENTITY         --wandb-entity             (default: the config's own)
#   ASSET_ID       --data.assets.asset-id     pins the norm-stats file (see the note below)
#   RESUME=1       continue from the last checkpoint in the run dir
#   OVERWRITE=1    wipe the run dir and start over
#   GPUS           CUDA_VISIBLE_DEVICES for this run, e.g. GPUS=0 or GPUS=0,1
#   MEM_FRACTION   XLA_PYTHON_CLIENT_MEM_FRACTION (default 1.0, as in the slurm jobs; lower it if
#                  you are sharing the GPU with someone)
#   LOG=1|<path>   tee to logs/<config>_<name>_<timestamp>.log (implied by BG=1)
#   BG=1           run detached under nohup and print the PID
#   DRY=1          print the command that would run and exit (checks the guards, starts nothing)
#
# NORM STATS are no longer a separate step: training resolves them itself (the content cache in
# training/norm_stats_cache.py finds an asset computed on this exact episode set wherever it was
# filed, and computes one on a genuine miss). The exception is ASSET_ID -- naming an asset
# explicitly is an instruction, so the cache steps aside and a missing file fails loudly. Compute it
# first in that case:  uv run python scripts/compute_norm_stats.py --config-name <config> --asset-id <id>
#
# EPISODE CONDITIONS are config names, not flags. Since 2026-09-11 the plain name already drops the
# failure give-up tails; `pi05_yam_lego_taxi_success` (successes only) and `..._withhoming` (the old
# default, give-up supervision kept) name the other two. Each carries its own norm stats and
# checkpoint dir. Do not reach for --data.success-only: a flag can be forgotten between two runs and
# a config name cannot, which is exactly how the deployed baseline ended up mislabelled once.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() { sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# Pick the interpreter: an already-built venv if there is one (fastest, and what the slurm jobs
# exec), otherwise let uv resolve it. --no-sync so a training launch never silently re-resolves deps.
pick_python() {
  local venv="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
  if [ -x "$venv/bin/python" ]; then
    PY=("$venv/bin/python" -u)
  elif command -v uv >/dev/null 2>&1; then
    PY=(uv run --no-sync python -u)
  else
    echo "error: no $venv/bin/python and no uv on PATH. Create the venv first (uv sync)." >&2
    exit 1
  fi
}

CONFIG=""
PASSTHROUGH=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list)
      pick_python
      PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
        "${PY[@]}" -c 'import openpi.training.config as c; print("\n".join(sorted(c._configs_dict())))'
      exit 0 ;;
    -*) PASSTHROUGH+=("$1") ;;
    *)  if [ -z "$CONFIG" ]; then CONFIG="$1"; else PASSTHROUGH+=("$1"); fi ;;
  esac
  shift
done

if [ -z "$CONFIG" ]; then
  echo "error: no config name given." >&2; echo >&2
  usage >&2; exit 1
fi

pick_python
EXP="${NAME:-${EXP:-${CONFIG#pi05_}}}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
RUN_DIR="$CKPT_DIR/$CONFIG/$EXP"

# A run dir with a saved step in it is somebody's training run. Make the caller say which they mean
# rather than guessing -- resuming when they wanted a fresh run wastes a day, and the reverse
# destroys one. A dir holding only metadata (a run that died before its first save) is not a run.
has_checkpoint() { [ -d "$1" ] && find "$1" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | grep -q .; }
FLAGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  FLAGS+=(--overwrite)
elif [ "${RESUME:-0}" = "1" ]; then
  FLAGS+=(--resume)
elif has_checkpoint "$RUN_DIR"; then
  echo "error: $RUN_DIR already holds saved checkpoints." >&2
  echo "       RESUME=1 to continue it, OVERWRITE=1 to discard it, or pick another NAME." >&2
  exit 1
elif [ -d "$RUN_DIR" ]; then
  echo "note: $RUN_DIR exists but holds no saved step (a run that died early) -> --overwrite"
  FLAGS+=(--overwrite)
fi

[ -n "${STEPS:-}" ]   && FLAGS+=(--num-train-steps "$STEPS")
[ -n "${SAVE:-}" ]    && FLAGS+=(--save-interval "$SAVE")
[ -n "${BATCH:-}" ]   && FLAGS+=(--batch-size "$BATCH")
[ -n "${PROJECT:-}" ] && FLAGS+=(--project-name "$PROJECT")
[ -n "${ENTITY:-}" ]  && FLAGS+=(--wandb-entity "$ENTITY")
[ -n "${ASSET_ID:-}" ] && FLAGS+=(--data.assets.asset-id "$ASSET_ID")

# The YAM/RoboCasa datasets decode three camera streams per sample, so the loader -- not the GPU --
# sets the step time, and the config default of 2 workers leaves the GPU idle. Half the cores is a
# reasonable share on a machine you are also using; the cap keeps memory sane, since each worker
# holds decoded frames.
if [ -n "${WORKERS:-}" ]; then
  FLAGS+=(--num-workers "$WORKERS")
else
  CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  AUTO=$(( CORES / 2 )); [ "$AUTO" -lt 1 ] && AUTO=1; [ "$AUTO" -gt 12 ] && AUTO=12
  FLAGS+=(--num-workers "$AUTO")
fi

export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-${XLA_PYTHON_CLIENT_MEM_FRACTION:-1.0}}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="$GPUS"

# HuggingFace datasets takes an flock per cache entry, and flock does not return on this cluster's
# NFS mounts -- a training start just hangs there with no error. Keep the arrow cache node-local and
# sweep yesterday's, since an exec'd run cannot clean up after itself.
if [ -z "${HF_DATASETS_CACHE:-}" ]; then
  export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_cache_$$"
  find "${TMPDIR:-/tmp}" -maxdepth 1 -name 'hf_datasets_cache_*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true
fi

echo "config     : $CONFIG"
echo "exp        : $EXP"
echo "checkpoints: $RUN_DIR"
echo "python     : ${PY[*]}"
echo "gpu        : ${CUDA_VISIBLE_DEVICES:-<all visible>}  (mem fraction $XLA_PYTHON_CLIENT_MEM_FRACTION)"
echo "flags      : ${FLAGS[*]} ${PASSTHROUGH[*]:-}"
echo

CMD=("${PY[@]}" scripts/train.py "$CONFIG" --exp-name="$EXP" --checkpoint-base-dir "$CKPT_DIR"
     "${FLAGS[@]}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"})

# LOG=1 picks a path; LOG=<path> uses it; BG implies logging, since a detached run with nowhere to
# write is a run you cannot debug.
if [ "${BG:-0}" = "1" ] && [ -z "${LOG:-}" ]; then LOG=1; fi
if [ -n "${LOG:-}" ]; then
  if [ "$LOG" = "1" ]; then
    mkdir -p "$REPO_ROOT/logs"
    LOG="$REPO_ROOT/logs/${CONFIG}_${EXP}_$(date +%Y%m%d_%H%M%S).log"
  else
    mkdir -p "$(dirname "$LOG")"
  fi
  echo "log        : $LOG"
fi

if [ "${DRY:-0}" = "1" ]; then
  printf 'would run:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

if [ "${BG:-0}" = "1" ]; then
  nohup "${CMD[@]}" >"$LOG" 2>&1 &
  echo "started in the background: pid $!"
  echo "  tail -f $LOG"
  exit 0
fi

if [ -n "${LOG:-}" ]; then
  exec "${CMD[@]}" 2>&1 | tee "$LOG"
fi
exec "${CMD[@]}"
