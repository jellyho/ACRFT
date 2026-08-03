#!/usr/bin/env bash
# Train Pi0RLT on the YAM lego-taxi dataset. Computes norm stats first if they are missing, then
# trains. Unlike the RoboCasa launcher there are no shared stats checked in - YAM's are per-dataset
# and computed on first run.
#
#   examples/robocasa/run_train_yam.sh                # relative-joint default
#   CONFIG=pi05_yam_lego_taxi_none_rlt examples/robocasa/run_train_yam.sh   # absolute joint
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-pi05_yam_lego_taxi_rlt}"
EXP="${EXP:-${CONFIG#pi05_}}"
PROJECT="${PROJECT:-yam-rlt}"
ENTITY="${ENTITY:-RSS-PFT_RLLAB}"

# Where the config looks for norm stats: assets_dirs/<repo_id>/norm_stats.json.
STATS="$REPO_ROOT/assets/$CONFIG/jellyho/yam_lego_taxi/norm_stats.json"

if [ -f "$STATS" ] && [ "${FORCE_NORM_STATS:-0}" != "1" ]; then
  echo "[1/2] norm stats present ($STATS), skipping"
else
  echo "[1/2] computing norm stats for $CONFIG (downloads the dataset if needed)"
  # Norm stats only accumulate state/actions in numpy - the sole GPU use is the loader's
  # host-to-device copy, which JAX_PLATFORMS=cpu removes. Keeps this step off the GPU entirely.
  if ! JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name "$CONFIG"; then
    echo "  !! norm stats failed"; exit 1
  fi
fi

echo "[2/2] train $CONFIG (exp $EXP)"
TRAIN_FLAGS=()
[ "${RESUME:-0}" = "1" ] && TRAIN_FLAGS+=(--resume)
[ "${OVERWRITE:-0}" = "1" ] && TRAIN_FLAGS+=(--overwrite)
# YAM decodes three camera streams per sample, so the loader is the bottleneck at the default of 2
# workers. Scale it with the CPUs the job actually got.
[ -n "${NUM_WORKERS:-}" ] && TRAIN_FLAGS+=(--num-workers "$NUM_WORKERS")
exec env XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-1.0}" \
  uv run python scripts/train.py "$CONFIG" \
  --exp-name="$EXP" --project-name "$PROJECT" --wandb-entity "$ENTITY" "${TRAIN_FLAGS[@]}"
