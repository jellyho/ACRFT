#!/usr/bin/env bash
# ============================================================================================
#  Stage 3 — train ONE critic variant on an annotated dataset, then run the offline diagnostics.
#
#  Normally submitted by slurm/sweep.sh as an array job, where each array index picks one line
#  of a manifest (RUN <tab> DATA <tab> FLAGS). It also runs standalone for a single variant:
#
#      RUN=base DATA=$ANNOT_ROOT/PrepareCoffee sbatch slurm/train_critic.sbatch
#      RUN=soft FLAGS="--v-agg soft --soft-tau 0.1" sbatch slurm/train_critic.sbatch
#
#  Partition/QOS below are the 48 GB tier; sweep.sh overrides them per tier on the command line
#  (sbatch CLI wins over #SBATCH). Override the log dir the same way with -o.
#
#  Env (all have defaults from slurm/env.sh):
#    RUN            variant name -> $CRITIC_RUNS/$SWEEP/$RUN                (required)
#    DATA           annotation dir (annotate_rlt.py output)                 (required)
#    FLAGS          extra train_rlt_critic.py flags for this variant
#    SWEEP          sweep name; groups the runs on disk and in wandb        (default manual)
#    STEPS          training steps                                          (default 200000)
#    TASK           RoboCasa task, used by the rollout eval                 (default PrepareCoffee)
#    ROLLOUT        1 = in-process critic-vs-VLA sim rollouts               (default 1)
#    ROLLOUT_EVERY  steps between rollout evals                             (default 50000)
#    ROLLOUT_TRIALS trials per rollout eval, per mode                       (default 20)
#    EVAL_EVERY     steps between the cheap within-state diagnostics        (default 5000)
#    SEED           seed                                                    (default 0)
#    WANDB          1 = log to $WANDB_PROJECT                               (default 1)
#
#  NOTE: train_rlt_critic.py has no resume path, so the job is NOT submitted with --requeue —
#  a preempted job would silently restart from step 0. Resubmit by hand if a node dies.
# ============================================================================================
#SBATCH --job-name=acrft_critic
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jellyho/acrft/logs/%x_%A_%a.out

set -euo pipefail

cd "${ACRFT_REPO:-/home/jellyho/ACRFT/ACRFT}"
source slurm/env.sh

# An array member takes its variant from the manifest; a bare submission takes it from the env.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ -f "${MANIFEST:-}" ]] || { echo "ERROR: array job needs MANIFEST=<file>"; exit 1; }
    line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
    [[ -n "$line" ]] || { echo "ERROR: manifest line ${SLURM_ARRAY_TASK_ID} is empty"; exit 1; }
    IFS=$'\t' read -r RUN DATA FLAGS <<< "$line"
fi

RUN="${RUN:?set RUN=<variant name>}"
DATA="${DATA:?set DATA=<annotation dir>}"
FLAGS="${FLAGS:-}"
SWEEP="${SWEEP:-manual}"
STEPS="${STEPS:-200000}"
TASK="${TASK:-PrepareCoffee}"
SEED="${SEED:-0}"
ROLLOUT="${ROLLOUT:-1}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50000}"
ROLLOUT_TRIALS="${ROLLOUT_TRIALS:-20}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
WANDB="${WANDB:-1}"
BATCH="${BATCH:-0}"                 # 0 = train_rlt_critic.py's own default (1024)

OUT="$CRITIC_RUNS/$SWEEP/$RUN"

echo "=== ACRFT stage-3 critic ==="
echo "sweep/run : $SWEEP/$RUN"
echo "node      : $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-?}   job: ${SLURM_JOB_ID:-?}"
echo "data      : $DATA"
echo "out       : $OUT"
echo "flags     : ${FLAGS:-<baseline>}"
echo "steps     : $STEPS"

# sweep.sh writes the seed into the manifest flags (that is how a multi-seed sweep varies it), so the
# env default only applies to a standalone submission. Adding both would work — argparse takes the
# last — but relying on that is a trap the next reader should not have to spot.
SEED_ARGS=()
[[ "$FLAGS" == *"--seed"* ]] || SEED_ARGS=(--seed "$SEED")

# --- preflight -------------------------------------------------------------------------------
# Every failure below is silent-but-fatal otherwise: a missing annotation column surfaces 20 minutes
# in as a shape error, and a CPU fallback runs ~100x slower while looking perfectly healthy.
#
# The venv check comes first because `uv run --no-sync` against a missing .venv CREATES an empty one
# and then fails with a bare ModuleNotFoundError — and with several array members starting at once,
# each would race to create it in the shared checkout.
[[ -x .venv/bin/python ]] || { echo "ERROR: no .venv — run 'ROLLOUT=1 slurm/setup.sh' on the login node first"; exit 1; }
[[ -f "$DATA/meta.json" ]] || { echo "ERROR: no meta.json in DATA=$DATA"; exit 1; }
grep -q '"stride": 1' "$DATA/meta.json" \
    || { echo "ERROR: $DATA was annotated with stride != 1; the per-prefix reward sums need every frame"; exit 1; }
for f in rl_token action_chunk base_action reward mc_return done episode_index; do
    [[ -f "$DATA/$f.dat" ]] || { echo "ERROR: $DATA/$f.dat missing (incomplete transfer?)"; exit 1; }
done
if ! uv run --no-sync python - <<'PYCHK'
import jax
devs = jax.devices()
print("jax devices:", devs)
assert any(d.platform == "gpu" for d in devs), "NO GPU visible to JAX — aborting (would run on CPU)"
PYCHK
then
    # Blame the NODE only when the hardware really is absent, never when we are at fault. Slurm
    # allocating a GPU that nvidia-smi cannot enumerate is a node fault. If nvidia-smi sees the card
    # and only jax does not, the problem is on our side (venv, CUDA plugin, LD_LIBRARY_PATH) and
    # recording the node would quietly retire a healthy machine for every future submission.
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "ERROR: no GPU was allocated to this job — a submission bug, not a node fault"
    elif ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        acrft_mark_bad_node "$(hostname -s)" "slurm allocated a GPU but nvidia-smi enumerates none"
        echo "ERROR: $(hostname -s) has no usable GPU — recorded in $ACRFT_BAD_NODES"
    else
        echo "ERROR: nvidia-smi sees a GPU but jax does not — OUR environment, node NOT recorded"
    fi
    exit 1
fi

# --- optional in-process sim rollout ----------------------------------------------------------
# This is the actual objective (does critic-guided selection beat the plain VLA?); the offline
# diagnostics are only its proxy. It keeps the annotated VLA resident, so it needs the checkpoint
# the tokens came from plus the sim deps (`uv sync --group eval`, see slurm/setup.sh).
ROLLOUT_ARGS=()
if [[ "$ROLLOUT" == "1" ]]; then
    [[ -d "$VLA_CKPT" ]] || {
        echo "ERROR: ROLLOUT=1 but VLA_CKPT=$VLA_CKPT is not a directory."
        echo "       Fetch the pi05 checkpoint the tokens were annotated from, or run the sweep"
        echo "       offline-only with ROLLOUT=0 and add the rollout arm once it is available."
        exit 1
    }
    uv run --no-sync python -c "import robosuite, mujoco" 2>/dev/null \
        || { echo "ERROR: sim deps missing — run 'ROLLOUT=1 slurm/setup.sh' on the login node"; exit 1; }
    ROLLOUT_ARGS=(
        --rollout-every "$ROLLOUT_EVERY"
        --rollout-trials "$ROLLOUT_TRIALS"
        --vla-checkpoint "$VLA_CKPT"
        --task "$TASK"
    )
    echo "rollout   : every $ROLLOUT_EVERY steps, $ROLLOUT_TRIALS trials, vla=$VLA_CKPT"
fi

# Only now, past every check — an empty run dir left by a job that died in preflight would show up
# in collect.py as a variant that ran and produced nothing.
mkdir -p "$OUT"

WANDB_ARGS=()
[[ "$WANDB" == "1" ]] && WANDB_ARGS=(
    --wandb-project "$WANDB_PROJECT"
    --wandb-entity "$WANDB_ENTITY"
    --wandb-group "$SWEEP"
    --wandb-name "$RUN"
)

# --- peak GPU memory sampler --------------------------------------------------------------------
# Whether a variant fits a 24 GB card is the difference between 106 eligible GPUs and 379, and it
# depends on flags (batch size, bootstrap-candidates, rollout) in ways that are easier to measure
# than to derive. One nvidia-smi sample every 10 s costs nothing and turns every run into evidence.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)"
echo "gpu       : ${GPU_NAME:-?} (${GPU_TOTAL:-?} MiB)"
# --- batch size, fixed at submit time -------------------------------------------------------------
# The peak allocation is the ensemble's target forward over B*P*N chunks, so batch size is the lever
# that fits a run to the card. Measured on an RTX 3090 (24576 MiB), ARQ baseline, ROLLOUT=0:
#
#   batch 1024  OOM   (asks for a 9.00 GiB tensor on top of 5.94 GB of resident data)
#   batch  512  OOM   (asks for 10.69 GiB — see below)
#   batch  256  peak 22344 MiB, ~3 it/s, completes
#
# Note batch 512 asks for a LARGER single allocation than batch 1024: XLA's rematerialization pass
# fires at 1024 and not at 512, so peak memory is not monotonic in batch size. Do not interpolate
# between these numbers — measure the setting you actually intend to run.
#
# It is never chosen from the hardware here. What a run computes has to be fixed when it is
# submitted, not by which node it landed on, or the same manifest line means different things on
# different nodes and the ablation ends up comparing runs that were never the same experiment.
FIT_ARGS=()
[[ "$BATCH" != "0" ]] && FIT_ARGS+=(--batch-size "$BATCH")
echo "batch     : ${BATCH/#0/1024 (script default)}"

PEAK_FILE="$(mktemp)"
(
    peak=0
    while :; do
        u="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)"
        [[ "$u" =~ ^[0-9]+$ ]] && ((u > peak)) && { peak=$u; echo "$peak" > "$PEAK_FILE"; }
        sleep 10
    done
) &
SAMPLER=$!
# Kill the sampler however the job ends, including a mid-training failure or an scancel.
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# --- train -------------------------------------------------------------------------------------
# $FLAGS is deliberately unquoted: it is a pre-tokenised flag string from the manifest.
# shellcheck disable=SC2086
uv run --no-sync scripts/train_rlt_critic.py \
    --data "$DATA" \
    --out "$OUT" \
    --steps "$STEPS" \
    --eval-every "$EVAL_EVERY" \
    --save-every "${SAVE_EVERY:-50000}" \
    "${SEED_ARGS[@]}" \
    "${FIT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    $FLAGS

# --- offline diagnostics ------------------------------------------------------------------------
# eval_rlt_critic.py reads the architecture (and the wandb run id) from the config.json the run just
# wrote, so the flags never have to be repeated here — and it appends diag/* onto the same wandb run.
uv run --no-sync scripts/eval_rlt_critic.py \
    --data "$DATA" \
    --params "$OUT/params.msgpack" \
    --out "$OUT/diag.json"

# Record what the run actually cost, so tier feasibility is read off measurements rather than
# re-derived. collect.py surfaces this column.
PEAK="$(cat "$PEAK_FILE" 2>/dev/null || echo 0)"
python3 - "$OUT/gpu.json" "$PEAK" "${GPU_NAME:-unknown}" "${GPU_TOTAL:-0}" <<'PY'
import json, sys
out, peak, name, total = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
json.dump({"gpu": name, "total_mib": total, "peak_mib": peak,
           "headroom_mib": total - peak, "fits_24gb": peak < 23000}, open(out, "w"), indent=2)
print(f"peak GPU memory {peak} MiB of {total} MiB on {name}"
      f"  -> {'fits' if peak < 23000 else 'does NOT fit'} a 24 GB card")
PY

echo "done: $OUT"
