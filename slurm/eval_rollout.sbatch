#!/usr/bin/env bash
# ============================================================================================
#  Sim-rollout evaluation of critic checkpoints, as a SEPARATE job from training.
#
#  Training writes params_<step>.msgpack every --save-every steps. Rolling those out in-process
#  (--rollout-every) forces the 3 B VLA to sit resident beside the data for the whole run, which
#  pins every training job to the 48 GB tier and cuts the eligible GPU pool from 379 to 106. Doing
#  it here instead lets training run anywhere and lets evaluation start the moment a checkpoint
#  lands, on its own GPU, for as many checkpoints as there are slots.
#
#  Evaluates BOTH modes on the identical scenes (same seed), because the number that matters is not
#  the critic's absolute success rate but whether it beats the VLA it is selecting for.
#
#      RUN_DIR=$CRITIC_RUNS/pro_main/base sbatch slurm/eval_rollout.sbatch
#      RUN_DIR=... STEPS_LIST="10000 20000 30000" TRIALS=20 sbatch slurm/eval_rollout.sbatch
#
#  Env: RUN_DIR (required) · STEPS_LIST (default: every params_*.msgpack + final) · TRIALS (20)
#       TASK (PrepareCoffee) · MODE (critic) · SEED (0)
# ============================================================================================
#SBATCH --job-name=acrft_roll
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/jellyho/acrft/logs/roll_%j.out

set -euo pipefail
cd "${ACRFT_REPO:-/home/jellyho/ACRFT/ACRFT}"
source slurm/env.sh

RUN_DIR="${RUN_DIR:?set RUN_DIR=<critic run dir holding params_*.msgpack>}"
TRIALS="${TRIALS:-20}"
TASK="${TASK:-PrepareCoffee}"
SEED="${SEED:-0}"
# Videos carry the HUD, which is the only place the critic's reasoning is visible: the spread of the
# N candidate values at each replan (flat spread = best-of-N is a coin flip), the value trace on a log
# axis (V = gamma^steps-to-goal, so a consistent critic traces a straight line), and the committed
# span per replan — the short-commitment behaviour, shown rather than inferred. NUM_VIDEOS is per mode.
NUM_VIDEOS="${NUM_VIDEOS:-0}"
VIDEO_ARGS=()
[[ "$NUM_VIDEOS" != "0" ]] && VIDEO_ARGS=(--video-dir "$RUN_DIR/videos" --num-videos "$NUM_VIDEOS")
# The _pardec_noprop checkpoint was trained with model flags the registered RoboCasa config does not
# carry (run_train_rlt.sh passes them on the command line and records them only in the exp dir name),
# so restoring it against the bare config fails on missing params. Replay them here.
VLA_OVERRIDE="${VLA_OVERRIDE:---vla-override rlt_decoder_mode=parallel --vla-override rlt_include_proprio=false}"

[[ -f "$RUN_DIR/config.json" ]] || { echo "ERROR: no config.json in $RUN_DIR"; exit 1; }
[[ -d "$VLA_CKPT" ]] || { echo "ERROR: VLA_CKPT=$VLA_CKPT missing"; exit 1; }
CONFIG="$(python3 -c "import json;print(json.load(open('$RUN_DIR/config.json')).get('vla_config') or '')")"
[[ -n "$CONFIG" ]] || CONFIG="$(python3 -c "import json;print(json.load(open('$RUN_DIR/config.json'))['data'])" | xargs -I{} python3 -c "import json;print(json.load(open('{}/meta.json'))['config'])")"

# Default to every checkpoint on disk, oldest first, so the success-vs-training-step curve is filled
# in rather than only its endpoint.
if [[ -z "${STEPS_LIST:-}" ]]; then
    mapfile -t CKPTS < <(ls "$RUN_DIR"/params_*.msgpack 2>/dev/null | sort -t_ -k2 -n)
    [[ -f "$RUN_DIR/params.msgpack" ]] && CKPTS+=("$RUN_DIR/params.msgpack")
else
    # 'final' names the end-of-training checkpoint, which is written as params.msgpack rather than
    # params_<steps>.msgpack (the periodic save is skipped on the last step).
    CKPTS=(); for s in $STEPS_LIST; do
        if [[ "$s" == "final" ]]; then CKPTS+=("$RUN_DIR/params.msgpack"); else CKPTS+=("$RUN_DIR/params_${s}.msgpack"); fi
    done
fi
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "ERROR: no params_*.msgpack in $RUN_DIR yet"; exit 1; }

_EXC="$(acrft_exclude_list)"; [[ -n "$_EXC" ]] && echo "excluded nodes: $_EXC"
echo "=== rollout eval ==="
echo "run    : $RUN_DIR"
echo "vla    : $CONFIG @ $VLA_CKPT"
echo "ckpts  : ${#CKPTS[@]}   trials: $TRIALS   task: $TASK   seed: $SEED"
uv run --no-sync python -c "import robosuite, mujoco" 2>/dev/null \
    || { echo "ERROR: sim deps missing — run 'ROLLOUT=1 slurm/setup.sh'"; exit 1; }
# Same rule as train_critic.sbatch: only a node that cannot enumerate its allocated GPU is recorded.
if ! uv run --no-sync python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" 2>/dev/null; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        acrft_mark_bad_node "$(hostname -s)" "slurm allocated a GPU but nvidia-smi enumerates none"
        echo "ERROR: $(hostname -s) has no usable GPU — recorded"
    else
        echo "ERROR: GPU unusable but not a node fault — node NOT recorded"
    fi
    exit 1
fi

OUT="$RUN_DIR/rollout"
mkdir -p "$OUT"
for ck in "${CKPTS[@]}"; do
    [[ -f "$ck" ]] || { echo "skip (absent): $ck"; continue; }
    tag="$(basename "$ck" .msgpack)"
    # Seed goes in the filename: repeating a checkpoint under different seeds is how the trial count
    # is raised without a longer job, and they must not overwrite each other.
    f="$OUT/${tag}_seed${SEED}.json"
    [[ -f "$f" ]] && { echo "have $tag"; continue; }
    echo "--- $tag"
    # All four modes in ONE invocation: the VLA is loaded once, and every mode sees the identical
    # scenes (--seed is fixed), which is what makes the comparison mean anything. The modes also
    # split the question the sweep cannot answer offline:
    #   vla     no critic at all — the baseline being selected for
    #   bon     best-of-N on the full chunk only — pure candidate selection, fixed commit length
    #   prefix  first candidate, critic picks only the commit length — pure adaptive chunking
    #   rand    a uniformly random candidate — the control that says whether the CRITIC picked well
    #           or merely that picking differently helped
    #   critic  joint arg-max over (candidate, prefix) — what deployment actually does
    # If `critic` beats `vla` but `bon` does not, the gain is commit length, not action choice.
    uv run --no-sync examples/robocasa/eval_critic.py \
        --config "$CONFIG" --checkpoint "$VLA_CKPT" --critic "$ck" \
        --task "$TASK" --num-trials "$TRIALS" --seed "$SEED" \
        --modes critic bon prefix rand vla --out "$f" $VLA_OVERRIDE "${VIDEO_ARGS[@]}" \
        || echo "  FAILED $tag"
done
echo "done: $OUT"
