#!/usr/bin/env bash
# ============================================================================================
#  Submit the critic ablation as one SLURM array job.
#
#  The sweep is one-factor-at-a-time from a single baseline, not a cross product: with the
#  in-process rollout eval on, each run is expensive, and the question each flag answers is
#  "does this knob move the ranking signal", which a full grid does not answer any better.
#  Axes (pick with AXES=, default all):
#
#    agg        the bootstrap's arg-max over N candidates x P prefixes — the reduction that the
#               recent work kept coming back to. v-agg / ens-agg / bootstrap-candidates /
#               target-noise all narrow or smooth that max-over-noisy-estimates.
#    discount   gamma. At 0.99 the effective horizon (~100 steps) is shorter than an episode, so
#               the goal is nearly invisible from a state's start. Needs the re-labelled dataset
#               copies — build them with slurm/make_discount_variant.py first.
#    structure  what the critic IS: arq vs qc, prefix granularity, scalar vs HL-Gauss.
#
#  Usage:
#    slurm/sweep.sh                                  # all axes, PrepareCoffee, 48 GB tier
#    SWEEP=abl2 AXES=agg slurm/sweep.sh              # aggregation axis only
#    DRYRUN=1 slurm/sweep.sh                         # print the manifest, submit nothing
#    SEEDS="0 1 2" AXES=agg slurm/sweep.sh           # repeat every variant over three seeds
#    TIER=pro6000 slurm/sweep.sh                     # 96 GB nodes
#
#  Env:
#    SWEEP      sweep name (run dirs + wandb group)     default critic_abl
#    DATA_NAME  dataset dir under $ANNOT_ROOT           default noprop
#    DATA       baseline annotation dir                 default $ANNOT_ROOT/$DATA_NAME
#    TASK       RoboCasa env for the rollout eval       default PrepareCoffee
#    DATA_G999 / DATA_G9995   discount-axis datasets    default ${DATA}_g999 / ${DATA}_g9995
#    AXES       agg,discount,structure                  default all
#    SEEDS      space-separated seed list               default 0
#    TIER       a6000|ampere|wide|pro6000|a100          default: a6000 if ROLLOUT=1 else wide
#    MAXPAR     max array members running at once       default 8
#    STEPS      training steps per variant              default 200000
#    ROLLOUT    1 = sim rollout eval inside the job     default 1
#    DRYRUN     1 = build the manifest and stop
# ============================================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source slurm/env.sh
acrft_mkdirs

SWEEP="${SWEEP:-critic_abl}"
# Two different things that must not share a variable: DATA_NAME is the directory under
# $ANNOT_ROOT (fetch_data.sh writes "noprop"), TASK is the RoboCasa env the rollout eval builds
# with make_env() and has to be a real task name.
DATA_NAME="${DATA_NAME:-noprop}"
TASK="${TASK:-PrepareCoffee}"
DATA="${DATA:-$ANNOT_ROOT/$DATA_NAME}"
DATA_G999="${DATA_G999:-${DATA}_g999}"
DATA_G9995="${DATA_G9995:-${DATA}_g9995}"
AXES="${AXES:-agg,discount,structure,target,stability,iql}"
SEEDS="${SEEDS:-0}"
MAXPAR="${MAXPAR:-8}"
STEPS="${STEPS:-200000}"
ROLLOUT="${ROLLOUT:-1}"
# ROLLOUT=1 keeps a 3 B VLA resident alongside the data, so it stays on the 48 GB tier. With the
# rollout off, `wide` opens 379 GPUs instead of 106 — on a cluster this busy that is the whole game.
TIER="${TIER:-$([[ "$ROLLOUT" == "1" ]] && echo a6000 || echo wide)}"
# Batch size is fixed here, once, for the whole sweep — never per node. A tier that can place a job
# on a 24 GB card has to run the batch that fits there (measured: 1024 and 512 both OOM on a 3090,
# 256 peaks at 22344 of 24576 MiB), and applying that to every variant is what keeps the comparison
# honest: the ablation only means something if the variants differ in one knob and nothing else.
case "$TIER" in
    ampere | wide) BATCH="${BATCH:-256}" ;;
    *) BATCH="${BATCH:-0}" ;;  # 0 = the script's own 1024
esac

IFS='|' read -r PART QOS <<< "$(acrft_tier "$TIER")"

[[ -f "$DATA/meta.json" ]] || { echo "ERROR: baseline annotation not found at $DATA (slurm/fetch_data.sh)"; exit 1; }
# The candidate count sizes the bootstrap-subset arm; grab it from the annotation rather than
# assuming it. `python3` is enough — this runs on the login node, before any GPU is involved.
N_CAND="$(python3 -c "import json,sys; print(json.load(open('$DATA/meta.json'))['num_samples'])")"

SWEEP_DIR="$CRITIC_RUNS/$SWEEP"
MANIFEST="$SWEEP_DIR/manifest.tsv"
mkdir -p "$SWEEP_DIR"
: > "$MANIFEST"

has_axis() { [[ ",$AXES," == *",$1,"* ]]; }

# Proprioception is not a flag any more: train_rlt_critic.py always loads it, because the `noprop`
# RLT token excludes proprio by design and a critic without it cannot know where the arm is.
BASE_FLAGS=""

# HL-Gauss and the mc_return floor were arms in the v3 sweep and are the baseline from v4 on. Neither
# was promoted because it beat scalar regression on the metric that matters — nothing did, since every
# v3 run sat at action_sensitivity ~0.0003 and demo-vs-candidate ranking at chance, which is an axis
# with no variance to rank methods along. They are promoted on the grounds that they are the
# better-founded defaults (a distributional head is the standard choice for value learning; a floor
# the behaviour policy demonstrably achieved cannot be wrong), so the ablations that follow are read
# against a defensible baseline rather than a stripped one. Set BASELINE_EXTRA="" to reproduce v3.
BASELINE_EXTRA="${BASELINE_EXTRA:---num-atoms 51 --mc-lower-bound}"
BASE_FLAGS="${BASE_FLAGS} ${BASELINE_EXTRA}"


# add <name> <flags> [data] [base_flags_override]
# The 4th argument replaces BASE_FLAGS entirely, for a variant that must DROP something every other
# run carries; prepending unconditionally would silently re-add it.
add() {
    local name="$1" flags="${2:-}" data="${3:-$DATA}" base="${4-$BASE_FLAGS}"
    if [[ ! -f "$data/meta.json" ]]; then
        echo "  skip $name — no annotation at $data"
        return
    fi
    if [[ ! -f "$data/proprio.dat" ]]; then
        echo "  skip $name — no proprio.dat at $data (run slurm/extract_proprio.py --data $data)"
        return
    fi
    flags="$base $flags"
    for s in $SEEDS; do
        local n="$name"
        [[ "$(echo "$SEEDS" | wc -w)" -gt 1 ]] && n="${name}_s${s}"
        printf '%s\t%s\t%s\n' "$n" "$data" "${flags} --seed ${s}" >> "$MANIFEST"
    done
}

# The baseline every other row is read against: ARQ, hard max over all N candidates, min over the
# ensemble, no target smoothing, scalar Q, macro group 2, and whatever gamma/value support the
# annotation's own meta.json records.
add base ""

if has_axis agg; then
    # --- how the bootstrap reduces over candidates -------------------------------------------
    add topm "--v-agg topm --top-m 3"          # average the 3 best instead of taking the max
    add soft "--v-agg soft --soft-tau 0.1"     # softmax-weighted, so a lone peak cannot own the target
    # --- how it reduces over the ensemble -----------------------------------------------------
    add lcb  "--ens-agg lcb --lcb-beta 1.0"    # mean - std, a graded pessimism instead of hard min
    # --- how many candidates the max ranges over ----------------------------------------------
    # Fewer items is both cheaper (the target forward dominates a step) and less biased upward;
    # resampling each step still visits all N over training. Taken as fractions of the annotation's
    # own N, because train_rlt_critic.py only subsamples when 0 < bootstrap_candidates < N — a
    # hardcoded 16 against an N=16 dataset is a silent no-op that duplicates the baseline.
    for n in $((N_CAND / 2)) $((N_CAND / 4)); do
        # 0 means "use all N", and n >= N does not subsample either — both would duplicate the
        # baseline under a different name. Only emit the arm when it actually narrows the max.
        if [[ "$n" -ge 2 && "$n" -lt "$N_CAND" ]]; then
            add "bc$n" "--bootstrap-candidates $n"
        else
            echo "  skip bc$n — no-op against N=$N_CAND"
        fi
    done
    # --- local smoothing of Q around each candidate --------------------------------------------
    add tn03 "--target-noise 0.3"
    add tn10 "--target-noise 1.0"
fi

if has_axis discount; then
    # Separate datasets, not flags: the discount also sets mc_return and the value support, and
    # train_rlt_critic.py deliberately reads all three from the annotation's meta.json so a
    # re-labelled dataset cannot be trained against the previous scheme's constants.
    add g999  "" "$DATA_G999"
    add g9995 "" "$DATA_G9995"
fi

if has_axis structure; then
    add qc   "--kind qc"                 # flat Q per chunk — no per-prefix head, no adaptive commit
    add mg4  "--macro-group-size 4"      # coarser prefixes: 4 heads instead of 8
    add mg8  "--macro-group-size 8"      # 2 heads
    # 4th arg replaces BASE_FLAGS: appending --num-atoms 1 after the baseline's 51 works only
    # because argparse keeps the last, which is a coincidence, not an intention.
    add scalarq "" "$DATA" "$(echo "$BASE_FLAGS" | sed 's/ *--num-atoms 51//')"
fi

if has_axis target; then
    # The only remaining place mc_return can touch the target. Unlike the terminal value — which the
    # bootstrap states exactly — this is a bound the DATA proves: the behaviour policy demonstrably
    # collected mc_return[t] from this state, so the optimum cannot be below it. That makes it a real
    # modelling choice rather than a correctness fix, and it is worth measuring precisely because the
    # measured target is inflated far from the goal (ratio 3.1x at 256-512 steps out): a floor helps
    # only where the target is too LOW, so if it changes anything here it is changing the early
    # transient, not the fixed point.
    add nofloor "" "$DATA" "$(echo "$BASE_FLAGS" | sed 's/ *--mc-lower-bound//')"
fi

if has_axis iql; then
    # IQL: learn a state value V(z) by expectile regression on the DEMONSTRATED action, and
    # bootstrap that instead of maxing Q over the stored candidates.
    #
    # This is the one change that removes the arg-max rather than softening it. Measured on the td
    # runs: the target over-estimates the true value gamma^d by 1.10x at 30-60 steps from the goal,
    # 1.55x at 120-250 and 5.07x beyond 250, and within every distance band the per-prefix targets
    # decline monotonically in h - so deployment's joint arg-max is reading estimation error, not
    # commitment length. Candidate-side knobs (topm/soft/bc*) only reweight that max; IQL deletes it.
    #
    # Cheaper too: the target forward over N=16 candidates is the dominant cost of a td step, and IQL
    # does not do it (measured ~2x the throughput at equal batch).
    #
    # tau brackets what V converges to: 0.5 is least squares (V -> mean Q under the behaviour policy,
    # i.e. no improvement at all), and higher tau weights over-shoots more, approaching max_a Q. If
    # the ranking signal is genuinely absent, every tau should land in the same place - which is
    # itself the answer.
    IQL_BASE="$BASE_FLAGS --objective iql"
    add iql_e50 "--expectile 0.50" "$DATA" "$IQL_BASE"
    add iql_e70 "--expectile 0.70" "$DATA" "$IQL_BASE"
    add iql_e90 "--expectile 0.90" "$DATA" "$IQL_BASE"
    add iql_e95 "--expectile 0.95" "$DATA" "$IQL_BASE"
    # Same objective without the prefix head, to separate "IQL helps" from "the prefix axis was the
    # problem" - the two are confounded in every arq run.
    add iql_qc  "--expectile 0.70 --kind qc" "$DATA" "$IQL_BASE"
fi

if has_axis stability; then
    # The three ways the bootstrap's upward bias is held down that were never varied. Measured on the
    # v3 runs: V is over-estimated by ~+0.025 at states far from the goal and by ~0 near it, which is
    # what tilts the per-prefix targets into a monotone decline and hands deployment's joint arg-max
    # to the shortest commitment. The candidate-side knobs (topm/soft/bc*) moved nothing, so the
    # remaining suspects are the two reductions nobody touched.
    #
    # K=2 is a weak `min`: the target maxes over N*P = 128 noisy estimates and then asks two critics
    # to pull it back. More members deepen the pessimism at linear cost (the target forward already
    # dominates the step).
    add k3 "--num-critics 3"
    add k5 "--num-critics 5"
    # No target network at all - the reference implementation's default. The Polyak copy is the usual
    # stabiliser, but it also traps an over-estimate for ~1/tau steps and adds a second timescale to
    # the fixed point, so it can preserve the bias it is credited with damping.
    add online "--bootstrap online"
    # How fast the copy tracks. 0.005 is ~200 steps of lag; these bracket it by an order of magnitude
    # each way, which separates "the target network helps" from "this particular lag helps".
    add tau001 "--target-tau 0.001"
    add tau05  "--target-tau 0.05"
fi


N="$(wc -l < "$MANIFEST")"
echo
echo "=== sweep $SWEEP: $N runs ==="
echo "tier      : $TIER -> -p $PART -q $QOS"
echo "data      : $DATA  (N=$N_CAND candidates)"
echo "steps     : $STEPS   rollout: $ROLLOUT   seeds: $SEEDS   batch: ${BATCH/#0/1024}"
echo "manifest  : $MANIFEST"
echo "out       : $SWEEP_DIR/<run>"
echo
nl -w3 -s'  ' "$MANIFEST" | sed 's/\t/  |  /g'
echo

if [[ "${DRYRUN:-0}" == "1" ]]; then
    echo "DRYRUN=1 — nothing submitted. Re-run without DRYRUN to submit."
    exit 0
fi
[[ "$N" -gt 0 ]] || { echo "ERROR: manifest is empty"; exit 1; }

# --export=ALL,... : the batch script re-sources slurm/env.sh, so only the per-sweep values that
# are not in env.sh have to be forwarded explicitly.
# DEPEND lets the sweep be queued behind a smoke test: `afterok:<jobid>` holds the whole array until
# that job succeeds, so a broken path costs one job instead of fourteen. The array still queues
# immediately, which on a saturated cluster is most of the wait won.
_EXC="$(acrft_exclude_list)"
[[ -n "$_EXC" ]] && echo "excluding  : $_EXC"

DEP_ARG=()
[[ -n "${DEPEND:-}" ]] && { DEP_ARG=(--dependency="$DEPEND"); echo "held until: $DEPEND"; }

# base_qos is non-preemptable but caps GPUs per user (8). Anything beyond that cap would simply sit
# in QOSMaxGRESPerUser until a slot frees, so the overflow goes to big_qos instead — more capacity,
# at the cost of being preemptable. That trade is real here: train_rlt_critic.py has no resume, so a
# preempted run restarts from step 0. The first MAXPAR tasks therefore take the safe queue and only
# the surplus takes the risky one. BIG_OVERFLOW=0 to keep everything on base_qos.
BIG_OVERFLOW="${BIG_OVERFLOW:-1}"
N_BASE="$N"
if [[ "$BIG_OVERFLOW" == "1" && "$QOS" == "base_qos" && "$N" -gt "$MAXPAR" ]]; then
    N_BASE="$MAXPAR"
    IFS='|' read -r BIG_PART BIG_QOS <<< "$(QOS=big_qos acrft_tier big)"
    echo "overflow  : tasks $((MAXPAR + 1))-$N -> -p $BIG_PART -q $BIG_QOS (preemptable)"
fi

# If our base_qos GPU allocation is already full (QOSMaxGRESPerUser is 8), tasks submitted there
# would sit pending behind our own jobs while big_qos has idle capacity. Spend the base quota only
# when some of it is actually free; otherwise the whole sweep goes to the preemptable tier, where
# train_critic.sbatch's checkpoint-resume makes preemption an inconvenience rather than a loss.
_BASE_BUSY="$(squeue -u "$USER" -h -o '%q' 2>/dev/null | grep -c base_qos || true)"
if [[ "$QOS" == "base_qos" && "$_BASE_BUSY" -ge 8 ]]; then
    echo "base_qos already holds $_BASE_BUSY of 8 GPUs — routing every task to $BIG_QOS instead"
    PART="$BIG_PART"; QOS="$BIG_QOS"; N_BASE="$N"
fi
jid="$(sbatch --parsable \
    --job-name="critic_${SWEEP}" \
    --partition="$PART" --qos="$QOS" \
    ${_EXC:+--exclude="$_EXC"} \
    --array="1-${N_BASE}%${MAXPAR}" \
    "${DEP_ARG[@]}" \
    --output="$SLURM_LOGS/${SWEEP}_%A_%a.out" \
    --export="ALL,ACRFT_REPO=$ACRFT_REPO,MANIFEST=$MANIFEST,SWEEP=$SWEEP,TASK=$TASK,STEPS=$STEPS,ROLLOUT=$ROLLOUT,BATCH=$BATCH" \
    slurm/train_critic.sbatch)"

echo "submitted array $jid  (tasks 1-$N_BASE on $QOS, max $MAXPAR concurrent)"
if [[ "$N_BASE" -lt "$N" ]]; then
    bjid="$(sbatch --parsable \
        --job-name="critic_${SWEEP}_big" \
        --partition="$BIG_PART" --qos="$BIG_QOS" \
        --array="$((N_BASE + 1))-${N}" \
        ${_EXC:+--exclude="$_EXC"} \
        "${DEP_ARG[@]}" \
        --output="$SLURM_LOGS/${SWEEP}_%A_%a.out" \
        --export="ALL,ACRFT_REPO=$ACRFT_REPO,MANIFEST=$MANIFEST,SWEEP=$SWEEP,TASK=$TASK,STEPS=$STEPS,ROLLOUT=$ROLLOUT,BATCH=$BATCH" \
        slurm/train_critic.sbatch)"
    echo "submitted array $bjid  (tasks $((N_BASE + 1))-$N on $BIG_QOS, preemptable)"
fi
echo "  squeue -u $USER"
echo "  tail -f $SLURM_LOGS/${SWEEP}_${jid}_1.out"
echo "  uv run slurm/collect.py $SWEEP   # summarise once the runs finish"
