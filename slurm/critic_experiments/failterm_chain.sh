#!/usr/bin/env bash
# The failure-terminal comparison, fired when the success-only BC run frees the GPU.
#
# Why: slurm/critic_ood.sbatch records that a failure's terminal makes V(s0) separate by episode
# outcome -- -1761.6 for failures against -1308.8 for successes, 44% of the target variance. A critic
# that can read "will this episode fail" off the state does not have to learn "which action is better
# here", which is the only thing it is for. --failure-terminal truncate removes that terminal: the
# operator giving up ends the RECORDING, not the MDP, so the state keeps a value and the
# success/fail label stops entering the target.
#
# The two critics trained on 2026-09-12 used the default (absorbing). This trains the same two
# configurations with truncate and scores all four, so the comparison is method-only-diff: same
# cache, same policy normalization, same hyperparameters, one flag apart.
#
#   1. score the two existing absorbing critics
#   2. train truncate FIXED  (macro-group-size 30) and score it
#   3. train truncate MACRO  (macro-group-size 5)  and score it
set -uo pipefail

REPO=/NHNHOME/jellyho/jellyho/ACRFT
BC_PID=3809133
BC_CKPT="$REPO/checkpoints/pi05_yam_cable_tie_success/yam_cable_tie_success"
BC_FINAL_STEP=200000

CACHE=/NHNHOME/jellyho/jellyho/pc_cache/cable_tie
CRITICS=/NHNHOME/jellyho/jellyho/critics
SCORES=/NHNHOME/jellyho/jellyho/critic_scores
ONSETS="$REPO/.scratch/yam_cable_tie_homing_onsets.json"
POLICY_CONFIG=pi05_yam_cable_tie_withhoming
PY="$REPO/.venv/bin/python -u"

cd "$REPO"
export LEROBOT_VIDEO_BACKEND=pyav
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export PYTHONPATH="$REPO:$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCORES"

say() { echo "[$(date +%F\ %T)] $*"; }

# Identical to the 09-12 runs except --failure-terminal, so the four critics differ by that alone.
common_args=(
  --cache "$CACHE"
  --homing-onsets "$ONSETS"
  --policy-config "$POLICY_CONFIG"
  --discount 0.999347 --preload
  --expectile 0.9
  --feat-dropout 0.1 --feat-noise 0.05
  --h-goal 30 --lr 1e-4
  --steps 200000 --save-every 50000
  --num-critics 2 --q-reduction min --seed 0
  --heldout-frac 0 --eval-every 5000
  --critic-arch independent --value-head categorical
  --reward-scheme cost_to_goal
  --wandb --wandb-group patch-critic-cable-tie
)

score() {  # score <critic dir> <label>
    say "--- scoring $2 ---"
    $PY scripts/score_critic_cached.py --critic "$1" --cache "$CACHE" --homing-onsets "$ONSETS" \
        > "$SCORES/$2.txt" 2>&1 \
        && say "    -> $SCORES/$2.txt" \
        || say "    scoring $2 FAILED (see $SCORES/$2.txt)"
}

[ -f "$ONSETS" ] || { say "ABORT: missing $ONSETS"; exit 1; }

say "waiting for BC pid $BC_PID"
while kill -0 "$BC_PID" 2>/dev/null; do sleep 60; done
say "BC pid $BC_PID exited"
if [ ! -d "$BC_CKPT/$BC_FINAL_STEP" ]; then
    say "NOTE: $BC_CKPT/$BC_FINAL_STEP missing -- the BC run did not finish."
    say "      That run is not an input here (this compares critics on an existing cache), so"
    say "      continuing; but do not read its checkpoints as a completed baseline."
fi

say "=== 1/3  score the existing absorbing critics ==="
score "$CRITICS/cable_tie_fixed_g30" absorbing_fixed_g30
score "$CRITICS/cable_tie_macro_g5"  absorbing_macro_g5

say "=== 2/3  truncate, FIXED chunk (macro-group-size 30) ==="
if $PY scripts/train_patch_critic_cached.py "${common_args[@]}" \
       --failure-terminal truncate --macro-group-size 30 \
       --wandb-name ct_trunc_fixed_g30 --out "$CRITICS/cable_tie_trunc_fixed_g30"; then
    score "$CRITICS/cable_tie_trunc_fixed_g30" truncate_fixed_g30
else
    say "truncate fixed FAILED; continuing to the macro run."
fi

say "=== 3/3  truncate, MACRO groups (macro-group-size 5) ==="
if $PY scripts/train_patch_critic_cached.py "${common_args[@]}" \
       --failure-terminal truncate --macro-group-size 5 \
       --wandb-name ct_trunc_macro_g5 --out "$CRITICS/cable_tie_trunc_macro_g5"; then
    score "$CRITICS/cable_tie_trunc_macro_g5" truncate_macro_g5
else
    say "truncate macro FAILED."
fi

say "=== finished; scores in $SCORES ==="
ls -1 "$SCORES" | sed 's/^/    /'
