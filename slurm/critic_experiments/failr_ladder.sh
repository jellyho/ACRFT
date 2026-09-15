#!/usr/bin/env bash
# The failure-anchor DEPTH ladder: between "keep the terminal at v_min" and "remove it entirely".
#
# What the first ladder established, scored on the same 4096 seeded frames over 157 episodes:
#
#                          within-ep Spearman   frac>0   AUC(success-vs-fail, early)
#   absorbing  fixed g30        +0.9100          0.955        1.0000
#   absorbing  macro g5         +0.9198          0.962        1.0000
#   truncate   fixed g30        +0.9988          1.000        0.1283
#   truncate   macro g5         +0.9994          1.000        0.1302
#
# The two axes move in opposite directions, which is the point: AUC 1.0 means the outcome label is
# readable off the state, and a critic that can read it does not have to rank actions. Removing the
# terminal costs the label and buys the ranking -- every one of 157 episodes correlates positively,
# against 4-5% negative for absorbing.
#
# But truncate's AUC is 0.13, not 0.5: failures now score HIGHER than successes. Their give-up tails
# are cut, so what remains is short, and with no terminal cost_to_goal reads "short" as "near the
# goal". Neither end is right. This ladder asks whether an intermediate anchor keeps the ranking
# while pulling the label back to chance -- the target is Spearman near truncate's AND AUC near 0.5.
#
#   v_min = -1531.4 (= -1/(1-discount), an episode-length constant, not a knob)
#   arms:  absorbing @ 0.50*v_min = -765.7
#          absorbing @ 0.25*v_min = -382.8
#
# Everything else is the 09-12 recipe verbatim, so each arm differs from the absorbing control by
# --failure-reward alone.
set -uo pipefail

REPO=/NHNHOME/jellyho/jellyho/ACRFT
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

common_args=(
  --cache "$CACHE" --homing-onsets "$ONSETS" --policy-config "$POLICY_CONFIG"
  --discount 0.999347 --preload --expectile 0.9
  --feat-dropout 0.1 --feat-noise 0.05
  --h-goal 30 --lr 1e-4 --steps 200000 --save-every 50000
  --num-critics 2 --q-reduction min --seed 0
  --heldout-frac 0 --eval-every 5000
  --critic-arch independent --value-head categorical --reward-scheme cost_to_goal
  --wandb --wandb-group patch-critic-cable-tie
)

# Both scorers, because they answer different questions and the pair is the finding: AUC says how
# much of the outcome label leaked in, ranking says whether the critic is any use at choosing.
score_both() {  # score_both <critic dir> <label>
    say "--- scoring $2 ---"
    $PY scripts/score_critic_cached.py --critic "$1" --cache "$CACHE" --homing-onsets "$ONSETS" \
        > "$SCORES/$2.txt" 2>&1 || say "    outcome-AUC scoring FAILED"
    $PY scripts/score_critic_ranking.py --critic "$1" --cache "$CACHE" --homing-onsets "$ONSETS" \
        --out "$SCORES/rank_$2.json" > "$SCORES/rank_$2.txt" 2>&1 || say "    ranking scoring FAILED"
    grep -E "spearman within|AUC\(success" "$SCORES/rank_$2.txt" 2>/dev/null | sed 's/^/    /'
}

run_arm() {  # run_arm <label> <failure-reward> <macro-group-size>
    local label="$1" failr="$2" mgs="$3"
    local out="$CRITICS/cable_tie_$label"
    if [ -f "$out/params.msgpack" ] || [ -d "$out/step_200000" ]; then
        say "=== $label already trained, scoring only ==="
    else
        say "=== training $label (failure-reward $failr, macro-group-size $mgs) ==="
        $PY scripts/train_patch_critic_cached.py "${common_args[@]}" \
            --failure-reward "$failr" --macro-group-size "$mgs" \
            --wandb-name "ct_$label" --out "$out" \
            || { say "$label FAILED; skipping its score"; return 1; }
    fi
    score_both "$out" "$label"
}

[ -f "$ONSETS" ] || { say "ABORT: missing $ONSETS"; exit 1; }

# Wait for any GPU job still running -- one card, and a second JAX process would just OOM.
while pgrep -u "$(id -u)" -f "train_patch_critic_cached.py" > /dev/null || \
      pgrep -u "$(id -u)" -f "scripts/train.py" > /dev/null; do
    sleep 120
done
say "GPU free; starting the failure-anchor ladder"

run_arm failr50_fixed_g30 -765.7 30
run_arm failr25_fixed_g30 -382.8 30

say "=== ladder finished ==="
say "compare with: absorbing (v_min) and truncate, already in $SCORES"
for f in "$SCORES"/rank_*.json; do
    $PY -c "
import json,sys,pathlib
d=json.load(open('$f'))
print(f\"  {pathlib.Path('$f').stem[5:]:34s} spearman_within {d['spearman_within_ep']:+.4f}  frac>0 {d['spearman_within_ep_frac_positive']:.3f}  AUC_early {d['auc_early']:.4f}\")" 2>/dev/null
done
