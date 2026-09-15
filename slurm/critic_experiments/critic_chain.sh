#!/usr/bin/env bash
# Fire the patch-critic pipeline the moment the cable-tie BC run finishes.
#
#   1. wait for the BC process to exit
#   2. REFUSE to continue unless it actually reached its last checkpoint -- a crashed run must not
#      silently seed a critic, because the critic would look trained and be wrong
#   3. frozen-DINOv2 feature cache, once (shared by both critics)
#   4. critic, FIXED chunk   (--macro-group-size == horizon 30: one value per whole chunk, best-of-N)
#   5. critic, MACRO groups  (--macro-group-size 5: per-prefix values, adaptive commitment)
#
# The hyperparameters are NOT this script's invention: they are slurm/critic_cable_tie.sbatch's
# recipe, which differs from train_patch_critic_cached.py's bare defaults in ways that matter --
# lr 1e-4 not 3e-4, expectile 0.9, and feature-space augmentation ON (--feat-dropout/--feat-noise
# default to 0.0, i.e. a default run has no augmentation at all). Running the defaults would have
# been a different experiment wearing the same name.
#
# Sequential on purpose: one GPU. Each stage aborts the chain on failure rather than feeding
# garbage forward.
set -uo pipefail

REPO=/NHNHOME/jellyho/jellyho/ACRFT
BC_PID=234866
BC_CKPT="$REPO/checkpoints/pi05_yam_cable_tie/yam_cable_tie"
BC_FINAL_STEP=200000

CACHE=/NHNHOME/jellyho/jellyho/pc_cache/cable_tie
OUT_FIXED=/NHNHOME/jellyho/jellyho/critics/cable_tie_fixed_g30
OUT_MACRO=/NHNHOME/jellyho/jellyho/critics/cable_tie_macro_g5
ONSETS="$REPO/.scratch/yam_cable_tie_homing_onsets.json"
# The policy this critic scores, named rather than path-resolved: --policy-config runs the same
# content-addressed norm-stats lookup training uses, so the critic normalizes actions exactly as
# that policy does. _withhoming is the pre-2026-09-11 cable-tie condition (give-up tails KEPT),
# which is what the running BC job trained on -- critic follows the POLICY, not the repo default.
POLICY_CONFIG=pi05_yam_cable_tie_withhoming
PY="$REPO/.venv/bin/python -u"

cd "$REPO"
export LEROBOT_VIDEO_BACKEND=pyav
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
# $REPO itself, not just src/: these scripts do `from scripts.train_patch_critic_clip import ...`,
# and the installed package only puts src/ on the path. Without this the critic dies at import.
export PYTHONPATH="$REPO:$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

say() { echo "[$(date +%F\ %T)] $*"; }

# One place, so the two runs cannot drift apart on anything but the group size.
common_args=(
  --cache "$CACHE"
  --homing-onsets "$ONSETS"
  --policy-config "$POLICY_CONFIG"
  --discount 0.999347 --preload
  --expectile 0.9
  --feat-dropout 0.1 --feat-noise 0.05   # feature-space augmentation; OFF by default in the script
  --h-goal 30 --lr 1e-4
  --steps 200000 --save-every 50000
  --num-critics 2 --q-reduction min --seed 0
  --heldout-frac 0 --eval-every 5000
  --critic-arch independent --value-head categorical
  --reward-scheme cost_to_goal
  --wandb --wandb-group patch-critic-cable-tie
)

[ -f "$ONSETS" ] || { say "ABORT: missing prerequisite $ONSETS"; exit 1; }

say "waiting for BC pid $BC_PID"
while kill -0 "$BC_PID" 2>/dev/null; do sleep 60; done
say "BC pid $BC_PID exited"

# The BC run is what every downstream stage is about; if it did not finish, stop here loudly.
if [ ! -d "$BC_CKPT/$BC_FINAL_STEP" ]; then
    say "ABORT: $BC_CKPT/$BC_FINAL_STEP is missing -- the BC run did not reach step $BC_FINAL_STEP."
    say "       saved steps: $(ls "$BC_CKPT" 2>/dev/null | tr '\n' ' ')"
    say "       Nothing downstream ran. Inspect the BC log, then rerun this script."
    exit 1
fi
say "BC reached $BC_FINAL_STEP; continuing"

say "=== 1/3  feature cache -> $CACHE  (frozen DINOv2, once per frame) ==="
if [ -f "$CACHE/meta.json" ]; then
    say "cache already present, skipping"
elif ! $PY scripts/cache_patch_features.py --repo-id jellyho/yam_cable_tie --out "$CACHE"; then
    say "ABORT: feature cache failed; neither critic ran."
    exit 1
fi
say "cache done: $(du -sh "$CACHE" 2>/dev/null | cut -f1)"

# FIXED first, as asked: one value for the whole 30-step chunk. It ranks whole candidate chunks and
# never splits one, so it is the simpler claim and the baseline the adaptive variant has to beat.
say "=== 2/3  critic FIXED chunk (macro-group-size 30) -> $OUT_FIXED ==="
$PY scripts/train_patch_critic_cached.py "${common_args[@]}" \
    --macro-group-size 30 --wandb-name ct_fixed_g30 --out "$OUT_FIXED" \
    || say "fixed-chunk critic FAILED; still attempting the macro-group run below."

# MACRO groups: a value per 5-step prefix, which is what adaptive commitment reads to decide how
# much of a chunk is still worth executing.
say "=== 3/3  critic MACRO groups (macro-group-size 5) -> $OUT_MACRO ==="
$PY scripts/train_patch_critic_cached.py "${common_args[@]}" \
    --macro-group-size 5 --wandb-name ct_macro_g5 --out "$OUT_MACRO" \
    || say "macro-group critic FAILED."

say "=== chain finished ==="
say "fixed : $OUT_FIXED"
say "macro : $OUT_MACRO"
