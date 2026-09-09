#!/usr/bin/env bash
# Patch critic for the YAM cable-tie task. It is now the only committed critic launcher: it was
# modelled on critic_ood.sbatch, and that one plus critic_cont / train_critic{,_nofloor,_tau9_noaug}
# / score_iql_pair were removed on 2026-09-09 -- every one of them --chdir'd into
# /data5/jellyho/ACRFT/openpi-alphaflow, whose scripts/ copies had diverged from this repo's (its
# train_patch_critic_cached.py makes --outcomes required). Recover them from git if that tree
# matters again.
#
#   sbatch --export=ALL,NAME=fixed_tau9_min_200k,K=2,RED=min,SEED=0,ARCH=independent,HO=0 \
#          slurm/critic_cable_tie.sbatch
#
# WHAT DIFFERS FROM THE LEGO RECIPE, AND WHY EACH ONE HAD TO CHANGE:
#
#  * NO --outcomes. Cable tie carries next.success / next.done in its own schema (150 success, 10
#    fail, 0 unknown), and the cache's meta.json freezes those verdicts at build time. The lego
#    sidecar .scratch/yam_outcomes_347.jsonl covers episodes 0..346 while cable tie's are 0..159, so
#    a copied --outcomes line is PREFERRED over the schema and silently trains this critic on lego
#    verdicts -- no shape error, no index error, only a WARNING that every lego log prints too.
#
#  * --discount 0.999347, not lego's 0.99964. This is an episode-length constant, not a taste knob:
#    v_min = -1/(1-discount) and --failure-reward defaults to v_min. Lego's value buys a 2777.8-frame
#    (92.6 s) horizon against its 2683-frame mean task; cable tie's mean task is 1479 frames (49.3 s),
#    1.8x shorter. Left at 0.99964 a typical cable-tie start sits at -1147 on a -2778 support, so the
#    critic would train on the top ~43% of its 101 atoms and resolve a 1479-frame task with 27.5-frame
#    bins. 0.999347 applies lego's own horizon-to-mean-task ratio (1.035) to the measured cable-tie
#    mean: v_min -1530.9 frames (51.0 s), bin 15.16, HL-Gauss sigma 11.37. Do NOT also pass --v-min or
#    --failure-reward; both derive from the discount. Consequence to remember when reading scores:
#    score_critic_cached.py's deep-atom threshold is 0.72*v_min = -1102 here against -2000 for lego,
#    so deep-atom mass is NOT comparable across the two tasks.
#
#  * --homing-onsets points at the cable-tie json, which had to be recomputed against a FIXED
#    detector. Every cable-tie episode ends [... 4. 4. 4. 4. 0.] -- one stray teleop frame after the
#    homing run -- and the old "onset = 1 + last teleop frame" rule therefore returned the episode
#    length for all 160, leaving every homing tail (5.77% of frames) labelled as task progress. Since
#    homing_onset is the denominator of the whole cost_to_goal target, that is not cosmetic. The same
#    bug affects 275 of 347 LEGO episodes. .scratch/yam_homing_onsets.json now holds the CORRECTED
#    onsets for lego, and the file the existing critics were trained on is preserved beside it as
#    .scratch/yam_homing_onsets_stale_pre_20260907.json -- do not train against that one.
#
#  * --preload, and --mem raised to 120G. The lego arms could not preload (138 GB of features); cable
#    tie's cache is 37 GB, so it fits in RAM and per-step NFS gathers become RAM reads.
#
#  * HO defaults to 0, not 0.15. Cable tie has 10 failure episodes against lego's 47; a stratified
#    0.15 split puts 1-2 of them on the held-out side, which cannot support ho_auc_early, and it
#    removes 15% of already-scarce failure signal from training -- where the absorbing terminal at
#    v_min is the only thing that populates the lower half of the support. Pass HO=0.15 explicitly
#    for a validation arm, and report n_fail_heldout with it.
#
# Everything else is a time at a shared 30 fps and carries over unchanged: --macro-group-size 30,
# --expectile 0.9, --h-goal 30 (1.00 s), --feat-dropout 0.1, --feat-noise 0.05, --lr 1e-4,
# --num-atoms 101, --truncate-homing all (default), --mc-floor (default).
#SBATCH --job-name=critic_ct
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/critic_ct_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
EDAC_ARGS=""
if [ -n "${EDAC:-}" ]; then EDAC_ARGS="--edac-weight ${EDAC}"; fi
INV_ARGS=""
if [ -n "${INV:-}" ]; then INV_ARGS="--inv-weight ${INV} --inv-max-k ${INVK:-90} --inv-bottleneck ${INVB:-64}"; fi
CQL_ARGS=""
if [ -n "${ACQL:-}" ]; then
  CQL_ARGS="--alpha-cql ${ACQL} --cql-negatives ${NEG:-shuffle} --cql-n ${CQLN:-8} --cql-batch ${CQLB:-64}"
  if [ -n "${BANK:-}" ]; then CQL_ARGS="${CQL_ARGS} --cql-bank ${BANK}"; fi
fi
echo "[$(date +%F\ %T)] critic_ct ${NAME:?} starting on $(hostname)"
# Direct venv interpreter, never `uv run` -- on node100 it hangs before Python starts (jobs
# 37235-37239 wrote zero-byte logs for two hours; logs/diag100_37263.log is the staged diagnostic).
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_patch_critic_cached.py \
  --cache /data1/jellyho/pc_cache/yam_cable_tie_s160 \
  --homing-onsets /data5/jellyho/ACRFT/openpi/.scratch/yam_cable_tie_homing_onsets.json \
  --norm-stats /data5/jellyho/ACRFT/openpi/assets/pi05_yam_cable_tie/jellyho/yam_cable_tie_s150/norm_stats.json \
  --discount "${DISCOUNT:-0.999347}" --preload \
  --macro-group-size 30 --expectile 0.9 --feat-dropout 0.1 --feat-noise 0.05 \
  --h-goal 30 --lr 1e-4 --steps "${STEPS:-200000}" \
  --num-critics "${K:-2}" --q-reduction "${RED:-min}" --seed "${SEED:-0}" \
  --heldout-frac "${HO:-0}" --eval-every "${EVERY:-5000}" \
  --critic-arch "${ARCH:-independent}" --value-head "${VH:-categorical}" --floq-steps "${FSTEPS:-8}" ${EDAC_ARGS} ${CQL_ARGS} ${INV_ARGS} \
  --wandb --wandb-name "ct_${NAME:?}" --wandb-group patch-critic-cable-tie \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_ct_s160_${NAME}"
