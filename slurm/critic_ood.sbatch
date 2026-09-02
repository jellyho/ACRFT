#!/usr/bin/env bash
# OOD-defence ablation on the patch critic. Everything follows the canonical 200k recipe of
# .scratch/patch_critic_yam_s347_fixed_tau9_min_200k (macro_group_size 30, expectile 0.9, h_goal 30,
# lr 1e-4, K=2, min reduction, aug on, mc_floor on) EXCEPT the one factor named by NAME, so the
# existing checkpoint is the control:
#
#   EDAC ladder (the design ACRFT-WS corrected). The matched pair must hold K FIXED and vary only
#   the EDAC term, or a movement in frac_outside_span cannot be attributed: EDAC diversifies
#   gradients ACROSS members, so pairing a K=2 arm against a K=10+EDAC arm changes ensemble size and
#   the penalty at once, and "a bigger ensemble is more pessimistic off-support" explains the result
#   just as well. K=32 because the shared trunk makes it nearly free (3.08 GB of activations against
#   2.79 at K=10 and 2.68 at K=2) and EDAC's effect grows with ensemble size, so testing at small K
#   hides it. Reduction is `mean`: with the pair matched on K, `min` would only make the absolute
#   numbers K-dependent for reasons unrelated to the penalty, and pessimism belongs at the serving
#   read (CriticQ.q_lcb) where beta is a free knob.
#
#   Labels are NEUTRAL on purpose -- ACRFT-WS scores the pair blind with probe_q_landscape.py, so the
#   arm names must not say which is which. The mapping lives in .scratch/edac_ladder_map.json.
#
#     sbatch --export=ALL,NAME=arm_a_s0,K=32,RED=mean,SEED=0            slurm/critic_ood.sbatch
#     sbatch --export=ALL,NAME=arm_b_s0,K=32,RED=mean,SEED=0,EDAC=1.0   slurm/critic_ood.sbatch
#     ... seeds 0,1,2 for both; the architecture control is one cheap run:
#     sbatch --export=ALL,NAME=arch_ctl_s0,K=2,RED=mean,SEED=0          slurm/critic_ood.sbatch
#   whose anchor is the ALREADY-TRAINED fully-independent K=2 critic
#   (.scratch/patch_critic_yam_s347_fixed_tau9_min_200k), so trunk sharing is tested without a new
#   job for the anchor. One or two seeds is enough there -- its job is to catch a GROSS effect of
#   sharing the trunk on grad_a Q, and a gross effect does not need three seeds to show.
#
#
# HOW TO READ THE SCORES, AND ONE TRAP. ACRFT-WS scores these with probe_q_landscape.py and reports
# frac_outside_span, required rho, and BoN excess against the on-support bias.
#
#   required rho is a RATIO OF AGGREGATES -- mean-over-episodes of dQ divided by mean-over-episodes
#   of the ensemble std, bootstrapped over episodes. It is NOT the mean of per-frame dQ/std. The
#   per-frame version explodes wherever two members nearly agree, because a near-zero denominator
#   enters the average: on the frozen nine it returns 7.80 / 14.30 / 29.1 where the correct values
#   are 2.27 / 2.27 / 2.50. If anything on this side ever computes rho, use the aggregate ratio.
#   ACRFT-WS hit this first and caught it; the wrong version would have shown a 1.8x mean-vs-min
#   difference that does not exist and sent us re-running the anchor for nothing.
#
#   The training reduction does NOT move these metrics, which is why arch_ctl_s0 (mean) is compared
#   against the existing independent K=2 anchor trained with min rather than a fresh matched anchor.
#   Measured on the one matched pair in the frozen nine (g5_tau9_mean vs g5_tau9_min, identical in
#   everything else): dQ at the box edge +1.5%, required rho at the box +0.4%, outside_span 0.0%,
#   with the rho CIs [1.75, 2.98] and [1.81, 3.02] almost entirely overlapping. The spread ACROSS the
#   nine critics is an order of magnitude larger (rho 1.64-2.50, outside_span 0.717-0.877). Caveat
#   from ACRFT-WS: that is n=1 recipe pair, enough to say a comparison will not trip over the
#   reduction, not enough to call the effect exactly zero.
#
#SBATCH --job-name=critic_ood
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/critic_ood_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
EDAC_ARGS=""
if [ -n "${EDAC:-}" ]; then EDAC_ARGS="--edac-weight ${EDAC}"; fi
CQL_ARGS=""
if [ -n "${ACQL:-}" ]; then
  CQL_ARGS="--alpha-cql ${ACQL} --cql-negatives ${NEG:-shuffle} --cql-n ${CQLN:-8} --cql-batch ${CQLB:-64}"
  if [ -n "${BANK:-}" ]; then CQL_ARGS="${CQL_ARGS} --cql-bank ${BANK}"; fi
fi
echo "[$(date +%T)] ${NAME:?} starting on $(hostname)"
# NOTE: we call the venv interpreter directly instead of `uv run --no-sync`. On node100, `uv run`
# hangs before it ever starts Python -- jobs 37235-37239 sat there for two hours writing a zero-byte
# log, and a staged diagnostic (logs/diag100_37263.log) showed the bash preamble printing and then
# both a 120s and a 180s `timeout uv run` expiring with no output, while the identical script
# finished in under a second on node200. With UV_PROJECT_ENVIRONMENT already pointing at the venv,
# `uv run --no-sync` only means "use this venv's python", so calling it directly is equivalent and
# removes the thing that wedges. The echo below exists so a silent log is diagnosable next time.
exec /data5/jellyho/ACRFT/openpi/.venv/bin/python -u scripts/train_patch_critic_cached.py \
  --cache /data1/jellyho/pc_cache/yam_s347 \
  --outcomes /data5/jellyho/ACRFT/openpi/.scratch/yam_outcomes_347.jsonl \
  --homing-onsets /data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json \
  --norm-stats /data5/jellyho/ACRFT/openpi/assets/pi05_yam_lego_taxi_rlt/jellyho/yam_lego_taxi_s300/norm_stats.json \
  --macro-group-size 30 --expectile 0.9 --feat-dropout 0.1 --feat-noise 0.05 \
  --h-goal 30 --lr 1e-4 --steps "${STEPS:-200000}" \
  --num-critics "${K:-2}" --q-reduction "${RED:-min}" --seed "${SEED:-0}" \
  --critic-arch "${ARCH:-shared}" ${EDAC_ARGS} ${CQL_ARGS} \
  --wandb --wandb-name "${NAME:?}" --wandb-group patch-critic-ood \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_${NAME}"
