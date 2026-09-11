#!/usr/bin/env bash
# Two readouts for one critic, so several critics can be compared on the same footing.
#
#   sbatch --export=ALL,CRITIC=.scratch/patch_critic_yam_s347_fixed_tau9_min_200k slurm/diag_critic.sbatch
#
# selection bias  -- within_state_candidate_std against between_state_std. The first is the only part
#                    a best-of-N choice can use: episode identity and task phase are constant across
#                    candidates at one state, so whatever the critic spends its range on between
#                    states cancels exactly there.
# score           -- success/fail ROC-AUC and the per-class value profile, i.e. how much of the
#                    critic's range goes to telling episodes apart rather than ranking actions.
#
# CACHE and ONSETS default to the lego set; pass them for cable tie. The onsets path is the canonical
# one, which now holds the CORRECTED file, so every critic is scored against the same true goal
# regardless of what it was trained on -- that is the point when comparing a homing-fixed critic
# against one trained before the fix.
#SBATCH --job-name=diag_critic
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/logs/diag_critic_%x_%j.slurm.log
export UV_PROJECT_ENVIRONMENT=/data5/jellyho/ACRFT/openpi/.venv
export PYTHONPATH=/data5/jellyho/ACRFT/openpi:/data5/jellyho/ACRFT/openpi/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
P=/data5/jellyho/ACRFT/openpi/.venv/bin/python
C="${CRITIC:?set CRITIC to a critic directory}"
TAG=$(basename "$C")
echo "[$(date +%F\ %T)] diag ${TAG} on $(hostname)"
$P -u scripts/diag_critic_selection_bias.py --critic "$C" \
  --cache "${CACHE:-/data1/jellyho/pc_cache/yam_s347}" \
  --out "/data5/jellyho/ACRFT/openpi/.scratch/extraction/diag_selection_bias_${TAG}.json" \
  --fig "/data5/jellyho/ACRFT/openpi/.scratch/extraction/fig_selection_bias_${TAG}.png"
echo "--- score ---"
exec $P -u scripts/score_critic_cached.py --critic "$C" \
  --cache "${CACHE:-/data1/jellyho/pc_cache/yam_s347}" \
  --homing-onsets "${ONSETS:-/data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json}" --stride 20
