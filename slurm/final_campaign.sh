#!/usr/bin/env bash
# ============================================================================================
#  FINAL campaign — the from-scratch, every-lesson-applied component sweep (2026-08-07).
#
#  Fixed across every arm: gamma 0.995 · 100k steps · batch 256 · lr 3e-4 · seed 0 · HLG
#  (atoms per arm) · mc_floor · action z-score · target tau 0.005 · IQL expectile 0.9.
#  Evaluation: per arm, 4 seeds (5000..5300) x 50 scenes, in-job paired with vla, stage logs.
#  Scene seeds are IDENTICAL across arms, so arm-vs-arm is also scene-paired.
#  Deployment is the SAME raw joint arg-max everywhere - the sweep isolates TRAINING factors.
#  Plus one 6-scene HUD-video job per arm.
#
#  Arms (14): center = mixed data, atoms 51, EMA target
#    A  method x bootstrap-op : td_max  td_soft  td_aqcmax  iql  qc
#    B  atoms                 : td_max_a101  td_max_a201  iql_a101  iql_a201
#    C  target network        : td_max_online  iql_online
#    D  data (demo-only)      : td_max_demo  iql_demo  qc_demo
#
#      bash slurm/final_campaign.sh            # submit everything
# ============================================================================================
set -euo pipefail
cd "$(dirname "$0")/.." && source slurm/env.sh
IFS='|' read -r BIG_PART BIG_QOS <<< "$(QOS=big_qos acrft_tier big)"
_EXC="$(acrft_exclude_list)"
MIX=$CACHE_DIR/annot/mixed
DEMO=$CACHE_DIR/annot/noprop
SWEEP=final
BASE_FLAGS="--batch-size 256 --lr 3e-4 --mc-lower-bound --seed 0 --save-every 25000 --eval-every 0"
EVAL_SEEDS="5000 5100 5200 5300"

declare -A DATA FLAGS
# ---- A. method x bootstrap
DATA[td_max]=$MIX;     FLAGS[td_max]="--objective td --boot-op max --num-atoms 51"
DATA[td_soft]=$MIX;    FLAGS[td_soft]="--objective td --boot-op softmax --boot-temp 0.05 --num-atoms 51"
DATA[td_aqcmax]=$MIX;  FLAGS[td_aqcmax]="--objective td --boot-op aqcmax --num-atoms 51"
DATA[iql]=$MIX;        FLAGS[iql]="--objective iql --expectile 0.9 --num-atoms 51"
DATA[qc]=$MIX;         FLAGS[qc]="--objective td --boot-op max --kind qc --num-atoms 51"
# ---- B. atoms
DATA[td_max_a101]=$MIX; FLAGS[td_max_a101]="--objective td --boot-op max --num-atoms 101"
DATA[td_max_a201]=$MIX; FLAGS[td_max_a201]="--objective td --boot-op max --num-atoms 201"
DATA[iql_a101]=$MIX;    FLAGS[iql_a101]="--objective iql --expectile 0.9 --num-atoms 101"
DATA[iql_a201]=$MIX;    FLAGS[iql_a201]="--objective iql --expectile 0.9 --num-atoms 201"
# ---- C. target network (online = bootstrap scored by online params, stop-gradient)
DATA[td_max_online]=$MIX; FLAGS[td_max_online]="--objective td --boot-op max --num-atoms 51 --bootstrap online"
DATA[iql_online]=$MIX;    FLAGS[iql_online]="--objective iql --expectile 0.9 --num-atoms 51 --bootstrap online"
# ---- D. demo-only
DATA[td_max_demo]=$DEMO; FLAGS[td_max_demo]="--objective td --boot-op max --num-atoms 51"
DATA[iql_demo]=$DEMO;    FLAGS[iql_demo]="--objective iql --expectile 0.9 --num-atoms 51"
DATA[qc_demo]=$DEMO;     FLAGS[qc_demo]="--objective td --boot-op max --kind qc --num-atoms 51"

EVAL_PARTS="${BIG_PART}"
EVAL_QOS="big_qos"
for ARM in td_max td_soft td_aqcmax iql qc td_max_a101 td_max_a201 iql_a101 iql_a201 \
           td_max_online iql_online td_max_demo iql_demo qc_demo; do
  TJ=$(RUN=$ARM DATA=${DATA[$ARM]} SWEEP=$SWEEP STEPS=100000 FLAGS="${FLAGS[$ARM]} $BASE_FLAGS" \
    sbatch --parsable --mem=96G ${_EXC:+--exclude="$_EXC"} \
    -o "$SLURM_LOGS/final_${ARM}_%j.out" slurm/train_critic.sbatch)
  CK=$CACHE_DIR/critic_runs/$SWEEP/$ARM/params.msgpack
  for SEED in $EVAL_SEEDS; do
    sbatch --parsable -J "fev_${ARM}_s$SEED" -p "$EVAL_PARTS" -q "$EVAL_QOS" \
      --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=16:00:00 --dependency=afterok:$TJ \
      ${_EXC:+--exclude="$_EXC"} -o "$SLURM_LOGS/fev_${ARM}_s${SEED}_%j.out" \
      --wrap "cd $ACRFT_REPO && source slurm/env.sh && uv run --no-sync examples/robocasa/eval_critic.py \
        --config pi05_robocasa_PrepareCoffee_rlt --checkpoint $VLA_CKPT --critic $CK \
        --task PrepareCoffee --num-trials 50 --seed $SEED --modes critic vla \
        --out $CACHE_DIR/critic_runs/$SWEEP/$ARM/rollout/f_s${SEED}.json \
        --vla-override rlt_decoder_mode=parallel --vla-override rlt_include_proprio=false" >/dev/null
  done
  sbatch --parsable -J "fvid_${ARM}" -p "$EVAL_PARTS" -q "$EVAL_QOS" \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=8:00:00 --dependency=afterok:$TJ \
    ${_EXC:+--exclude="$_EXC"} -o "$SLURM_LOGS/fvid_${ARM}_%j.out" \
    --wrap "cd $ACRFT_REPO && source slurm/env.sh && uv run --no-sync examples/robocasa/eval_critic.py \
      --config pi05_robocasa_PrepareCoffee_rlt --checkpoint $VLA_CKPT --critic $CK \
      --task PrepareCoffee --num-trials 6 --seed 5000 --modes critic vla \
      --video-dir $CACHE_DIR/videos_final/$ARM --num-videos 6 \
      --out $CACHE_DIR/critic_runs/$SWEEP/$ARM/rollout/video_s5000.json \
      --vla-override rlt_decoder_mode=parallel --vla-override rlt_include_proprio=false" >/dev/null
  echo "$ARM: train=$TJ + eval x4 + video"
done
echo "FINAL campaign submitted: 14 trainings, 56 evals, 14 video jobs"
