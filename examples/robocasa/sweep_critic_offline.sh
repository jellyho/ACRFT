#!/usr/bin/env bash
# Offline critic ablation for a cluster WITHOUT the simulator: train + diagnostics only, logged to
# wandb. See docs/critic_on_another_cluster.md. No robosuite/robocasa needed - --rollout-every is off.
#
#   DATA=$(python -c "from huggingface_hub import snapshot_download as d; print(d('jellyho/acrft-annot-noprop', repo_type='dataset'))")
#   bash examples/robocasa/sweep_critic_offline.sh "$DATA"
#
# Override the launcher for your scheduler; this uses plain background processes on one node.
set -uo pipefail
DATA="${1:?pass the snapshot_download path of jellyho/acrft-annot-noprop}"
OUT="${OUT:-./out}"
STEPS="${STEPS:-100000}"
PROJ="${PROJ:-acrft-critic}"
mkdir -p "$OUT"

CT="--num-atoms 51 --terminal-uses-mc --bootstrap online --mc-lower-bound"
run() {  # $1=tag  $2..=flags
  tag=$1; shift
  echo ">> $tag"
  uv run python scripts/train_rlt_critic.py --data "$DATA" --steps "$STEPS" --batch-size 256 \
    $CT --wandb-project "$PROJ" --wandb-group offline --wandb-name "$tag" --out "$OUT/$tag" "$@" \
  && uv run python scripts/eval_rlt_critic.py --data "$DATA" \
    --params "$OUT/$tag/params.msgpack" --num-states 4096 --out "$OUT/$tag/diag.json"
}

# architecture
run arq     --kind arq
run qc      --kind qc
# capacity
run arq_big --kind arq --num-layers 4 --num-heads 8 --head-dim 64 --mlp-dim 2048
# ensemble
run arq_k4  --kind arq --num-critics 4
# aggregation
run arq_lcb --kind arq --ens-agg lcb --lcb-beta 1.0
run arq_top --kind arq --v-agg topm --top-m 3
