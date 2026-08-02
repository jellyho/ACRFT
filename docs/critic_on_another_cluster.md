# Running critic ablations on another cluster

The stage-2 critic is small and VRAM-light, and its data lives on the Hub, so it can train wherever
GPUs are free. It has **no simulator dependency** unless you turn on in-loop rollout: `train_rlt_critic.py`
and `eval_rlt_critic.py` import only jax / numpy / optax / flax and the critic module. robosuite and
robocasa are imported lazily, only when `--rollout-every` is set — leave it off and none of that is needed.

## One-time setup

```bash
git clone https://github.com/jellyho/ACRFT.git && cd ACRFT
git checkout fix/probe-eval-jit
uv sync                      # jax-cuda etc. from the lockfile
```

## Get the annotation (once)

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("jellyho/acrft-annot-noprop", repo_type="dataset"))
PY
```

That path (call it `$DATA`) holds the `.dat` memmaps + `meta.json` the trainer reads directly.
Tokens are from `pardec_noprop @ 70k`, 279k frames, N=16 (+8 held-out), terminal-sparse reward,
support `[0, 1]`.

## Train + diagnose (offline, no GPU sim)

```bash
uv run python scripts/train_rlt_critic.py --data $DATA --kind arq --steps 100000 \
  --batch-size 256 --num-atoms 51 --terminal-uses-mc --bootstrap online --mc-lower-bound \
  --wandb-project acrft-critic --wandb-group <sweep-name> --out ./out/<run>

uv run python scripts/eval_rlt_critic.py --data $DATA \
  --params ./out/<run>/params.msgpack --num-states 4096 --out ./out/<run>/diag.json
```

wandb logs the training curves and the within-state diagnostics; `diag.json` has the final numbers.
Point `--wandb-project acrft-critic` at the same project everywhere so sweeps collect together.

## What this cluster can and cannot do

- **Can**: every critic hyperparameter — `--kind {arq,qc}`, capacity (`--num-layers/--num-heads/--head-dim/--mlp-dim`, `--hidden-dims`), ensemble (`--num-critics`), aggregation (`--ens-agg`, `--v-agg`, `--top-m`), target smoothing (`--target-noise`), and the offline diagnostics.
- **Cannot**: rollout success. `--rollout-every` needs the VLA checkpoint and robosuite; run that on the sim cluster. The offline within-state metrics are the proxy; the final judgement is rollout.

## The VLA checkpoint (only if you want rollout there too)

The annotation's VLA is on the Hub as well; you also need robosuite + robocasa installed:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("jellyho/pi05-robocasa-prepcoffee-rlt-pardec-noprop-70k"))
PY
```
Pass that dir's `params` to `--vla-checkpoint` and add `--rollout-every 25000 --rollout-trials 20`.

## Discount sweep (re-labels the return)

Discount changes `mc_return`, so it needs a re-label, not just a flag:

```bash
uv run python scripts/relabel_reward.py --data $DATA --scheme sparse --discount 0.999
# then train with the same --data; discount and support are read from meta.json
```
