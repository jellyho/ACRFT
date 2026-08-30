# Deploying the YAM policy behind the patch-critic

How to serve the YAM `lego_taxi` VLA with the standalone **patch-critic** in front of it, either as
best-of-N or as adaptive chunking. Also: how to obtain, verify, and (re)train the critic.

Everything here is CPU/GPU work on the cluster — **submit through SLURM** (`srun`/`sbatch`); do not grab
`CUDA_VISIBLE_DEVICES` directly.

---

## 1. What the stack is

```
robot client ──websocket──► serve_policy.py --critic
                              ├─ base VLA (pi05)            sample N chunks in ONE backbone pass
                              └─ patch-critic (frozen DINOv2 + ARQ heads)   score them
                                   └─ mode=bon       → return the argmax-Q chunk (full H)
                                      mode=adaptive  → return that chunk truncated to its
                                                       highest-value commitment prefix K, then replan
```

The critic is **independent of the VLA**: its observation is a grid of frozen DINOv2 patch tokens over
the robot cameras, not a VLA token. That is what lets it be trained and scored offline without ever
running the policy.

## 2. Get the critic

Checkpoints live in the (private) model repo **`jellyho/patch_critic_yam_lego_taxi`**:

| folder | `macro_group_size` | prefixes it scores | use |
|---|---|---|---|
| `g5_s347` | 5 | k = 5, 10, 15, 20, 25, 30 | `--mode adaptive` (needs per-prefix values) |
| `fixed_s347` | 30 | k = 30 only | `--mode bon` control (method-only diff vs `g5`) |

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("jellyho/patch_critic_yam_lego_taxi", repo_type="model",
                  local_dir="/data5/jellyho/critics/yam")
PY
```

Each folder holds `params.msgpack` (Q ensemble), `v_params.msgpack` (state-value net) and `config.json`
(everything needed to rebuild the modules). `--critic` takes the **folder**.

## 3. Serve

```bash
srun -p debug --gres=gpu:L40S:1 --cpus-per-task=8 --mem=64G -t 08:00:00 \
  bash -lc 'cd /data5/jellyho/ACRFT/openpi && \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    uv run --no-sync python scripts/serve_policy.py --critic \
      --config pi05_yam_lego_taxi \
      --checkpoint checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/125000 \  # any saved BC step; HF backup: jellyho/pi05_yam_lego_taxi_h30_s300
      --critic /data5/jellyho/critics/yam/g5_s347 \
      --mode adaptive --num-samples 8 --port 8000'
```

Wait for `Serving ... listening on 0.0.0.0:8000`. The base checkpoint above is the s300 run; swap in
whichever step you are evaluating, and **record it in the report** — a value comparison is only valid
between method-only-diff checkpoints.

### In-process, without a server

The same wrapper can be built directly — this is exactly the path the deploy test exercises, so it is
the quickest way to check a new checkpoint actually loads and infers:

```python
from huggingface_hub import snapshot_download
from openpi.policies import policy_config
from openpi.policies.patch_critic_policy import PatchCriticSelectPolicy
from openpi.training import config as _config

snapshot_download("jellyho/patch_critic_yam_lego_taxi", repo_type="model",
                  allow_patterns="fixed_pi05_s347/*", local_dir="/data5/jellyho/critics/yam")

policy = policy_config.create_trained_policy(
    _config.get_config("pi05_yam_lego_taxi"),  # BC config (the deploy default going forward)
    "checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/125000",
)
wrapped = PatchCriticSelectPolicy(
    policy, "/data5/jellyho/critics/yam/fixed_pi05_s347", mode="bon", default_samples=8,
)

out = wrapped.infer(obs)   # 3 camera images + observation/state + prompt
out["actions"]             # (30, 14)  selected chunk
out["critic_scores"]       # (30, N)   value of each candidate
out["critic_choice"]       # (30, 1)   which candidate won
```

The runnable version is `.scratch/deploy_test.py`: it downloads into a directory where the repo's
asset paths do **not** resolve, which is what proves the checkpoint is self-contained.

`--mode adaptive` needs a checkpoint with more than one prefix, so it cannot be used with a
`macro_group_size = 30` critic.

## 4. Client contract

The wrapper is **opt-in per request**, so the same server can serve the plain VLA and the value-guided
policy in one session (that is how a paired comparison is run):

| key | meaning |
|---|---|
| `critic_select` | `False`/absent → plain VLA chunk. `True` → critic selection. |
| `num_samples` | candidates to score this request (defaults to `--num-samples`). |

In `adaptive` mode the returned chunk is **already truncated** to the selected commitment length, so the
client simply executes what it receives and calls again — no protocol change is needed.

## 5. Verify a critic BEFORE deploying

Never deploy an unscored critic. Scoring reads the precomputed feature cache, so it needs **no video
decode and no DINOv2 forward** and finishes in minutes:

```bash
srun -p debug --gres=gpu:L40S:1 --cpus-per-task=8 --mem=64G -t 40 \
  bash -lc 'cd /data5/jellyho/ACRFT/openpi && XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-sync python scripts/score_critic_cached.py \
      --critic /data5/jellyho/critics/yam/g5_s347 \
      --cache /data1/jellyho/pc_cache/yam_s347 \
      --outcomes .scratch/yam_outcomes_347.jsonl \
      --homing-onsets .scratch/yam_homing_onsets.json --stride 15'
```

It prints success-vs-failure ROC-AUC (mean / max / last value), the per-class value profile, and the
**deep-atom mass** — the fraction of probability the critic puts below `0.72·v_min`. That last number is
the diagnostic that caught the critic compressing all its values into the top third of its support.

## 6. Rebuilding the data pipeline (only if you retrain)

```bash
# 1) which episodes succeeded/failed  ->  .scratch/yam_outcomes_347.jsonl
# 2) where each failure starts homing (control_mode 0=teleop, 4=homing)
uv run --no-sync python scripts/compute_homing_onsets.py ...   # -> .scratch/yam_homing_onsets.json

# 3) precompute frozen DINOv2 patch features once (938k frames, ~130 GB, put it on /data1)
sbatch ... --wrap 'uv run --no-sync python scripts/cache_patch_features.py \
    --outcomes .scratch/yam_outcomes_347.jsonl --out /data1/jellyho/pc_cache/yam_s347 \
    --clip-len 120 --num-workers 16 --encode-batch 8'

# 4) train from the cache (~4x faster than decoding video each step)
sbatch ... --wrap 'uv run --no-sync python scripts/train_patch_critic_cached.py \
    --cache /data1/jellyho/pc_cache/yam_s347 --outcomes .scratch/yam_outcomes_347.jsonl \
    --homing-onsets .scratch/yam_homing_onsets.json --preload \
    --macro-group-size 5 --batch 256 --steps 20000 --wandb --out .scratch/<run>'
```

`--preload` pulls the cache into RAM (the node has ~1 TB); without it every step is an NFS gather.
`--macro-group-size 30` reproduces the `fixed` control. Video decode uses openpi's **pyav** backend
(`LEROBOT_VIDEO_BACKEND=pyav`), which ships its own FFmpeg and therefore works on every node —
torchcodec needs system `libavutil`, which the older nodes lack.

## 6b. The input contract (read this before training or serving a critic)

A critic is only meaningful in the input space it was trained in, so every checkpoint now records that
space in `config.json`'s `input_spec`, with `norm_stats.json` alongside as the reference distribution.
The server validates the contract at load and refuses a mismatch.

| `normalization` | what the critic eats | how the server feeds it |
|---|---|---|
| `pi05` (default) | pi05-normalized **joint deltas** + normalized state — the base VLA's own space | the sampler's output **directly**, no conversion |
| `raw` (legacy) | absolute joint targets, unnormalized 42-d state | via `_output_transform` (un-normalize + un-delta) |

Prefer `pi05`. It removes the conversion between the sampler and the critic entirely — which is where
the bug below lived — and it fixes conditioning: raw state channels span std 0.096–3.124 (32×), the
normalized ones 0.265–0.650 (2.5×). It needs no re-caching, because the delta and normalization are
computed on the fly from the cached raw state and actions:

```bash
--input-mode pi05 --norm-stats assets/pi05_yam_lego_taxi_rlt/jellyho/yam_lego_taxi_s300/norm_stats.json
```

Use the **same** `norm_stats.json` as the base checkpoint being served. A pre-contract checkpoint can
be stamped with `scripts/backfill_critic_spec.py` (it is raw-space by definition).

## 7. Known limitations — read before trusting a number

- **Failure values are not calibrated.** The terminal `v_min` anchor deepens them but they do not reach
  the floor (−2778); the anchor supervises only the last frames directly and propagates by bootstrap.
  Use failure values **ordinally**, not as magnitudes.
- **The critic under-uses its deep atoms** (almost no mass below ≈ −2000), so absolute values are
  compressed. Ranking is far more trustworthy than value.
- **Hindsight leakage.** The training data is human teleoperation, i.e. *closed-loop*: later actions were
  chosen after seeing the intervening states, so conditioning a chunked critic on a whole chunk leaks
  information about outcomes and biases the estimate (Li, Park & Levine, arXiv:2512.10926). The bias is
  expected to **grow with the commitment length**, so `--mode adaptive`'s cross-prefix comparison is
  **provisional** until that bias is measured.
- **Every number produced by the serving path before 2026-08-19 is void.** The wrapper passed
  `state=zeros` into the output transform, so `JointAbsoluteActions` (`actions[..., i] +=
  state[..., ref[i]]`) was a no-op and the critic scored joint **deltas** (mean ≈ 0, std ≈ 0.25) while
  it had been trained on **absolute** joint targets (mean ≈ 1.7, std ≈ 0.44). It failed silently. Fixed,
  and made unrepeatable by the shared-preprocessing path in §6b — but any earlier `bon`/`adaptive`
  comparison must be re-run.
- **The two checkpoints in the HF repo are raw-space** (`g5_s347`, `fixed_s347`) and were only ever
  served through that broken path. Re-train with `--input-mode pi05` before deploying.
- **`adaptive` has not been validated end-to-end on the robot** in this repo — the server loads and
  serves, and the critic is scored offline, but no paired robot evaluation of `adaptive` vs `bon` vs plain
  VLA has been reported yet. Treat it as ready-to-test, not as a validated result.

## 8. Related

- Critic modules: `src/openpi/patch_critic/{backbone,critic}.py`
- Serving wrapper: `src/openpi/policies/patch_critic_policy.py`
- Reports (method, figures, judgements): <https://huggingface.co/spaces/jellyho/acrft-reports>
