# ACRFT — RL Tokens & Adaptive Q-Chunking on top of π₀.₅

A research fork of [openpi](https://github.com/Physical-Intelligence/openpi) that adds an end-to-end
pipeline for learning a **compact state representation (the "RL token")** on top of a frozen/finetuned
π₀.₅ VLA, and training an **offline value critic** on it — implementing and extending
[*RL Token: Bootstrapping Online RL with VLA Models*](https://arxiv.org/abs/2604.23073) and the
ACSAC adaptive-chunk critic.

Everything runs on **RoboCasa 365** kitchen tasks (LeRobot v3.0 data), which is one supported
environment section, not the focus — the focus is the RLT → critic pipeline, which is
task-agnostic.

> New to the repo? Read [The pipeline in one page](#the-pipeline-in-one-page), then jump to
> [Quickstart](#quickstart). openpi's own model/setup docs are preserved in
> [`docs/openpi_upstream.md`](docs/openpi_upstream.md).

---

## The pipeline in one page

The goal: turn a VLA's internal features into a small vector `z_rl` that a lightweight RL critic can
consume instead of pixels, then score/plan action chunks with that critic — **without ever running
the 2B backbone during RL**.

```
                 ┌─────────────────────────────────────────────────────────────────┐
   Stage 0       │  π₀.₅ base checkpoint  (gs://openpi-assets/checkpoints/pi05_base)  │
                 └─────────────────────────────────────────────────────────────────┘
                                          │  BC fine-tune  +  RLT bottleneck (joint)
                                          ▼
   Stage 1     Pi0RLT training            scripts/train.py  ·  examples/robocasa/run_train_rlt.sh
   ─────────   Learns z_rl (2048-d) jointly with the BC (flow-matching) loss, from ONE backbone
               forward. A stop-gradient bottleneck reconstructs the image tokens and/or predicts
               task progress. Optional: a latent BC "probe" head + in-process sim eval every 10k.
                                          │  frozen VLA
                                          ▼
   Stage 2     Annotation                 examples/robocasa/annotate_rlt.py
   ─────────   Runs the trained VLA over every demo frame ONCE and writes flat arrays:
               rl_token[2048], action_chunk[H,12], base_action[N,H,12] candidates, reward, progress.
                                          │  numeric arrays, no video, no VLA
                                          ▼
   Stage 3     Critic training            scripts/train_rlt_critic.py
   ─────────   QC (flat Q per chunk) or ARQ (per-prefix Q, causal transformer) critic, scalar or
               HL-Gauss distributional. Data lives on the GPU; updates are fused with lax.scan.
                                          │
                                          ▼
   Stage 4     Deployment (planned)       score N candidate chunks, pick (chunk, commit-length)
```

Key design choices, and where they live:

| Idea | Where | One-liner |
|---|---|---|
| Language-conditioned RL token | `models/pi0_rlt.py` | token reads the image-token hidden states of the full image+language prefix |
| Joint BC + RLT, one forward | `Pi0RLT.compute_loss` | RLT reuses the BC forward's prefix output; ≈1.12× BC step cost |
| Stop-gradient readout (default) | `rlt_backbone_gradient=False` | RLT loss never reshapes the VLM; BC alone shapes it. **Backbone-grad fails** (see results) |
| Progress = per-episode time-to-success | `training/progress.py` | 0 at start → 1 at success, normalized per demo (not a discounted value) |
| Latent BC probe | `rlt_bc_probe=True` | a small flow-matching head on the *frozen* token; measures how much policy the latent alone recovers |
| QC vs ARQ critic | `rlt_critic/critic.py` | ARQ returns a value per prefix length → adaptive chunk commit at deploy |

---

## Quickstart

### Install (same as openpi)

```bash
git clone --recurse-submodules <this repo>            # or: git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv sync --group eval                                  # sim deps (robosuite/mujoco) for evaluation
```

Requires an NVIDIA GPU with ≥70 GB for full fine-tuning (π₀.₅ is 3B); L40S/48 GB is enough for
inference and evaluation. See [`docs/openpi_upstream.md`](docs/openpi_upstream.md#requirements) for the
full matrix.

### Data

RoboCasa 365 demos, converted to LeRobot v3.0 — see [examples/robocasa/README.md](examples/robocasa/README.md)
for download + conversion. Normalization stats for all 50 tasks are **checked into the repo**
(`examples/robocasa/norm_stats/`), so a fresh clone can train without downloading a dataset just to
compute them.

### Run the pipeline (single task, e.g. PrepareCoffee)

```bash
# Stage 1 — RLT training (node with ≥70 GB VRAM)
examples/robocasa/run_train_rlt.sh PrepareCoffee

# Stage 2 — annotate the trained checkpoint into flat arrays
uv run examples/robocasa/annotate_rlt.py \
    --config pi05_robocasa_PrepareCoffee_rlt \
    --checkpoint checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt/100000 \
    --out data/rlt_critic/PrepareCoffee

# Stage 3 — critic (fits on any GPU; the data is a few GB, resident on-device)
uv run scripts/train_rlt_critic.py --data data/rlt_critic/PrepareCoffee --kind arq

# Evaluate a policy checkpoint (sim rollouts, every checkpoint)
examples/robocasa/run_eval.sh PrepareCoffee

# Serve a trained checkpoint to a robot / sim client (see "Serving a policy")
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_robocasa_PrepareCoffee_rlt \
    --policy.dir checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt/100000
```

---

## Looking at a rollout

A deployed rollout is a dataset; `misc/` turns one into a video — the candidate chunks the policy
proposed, which one the critic picked, how far ahead it committed, and the value it was scored on,
overlaid on the three cameras.

```bash
misc/yam-misc render-gui        # pick a dataset and an episode, press Render
misc/yam-misc render-samples --repo-id lerobot_rollout/<name> --root ~/lerobot_rollout --episode 0
```

Self-contained: it carries its own copy of the arm's kinematics and the rig's agentview extrinsics,
so it needs nothing from the robot repo. See [misc/README.md](misc/README.md) — including which of
those files are snapshots to re-copy after a recalibration.

## Stage 1 — RLT training

`examples/robocasa/run_train_rlt.sh <Task> [<Task> ...] [flags]` wraps `scripts/train.py`. It
computes/checks norm stats, then trains `pi05_robocasa_<Task>_rlt`. Every ablation flag tags the
experiment name so runs never share a checkpoint dir.

```bash
examples/robocasa/run_train_rlt.sh PrepareCoffee                        # baseline: recon, stop-grad, proprio
examples/robocasa/run_train_rlt.sh PrepareCoffee --objective progress   # progress objective
examples/robocasa/run_train_rlt.sh PrepareCoffee --no-proprio           # image+language-only token
examples/robocasa/run_train_rlt.sh PrepareCoffee --backbone-grad        # (ablation: known to fail)
examples/robocasa/run_train_rlt.sh PrepareCoffee --help                 # full flag list
```

The switches (each also has an env-var form) map to `Pi0RLTConfig` fields:

| Flag | Config field | What it does | Default |
|---|---|---|---|
| `--objective STR` | `rlt_objective` | `reconstruction` / `progress` / `reconstruction+progress` | reconstruction |
| `--scalar-head` | `rlt_progress_head` | progress head = MSE regression instead of HL-Gauss histogram | distributional |
| `--no-proprio` | `rlt_include_proprio` | drop the proprio token from the bottleneck (paper-faithful; critic must supply proprio) | keep |
| `--backbone-grad` | `rlt_backbone_gradient` | let the RLT loss reshape the VLM backbone | off (readout) |
| `--parallel-decoder` | `rlt_decoder_mode` | decode every token from `z_rl` alone (no teacher forcing) — prevents context-bypass | autoregressive |
| `--loss-weight F` | `rlt_loss_weight` | weight of the RLT loss vs BC | 1.0 |

**How to choose:** start from the default (reconstruction, stop-grad, proprio). Compare objectives
with `--objective`; test the proprio and decoder axes one flag at a time (not a full grid). Judge a
run on **`bc_loss`** (policy quality — every config logs it, comparable to a plain BC run) plus
**`rlt/probe_progress_r2`** and the sim eval curves, *not* on `loss_recon` (a low recon loss can just
mean the decoder is ignoring the token — watch `rlt/bypass_ratio`, which should be ≫1).

Monitoring is logged to wandb (project `acrft`): per-step loss components, a periodic
participation-ratio / bypass-ratio monitor, and an embedding visualization (PCA + t-SNE/UMAP
trajectories with camera-view-on-hover). With `rlt_bc_probe` / `rlt_probe_eval_interval` set (on by
default for the RLT configs), it also runs an in-process headless sim eval of the VLA and the latent
probe every 10k steps. The **RLT metrics guide** on the [dashboard](examples/robocasa/site/index.html)
explains how to read each metric.

## Stage 2 — annotation

`examples/robocasa/annotate_rlt.py` runs the trained VLA over every frame once and writes memmap
arrays (resumable, `--resume`). Notable flags:

| Flag | Meaning |
|---|---|
| `--num-samples N` | base-policy action candidates per frame (default 32) |
| `--stride K` | keep every K-th frame (default 1; the critic needs stride 1) |
| `--dtype` | `float32` (default) / `float16` / `bfloat16` — 16-bit halves the files (6.7 → 3.4 GB/task) |
| `--num-flow-steps` | flow-matching denoising steps for the candidates (default 10) |

## Stage 3 — critic

`scripts/train_rlt_critic.py --data <annot dir> --kind {qc,arq}`. The whole (numeric) dataset is
loaded onto the GPU once; there is no data loader. Key flags: `--num-atoms` (1 = scalar Q, >1 =
HL-Gauss distributional), `--macro-group-size` (ARQ: steps per prefix token), `--num-critics`
(ensemble), plus capacity knobs (`--hidden-dims`, `--num-layers`, …).

## Evaluation

`examples/robocasa/run_eval.sh <Task>` evaluates every checkpoint of a run with N sim rollouts each,
writing `summary.csv` + a plot. Override the config/exp to evaluate an RLT run:

```bash
CONFIG=pi05_robocasa_PrepareCoffee_rlt EXP=PrepareCoffee_rlt examples/robocasa/run_eval.sh PrepareCoffee
uv run examples/robocasa/plot_eval.py --task PrepareCoffee   # success-rate-vs-checkpoint plot
```

---

## Serving a policy

`scripts/serve_policy.py` loads a checkpoint and serves it over a websocket, so a robot (or a
sim client) can send an observation and get an action chunk back. It speaks openpi's protocol,
so any openpi client works unchanged.

`uv run` resolves the project from the working directory, so run it from **this repo** —
from a robot/client checkout it tries to build that project's dependencies instead and fails
with something unrelated (`Failed to build ruckig`).

A backslash must be the LAST character on its line. A trailing space after it escapes the
space instead of the newline, which hands tyro an argument of `' '` — `invalid choice: ' '` —
and then runs the next line as its own command. Copy carefully, or paste the one-line form:

```bash
cd /path/to/ACRFT
uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_yam_lego_taxi_rlt --policy.dir <checkpoint>/100000
```

```bash
cd /path/to/ACRFT

# YAM, relative-joint actions (the default convention)
uv run scripts/serve_policy.py \
    --port 8000 \
    --execute-steps 50 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_h50 \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_bc_s300_h50/200000

uv run scripts/serve_policy.py \
    --port 8000 \
    --execute-steps 30 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_bc_s300_h30/200000

# with a patch critic scoring 8 candidates. --critic takes the critic DIRECTORY; the server
# reads its config to tell a patch critic from an RLT one and to derive adaptive vs bon.
uv run scripts/serve_policy.py \
    --port 8000 \
    --num-samples 8 \
    --critic ~/hf_utils_downloads/acrft-yam-critics/patch_critic_yam_s347_fixed_200k \
    --critic-mode bon \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_bc_s300_h30/200000


uv run scripts/serve_policy.py \
    --num-samples 16 \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_rlt \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_rlt_s300/200000


# same checkpoint family, absolute joint targets
uv run scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_none_rlt \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_none_rlt_s200/100000
```

It is up when the log reads:

```
Serving pi05_yam_lego_taxi_rlt: {'action_horizon': 30, 'supports_multi_sample': True}
Creating server (host: ..., ip: ...)
```

`action_horizon` is read off the train config and sent to the client as metadata, so nobody has
to hard-code the chunk size per robot. A checkpoint trained at 30 served to a client assuming 16
raises nothing — it silently throws away half of every chunk.

### `--policy.asset-id` (usually unnecessary)

Norm stats live inside the checkpoint at `assets/<asset_id>/norm_stats.json`, and `asset_id`
defaults to the data config's `repo_id`. That default does not hold for the data-scaling study:
each point trains on a different subset of episodes and so needs its own stats, which is why
`compute_norm_stats.py` takes `--asset-id` and `train.py` takes the matching
`--data.assets.asset-id`. Nothing in the checkpoint records which name was used, so serving
cannot derive it.

It does not have to. A save writes exactly one `asset_id`, so a checkpoint has exactly one set
of norm stats, and when the config's name does not match, the only one present is the only one
it could have meant. Loading takes it and says so:

```
WARNING No norm stats under asset_id 'jellyho/yam_lego_taxi'; using the only ones this
        checkpoint has, at .../assets/jellyho/yam_lego_taxi_s200.
```

So the commands above work without the flag. Pass it to pin the choice and silence the warning,
or when a checkpoint really does carry several — which takes deliberately saving twice into one
step directory. Then the choice matters, and guessing would quietly normalise with the wrong
statistics rather than fail, so loading refuses and lists them:

```
FileNotFoundError: No norm stats under asset_id 'jellyho/yam_lego_taxi', and this checkpoint
has several to choose from: jellyho/a, jellyho/b. Pass the right one (serving: --policy.asset-id).
```

Only the checkpoint step you serve has to be on disk (~13 GB), not the whole run.

---

## Serving a policy behind the patch-critic

The **patch-critic** is a value function that is independent of the VLA: it scores candidate action
chunks from a grid of frozen DINOv2 patch tokens, so it can be trained and evaluated offline without
ever running the policy. Put it in front of a checkpoint and the server samples N chunks in one
backbone pass and returns the one the critic likes best (`bon`), or that chunk truncated to its
highest-value commitment prefix (`adaptive`).

Checkpoints live in **`jellyho/patch_critic_yam_lego_taxi`**. Bring the critic folder and the base
checkpoint it was trained against — the critic embeds its own copy of the base norm stats, and the
server compares them against the policy it is actually serving, so a mismatched base raises at load
instead of silently degrading.

```bash
cd /path/to/ACRFT

uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("jellyho/patch_critic_yam_lego_taxi", repo_type="model",
                  allow_patterns="fixed_pi05_s347/*", local_dir="/data5/jellyho/critics/yam")
PY

srun -p debug --gres=gpu:L40S:1 --cpus-per-task=8 --mem=64G -t 08:00:00 \
  bash -lc 'cd /path/to/ACRFT && XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-sync python scripts/serve_policy.py \
      --port 8000 --num-samples 8 \
      --critic /data5/jellyho/critics/yam/fixed_pi05_s347 \
      --critic-mode bon \
      policy:checkpoint \
      --policy.config pi05_yam_lego_taxi_rlt \
      --policy.dir checkpoints/pi05_yam_lego_taxi_rlt/yam_lego_taxi_rlt_s300_successonly/280000'
```

```bash
uv run scripts/serve_policy.py \
    --port 8000 \
    --num-samples 8 \
    --critic ~/hf_utils_downloads/patch_critic_yam_lego_taxi/fixed_pi05_s347 \
    --critic-mode bon \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_bc_s300_h30/200000

uv run scripts/serve_policy.py \
    --port 8000 \
    --num-samples 8 \
    --critic ~/hf_utils_downloads/patch_critic_yam_lego_taxi/fixed_pi05_s347 \
    --critic-mode bon \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_alphaflow \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_alphaflow/200000
```

`--num-steps` sets the denoising iterations per chunk; unset leaves the model's own default
(alphaflow 1, since it is trained on mean velocity; pi05 10). It is charged per replan, so it
compounds with `--execute-steps`:

```bash
# commit 5 steps at a time instead of 30, and pay one denoising step per replan
uv run scripts/serve_policy.py --port 8000 --execute-steps 5 --num-steps 1 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_alphaflow \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_alphaflow/200000
```

Or build the wrapper in-process, which is the fastest way to check a new critic loads and infers:

```python
from openpi.policies import policy_config
from openpi.policies.patch_critic_policy import PatchCriticSelectPolicy
from openpi.training import config as _config

policy = policy_config.create_trained_policy(
    _config.get_config("pi05_yam_lego_taxi_rlt"), "<base checkpoint>")
wrapped = PatchCriticSelectPolicy(
    policy, "/data5/jellyho/critics/yam/fixed_pi05_s347", mode="bon", default_samples=8)

out = wrapped.infer(obs)   # 3 camera images + observation/state + prompt
out["actions"]             # (30, 14)  selected chunk
out["critic_scores"]       # (30, N)   value of each candidate
```

`--mode adaptive` needs a critic with **more than one commitment prefix**, so it does not work with a
`macro_group_size = 30` checkpoint such as `fixed_pi05_s347`.

**Before trusting any number from this path**, read
[`docs/deploy_yam_patch_critic.md`](docs/deploy_yam_patch_critic.md) — it lists what has and has not
been validated. In particular the published critics have not yet been evaluated on the robot, and the
`*_s347` (raw-units) folders were served through a path that scored the wrong action space, so numbers
from those are void.

---

## Results so far (PrepareCoffee, 50 trials/checkpoint)

| Variant | best success | takeaway |
|---|---|---|
| BC (full finetune) | 62% @ 50k | baseline |
| **RLT (stop-grad)** | **78% @ 60k** | ≈ BC through 40k, ahead at the peak — the token does not hurt the policy |
| RLT + backbone-grad | 22% (mostly <10%) | letting the RLT loss into the backbone **breaks** the policy |

So the RLT bottleneck can be learned jointly with BC at no policy cost (stop-gradient), and
backbone-gradient is a clear negative. Critic and deployment results are in progress.

---

## Repo layout (what's new here)

```
src/openpi/models/pi0_rlt.py             Pi0RLT: RL-token bottleneck, progress head, latent BC probe
src/openpi/training/progress.py per-episode time-to-success progress labels
src/openpi/rlt_critic/critic.py          QC / ARQ critics (scalar + HL-Gauss), ensemble
src/openpi/patch_critic/critic.py        VLA-independent patch critic (frozen DINOv2 + per-prefix ARQ)
src/openpi/patch_critic/preproc.py       shares the base VLA's state/action preprocessing
src/openpi/patch_critic/spec.py          the critic's input contract, validated at serve time
src/openpi/policies/patch_critic_policy.py  best-of-N / adaptive-chunk serving wrapper
scripts/cache_patch_features.py          precompute frozen DINOv2 features once (~4x faster training)
scripts/train_patch_critic_cached.py     patch-critic training from that cache
scripts/score_critic_cached.py           success-vs-failure AUC + deep-atom diagnostics
docs/deploy_yam_patch_critic.md          deploying the critic: contract, verification, limitations
scripts/train_rlt_critic.py              Stage 3 critic training (GPU-resident, lax.scan)
scripts/train.py                         Stage 1 training loop (+ RLT monitors, probe sim eval)
examples/robocasa/run_train_rlt.sh       Stage 1 launcher with ablation flags
examples/robocasa/annotate_rlt.py        Stage 2 annotation
examples/robocasa/rollout.py             shared RoboCasa rollout utils (main.py + inline eval)
examples/robocasa/                       RoboCasa data prep, eval, dashboard  (see its README)
docs/openpi_upstream.md                  the original openpi README (install, base models, PyTorch)
```

---

## Upstream openpi

This fork keeps openpi's models (π₀ / π₀-FAST / π₀.₅), training/serving infra, and PyTorch support
intact. For base-model checkpoints, generic fine-tuning, inference servers, Docker, multi-GPU, and
precision settings, see **[`docs/openpi_upstream.md`](docs/openpi_upstream.md)**.
