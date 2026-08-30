# ACRFT — adaptive action chunking on top of π₀.₅

A research fork of [openpi](https://github.com/Physical-Intelligence/openpi) for **deciding how far
ahead to commit**. A VLA emits an action chunk and the robot flies all of it; committing less means
reacting sooner but replanning more, and the right length is not a constant — it depends on the
state. So we fit an offline value critic over action chunks and let it choose, per replan, which of
N candidate chunks to execute and how much of it to commit to.

The robot is a bimanual **YAM** (real hardware, three cameras); the policy is a plain π₀.₅ BC
finetune; the critic is a small transformer over **frozen DINOv2 patch features**, trained with IQL
on the same teleop set the policy was finetuned on — successes *and* failures.

> New to the repo? Read [The pipeline in one page](#the-pipeline-in-one-page), then jump to
> [Quickstart](#quickstart). openpi's own model/setup docs are preserved in
> [`docs/openpi_upstream.md`](docs/openpi_upstream.md).

> **The RL-token (RLT) pipeline this repo started as is legacy.** The RLT bottleneck existed to
> feed a critic a compact state vector; the patch critic reads frozen DINOv2 instead and needs
> nothing from inside the VLA, so the base policy went back to being an ordinary BC finetune. The
> code and docs are still here — see [Legacy: the RLT pipeline](#legacy-the-rlt-pipeline) — and
> RoboCasa remains as a sim environment, but neither is where the work is.

---

## The pipeline in one page

```
   Stage 0     π₀.₅ base checkpoint            gs://openpi-assets/checkpoints/pi05_base
                          │  BC finetune on YAM teleop (no bottleneck, no auxiliary loss)
                          ▼
   Stage 1     Base policy                     scripts/train.py --config pi05_yam_lego_taxi
   ─────────   An ordinary π₀.₅ flow-matching policy: horizon 30, relative joint actions. It is
               never modified again -- everything downstream reads it, nothing writes to it.
                          │  the demos, plus their success/fail labels (outcomes.jsonl)
                          ▼
   Stage 2     Critic data                     scripts/convert_yam_to_patchcritic.py
   ─────────   3 cameras at 224, state[42], action[14], sparse terminal reward from the labels.
               FAILURES ARE THE POINT: cost-to-goal needs the negatives, and they are the scarce
               half (s347 = 300 success / 47 fail).
                          │  frozen DINOv2, run once per frame
                          ▼
               Feature cache                   scripts/cache_patch_features.py
   ─────────   The backbone never trains, so its output never changes -- caching the pooled tokens
               makes critic training ~20-40x faster and byte-identical.
                          │
                          ▼
   Stage 3     Critic training                 scripts/train_patch_critic_cached.py
   ─────────   IQL, cost-to-goal reward, HL-Gauss distributional head. Fixed (one value per whole
               chunk) or ARQ (one per macro-group prefix) -- the second is what makes the
               commitment adaptive.
                          │
                          ▼
   Stage 4     Deployment                      scripts/serve_policy.py --critic
   ─────────   N candidates from ONE backbone pass, scored in the critic's own units. best-of-N
               picks the chunk; adaptive also picks how much of it to commit before replanning.
                          │
                          ▼
   Stage 5     Analysis                        misc/yam-misc {render-bulk, stats, plots}
   ─────────   What the run did, in video and in numbers: commitment lengths, the critic's margin
               over the runner-up, and the discontinuity at each replan boundary.
```

Key design choices, and where they live:

| Idea | Where | One-liner |
|---|---|---|
| Critic reads pixels, not the VLA | `patch_critic/backbone.py` | frozen DINOv2 patches, so the critic is independent of the policy it scores |
| Per-prefix value (ARQ) | `patch_critic/critic.py` | one value per macro-group prefix → the commitment length is a choice, not a constant |
| Cost-to-goal reward | `train_patch_critic_cached.py` | −1/step to an absorbing goal; values are comparable across states without a reward model |
| N candidates, one backbone pass | `Pi0.sample_n_actions` | the VLM prefix runs once and its KV cache is tiled across the noise draws |
| Scored in the critic's units | `policies/patch_critic_policy.py` | candidates are decoded to joint targets and re-normalized with the critic's own stats, so policy and critic need not share them |
| Verdict at the handle | (robot repo) | every rollout is labelled success/fail on the arm, which is what `outcomes.jsonl` is |

---

## Quickstart

### Install (same as openpi)

```bash
git clone --recurse-submodules <this repo>            # or: git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv sync --group eval                                  # sim deps (robosuite/mujoco), for the RoboCasa path
```

Requires an NVIDIA GPU with ≥70 GB for full fine-tuning (π₀.₅ is 3B); L40S/48 GB is enough for
inference, serving and critic training. See
[`docs/openpi_upstream.md`](docs/openpi_upstream.md#requirements) for the full matrix.

### Data

YAM bimanual teleop as LeRobot v3.0, with an `outcomes.jsonl` sidecar giving each episode a
success/fail label — written by the robot repo's recorder as the operator ends each rollout at the
handle. `success_only` training resolves its episode list from that file; the critic uses both
halves.

RoboCasa 365 demos (for the sim path) are in
[examples/robocasa/README.md](examples/robocasa/README.md).

### Run the pipeline

```bash
# Stage 1 — BC finetune (node with ≥70 GB VRAM)
uv run scripts/train.py pi05_yam_lego_taxi --exp-name yam_bc

# Stage 2 — critic data, then the frozen-feature cache
uv run python scripts/convert_yam_to_patchcritic.py \
    --repo-id jellyho/yam_lego_taxi --root ~/lerobot_data \
    --outcomes ~/lerobot_data/yam_lego_taxi/outcomes.jsonl \
    --max-frames 160000 --out ~/pc_rollouts/lego_taxi
uv run python scripts/cache_patch_features.py --data ~/pc_rollouts/lego_taxi --out ~/pc_cache/yam_s347

# Stage 3 — critic (any GPU; the features are a memmap, the model is a small transformer)
uv run python scripts/train_patch_critic_cached.py --cache ~/pc_cache/yam_s347 \
    --macro-group-size 5 --reward-scheme cost_to_goal      # 5 → adaptive; 30 → fixed/best-of-N

# Stage 4 — serve it (see "Serving a policy behind the patch-critic")
uv run scripts/serve_policy.py --port 8000 --num-samples 8 \
    --critic ~/hf_utils_downloads/acrft-yam-critics/patch_critic_yam_s347_g5_200k \
    policy:checkpoint --policy.config pi05_yam_lego_taxi --policy.dir <ckpt>

# Stage 5 — look at what happened
misc/yam-misc stats --root ~/lerobot_data --repo-id <run> --per-episode
```

---

## Policy-extraction arms

Ten ways to turn the frozen BC policy plus the frozen critic into a better policy — best-of-N,
IDQL's implicit policy, QPILOTS test-time steering, latent actors, and the weight-only arms
(AWR / CFGRL / DQL / QAM / FlowDPG / FQL-X). They all serve through the one entry point; the
command for each is in [docs/extraction_arms.md](docs/extraction_arms.md), along with the three
things that are easy to get wrong.

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

## Legacy: the RLT pipeline

Everything from here to [Serving a policy](#serving-a-policy) is the **RL-token** pipeline the repo
started as: a bottleneck trained jointly with BC produces a compact `z_rl`, an annotation pass
freezes it into flat arrays, and a critic trains on those. It is superseded by the patch critic —
which needs nothing from inside the VLA, so the base policy no longer carries a bottleneck at all —
but the code runs and the RoboCasa results below are what motivated the switch.

### Stage 1 — RLT training

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

### Stage 2 — annotation

`examples/robocasa/annotate_rlt.py` runs the trained VLA over every frame once and writes memmap
arrays (resumable, `--resume`). Notable flags:

| Flag | Meaning |
|---|---|
| `--num-samples N` | base-policy action candidates per frame (default 32) |
| `--stride K` | keep every K-th frame (default 1; the critic needs stride 1) |
| `--dtype` | `float32` (default) / `float16` / `bfloat16` — 16-bit halves the files (6.7 → 3.4 GB/task) |
| `--num-flow-steps` | flow-matching denoising steps for the candidates (default 10) |

### Stage 3 — RLT critic

`scripts/train_rlt_critic.py --data <annot dir> --kind {qc,arq}`. The whole (numeric) dataset is
loaded onto the GPU once; there is no data loader. Key flags: `--num-atoms` (1 = scalar Q, >1 =
HL-Gauss distributional), `--macro-group-size` (ARQ: steps per prefix token), `--num-critics`
(ensemble), plus capacity knobs (`--hidden-dims`, `--num-layers`, …).

### Evaluation (RoboCasa sim)

`examples/robocasa/run_eval.sh <Task>` evaluates every checkpoint of a run with N sim rollouts each,
writing `summary.csv` + a plot. Override the config/exp to evaluate an RLT run:

```bash
CONFIG=pi05_robocasa_PrepareCoffee_rlt EXP=PrepareCoffee_rlt examples/robocasa/run_eval.sh PrepareCoffee
uv run examples/robocasa/plot_eval.py --task PrepareCoffee   # success-rate-vs-checkpoint plot
```

### RoboCasa results (PrepareCoffee, 50 trials/checkpoint)

| Variant | best success | takeaway |
|---|---|---|
| BC (full finetune) | 62% @ 50k | baseline |
| **RLT (stop-grad)** | **78% @ 60k** | ≈ BC through 40k, ahead at the peak — the token does not hurt the policy |
| RLT + backbone-grad | 22% (mostly <10%) | letting the RLT loss into the backbone **breaks** the policy |

So the RLT bottleneck can be learned jointly with BC at no policy cost (stop-gradient), and
backbone-gradient is a clear negative — the token was free, but it was also not necessary. That is
what sent the critic to frozen DINOv2 patches instead: the same per-prefix value, with no claim on
the policy's internals and no retraining of the VLA when the critic changes.

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

# adaptive commitment: the critic carves the chunk into macro groups and commits to the
# highest-value prefix, so each reply is as long as it is still worth executing.
uv run scripts/serve_policy.py \
    --port 8000 \
    --num-samples 8 \
    --critic ~/hf_utils_downloads/acrft-yam-critics/patch_critic_yam_s347_g5_200k \
    --critic-mode adaptive \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi \
    --policy.dir ~/hf_utils_downloads/pi05_yam_lego_taxi_bc_s300_h30/200000
```

**The critic and the policy need not share norm stats.** Candidates are decoded to absolute joint
targets and re-normalized with the critic's own statistics, so any pairing is scored in the units
the critic learned. A mismatch logs a warning rather than refusing to load, and the warning is
worth reading for what it actually says: the critic was fitted against a different base policy, so
its values are a claim about transfer. The units are not in question.

The critic's horizon need not match either — a shorter critic scores the first C steps of a longer
proposal, exactly (the joint delta at step k is taken against the same base state whatever the
chunk length), and best-of-N then commits only what was scored.

```bash


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

## Repo layout (what's new here)

**The critic**
```
src/openpi/patch_critic/critic.py        VLA-independent patch critic (frozen DINOv2 + per-prefix ARQ)
src/openpi/patch_critic/backbone.py      the frozen DINOv2 feature extractor
src/openpi/patch_critic/preproc.py       shares the base VLA's state/action preprocessing
src/openpi/patch_critic/spec.py          the critic's input contract, validated at serve time
scripts/convert_yam_to_patchcritic.py    LeRobot demos + outcomes.jsonl -> per-step transitions
scripts/cache_patch_features.py          precompute the frozen features once (~20-40x faster training)
scripts/train_patch_critic_cached.py     patch-critic training from that cache
scripts/score_critic_cached.py           success-vs-failure AUC + deep-atom diagnostics
docs/deploy_yam_patch_critic.md          deploying the critic: contract, verification, limitations
```

**Serving**
```
src/openpi/policies/patch_critic_policy.py  best-of-N / adaptive-chunk serving wrapper
src/openpi/policies/policy.py            MultiSample / TruncateChunk wrappers, the robot-side probe
src/openpi/models/pi0.py                 sample_n_actions: N candidates from ONE prefix pass
scripts/serve_policy.py                  one server for plain, multi-sample and critic-guided serving
```

**Looking at what happened**
```
misc/                                    self-contained: render, bulk render, stats, figures
misc/rollout_stats.py                    commitment lengths, latency, the critic's margin, the splice
misc/stats_plots.py                      those numbers as figures, in the house plot style
```

**Legacy (RLT / RoboCasa)**
```
src/openpi/models/pi0_rlt.py             Pi0RLT: RL-token bottleneck, progress head, latent BC probe
src/openpi/rlt_critic/critic.py          QC / ARQ critics (scalar + HL-Gauss), ensemble
src/openpi/training/progress.py          per-episode progress labels; also resolves success_only
scripts/train_rlt_critic.py              RLT critic training (GPU-resident, lax.scan)
examples/robocasa/run_train_rlt.sh       RLT launcher with ablation flags
examples/robocasa/annotate_rlt.py        RLT annotation
examples/robocasa/                       RoboCasa data prep, eval, dashboard  (see its README)
```

```
scripts/train.py                         the training loop, shared by every config
docs/openpi_upstream.md                  the original openpi README (install, base models, PyTorch)
```

---

## Upstream openpi

This fork keeps openpi's models (π₀ / π₀-FAST / π₀.₅), training/serving infra, and PyTorch support
intact. For base-model checkpoints, generic fine-tuning, inference servers, Docker, multi-GPU, and
precision settings, see **[`docs/openpi_upstream.md`](docs/openpi_upstream.md)**.
