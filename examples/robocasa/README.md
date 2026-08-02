# RoboCasa 365 — target-split human demos → LeRobot v3.0

This example downloads the **human demonstrations for the 50 RoboCasa 365 `target` tasks** and
converts them to **LeRobot dataset format v3.0** (`CODEBASE_VERSION = "v3.0"`, the format openpi
now uses via `lerobot==0.4.4`).

RoboCasa is vendored as a git submodule at [`third_party/robocasa`](../../third_party/robocasa)
(branch `robocasa365_release`). The set of target tasks is read from its registry; the actual
download paths are discovered from the HF repo, so the script is robust to path/date drift.

> **Task count:** the registry lists 51 tasks with a `target` split, but only **50** are
> published in the HF dataset (`DessertAssembly` has a registry entry but no uploaded data).
> The script processes the 50 that exist and warns about the missing one.

## End-to-end pipeline (reproducible)

Every step is a committed, idempotent script — run them in order (details in the sections below):

```bash
git submodule update --init third_party/robocasa
GIT_LFS_SKIP_SMUDGE=1 uv sync                          # openpi training venv

# 1. Data: download the 50 target/human demos and convert to LeRobot v3.0 (optionally push to Hub)
uv run examples/robocasa/prepare_robocasa365.py --output-dir /data5/jellyho/robocasa365

# 2. Train a task (computes SHARED norm stats across all tasks on first run, then trains)
examples/robocasa/run_train.sh PrepareCoffee           # -> checkpoints/pi05_robocasa_PrepareCoffee/...

# 3. Eval: one-time sim-env setup (deps + ~10 GB assets), then sweep every checkpoint
examples/robocasa/setup_eval_env.sh
examples/robocasa/run_eval.sh PrepareCoffee            # -> eval/.../summary.csv + videos
```

## Which HF dataset?

RoboCasa's own `download_datasets.py` points at `nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos`,
which is **not published** (HF returns `RepoNotFound`). The live, **public** mirror is
[`nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos)
— that is what this script uses. (It also drops the `v1.0/` path prefix the registry uses.)

## Why no MuJoCo replay?

That HF dataset already hosts **pre-converted LeRobot datasets** (`<path>/lerobot.tar`).
RoboCasa produced them with `lerobot==0.3.3`, which writes the **v2.1** format. So the pipeline
is just:

```
download lerobot.tar (v2.1)  →  extract  →  upgrade v2.1 → v3.0
```

No `robosuite` / `mujoco` / sim environment is required, and you do **not** need the heavy
`convert_hdf5_lerobot.py` replay path.

## Prerequisites

1. Initialize the submodule (registry source):
   ```bash
   git submodule update --init third_party/robocasa
   ```
2. Install openpi (provides both `huggingface_hub` and `lerobot==0.4.4`):
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   ```

The dataset is **public** — no Hugging Face login or terms acceptance is required. (If a token
is configured it will simply be used.)

## Usage

Everything runs inside the openpi venv (no separate environment needed):

```bash
# All 50 published target tasks -> <output-dir>/<Task> as LeRobot v3.0 datasets:
uv run examples/robocasa/prepare_robocasa365.py --output-dir /data5/jellyho/robocasa365

# A subset (task names must be target-split tasks):
uv run examples/robocasa/prepare_robocasa365.py --output-dir /data/rc365 \
    --tasks OpenDrawer CloseFridge PickPlaceCounterToCabinet

# Smoke test on the first task only:
uv run examples/robocasa/prepare_robocasa365.py --output-dir /data/rc365 --limit 1
```

Useful flags:

| flag | effect |
|------|--------|
| `--download-only` | download + extract, skip the v3.0 upgrade |
| `--convert-only`  | skip download, only run the v3.0 upgrade on already-extracted datasets |
| `--overwrite`     | redownload/extract even if `<output-dir>/<Task>` exists |
| `--keep-backup`   | keep the pre-upgrade v2.1 copy (`<Task>_old`); removed by default to save disk |
| `--limit N`       | only process the first N tasks |

Each task is processed independently; failures are logged and reported at the end without
aborting the whole run. **Resume is automatic:** a task already converted to v3.0 on disk is
skipped entirely (no redownload, no reconversion) unless `--overwrite` is given.

## Push to the Hugging Face Hub (optional)

Add `--push-to-hub` to upload each converted v3.0 dataset and gather them into a HF collection.
Because the push pass runs *after* the (resume-aware) convert loop, you can run it on an
already-finished output dir — conversions are skipped and only the upload happens:

```bash
# Upload the 50 datasets (public) and create a collection, after conversion is done:
uv run examples/robocasa/prepare_robocasa365.py --output-dir /data5/jellyho/robocasa365 --push-to-hub
```

- One dataset repo per task: **`<hf-user>/<prefix>-<Task>`** (e.g. `jellyho/robocasa365-CloseBlenderLid`).
  Each repo is tagged `v3.0` so `LeRobotDataset(repo_id)` resolves it.
- A proper **LeRobot dataset card** (README) is generated for each repo from `meta/info.json`.
- A collection titled *"RoboCasa 365 Target (Human) — LeRobot v3.0"* is created and all repos added.

To (re)generate just the dataset cards on already-uploaded repos without re-uploading data:

```bash
uv run examples/robocasa/prepare_robocasa365.py --output-dir /data5/jellyho/robocasa365 --push-cards-only
```

Requires a HF token with **write** access (`huggingface-cli login`, or `HF_TOKEN`).

| flag | effect |
|------|--------|
| `--push-to-hub`        | enable the upload + collection pass |
| `--hf-user NAME`       | target user/org (default: the logged-in user) |
| `--repo-prefix PREFIX` | repo name prefix (default `robocasa365`) |
| `--private`            | create repos and collection as private (default: public) |
| `--collection-title T` | collection title |
| `--no-collection`      | upload datasets but skip collection creation |

Uploading is video-backed and large (tens–hundreds of GB total); expect it to take a while.

## Dataset schema (metadata)

Each converted task is a **LeRobot v3.0** dataset (`CODEBASE_VERSION = "v3.0"`), robot
`PandaOmron` (mobile manipulator), **20 fps**, ~**500 human demos** per task (target split).

**Cameras** — three `256×256` RGB video streams:

| key | view |
|-----|------|
| `observation.images.robot0_agentview_left`  | exterior (third-person) |
| `observation.images.robot0_agentview_right` | exterior (second) |
| `observation.images.robot0_eye_in_hand`     | wrist |

**`observation.state`** — 16-d float:

| idx | field |
|-----|-------|
| 0:3   | base position |
| 3:7   | base rotation (quaternion) |
| 7:10  | end-effector position (relative) |
| 10:14 | end-effector rotation (relative) |
| 14:16 | gripper qpos |

**`action`** — 12-d float (delta end-effector + absolute gripper):

| idx | field |
|-----|-------|
| 0:4   | base motion |
| 4:5   | control mode (discrete) |
| 5:8   | end-effector position |
| 8:11  | end-effector rotation |
| 11:12 | gripper (close) |

**Other fields:** `next.reward`, `next.done`, `annotation.human.task_description` /
`annotation.human.task_name` (int indices), plus the standard LeRobot bookkeeping
(`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`). The natural-language
instruction lives in `meta/tasks.parquet` (e.g. *"Close the lid blender by securely placing the
lid on top."*) and is surfaced as the `prompt` when `prompt_from_task=True`.

## Training

The model-input mapping lives in
[`src/openpi/policies/robocasa_policy.py`](../../src/openpi/policies/robocasa_policy.py)
(3 cameras → `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`; 16-d state and 12-d action
passed through and padded to the model action dim). π0.5 configs are registered in
[`src/openpi/training/config.py`](../../src/openpi/training/config.py):

- **`pi05_robocasa`** — the default (currently PrepareCoffee), pi05, constant `5e-5` after a 1k
  warmup, batch 32, 100k steps, checkpoint every 10k.
- **`pi05_robocasa_<Task>`** — one config per target task (same recipe), so norm-stats and training
  can be run for any task by config name.

### Easy per-task runs

```bash
# One task (computes norm stats, then trains):
examples/robocasa/run_train.sh PrepareCoffee

# Several tasks, sequentially:
examples/robocasa/run_train.sh PrepareCoffee OpenDrawer TurnOnMicrowave
```

The script symlinks the local converted datasets (`/data5/jellyho/robocasa365/<Task>`) into the
LeRobot cache under the Hub id, so training reuses them instead of re-downloading from the Hub.

**Normalization is shared across all tasks.** `compute_shared_norm_stats.py` computes one
mean/std/quantile set over every task and writes it into each per-task config's asset dir. This is
required: per-task stats are ill-conditioned for near-stationary tasks (e.g. WashLettuce, where
base motion / control-mode are ~constant → a `q99-q01` range of ~0 that amplifies rare values
~1e6× and blows up the loss). RoboCasa is one robot with one action space, so a shared range is
correct. The script runs once on the first task (writes all 50) and is auto-skipped afterward
(`FORCE_NORM_STATS=1` to recompute).
Env overrides: `EXP_SUFFIX` (exp name suffix), `SKIP_NORM_STATS=1` (always skip),
`FORCE_NORM_STATS=1` (recompute), `ROBOCASA_LOCAL_DIR`, `HF_USER`.
Checkpoints land in `checkpoints/pi05_robocasa_<Task>/<Task>_<EXP_SUFFIX>/`.

Equivalently, by hand:

```bash
uv run scripts/compute_norm_stats.py --config-name=pi05_robocasa_PrepareCoffee
uv run scripts/train.py pi05_robocasa_PrepareCoffee --exp-name=PrepareCoffee_run
```

### Monitoring action-distribution collapse

pi05 is a flow-matching policy, so for a fixed observation it maps different noise samples to
different action chunks. Training logs (to wandb) how much those samples spread — a shrinking
spread means the policy is collapsing to a single action per state (overfitting):

- `action_dist/sample_std` — mean spread of `K` sampled action chunks for the same observations
  (the key signal; watch for it trending toward 0).
- `action_dist/data_std` — spread of the ground-truth actions across the batch (scale reference).
- `action_dist/sample_to_data_ratio` — the two above, as a ratio.

Enabled in the RoboCasa configs via `action_dist_interval` (default every 1000 steps, `K` =
`action_dist_num_samples` = 32; sampled sequentially so cost/memory stay bounded). Set
`action_dist_interval=0` to disable.

Notes:
- `action_horizon` (default 10) is the predicted chunk length at 20 fps — tune per task.
- RoboCasa actions are already deltas, so no delta conversion is applied. If yours are absolute,
  set `extra_delta_transform=True` on the data config.
- Training uses a **single** LeRobot dataset (one task) per run. Multi-task training would need a
  merged dataset — see the roadmap.

## Inference / evaluation in sim

Rollouts use openpi's server/client split: a **policy server** (openpi training venv) and a
**RoboCasa eval client** ([`main.py`](main.py)) that drives the sim. Keep them in separate venvs
— robosuite pins numpy<2, which conflicts with the LeRobot v3.0 / numpy≥2 training stack.

```bash
# 1) Serve a trained checkpoint (openpi venv):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_robocasa --policy.dir /path/to/checkpoint

# 2) Roll out in sim (a venv with robocasa + robosuite + openpi-client installed):
python examples/robocasa/main.py --task PickPlaceCounterToCabinet --host <server-host> \
    --num-trials 10 --video-dir /tmp/robocasa_eval
```

The client builds the model input from live env observations (three `256²` cameras, the 16-d
`observation.state` from the robot proprio keys, and the task's language instruction), reorders the
returned 12-d action from LeRobot order into the env's expected order, and steps the env until
success or the task horizon. It reports a per-task success rate and can save rollout videos.

> Note: the client flips camera frames vertically to match the (upright) training data — if
> rollouts look wrong, verify orientation against a dataset frame. RoboCasa's sim env
> (robosuite + MuJoCo + kitchen assets) must be installed in the client venv.

### Sweep every checkpoint

[`run_eval.sh`](run_eval.sh) evaluates all checkpoints of a run — for each one it starts a
server, runs `NUM_TRIALS` rollouts, stops the server, and finally writes a success-rate summary:

```bash
# client venv must have robosuite + robocasa + openpi-client:
EVAL_PYTHON=/path/to/robocasa-venv/bin/python \
    examples/robocasa/run_eval.sh PrepareCoffee
```

Per checkpoint it writes `eval/<config>/<exp>/<step>/`:
`results.json` (success count/rate + per-trial), `videos/` (up to `NUM_VIDEOS` rollout mp4s),
and `server.log` / `client.log`. A combined `eval/<config>/<exp>/summary.csv`
(`step,successes,num_trials,success_rate`) is written at the end.

Env overrides: `NUM_TRIALS=50`, `NUM_VIDEOS=5`, `PORT=8000`, `EXP_SUFFIX=run`, `SEED=0`,
`STEPS="10000 20000"` (subset), `OUT_DIR`. The server runs via `uv run` (openpi venv); only the
client uses `EVAL_PYTHON`.

Plot the sweep (`summary.csv` -> `summary.png`, success rate vs checkpoint):

```bash
uv run examples/robocasa/plot_eval.py --task PrepareCoffee
```

**Client env + assets** — one reproducible, idempotent setup script does everything (installs the
sim deps into the openpi `.venv` and downloads all assets):

```bash
examples/robocasa/setup_eval_env.sh
```

What it sets up, and why (all captured so it reproduces cleanly):
- **`uv sync --group eval`** — the sim rollout runs from the same `.venv`; the `eval` group adds
  `mujoco==3.3.1`, `h5py`, pins `numpy==2.2.5` (robocasa asserts this exact version), and pins
  `robosuite` to its **`robocasa-dev`** branch commit (RoboCasa's kitchen env needs
  `load_model_on_init` / `get_elements`, absent from the PyPI release and master). A uv
  `override-dependencies` entry forces `numpy==2.2.5` past robosuite's `mink` dep (`numpy<2`, only
  used for whole-body IK, unused here). `robocasa` and `openpi_client` are added via `PYTHONPATH`
  (not installed); `run_eval.sh` defaults `EVAL_PYTHON` to `.venv/bin/python`.
- **Kitchen assets (~10 GB)** — the official `download_kitchen_assets.py` for textures/objaverse/
  aigen, plus [`download_lightwheel_assets.py`](download_lightwheel_assets.py) for the lightwheel
  objects/fixtures (NVIDIA renamed + restructured that HF repo into per-object zips, so the official
  script 404s — this fetches the same assets from the new layout).

Headless rendering uses `MUJOCO_GL=egl` (set by `run_eval.sh`).

## Task overview / dashboard

A browsable dashboard covers all 50 target tasks (thumbnail + full-trajectory video, category,
horizon, description, single/multi language-instruction badge, and on-disk status) plus project
status, roadmap, and the dataset schema. It's published via **GitHub Pages** from
`examples/robocasa/site/` (see `.github/workflows/pages.yml`) — enable *Settings → Pages → Source:
GitHub Actions* once, and it deploys to `https://<user>.github.io/<repo>/`.

Rebuild it after adding tasks or updating project status:

```bash
# 1) Cut each task's episode-0 video (exact boundaries from the LeRobot episode metadata)
#    and sample a poster frame:
uv run examples/robocasa/make_previews.py --output-dir /data5/jellyho/robocasa365
# (optional) refresh the single/multi instruction counts:
uv run examples/robocasa/make_previews.py --instruction-counts

# 2) Build the site page (default mode):
uv run examples/robocasa/gen_dashboard.py --output-dir /data5/jellyho/robocasa365

# 3) Preview locally, or commit site/ and push to redeploy Pages:
cd examples/robocasa/site && python -m http.server 8000   # → http://localhost:8000/
```

The page embeds poster thumbnails inline and references `site/videos/<Task>.mp4` for playback;
both `site/index.html` and the videos (~150 KB–1 MB each) are committed so Pages can serve them.

> `gen_dashboard.py --mode artifact` can also emit a single self-contained HTML file (thumbnails
> embedded, no videos) if you ever want a portable one-file version.

## Output layout

```
<output-dir>/
  CloseBlenderLid/         # LeRobot v3.0 dataset (meta/, data/, videos/)
  CloseFridge/
  ...
  ArrangeTea/
```

Feature schema (from RoboCasa's converter): three video streams
(`observation.images.robot0_{eye_in_hand,agentview_left,agentview_right}`),
`observation.state` (16-d), `action` (12-d), plus `next.reward`/`next.done` and
`annotation.human.*` indices.

## Notes / caveats

- **Disk**: the `target` split is 500 human demos per task (video-backed). Downloading all 50
  tasks is on the order of tens–hundreds of GB. The v3.0 upgrade temporarily needs extra space
  for the `<Task>_old` backup (auto-removed unless `--keep-backup`). Point `--output-dir` at a
  large volume.
- **Training integration**: to train on these datasets you still need a RoboCasa `DataConfig`
  (repack/normalization transforms mapping the schema above into openpi model inputs). That is
  not included here — ask if you want it added to `src/openpi/training/config.py`.
- The `target`-split human demos are the evaluation/fine-tuning set. The much larger `pretrain`
  split (and MimicGen sources) can be fetched by extending this script or using
  `third_party/robocasa/robocasa/scripts/download_datasets.py`.

## YAM bimanual (jellyho/yam_lego_taxi)

Pi0RLT on the YAM two-arm dataset. Defaults are the RoboCasa-best RLT variant (parallel decoder,
no-proprio) with a joint-space delta action; no sim, so no rollout/probe eval and no probe actor.
Three configs differ only in the action delta convention:

    pi05_yam_lego_taxi_rlt        joint delta (subtract current joint position; grippers absolute) [default]
    pi05_yam_lego_taxi_none_rlt   absolute joint targets
    pi05_yam_lego_taxi_umi_rlt    frame-relative EEF (UMI), for pose-action data

Train (norm stats are computed on first run, no shared-asset file needed):

    uv run scripts/compute_norm_stats.py pi05_yam_lego_taxi_rlt
    XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 uv run scripts/train.py pi05_yam_lego_taxi_rlt --exp-name=yam_joint

The action is 14-d joint space (per arm: 6 joints + 1 gripper); success is episode-level
(outcomes.jsonl), so there is no per-frame reward and the progress objective is off.
