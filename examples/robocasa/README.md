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

A π0.5 config is wired up in [`src/openpi/training/config.py`](../../src/openpi/training/config.py):
`pi05_robocasa` (full fine-tune) and `pi05_robocasa_low_mem_finetune` (LoRA). The mapping from the
schema above into model inputs lives in
[`src/openpi/policies/robocasa_policy.py`](../../src/openpi/policies/robocasa_policy.py)
(3 cameras → `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`; 16-d state and 12-d action
passed through and padded to the model action dim).

Pick the dataset via `repo_id`:

```bash
# Option A — train from a pushed Hub dataset:
#   set data.repo_id="<user>/robocasa365-<Task>" in the config, then:
uv run scripts/compute_norm_stats.py --config-name=pi05_robocasa
uv run scripts/train.py pi05_robocasa --exp-name=my_run

# Option B — train from the local converted dir (no upload):
export HF_LEROBOT_HOME=/data5/jellyho/robocasa365
#   set data.repo_id="<Task>" (bare task name) in the config, then run the two commands above.
```

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

## Task overview / dashboard

For a browsable overview of all 50 target tasks (thumbnail, category, horizon, description, and
on-disk conversion status) plus project status and roadmap, generate the dashboard page:

```bash
uv run examples/robocasa/gen_dashboard.py \
    --output-dir /data5/jellyho/robocasa365 \
    --out examples/robocasa/dashboard.html
```

It re-scans the dataset dir and embeds any thumbnails found under `examples/robocasa/assets/`
(one `<Task>.jpg` per task, sampled from the converted agent-view videos). Re-run it to refresh
as more tasks finish converting or as the project evolves.

### Two modes

- **`--mode artifact`** (default) → `dashboard.html`: fully self-contained (thumbnails embedded as
  data URIs, no external files). This is what gets published as a claude.ai artifact.
- **`--mode site`** → `site/index.html`: references `site/videos/<Task>.mp4` so you can **play the
  full episode-0 trajectory** for each task. Full videos are too large to embed in a self-contained
  artifact, so this mode is meant to be served locally (or via GitHub Pages).

Generate the full-trajectory videos first, then build and serve the site:

```bash
# Cut each task's episode-0 video (exact boundaries from the LeRobot episode metadata):
uv run examples/robocasa/make_previews.py --output-dir /data5/jellyho/robocasa365

# Build the site page and serve it (videos play in-browser with controls):
uv run examples/robocasa/gen_dashboard.py --mode site
cd examples/robocasa/site && python -m http.server 8000   # → http://localhost:8000/
```

The videos (~150 KB–1 MB each, `site/videos/`) are **git-ignored** — regenerate them locally with
`make_previews.py`. To publish the site on GitHub Pages, commit `site/` (including videos, or host
them elsewhere) and enable Pages for the directory.

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
