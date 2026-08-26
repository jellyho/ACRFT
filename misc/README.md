# `misc/` — rollout rendering

Draw what a deployed policy actually did: the candidate chunks it considered, which one the critic
picked, how far ahead it committed, and the value it was scored on — overlaid on the three cameras
of a recorded episode.

Self-contained. Nothing here imports the robot repo (see [What it carries](#what-it-carries)).

## Open the GUI

```bash
misc/yam-misc render-gui
```

Pick a dataset, pick an episode, press **Render**. The picker lists folders that hold a LeRobot
dataset (`meta/info.json`), so the renderer's own `*_renders` output folders stay out of it.

```
root      [~/lerobot_data                        ] [ … ]
dataset   [yam_s300_rel_200k_g5               ▾]
episode   [── all episodes · render every one, then zip ──  ▾]
overlay   [samples ▾]
options   speed [4.0]  panel height [360]  ☑ critic value strip  ☑ chunk length strip
output    [~/yam_s300_rel_200k_g5_renders         ]

6 episode(s) at 30 fps · 8 candidates · critic scores · ADAPTIVE (full candidates recorded) ·
replan boundaries from the run

                    [ Render ]  [ Stats ]
```

The line under the form is read off the dataset, not typed in: how many candidates it holds,
whether a critic scored them, whether the run was adaptive, and whether it recorded its own replan
boundaries. Those are the numbers that are painful to get wrong from the command line — a mistyped
candidate count is a reshape error, and a mistyped horizon *silently* draws every chunk in the
wrong place.

The first entry in the episode list renders **every** episode and zips the folder — the same
batch the command line runs (see [A whole dataset at once](#a-whole-dataset-at-once)), so it skips
what is already rendered, survives an episode that cannot be drawn, and names the ones that failed.
The progress bar then spans the whole batch rather than sweeping 0–100 once per episode.

Rendering runs on a worker thread, so the window stays live and a failure lands in the window
rather than a terminal you have closed.

## Or from the command line

```bash
misc/yam-misc render-samples \
    --repo-id lerobot_rollout/yam_s300_rel_200k_g5 \
    --root ~/lerobot_rollout \
    --episode 3 --out ~/rollout.mp4
```

`--root` is the **parent** of the dataset folder: the dataset is `<root>/<last segment of
--repo-id>`. Get it wrong and LeRobot decides the dataset must be on the Hub and fails with a 404
rather than "no such directory".

Everything else is optional. `--candidates` and `--horizon` are read from the recording unless you
pass them; `--source action` draws the single executed trajectory instead of the fan, which works
on any LeRobot recording (teleop, demo, replay).

## A whole dataset at once

```bash
misc/yam-misc render-bulk \
    --repo-id lerobot_rollout/yam_s300_rel_200k_g5 \
    --root ~/lerobot_rollout
```

Renders every episode to `~/<dataset>_renders/<dataset>_ep000.mp4`, then zips the folder next to
it. Takes every option `render-samples` does, plus:

| | |
|---|---|
| `--episodes` | `all` (default), `3`, `0-9`, `0,3,5-7`. An out-of-range index is an error, not a trim -- asking for `0-49` of a 20-episode run means one of the numbers is wrong, and quietly rendering 20 hides which. |
| `--out-dir` | where the mp4s go (default `~/<dataset>_renders`) |
| `--zip` / `--no-zip` | zip path (default `<out-dir>.zip`), or skip it |
| `--overwrite` | re-render episodes whose mp4 already exists |

Two things a long batch depends on: **a failing episode does not end it** (episode 17 recording no
candidates is not a reason to lose 18..40 -- failures are collected and reported at the end, and
the exit code is non-zero), and **a killed batch resumes** (existing renders are skipped, so
re-running after a Ctrl-C picks up where it stopped).

The zip is stored rather than deflated: the payload is h264 in mp4, already compressed, so
deflating spends CPU on every byte to save roughly none.

## What a run did, in numbers

```bash
misc/yam-misc stats --root ~/lerobot_data --repo-id run_a run_b run_c
misc/yam-misc stats --root ~/lerobot_data --all --json ~/stats.json
```

```
dataset                     eps  sec      replans   chunk len    range  replan/s     infer p50  spread         jump@bnd       jump@in
yam_s300_rel_200k_fixed     5    55 ± 36  56 ± 36   29.8 ± 0.3   8-30   1.01 ± 0.01  164 ± 15   18.42 ± 15.96  0.405 ± 0.154  0.071 ± 0.023
yam_s300_rel_200k_g5        6    36 ± 17  94 ± 32   11.1 ± 2.9   5-30   2.92 ± 1.04  157 ± 1     8.67 ± 4.50   0.205 ± 0.087  0.066 ± 0.019
```

Several runs side by side is the point: that is how a critic arm gets compared to another without
watching forty videos. Above, the fixed critic commits 29.8 steps on average (one group, so it can
only ever choose the whole chunk) while the adaptive one commits 11.1 and replans three times as
often — and pays for it at the splices, where its boundary jumps are smaller but far more numerous.

The GUI's **Stats** button runs the same summary for the selected dataset (every episode, whatever
the episode row says — a summary of one episode is mostly its own row repeated), and adds the
per-episode rows and the commitment histogram that `--per-episode` prints.

Every number is recomputed from the recording. Aggregates are **episode-level means with a 95%
t-CI**, never a mean over pooled frames — pooling weights long episodes more, and its spread
describes frames rather than runs. A single episode reports no interval instead of zero.

| | |
|---|---|
| `chunk len`, `range`, `replan/s` | how far ahead each reply committed, from `policy.chunk_index` — the same boundaries the renderer draws |
| `infer p50/p95`, `delay ticks` | from `policy.infer_ms` / `policy.delay_ticks` |
| `spread`, `adv`, `pick#0` | best-minus-worst candidate value per replan, best minus runner-up, and how often the first sample won. A near-zero spread means the critic saw nothing to choose between |
| `jump@bnd`, `jump@in` | 95th pct of the largest single-joint step across a replan boundary vs inside a chunk. The boundary number means nothing alone — a splice artefact is a boundary step much larger than an ordinary one |

Columns no dataset has are dropped, so a table of teleop recordings carries no empty critic
columns. `--json` writes the full per-episode numbers.

## What the frame shows

```
[ wrist_left ] [ agentview ] [ wrist_right ]
[     critic value — picked (line) vs candidate spread (band)     ]
[     chunk length — steps the reply carried                      ]
```

- **The fan** is every candidate chunk, projected through forward kinematics. Candidates are
  coloured by the critic's own ranking, cold (rejected) to warm (preferred), normalised per replan
  — the absolute values are arbitrary (cost-to-goal runs to −2777); what matters at each decision
  is which of *these* it preferred.
- **The highlighted path** is the candidate that was executed, from `critic_choice`. Small white
  dots mark the macro-group boundaries: the granularity an adaptive commitment can stop at.
- **A faint tail** on that path is the part the model proposed and the critic declined to commit
  to. Only adaptive runs have one; in `bon` the executed chunk *is* the whole proposal.
- **The header** carries the decision in numbers:
  `critic: #3 of 8  Q -1500.22  (best by +1.15, spread 3.11)`. A near-zero spread means the critic
  saw nothing to choose between, which is worth seeing.
- **The strips** share the cameras' time axis, so they read against each other and against the
  footage. Both are drawn only when the recording has them.

## What it carries

The robot repo owns the robot. What this tool needs from it is copied in and pinned:

| | |
|---|---|
| `models/yam_kinematics.xml` | The YAM + linear-4310 kinematic tree — bodies, joints and sites, with 7.4 MB of visual/collision meshes stripped. Forward kinematics never reads a mesh: verified bit-identical (max \|difference\| **0.0** over 200 random configurations, across the whole `WristCameraGeometry` API). |
| `agentview_extrinsics.yaml` | The rig's calibrated `base_T_agentview` per arm, from the board-on-gripper solve. |
| `viz/` | Forward kinematics and camera projection. |
| `dataset_reader.py` | LeRobot access: metadata first, one episode loaded lazily. |

**Both data files are snapshots.** Re-copy them after a recalibration or a change to the arm model,
or the overlay will keep projecting through the old geometry — silently, since a wrong extrinsic
still draws a plausible-looking fan.

## Environment

Runs in the workstation conda env (`yam_ws` by default, `YAM_WS_ENV` to override), which has
PyQt5 / lerobot / mujoco / mink. ACRFT's own `.venv` does not have PyQt5 or mink.

## Tests

```bash
conda activate yam_ws   # NOT `uv run`: ACRFT's .venv has no mink or PyQt5, and those
                        # two modules are importorskip'd, so the suite goes quiet rather
                        # than red -- 9 of 29 tests would run and still report success.
QT_QPA_PLATFORM=offscreen PYTHONPATH=$PWD python -m pytest misc/ -q
```

These cover the parts that are pure functions of a recording: the value ramp, the commitment
arithmetic, the chunk series, the strip painter, and what the GUI hands the renderer. The
end-to-end path is checked against real recordings instead — most recently by rendering the same
episode from this copy and from the original in the robot repo and diffing the frames (identical,
max \|difference\| 0).
