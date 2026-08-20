# `misc/` — rollout rendering

Draw what a deployed policy actually did: the candidate chunks it considered, which one the critic
picked, how far ahead it committed, and the value it was scored on — overlaid on the three cameras
of a recorded episode.

Self-contained. Nothing here imports the robot repo (see [What it carries](#what-it-carries)).

## Open the GUI

```bash
misc/yam-misc render-gui
```

Pick a dataset, pick an episode, press **Render**.

```
root      [~/lerobot_rollout                     ] [ … ]
dataset   [yam_s300_rel_200k_g5               ▾]
episode   [episode 3 · 1580 frames (53s)      ▾]
overlay   [samples ▾]
options   speed [4.0]  panel height [360]  ☑ critic value strip  ☑ chunk length strip
output    [~/yam_s300_rel_200k_g5_ep3.mp4         ]

6 episode(s) at 30 fps · 8 candidates · critic scores · ADAPTIVE (full candidates recorded) ·
replan boundaries from the run

                         [ Render ]
```

The line under the form is read off the dataset, not typed in: how many candidates it holds,
whether a critic scored them, whether the run was adaptive, and whether it recorded its own replan
boundaries. Those are the numbers that are painful to get wrong from the command line — a mistyped
candidate count is a reshape error, and a mistyped horizon *silently* draws every chunk in the
wrong place.

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
QT_QPA_PLATFORM=offscreen PYTHONPATH=$PWD python -m pytest misc/ -q
```

These cover the parts that are pure functions of a recording: the value ramp, the commitment
arithmetic, the chunk series, the strip painter, and what the GUI hands the renderer. The
end-to-end path is checked against real recordings instead — most recently by rendering the same
episode from this copy and from the original in the robot repo and diffing the frames (identical,
max \|difference\| 0).
