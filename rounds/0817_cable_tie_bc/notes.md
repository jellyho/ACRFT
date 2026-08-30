# 0817_cable_tie_bc — plain BC finetune of pi0.5 on YAM cable-tie teleop

Append-only log. Reports get rewritten; this file does not.

## 2026-08-17 — round opened

**Goal.** A plain behaviour-cloning pi0.5 policy on the cable-tie YAM teleop set,
with no RLT bottleneck. This is both the deployable policy and the baseline the
RLT runs are measured against.

**Why a new config.** The repo had YAM registered only as `pi05_yam_lego_taxi_rlt`
and `pi05_yam_lego_taxi_none_rlt`. There was no plain-BC YAM config, so
`pi05_yam_cable_tie` was added (`src/openpi/training/config.py:1380`). No existing
config was modified.

**Data.** `/NHNHOME/WORKSPACE/gwanwoo/rl_specialist/datasets/lerobot_cable_tie_100_clean`,
prepared by a separate session from `lerobot_cable_tie_100`. 100 success episodes
of 105, each cut at the first `observation.control_mode == 4` frame (the operator's
homing motion, where their hand enters the cameras). 157,023 frames of the source
171,378.

Verified independently before training, not taken on trust:

- no `control_mode == 4` rows remain; global `index` contiguous; `frame_index`
  restarts at 0 per episode; 100 episodes.
- the schema matches `YAMInputs`/`YAMOutputs` with no code change.
- the joint-delta transform reproduces `raw_action - state_ref` numerically on
  six sampled dimensions, grippers left absolute.
- the last retained agentview frame of episodes 0/25/50/99 shows no hand
  (`figs/` not kept — checked visually at build time).

**Two discrepancies found in the clean dataset's own README**, cosmetic, reported
to the data owner: it states the mode-4 run is 57–114 frames (measured: 56–125)
and that 0–2 mode-0 frames trail it (measured: exactly 1, in all 105 episodes).
The cut is value-driven so neither affects the result.

**Decision: action chunks are left padded at the episode tail.** Cutting at the
mode-4 onset without an extra `action_horizon - 1` margin means the last 29
samples of each episode request actions past the new episode end. LeRobot clamps
and flags them via `action_is_pad`; openpi has no reference to that key, so those
targets train as if real. That is 2,900 of 157,023 samples (1.85%), and the
repeated target is "hold at the completion pose" — benign for BC, and strictly
better than the homing motion it replaced. Left as is deliberately.

**Not checked.** The agentview mid-file frame-drop bug from
`docs/yam_dataset_issues.md` was not re-verified on this dataset;
`scan_yam_video_alignment.py` was not run. The clean dataset's README asserts no
encoder frame loss. If the trained policy misbehaves, suspect this first.

**Launch.** step 0 at 2026-08-17 18:58, seed 42, commit — config change was
uncommitted in the working tree at launch (`src/openpi/training/config.py`, +42
lines); this needs a hash recorded before the run is cited.

    HF_LEROBOT_HOME=/NHNHOME/WORKSPACE/gwanwoo/rl_specialist/cache/huggingface/lerobot \
    XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 uv run python scripts/train.py pi05_yam_cable_tie \
        --exp-name=cable_tie_bc --project-name cable-tie-bc --num-workers 8

wandb: https://wandb.ai/RSS-PFT_RLLAB/cable-tie-bc/runs/l1gyxtbg
checkpoints: `checkpoints/pi05_yam_cable_tie/cable_tie_bc/<step>`, 42 GB each,
all retained (`keep_period=5000` default vs `save_interval=10000`).

**Open question put to the data owner:** run to 100k (~30 h) or evaluate the 30k
checkpoint first. Unanswered at time of writing; the run continues either way.

## 2026-08-18 03:07 — stop at 50k

Supersedes the "run to 100k" plan in the entry above. The data owner decided 50k
is enough; the run was left going rather than relaunched.

**Why stopping early is clean here.** The schedule is
`CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5)`
— peak equals decay, so the learning rate is *constant* at 5e-5 after warmup.
There is no unfinished decay ramp, and the weights at step 50k are what a run
configured for 50k would have produced. This would not hold for a genuinely
decaying schedule.

50k steps at batch 32 over 157,023 frames is 10.2 epochs.

**Mechanism.** `stop_at_50k.sh`, launched detached (`setsid`, session leader
624403) against training pid 579727. It waits for the final `50000` checkpoint
directory — Orbax renames from `.orbax-checkpoint-tmp-*` only after the async
save finalizes — then takes a 180 s grace period and sends SIGTERM so wandb can
flush, escalating to SIGKILL only if the process ignores it. It also exits
harmlessly if training dies on its own first. Progress is appended to
`stop_at_50k.log`.

Loss at the decision point: step 25k, `bc_loss` 0.0055, improvement per 5k steps
down to 0.0007 from 0.0015 at 10–15k.

## 2026-08-18 10:07 — run finished at 50k

Training received SIGTERM from the watchdog at 10:06:57 and exited with 143, three
minutes after the 50k checkpoint finalized at 10:03:50. GPU released (0 MiB). No
straggler processes.

Five checkpoints retained, 42 GB each, 208 GB total:
10000, 20000, 30000, 40000, 50000.

BC loss by step (training-batch scalar on dataset states and dataset actions —
there is no validation split in this path, so it cannot show overfitting):

    0      0.0818
    10000  0.0088
    20000  0.0062
    25000  0.0055
    30000  0.0048
    35000  0.0048
    40000  0.0042
    45000  0.0040
    50000  0.0041

The flat reading at 30k–35k was noise at 5k sampling, not convergence; the loss
resumed falling to 0.0040 at 45k and is flat-to-noisy over the last 10k steps.

**Recommendation carried into evaluation:** roll out both the 30k and the 50k
checkpoint. Their training losses differ by 0.0007, which cannot distinguish them,
and with 100 demos over 10 epochs and no validation split, the later checkpoint is
not automatically the better policy.

**Checkpoint 50000 verified deployable.** Loaded via `create_trained_policy` and
run on one training frame: output `(30, 14) float32`, max |prediction − current
joint angle| 0.031 rad (small, so the served policy does return ABSOLUTE joint
targets — the runtime must not add state back), max |prediction − demonstrated
action| 0.011 rad. In-distribution, so this checks the plumbing, not generalization.

**Gotcha found while verifying:** `create_trained_policy` does not attach the data
config's repack transform (`policy_config.py:45` defaults to an empty Group), so
the served policy expects the repacked key namespace, not LeRobot column names:
`observation/image` (agentview), `observation/wrist_image` (left),
`observation/image_right` (right), `observation/state`, `prompt`. The left/right
wrist keys follow different naming conventions, so swapping them is easy and fails
silently. Added to the report as its own deployment section.
