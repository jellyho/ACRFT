# `jellyho/yam_lego_taxi` — two bugs in the LeRobot export

Found while training on the dataset. Both are on the **writing** side, both are silent until a
consumer hits them, and both are cheap to fix at the source. Neither is a training-code problem.

Measured on the copy at `jellyho/yam_lego_taxi` (LeRobot v3.0, 119 episodes, 363,423 frames, 30 fps,
3 cameras). Everything below is reproducible with `scripts/scan_yam_video_alignment.py`.

---

## Bug 1 — the `agentview` stream drops frames *mid-file*, silently misaligning later episodes

Frame counts per video file, decoded with torchcodec:

| file | wrist_left | wrist_right | **agentview** | missing |
|---|---|---|---|---|
| file-001.mp4 | 64300 | 64300 | **64299** | −1 |
| file-004.mp4 | 6499 | 6499 | **6498** | −1 |
| file-006.mp4 | 47066 | 47066 | **47064** | −2 |
| file-007.mp4 | 15823 | 15823 | **15822** | −1 |

The other 8 files agree exactly across all three cameras. So the parquet says an episode has N frames
and two cameras deliver N, but `agentview` delivers N−1 or N−2.

**The visible symptom is a crash.** Training dies mid-run when the sampler reaches the last episode
of one of those files:

```
RuntimeError: Invalid frame index=47064 for streamIndex=0 numFrames=47064
```

**The invisible symptom is worse, and it is the real problem.** The frames are *not* lost at the end
of the file. Episodes are concatenated into one mp4, so each join is a hard visual cut; comparing
where that cut actually lands against where `from_timestamp` predicts gives the stream's accumulated
frame offset at every boundary in the file. For file-006:

| episode | 50–54 | 55–58 | 59–66 |
|---|---|---|---|
| agentview offset | 0 | **−1** | **−2** |

(cut sharpness 11–29× the local baseline at every boundary, so these are unambiguous; the wrist
streams measure 0 everywhere, as expected from their exact frame counts.)

One frame was lost during episode 54 and another during episode 58. **Every episode after a drop had
its agentview video silently shifted against its own states and actions** — 12 of file-006's 17
episodes — and nothing anywhere reported it. The crash only happens because the *last* episode of the
file runs off the end; had the file ended on a boundary, there would have been no crash at all and
the misalignment would have gone straight into training.

For files 001, 004 and 007 every boundary measures 0 and the shortfall is 1, so their single drop is
inside the file's last episode (22, 41 and 71 respectively).

**Cause.** Since drops occur in the middle of the stream, an unflushed encoder is ruled out — that
can only ever lose frames at the tail. What is left is frames being dropped on the way in:

1. **ffmpeg dropping frames it considers duplicates.** With the default `-fps_mode cfr` / `-vsync 1`,
   input frames whose PTS collide after rounding to the output timebase are dropped, anywhere in the
   stream, and the remaining frames are renumbered onto a dense grid — which is exactly what the
   files show (`ffprobe` finds no PTS hole; the timeline is contiguous 0…N−1).
2. **Frame grabs being missed at capture** and the writer simply never seeing them. Same outcome if
   the writer trusts its own frame counter rather than the camera's.

`agentview` is the only stream affected, so whatever it is, it is specific to that camera's path —
plausibly a slightly different or less stable capture rate than the wrist cameras.

**How to fix**

- Assign explicit, monotonic PTS rather than letting the encoder infer them from wall-clock:
  `frame.pts = i` with `stream.time_base = Fraction(1, fps)`, and encode with `-fps_mode passthrough`
  (`-vsync 0`) so nothing is dropped or duplicated.
- Flush the encoder before closing the container (`stream.encode(None)`). Not the cause here, but
  it is the other way to lose frames and costs one line.
- **Add a post-write assertion.** This is the important one — it turns a silent corruption into a
  loud failure at write time:

  ```python
  from torchcodec.decoders import VideoDecoder
  for key in video_keys:
      n = VideoDecoder(str(path_for(key))).metadata.num_frames
      expected = sum(ep_length for ep in episodes_in(path_for(key)))
      assert n == expected, f"{key}: encoded {n} frames, parquet says {expected}"
  ```

  Every camera in every file, every time. It costs seconds and would have caught this before upload.
- Worth knowing: a count check alone is necessary but not sufficient. A stream that drops one frame
  and duplicates another has the right total and is misaligned in between. If the capture path can do
  that, log the camera's own frame counter per episode and assert against it too.

---

## Bug 2 — `timestamp` is stored as `float32`, which breaks the loader past ~840 s

`meta/info.json` declares:

```json
"timestamp": {"dtype": "float32", "shape": [1], "names": null}
```

LeRobot looks up video frames by timestamp and checks the match against `tolerance_s`, default
`1e-4` s. But float32 resolution degrades with magnitude:

| t | float32 ulp |
|---|---|
| 688 s | 6.1e-05 |
| **1207 s** | **1.22e-04** |

Past **~840 s the representation error alone exceeds the default tolerance**, so the check cannot
pass no matter how good the capture was. Episodes are concatenated into multi-hour files here, and
`from_timestamp` offsets reach ~2100 s — well past the limit.

**Symptom.** Every consumer using default settings aborts before training starts:

```
FrameTimestampError: One or several query timestamps unexpectedly violate the tolerance
(tensor([0.0001]) > tolerance_s=0.0001)
queried timestamps: tensor([1207.2334])
loaded timestamps:  tensor([1207.2333])
```

Note the two values differ in the last digit only — this is *pure storage precision*, not a capture
problem. Measured against the nominal grid, the actual timing is excellent:

```
timestamp - frame_index/fps :  max|dev| = 7.1e-06 s  (0.0002 frame periods)
```

So the data is fine; only the dtype is wrong. It is easy to misread this error as "our timestamps
are jittery" and start chasing the capture pipeline — they are not.

**How to fix**

- Write `timestamp` as **`float64`**. At 2100 s a float64 ulp is ~2e-13, eight orders of magnitude
  inside the tolerance. This is a one-line dtype change in the feature spec and costs 4 bytes/frame
  (~1.5 MB across this dataset).
- Consider also storing `timestamp = frame_index / fps` exactly, and keeping the raw capture clock in
  a separate column (e.g. `capture_time`) if it is needed for analysis. The lookup grid is
  `frame_index / fps` anyway, so deriving it removes a whole class of drift questions — and the raw
  clock stays available for diagnosing dropped frames.

**One caveat that is not yours.** float64 storage does not fully close this, because LeRobot itself
does the comparison in float32: `decode_video_frames_torchcodec` builds `torch.tensor(timestamps)`
from a Python list, which defaults to `torch.float32`. Two float64 values a picosecond apart can
round to adjacent float32 values 1.22e-4 apart and fail a 1e-4 tolerance. After converting this
dataset to float64 exactly 2 of ~600 sampled frames still tripped it. Consumers need a tolerance of a
few float32 ulp at their largest timestamp (openpi now computes this in
`data_loader._float32_safe_tolerance`). Worth an upstream issue against LeRobot; nothing to change on
the writing side beyond the dtype.

---

## Summary of the asks

1. Fixed-PTS, `passthrough` encoding (plus a flush) so no frame is dropped or duplicated anywhere in
   the stream — not just at the end.
2. Assert `num_frames == expected` per camera per file at write time, and fail loudly.
3. Store `timestamp` as float64 (ideally as the exact `frame_index / fps` grid).

(1) and (2) require a re-export of at least files 001/004/006/007; (3) requires rewriting the parquet
metadata but not the videos.

---

## What we did locally, in the meantime

`scripts/repair_yam_dataset.py` fixes the downloaded copy in place, without re-encoding video:

- **Bug 1** — each episode's `from_timestamp`/`to_timestamp` is shifted by the frame offset measured
  at its own end, which realigns it with the stream as encoded. 16 (episode, camera) pairs shifted.
  Re-running the boundary scan afterwards measures 0 offset everywhere except episodes 54 and 58,
  which contain a drop internally and therefore keep a residual of exactly 1 frame (33 ms) over the
  part before the splice. The five episodes with such a residual are 22, 41, 54, 58 and 71.
- **Bug 2** — `timestamp` rewritten as float64 on the exact `frame_index / fps` grid (max change from
  the stored value: 7.1e-06 s), and `info.json` updated to match.

Originals are kept next to each rewritten file as `*.orig`. Verified afterwards: all 119 episodes
load — head, middle and last five frames of each — through openpi's dataset path with no decode
failures and no tolerance violations. This is a local patch, not a substitute for a correct
re-export: it cannot recover the frames that were never encoded.
