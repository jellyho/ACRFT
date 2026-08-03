# `jellyho/yam_lego_taxi` — two bugs in the LeRobot export

Found while training on the dataset. Both are in the **writing** side, both are silent until a
consumer hits them, and both are cheap to fix at the source. Neither is a training-code problem.

Measured on the copy at `jellyho/yam_lego_taxi` (LeRobot v3.0, 119 episodes, 363,423 frames, 30 fps,
3 cameras).

---

## Bug 1 — the `agentview` video is short by 1–2 frames in 4 of the 12 files

Frame counts per video file, decoded with torchcodec:

| file | wrist_left | wrist_right | **agentview** | missing |
|---|---|---|---|---|
| file-001.mp4 | 64300 | 64300 | **64299** | −1 |
| file-004.mp4 | 6499 | 6499 | **6498** | −1 |
| file-006.mp4 | 47066 | 47066 | **47064** | −2 |
| file-007.mp4 | 15823 | 15823 | **15822** | −1 |

The other 8 files agree exactly across all three cameras. So the parquet says an episode has N frames
and two cameras deliver N, but `agentview` delivers N−1 or N−2.

**Symptom.** Training crashes mid-run when the sampler reaches one of those episodes:

```
RuntimeError: Invalid frame index=47064 for streamIndex=0 numFrames=47064
```

That is episode 66 (the last in file-006) asking for the frame one past the end. Episodes 41 and 66
are unloadable; the dataset is unusable for training without a workaround.

**Good news on severity.** The lost frames are at the *end* of each file, not the middle: only the
last episode of each short file overshoots, so per-episode video/state alignment elsewhere is intact.
Had frames been dropped mid-file, every subsequent episode's video would be silently offset — same
crash-free appearance, corrupted data. Worth making sure the fix does not just paper over that case.

**Likely cause.** `agentview` is the only stream affected, so it is specific to that camera's
encoding path. The two usual suspects:

1. **The encoder is not flushed before the container is closed.** In PyAV this is the missing
   `for packet in stream.encode(None): container.mux(packet)` before `container.close()`; buffered
   B-frames at the end of the file are silently discarded. The 1–2 frame magnitude matches a
   lookahead buffer almost exactly.
2. **ffmpeg dropping frames it considers duplicates.** With the default `-fps_mode cfr` / `-vsync 1`,
   input frames whose PTS collide after rounding to the output timebase get dropped. If agentview is
   captured at a slightly different or less stable rate than the wrist cameras, this hits it alone.

**How to fix**

- Assign explicit, monotonic PTS rather than letting the encoder infer them from wall-clock:
  `frame.pts = i` with `stream.time_base = Fraction(1, fps)`, and encode with `-fps_mode passthrough`
  (`-vsync 0`) so nothing is dropped or duplicated.
- Flush the encoder: `stream.encode(None)` (or `ffmpeg` process wait-to-EOF) **before** closing.
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
timestamp - frame_index/fps :  max|dev| = 7e-06 s  (0.0002 frame periods)
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

---

## Summary of the asks

1. Flush + fixed-PTS encoding so every camera's frame count matches the parquet.
2. Assert `num_frames == expected` per camera per file at write time, and fail loudly.
3. Store `timestamp` as float64 (ideally as the exact `frame_index / fps` grid).

(1) and (2) require a re-export of at least files 001/004/006/007; (3) requires rewriting the parquet
metadata but not the videos.
