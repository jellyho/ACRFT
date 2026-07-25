"""Task-progress labels for LeRobot datasets, derived from a sparse success signal.

Dataset-agnostic: nothing here knows about a particular robot or task suite. A dataset qualifies if
it is a LeRobot v3 dataset; the success signal is whichever column ``reward_key`` names (default
``next.reward``, the LeRobot convention). Demos typically carry 0 for most of the episode and flip to
1 once the task is satisfied (staying 1 for a short tail, with ``next.done`` marking the final
frame). We take the FIRST frame where it fires as the success moment - a real success signal, unlike
"end of episode", which also covers the trailing frames after the task is already done (and, in a
couple of episodes, a success that is briefly lost and regained).

Datasets with no such column still work: every episode then treats its final frame as the goal, which
is the usual assumption for demonstration data that is successful by construction.

Progress measures "how far through the task are we", normalized PER EPISODE so that every demo runs
0 at its start to 1 at its success:

    progress = clip(frame_index / onset_frame, 0, 1)     # 0 at start, 1 at/after success

Per-episode (not global) normalization is deliberate. The same task takes a different number of
frames in different demos (operator speed, hesitation), so an ABSOLUTE frames-to-success target
would give two visually-identical "just started, nothing done" frames different progress just
because one demo happened to be longer -- which is nonsensical for a task-progress signal, and also
injects label noise (near-identical inputs, contradictory targets). Dividing each episode by its own
time-to-success makes "half done" mean the same thing regardless of that demo's duration.

This is intentionally NOT a value proxy: the discounted return-to-go (and its discount) belongs to
the downstream critic, which is tuned separately. Here progress only shapes the token to encode task
phase. (A ``mode="global"`` fallback keeps the old absolute-time definition available.)

Only the per-episode onsets are needed at train time (a few hundred ints), so the reward column is
scanned once and cached next to the dataset.
"""

import dataclasses
import logging
import pathlib

import numpy as np

import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

# Percentile of per-episode time-to-success used as the normalization horizon.
_HORIZON_PERCENTILE = 99.0


@dataclasses.dataclass(frozen=True)
class ProgressLabels:
    """Per-episode success onsets + the (per_episode | global) normalization mode."""

    onsets: np.ndarray  # [num_episodes] int32, first frame where the reward fires
    delta_max: float  # only used by mode="global": dataset-wide time-to-success horizon
    mode: str = "per_episode"  # "per_episode": frame/onset in [0,1]; "global": 1 - delta/delta_max

    def progress(self, episode_index, frame_index) -> np.ndarray:
        onset = self.onsets[np.asarray(episode_index, dtype=np.int64)]
        frame = np.asarray(frame_index, dtype=np.int64)
        if self.mode == "per_episode":
            # 0 at the episode's start, 1 at/after its own success moment.
            return np.clip(frame / np.maximum(onset, 1), 0.0, 1.0).astype(np.float32)
        if self.mode == "global":
            delta = np.maximum(0, onset - frame)
            return (1.0 - np.minimum(delta, self.delta_max) / self.delta_max).astype(np.float32)
        raise ValueError(f"unknown progress mode {self.mode!r}")


def read_reward_column(root: pathlib.Path, total_frames: int, reward_key: str = "next.reward") -> np.ndarray | None:
    """Read the (index, ``reward_key``) columns out of the LeRobot v3 parquet files.

    Returns None if the dataset has no such column, so callers can fall back to "episode end is the
    goal" instead of failing on datasets that carry no reward.
    """
    import pyarrow.dataset as pads

    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot v3 parquet files under {root}/data/chunk-*/file-*.parquet")
    dataset = pads.dataset(files, format="parquet")
    if reward_key not in dataset.schema.names:
        return None
    table = dataset.to_table(columns=["index", reward_key])
    reward = np.zeros(total_frames, dtype=np.float32)
    reward[table["index"].to_numpy()] = table[reward_key].to_numpy()
    return reward


def compute_progress_labels(
    repo_id: str, *, reward_key: str = "next.reward", mode: str = "per_episode", use_cache: bool = True
) -> ProgressLabels:
    """Success onsets for every episode of ``repo_id`` (cached next to the dataset).

    Works for any LeRobot v3 dataset: ``reward_key`` names the sparse success column, and a dataset
    without it falls back to treating each episode's final frame as the goal.
    """
    from lerobot.datasets import lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    root = pathlib.Path(meta.root)
    cache = root / "meta" / "openpi_progress_onsets.npz"
    if use_cache and cache.exists():
        with np.load(cache) as d:
            if len(d["onsets"]) == meta.total_episodes:
                return ProgressLabels(onsets=d["onsets"], delta_max=float(d["delta_max"]), mode=mode)

    reward = read_reward_column(root, meta.total_frames, reward_key)
    lo = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    hi = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)

    onsets = np.empty(len(lo), dtype=np.int32)
    n_no_success = 0
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        fired = np.flatnonzero(reward[a:b]) if reward is not None else []
        if len(fired):
            onsets[i] = int(fired[0])
        else:
            # No success signal for this episode: treat the last frame as the goal so the label stays
            # defined (also the whole-dataset fallback when there is no reward column at all).
            onsets[i] = int(b - a - 1)
            n_no_success += 1
    if reward is None:
        logger.info(f"No '{reward_key}' column in {repo_id}; using each episode's final frame as the goal.")

    delta_max = float(max(1.0, np.percentile(onsets, _HORIZON_PERCENTILE)))
    logger.info(
        f"Progress labels for {repo_id}: {len(onsets)} episodes "
        f"({n_no_success} without a success reward), time-to-success "
        f"min={onsets.min()} p50={int(np.percentile(onsets, 50))} max={onsets.max()}, "
        f"horizon delta_max={delta_max:.0f} frames"
    )

    if use_cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, onsets=onsets, delta_max=delta_max)
        except OSError as e:  # a read-only dataset dir is fine, we just recompute next time
            logger.warning(f"Could not cache progress labels to {cache}: {e}")

    return ProgressLabels(onsets=onsets, delta_max=delta_max, mode=mode)


@dataclasses.dataclass(frozen=True)
class AddProgress(_transforms.DataTransformFn):
    """Adds a scalar ``progress`` label to a raw LeRobot item, from its episode/frame index.

    Runs before the repack transform, while the raw LeRobot keys are still present. ``progress`` is a
    TRAINING-only label, so this is a no-op at inference/rollout time, where the raw item carries no
    ``episode_index``/``frame_index`` (the same transform chain is reused for serving).
    """

    labels: ProgressLabels

    def __call__(self, data: dict) -> dict:
        if "episode_index" not in data or "frame_index" not in data:
            return data
        return {
            **data,
            "progress": self.labels.progress(data["episode_index"], data["frame_index"]),
        }
