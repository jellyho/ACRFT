"""Reusable critic-training data loader, built on openpi's LeRobot stack (pyav decode, delta_timestamps).

A distributional / IQL patch critic needs, per transition: the current observation, the H-step action
chunk, and the FUTURE observations at the bootstrap prefixes {g, 2g, ..., H}. Loading a per-transition
16-frame window would re-decode every frame ~16x (it is the "current" of one transition and a "future"
of ~15 others). Instead we load overlapping CLIPS of ``clip_len`` contiguous frames -- each frame
decoded ONCE by LeRobot+pyav -- and explode each clip into ``clip_len - horizon`` transitions
downstream, so a frame is decoded ~``clip_len / (clip_len - horizon)``x (~1.5-2x, not 16x).

This reuses openpi's verified decode path: the same ``LeRobotDataset`` construction as
``openpi.training.data_loader.create_torch_dataset`` (pyav video backend so no FFmpeg shared libs are
needed on compute nodes; float32-safe timestamp tolerance), only with a clip-shaped ``delta_timestamps``
and strided per-episode clip starts. Wrap it in a plain ``torch.utils.data.DataLoader`` (num_workers)
exactly like openpi does.

    ds = CriticClipDataset(repo_id="jellyho/yam_lego_taxi", root=..., episodes=labeled,
                           image_keys=[...], state_key="observation.state", action_key="action",
                           horizon=30, clip_len=60, stride=30, outcomes={...})
    dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, shuffle=True,
                                     collate_fn=ds.collate, drop_last=True)
    for clip in dl:  # clip["images"]: [B, clip_len, ncam, 3, H0, W0]; clip["action"]: [B, clip_len+H, ad]; ...
        ...
"""

from __future__ import annotations

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

from openpi.training import data_loader as _dl


class CriticClipDataset(torch.utils.data.Dataset):
    """Yields fixed-length contiguous clips (never crossing an episode boundary) so a critic trainer can
    explode each into transitions with in-clip bootstrap futures -- decoding every frame only once."""

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | None = None,
        episodes: list[int] | None,
        image_keys: list[str],
        state_key: str = "observation.state",
        action_key: str = "action",
        horizon: int = 30,
        clip_len: int = 60,
        stride: int | None = None,
        outcomes: dict[int, str] | None = None,
        tolerance_s: float | None = None,
        img_size: int = 224,
        homing_onsets: dict | None = None,
    ):
        # FAILURE episodes: truncate the trailing "homing" return-to-home frames so they are not
        # sampled as currents and not used as bootstrap successors (a failure's homing = arms back near
        # home / task-not-done, which collides with SUCCESS starts and would mislabel the critic).
        # homing_onsets maps str(episode) -> {"homing_onset": k}; SUCCESS episodes are left untouched.
        self.homing_onsets = homing_onsets or {}
        self.img_size = img_size
        self.image_keys = list(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.horizon = horizon
        self.clip_len = clip_len
        self.stride = stride if stride is not None else max(1, clip_len - horizon)
        self.outcomes = outcomes or {}

        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        fps = meta.fps
        self.fps = fps
        # Clip delta_timestamps: images + state at the clip's frames [0..clip_len); actions out to
        # clip_len+horizon so every in-clip frame j has its own H-step chunk action[j:j+H].
        img_offsets = [t / fps for t in range(clip_len)]
        dt = dict.fromkeys([*self.image_keys, state_key], img_offsets)
        dt[action_key] = [t / fps for t in range(clip_len + horizon)]
        if tolerance_s is None:
            tolerance_s = _dl._float32_safe_tolerance(fps, _dl._max_video_timestamp(meta))
        self.ds = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=dt,
            tolerance_s=tolerance_s,
            video_backend="pyav",  # ships its own FFmpeg -> works on every node (openpi's fix)
        )
        # Per-episode [from, to) in GLOBAL frame index, then strided clip starts inside each episode.
        epds = self.ds.meta.episodes
        starts = {int(e): int(f) for e, f in zip(epds["episode_index"], epds["dataset_from_index"], strict=True)}
        ends = {int(e): int(t) for e, t in zip(epds["episode_index"], epds["dataset_to_index"], strict=True)}
        keep = episodes if episodes is not None else list(starts)
        self.clip_starts: list[tuple[int, int, int]] = []  # (global_start, ep_id, effective_len)
        for e in keep:
            s, t = starts[int(e)], ends[int(e)]
            full_len = t - s
            # FAILURE: cut the homing tail -> effective length = homing_onset. SUCCESS: keep full length.
            eff_len = full_len
            if self.outcomes.get(int(e)) != "success" and str(int(e)) in self.homing_onsets:
                eff_len = int(self.homing_onsets[str(int(e))]["homing_onset"])
            # clip starts (=current frames) only within the task span; the tail clip's missing futures
            # are masked by is_pad, and futures past eff_len are masked by the trainer via eff_len.
            for c in range(s, s + eff_len, self.stride):
                self.clip_starts.append((c, int(e), eff_len))

    def _resize_clip(self, sample):
        import cv2

        cv2.setNumThreads(0)  # each DataLoader worker is already a process; don't oversubscribe
        s = self.img_size
        out = np.empty((self.clip_len, len(self.image_keys), s, s, 3), np.uint8)
        for c, key in enumerate(self.image_keys):
            v = np.asarray(sample[key], np.float32)  # [clip_len, 3, H0, W0], CHW float[0,1]
            v = np.transpose(v, (0, 2, 3, 1))  # -> [clip_len, H0, W0, 3]
            v = (np.clip(v, 0, 1) * 255).astype(np.uint8)
            for t in range(v.shape[0]):
                out[t, c] = cv2.resize(v[t], (s, s), interpolation=cv2.INTER_AREA)
        return out

    def __len__(self):
        return len(self.clip_starts)

    def __getitem__(self, i):
        gstart, ep_id, ep_len = self.clip_starts[i]
        sample = self.ds[gstart]
        # frame position of the clip start within its episode (LeRobot provides it directly); used for
        # the analytic cost-to-goal target.
        clip_pos0 = int(np.asarray(sample["frame_index"]))
        # Resize to img_size HERE, in the (parallel) worker: LeRobot gives CHW float[0,1] at native res;
        # emit HWC uint8 at img_size so the batch that crosses to the GPU is ~6x smaller than native f32.
        imgs = self._resize_clip(sample)  # [clip_len, ncam, img_size, img_size, 3] uint8
        state = np.asarray(sample[self.state_key], np.float32)  # [clip_len, state_dim]
        action = np.asarray(sample[self.action_key], np.float32)  # [clip_len+H, action_dim]
        img_pad = np.asarray(sample[f"{self.image_keys[0]}_is_pad"], np.bool_)  # [clip_len]
        return {
            "images": imgs,
            "state": state,
            "action": action,
            "img_pad": img_pad,
            "clip_pos0": np.int64(clip_pos0),
            "ep_len": np.int64(ep_len),
            "is_success": np.bool_(self.outcomes.get(ep_id) == "success"),
        }

    @staticmethod
    def collate(batch):
        return {
            "images": torch.from_numpy(np.stack([b["images"] for b in batch])),
            "state": torch.from_numpy(np.stack([b["state"] for b in batch])),
            "action": torch.from_numpy(np.stack([b["action"] for b in batch])),
            "img_pad": torch.from_numpy(np.stack([b["img_pad"] for b in batch])),
            "clip_pos0": torch.from_numpy(np.stack([b["clip_pos0"] for b in batch])),
            "ep_len": torch.from_numpy(np.stack([b["ep_len"] for b in batch])),
            "is_success": torch.from_numpy(np.stack([b["is_success"] for b in batch])),
        }
