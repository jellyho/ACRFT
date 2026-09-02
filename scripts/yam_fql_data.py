"""YAM chunk-transition loader for FQL: (obs_t, a_{t:t+H}, R_chunk, obs_{t+H}, done, MC).

Reuses the exact openpi pipeline the BC checkpoint was trained with -- same LeRobotDataset decode
(pyav), same YAM repack/delta/normalize/tokenize transform stack, norm stats loaded from the BC
checkpoint's own assets -- so the frozen flow expert sees distributions identical to its training.
The one addition over the BC loader is the SECOND decode at t+H (the chunk's successor frame),
fetched through the same delta_timestamps query, so each __getitem__ costs two frame decodes.

Reward/return conventions are the house patch-critic ones (train_patch_critic_clip.py), so the FQL
critic and the patch critic speak the same value scale:
  cost_to_goal   r = -1 per step; the last h_goal frames of a SUCCESS episode are the absorbing
                 goal (r = 0, done). gamma = 0.99964, v_min = -1/(1-gamma) ~= -2778.
  FAILURE        episode truncated at its homing onset (compute_homing_onsets.py); the last task
                 frame is an absorbing terminal anchored at failure_reward (v_min by default).
                 MC is pinned to v_min (the true return is unobserved, v_min is the safe floor).
Chunk quantities follow: R = sum_{i<H} gamma^i r_{t+i} (absorbing beyond terminal), the backup in
the trainer uses gamma^H, done = successor at/past the absorbing region, MC = per-step discounted
return-to-go at t (a valid floor for Q(s_t, a_chunk) when the floor is enabled).
"""

import dataclasses
import json
import pathlib

import numpy as np
import torch

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.training.outcomes as _outcomes
import openpi.transforms as _transforms

try:
    from lerobot.datasets import lerobot_dataset
except ImportError:  # lerobot < 0.4
    from lerobot.common.datasets import lerobot_dataset


def _episode_tables(ds_root, outcomes_path, homing_path, h_goal, discount, v_min, failure_reward):
    """Per-episode arrays: length, success flag, terminal frame index, in exact house conventions.

    Verdicts and lengths come from the dataset's own metadata (``next.success`` / ``next.done`` per-episode
    stats); ``outcomes_path`` names a legacy sidecar and is only honoured when given explicitly."""
    outcomes = _outcomes.load_outcomes(ds_root, legacy_jsonl=outcomes_path)
    lengths = _outcomes.episode_lengths(ds_root)
    homing = json.loads(pathlib.Path(homing_path).read_text())
    n_ep = len(lengths)
    length = np.zeros(n_ep, np.int64)
    succ = np.zeros(n_ep, bool)
    term = np.zeros(n_ep, np.int64)  # first absorbing frame: T - h_goal (success) | onset - 1 (failure)
    for e, n in lengths.items():
        length[e] = int(n)
        succ[e] = outcomes.get(e) == "success"
        if succ[e]:
            term[e] = max(length[e] - h_goal, 1)
        else:
            onset = int(homing[str(e)]["homing_onset"])
            term[e] = max(min(onset, length[e]) - 1, 1)
    return length, succ, term


class YamFQLTransitions(torch.utils.data.Dataset):
    """Map-style dataset over valid chunk base frames of the 347-episode YAM set."""

    def __init__(
        self,
        *,
        repo_id: str,
        root: str,
        horizon: int,
        bc_assets_dir: str,
        asset_id: str = "jellyho/yam_lego_taxi",
        homing_path: str,
        outcomes_path: str | None = None,
        h_goal: int = 3,
        discount: float = 0.99964,
        failure_reward: float | None = None,
    ):
        self.horizon = horizon
        self.discount = discount
        self.v_min = -1.0 / (1.0 - discount)
        self.failure_reward = self.v_min if failure_reward is None else failure_reward

        # the BC config IS the transform recipe; only the norm-stats source moves to the checkpoint
        cfg = _config.get_config("pi05_yam_lego_taxi")
        factory = dataclasses.replace(
            cfg.data, assets=_config.AssetsConfig(assets_dir=bc_assets_dir, asset_id=asset_id)
        )
        data_config = factory.create(cfg.assets_dirs, cfg.model)
        if data_config.norm_stats is None:
            raise ValueError(f"no norm stats under {bc_assets_dir}/{asset_id}")
        self._tf = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )

        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        fps = meta.fps
        self._cams = ("observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right")
        succ_dt = horizon / fps  # the successor frame, decoded through the same episode-aware query
        delta_timestamps = {cam: [0.0, succ_dt] for cam in self._cams}
        delta_timestamps["observation.state"] = [0.0, succ_dt]
        delta_timestamps["action"] = [i / fps for i in range(horizon)]
        ds = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            tolerance_s=_data_loader._float32_safe_tolerance(fps, _data_loader._max_video_timestamp(meta)),
            video_backend="pyav",
        )
        if data_config.prompt_from_task:
            ds = _data_loader.TransformedDataset(
                ds, [_transforms.PromptFromLeRobotTask(_data_loader._task_index_to_prompt(meta))]
            )
        self._ds = ds

        self._len, self._succ, self._term = _episode_tables(
            pathlib.Path(root) / repo_id, outcomes_path, homing_path, h_goal, discount, self.v_min, self.failure_reward
        )
        # LeRobot v3 global frame order is episode-major; starts from the episode lengths
        self._ep_start = np.concatenate([[0], np.cumsum(self._len)[:-1]])
        if int(self._len.sum()) != meta.total_frames:
            raise ValueError(f"outcomes frames {int(self._len.sum())} != dataset frames {meta.total_frames}")
        # valid base frames: success uses every frame (chunks inside the goal teach Q(goal)=0);
        # failure stops before its absorbing terminal
        idx = []
        for e in range(len(self._len)):
            last = self._len[e] if self._succ[e] else self._term[e]
            idx.append(self._ep_start[e] + np.arange(last))
        self._base = np.concatenate(idx)

    def __len__(self):
        return len(self._base)

    def _targets(self, e: int, t: int):
        g, h = self.discount, self.horizon
        term = int(self._term[e])
        n_neg = int(np.clip(term - t, 0, h))  # -1-reward steps inside the chunk
        r = -(1.0 - g**n_neg) / (1.0 - g)
        done = float(t + h >= term)
        if self._succ[e]:
            mc = -(1.0 - g ** max(term - t, 0)) / (1.0 - g)
        else:
            if t + h >= term:  # chunk consumes (or lands on) the absorbing failure terminal
                r += (g ** min(term - t, h)) * self.failure_reward
            mc = self.v_min
        return np.float32(r), np.float32(done), np.float32(mc)

    def __getitem__(self, i: int):
        base = int(self._base[i])
        e = int(np.searchsorted(self._ep_start, base, side="right") - 1)
        t = base - int(self._ep_start[e])
        item = self._ds[base]

        def frame(k):
            raw = {cam: np.asarray(item[cam])[k] for cam in self._cams}
            raw["observation.state"] = np.asarray(item["observation.state"])[k]
            raw["action"] = np.asarray(item["action"])
            raw["prompt"] = item["prompt"]
            return self._tf(raw)

        cur, nxt = frame(0), frame(1)
        r, done, mc = self._targets(e, t)
        out = {f"obs/{k}": v for k, v in cur.items() if k != "actions"}
        out.update({f"nobs/{k}": v for k, v in nxt.items() if k != "actions"})
        out.update({"actions": cur["actions"], "reward": r, "done": done, "mc": mc})
        return out


def make_loader(dataset, *, batch_size, num_workers, seed=0):
    """Infinite iterator of (obs, act, rew, nobs, done, mc) numpy batches for train_fql's step fns."""
    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )

    def _to_np(v):
        # collate keeps nested dicts (e.g. image: {cam: tensor}); floats go to f32 for the jit
        if isinstance(v, dict):
            return {k: _to_np(x) for k, x in v.items()}
        arr = np.asarray(v)
        return arr.astype(np.float32) if arr.dtype == np.float64 else arr

    def _split(batch, prefix):
        sub = {k[len(prefix) :]: _to_np(v) for k, v in batch.items() if k.startswith(prefix)}
        return _model.Observation.from_dict(sub)

    def gen():
        while True:
            for batch in torch_loader:
                obs = _split(batch, "obs/")
                nobs = _split(batch, "nobs/")
                yield (
                    obs,
                    np.asarray(batch["actions"], np.float32),
                    np.asarray(batch["reward"], np.float32),
                    nobs,
                    np.asarray(batch["done"], np.float32),
                    np.asarray(batch["mc"], np.float32),
                )

    return gen()
