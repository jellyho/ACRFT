"""Shared dataset for pi0.5-scale policy extraction: BC samples + per-transition critic annotations.

A sample i of the full 347-episode YAM dataset (episodes=None) has torch-dataset index == LeRobot
global frame index == patch-cache row index (the cache was built over the same dataset in order;
cache N == dataset total_frames == 937,993), so critic-side annotations (advantage arrays, cached
DINO features) are addressed by the SAME index the loader yields. Transforms mirror the BC recipe
(pi05_yam_lego_taxi) so the policy trains in exactly its pretraining input space.
"""

import dataclasses

import numpy as np
import torch

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def make_bc_dataset(bc_assets_dir: str, asset_id: str = "jellyho/yam_lego_taxi"):
    """The exact BC pipeline (transforms + norm stats from the BC checkpoint's assets)."""
    cfg = _config.get_config("pi05_yam_lego_taxi")
    factory = dataclasses.replace(cfg.data, assets=_config.AssetsConfig(assets_dir=bc_assets_dir, asset_id=asset_id))
    data_config = factory.create(cfg.assets_dirs, cfg.model)
    data_config = dataclasses.replace(data_config, episodes=None)  # extraction sees ALL 347 episodes
    if data_config.norm_stats is None:
        raise ValueError(f"no norm stats under {bc_assets_dir}/{asset_id}")
    ds = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    return _data_loader.transform_dataset(ds, data_config), cfg


class AnnotatedBC(torch.utils.data.Dataset):
    """BC sample + any per-frame annotation arrays (advantage, labels), keyed by the same index."""

    def __init__(self, base, annotations: dict[str, np.ndarray]):
        self._base = base
        self._ann = annotations
        n = len(base)
        for k, v in annotations.items():
            if len(v) != n:
                raise ValueError(f"annotation '{k}' has {len(v)} rows, dataset has {n}")

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        item = dict(self._base[i])
        out = dict(item)
        for k, v in self._ann.items():
            out[f"ann/{k}"] = np.float32(v[i])
        out["ann/idx"] = np.int64(i)
        return out


def make_loader(dataset, *, batch_size, num_workers, seed=0):
    """Infinite iterator of (Observation, actions[f32], {ann}) batches."""
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
        if isinstance(v, dict):
            return {k: _to_np(x) for k, x in v.items()}
        arr = np.asarray(v)
        return arr.astype(np.float32) if arr.dtype == np.float64 else arr

    def gen():
        while True:
            for batch in torch_loader:
                ann = {k[4:]: np.asarray(v) for k, v in batch.items() if k.startswith("ann/")}
                obs = _model.Observation.from_dict(
                    {k: _to_np(v) for k, v in batch.items() if k != "actions" and not k.startswith("ann/")}
                )
                yield obs, np.asarray(batch["actions"], np.float32), ann

    return gen()
