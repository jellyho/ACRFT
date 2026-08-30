"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    # Only state/actions are accumulated, so skip decoding the camera videos entirely.
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config, skip_videos=True)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    success_only: bool = False,
    asset_id: str | None = None,
):
    """Compute norm stats, optionally on a success-only subset and into a custom asset dir.

    ``success_only`` and ``asset_id`` exist for the data-scaling study: each scaling point trains on
    a DIFFERENT set of episodes (its own success-only subset), so it must get its OWN norm stats
    rather than sharing one file. ``asset_id`` isolates the output directory per point, and stats are
    written to ``assets_dirs/asset_id`` — the exact path the training run then LOADS from (train
    passes the matching ``--data.assets.asset-id``).
    """
    config = _config.get_config(config_name)
    if success_only or asset_id is not None:
        replaced = {}
        if success_only:
            if not hasattr(config.data, "success_only"):
                raise ValueError(f"{type(config.data).__name__} has no success_only field")
            replaced["success_only"] = True
        if asset_id is not None:
            replaced["assets"] = dataclasses.replace(config.data.assets, asset_id=asset_id)
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, **replaced))
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Write where the training run will LOAD from: assets_dirs/asset_id (asset_id defaults to repo_id,
    # so the un-overridden path is unchanged).
    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    # provenance stamp: which dataset generation these stats describe -- the training loader
    # cross-checks this against the live dataset and refuses stale stats (data_loader.py).
    import json as _json

    try:
        import lerobot.datasets.lerobot_dataset as _lds

        _meta = _lds.LeRobotDatasetMetadata(data_config.repo_id)
        _prov = {
            "computed_on": {
                "repo_id": data_config.repo_id,
                "total_episodes": _meta.total_episodes,
                "total_frames": _meta.total_frames,
                "episodes_subset": sorted(data_config.episodes) if data_config.episodes else "all",
            }
        }
        (output_path / "provenance.json").write_text(_json.dumps(_prov, indent=2))
    except Exception as e:  # provenance is best-effort; stats themselves are already saved
        print(f"provenance stamp skipped: {e}")


if __name__ == "__main__":
    tyro.cli(main)
