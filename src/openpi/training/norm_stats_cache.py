"""Norm stats as a cache keyed on what they were computed from, not on a name someone remembers.

The problem this replaces. `asset_id` defaults to the repo id (config.py), so stats are found by
NAME. Two runs of one config that train on different episode subsets -- `--data.success-only` and
not -- resolve to the same name and silently share one file. Measured on the YAM lego assets, the
success-only and all-episode stats differ by up to 2.5% in action q99: enough to mis-normalize
everything, never enough to NaN or to move the loss curve. The same shape of error, at 37% rather
than 2.5%, is what the alpha-Flow 08-23 incident was.

The fix is a cache, not a warning. The key covers what actually determines the statistics and can
vary WITHIN a config -- the episode subset and the action horizon. The config name does not need to
be in the key because it is already in the path (`assets_dirs` is `assets/<config name>`), and
neither does delta_mode, which has its own config name. On a hit nothing happens; on a miss the
stats are recomputed, which is affordable precisely here: the pass sets `skip_videos=True`, so it
reads the low-dimensional fields and never decodes a frame.

Recomputed stats are written to a key-suffixed sibling directory rather than over the primary asset,
so a hand-managed asset is never clobbered and two subsets of one config stop overwriting each other.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib

import numpy as np
import tqdm

from openpi.shared import normalize as _normalize
from openpi.training import config as _config

logger = logging.getLogger(__name__)


def stats_key(data_config: _config.DataConfig, action_horizon: int) -> str:
    """What the statistics depend on and can differ within one config."""
    payload = {
        "repo_id": data_config.repo_id,
        "episodes": sorted(data_config.episodes) if data_config.episodes else "all",
        "action_horizon": int(action_horizon),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _provenance_key(d: pathlib.Path) -> str | None:
    p = d / "provenance.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("stats_key")
    except (OSError, ValueError):
        return None


def ensure_norm_stats(
    config: _config.TrainConfig, data_config: _config.DataConfig, *, max_frames: int | None = None
) -> _config.DataConfig:
    """Return `data_config` with statistics that provably match what this run trains on.

    Order of preference: a key-stamped sibling (a previous miss, now a hit), then the primary asset
    if IT carries the right key, then recompute. An unstamped primary asset is not trusted -- it is
    exactly the ambiguous case -- but it is also not deleted.
    """
    # An EXPLICIT --data.assets.asset-id is an instruction, not a guess: it is how you pin a run to
    # the exact statistics an earlier run used so the comparison is schedule-only. Second-guessing it
    # would silently break the thing it exists to guarantee. The cache applies only where asset_id
    # fell back to the repo id, which is where the ambiguity lives.
    if getattr(config.data, "assets", None) is not None and config.data.assets.asset_id is not None:
        logger.info(
            "norm stats: asset_id '%s' was named explicitly, using it as given (the episode-subset "
            "cache applies only to the repo-id fallback)",
            config.data.assets.asset_id,
        )
        return data_config

    key = stats_key(data_config, config.model.action_horizon)
    asset = data_config.asset_id or data_config.repo_id
    base = pathlib.Path(config.assets_dirs)
    primary, keyed = base / asset, base / f"{asset}__{key}"

    for d in (keyed, primary):
        if d.is_dir() and _provenance_key(d) == key:
            logger.info("norm stats: cache hit for key %s at %s", key, d)
            return dataclasses.replace(data_config, norm_stats=_normalize.load(d), asset_id=str(d.relative_to(base)))

    why = "no stamped stats for this episode subset" if primary.is_dir() else "no stats on disk"
    logger.warning(
        "norm stats: %s (key %s, repo %s, %s) -- recomputing into %s. This is the case a name-keyed "
        "asset cannot distinguish: the same config with and without --data.success-only resolves to "
        "the same name and would otherwise share one file.",
        why,
        key,
        data_config.repo_id,
        f"{len(data_config.episodes)} episodes" if data_config.episodes else "all episodes",
        keyed.name,
    )
    stats = _compute(config, data_config, max_frames=max_frames)
    keyed.mkdir(parents=True, exist_ok=True)
    _normalize.save(keyed, stats)
    (keyed / "provenance.json").write_text(
        json.dumps(
            {
                "stats_key": key,
                "computed_on": {
                    "repo_id": data_config.repo_id,
                    "episodes_subset": sorted(data_config.episodes) if data_config.episodes else "all",
                    "action_horizon": int(config.model.action_horizon),
                },
            },
            indent=1,
        )
    )
    return dataclasses.replace(data_config, norm_stats=stats, asset_id=str(keyed.relative_to(base)))


def _compute(config: _config.TrainConfig, data_config: _config.DataConfig, *, max_frames: int | None):
    from openpi.training import data_loader as _dl  # local: data_loader imports this module

    # skip_videos is what makes recompute-on-miss affordable at all -- the frames are never decoded.
    dataset = _dl.create_torch_dataset(data_config, config.model.action_horizon, config.model, skip_videos=True)
    dataset = _dl.TransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    )
    n = len(dataset) if max_frames is None else min(len(dataset), max_frames)
    nb = max(n // config.batch_size, 1)
    loader = _dl.TorchDataLoader(
        dataset, local_batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, num_batches=nb
    )
    running = {k: _normalize.RunningStats() for k in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=nb, desc=f"norm stats ({data_config.repo_id})"):
        for k, r in running.items():
            r.update(np.asarray(batch[k]))
    return {k: r.get_statistics() for k, r in running.items()}


__all__ = ["ensure_norm_stats", "stats_key"]
