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
import re
import shutil

import numpy as np
import tqdm

from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _RemoveStrings(_transforms.DataTransformFn):
    """Drop non-numeric fields before collation (the prompt), as compute_norm_stats.py does."""

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _transform_fingerprint(data_config: _config.DataConfig) -> list:
    """Identify the transform pipeline `_compute` runs, since the statistics are OF its output.

    `delta_mode` is the case that matters: pi05_yam_lego_taxi pushes JointDeltaActions and
    pi05_yam_lego_taxi_none does not, so one produces ~0-centred joint deltas and the other
    radians-scale absolute targets from the same episodes of the same repo. Without this in the key
    they collide, and because the search deliberately crosses config directories, the collision is
    not hypothetical -- a joint run loads the absolute stats and normalizes every action by a mean
    1.74 rad off with 2.3-2.5x the std, reported as a proven match.
    """
    out = []
    for t in [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]:
        fields = {}
        for k, v in vars(t).items():
            if k.startswith("_"):
                continue
            try:
                fields[k] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else repr(v)
            except (TypeError, ValueError):
                fields[k] = repr(v)
        out.append([type(t).__name__, sorted(fields.items())])
    return out


def stats_key(data_config: _config.DataConfig, action_horizon: int) -> str:
    """What the statistics depend on and can differ within one config."""
    payload = {
        "repo_id": data_config.repo_id,
        "episodes": sorted(data_config.episodes) if data_config.episodes else "all",
        "action_horizon": int(action_horizon),
        # The config NAME is not in the key on purpose -- stats belong to data, not to a config -- so
        # everything a config does that changes the numbers has to be in here explicitly.
        "transforms": _transform_fingerprint(data_config),
        # A frame filter changes the pool the statistics are computed over while leaving `episodes`
        # untouched, so without this a run with the failure give-up tails cut and one without hash
        # identically and share one asset.
        "drop_failure_homing": bool(data_config.drop_failure_homing),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode()).hexdigest()[:12]


def _provenance(d: pathlib.Path) -> dict:
    p = d / "provenance.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _provenance_key(d: pathlib.Path) -> str | None:
    return _provenance(d).get("stats_key")


_COUNT = re.compile(r"\b(\d+)\s*(?:/\s*\d+)?\b")


def _declared_episode_count(subset) -> int | None:
    """How many episodes a provenance record says its stats were computed on.

    Stats written before this cache existed describe their subset in prose -- the YAM lego h30 asset
    says "success-only (300/347 via outcomes.jsonl)". That sentence is unambiguous to a person and
    the count is the one machine-checkable thing in it, so it is what we match on.
    """
    if isinstance(subset, list):
        return len(subset)
    if isinstance(subset, str):
        if subset == "all":
            return None
        m = _COUNT.search(subset)
        return int(m.group(1)) if m else None
    return None


def _match(prov: dict, data_config: _config.DataConfig, key: str, horizon: int) -> tuple[int, str] | None:
    """Score how well an on-disk asset matches this run: 2 = proven, 1 = declared, None = no."""
    if prov.get("stats_key") == key:
        return 2, f"stats_key {key}"
    on = prov.get("computed_on")
    if not isinstance(on, dict) or on.get("repo_id") != data_config.repo_id:
        return None
    # A capped pass is not these statistics, whatever its episode list says. It saw a prefix of the
    # subset, so its means and stds land close while its quantiles do not -- and quantiles are the
    # scale under pi05 normalisation. Refusing it here is what keeps the cache from handing a run a
    # file that describes the right episodes and the wrong distribution.
    if on.get("full_pass") is False:
        return None
    # A horizon the record does not state cannot disagree with ours. Norm stats are marginals over
    # action dimensions, so the horizon barely moves them, but a stated one that differs is a real
    # signal that these are some other run's statistics.
    stated_h = on.get("action_horizon")
    if stated_h is not None and int(stated_h) != int(horizon):
        return None
    want = sorted(data_config.episodes) if data_config.episodes else None
    subset = on.get("episodes_subset")
    if isinstance(subset, list):
        return (2, f"episode list ({len(subset)})") if sorted(subset) == (want or []) else None
    if want is None:
        # Prose counts here too. This branch used to accept only the literal "all", which rejected
        # the very record this repo writes: the corrected s300h30 provenance says "all 347 episodes",
        # so an all-episodes run could not match its own stats and fell through to a recompute.
        if isinstance(subset, str) and re.search(r"\ball\b", subset):
            return 1, f"declared: {subset!r}"
        return None
    n = _declared_episode_count(subset)
    return (1, f"declared {n} episodes: {subset!r}") if n == len(want) else None


def _search(root: pathlib.Path, data_config: _config.DataConfig, key: str, horizon: int):
    """Find stats anywhere under the assets root that were computed on what this run trains on.

    Searching across config directories is the point, not a convenience. The statistics depend on the
    dataset and the episode subset, never on which config consumes them, but `asset_id` defaults to
    the repo id under `assets/<config name>/`, so an asset computed for one config is invisible to
    the next by name alone. That invisibility is what forced runs to hard-code
    `--data.assets.asset-id` in their launcher, and a hard-coded id is exactly what goes stale.
    """
    hits = []
    for stats in sorted(root.glob("*/**/norm_stats.json")):
        d = stats.parent
        m = _match(_provenance(d), data_config, key, horizon)
        if m:
            hits.append((m[0], m[1], d))
    if not hits:
        return None
    best = max(h[0] for h in hits)
    hits = [h for h in hits if h[0] == best]
    if len(hits) > 1:
        loaded = [_normalize.load(d) for _, _, d in hits]
        if any(_digest(x) != _digest(loaded[0]) for x in loaded[1:]):
            raise ValueError(
                "norm stats: "
                + str(len(hits))
                + " assets claim to describe this run's data but hold DIFFERENT numbers: "
                + ", ".join(str(d) for _, _, d in hits)
                + ". Delete or re-stamp the stale ones, or pin the right one with "
                "--data.assets.asset-id."
            )
    return hits[0]


def _digest(stats) -> str:
    flat = {k: [v.mean.tolist(), v.std.tolist()] for k, v in sorted(stats.items())}
    return hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()


def _resolved(data_config: _config.DataConfig, d: pathlib.Path, base: pathlib.Path, stats) -> _config.DataConfig:
    """Point the config at `d` -- statistics, asset id AND provenance.

    Carrying the provenance over is not bookkeeping. `create_base_config` read a provenance record
    for the asset it resolved BY NAME, and `_check_norm_stats_provenance` runs afterwards on whatever
    the config holds. Leave it behind and that check validates a file this run no longer uses: it
    aborts a run whose statistics are right (a success-only run resolving the 300-episode asset dies
    on the all-episode record left over from the repo-id fallback), and, worse, the generation guard
    -- total_episodes / total_frames against the live dataset -- never runs on the file that IS used.
    """
    return dataclasses.replace(
        data_config,
        norm_stats=stats,
        asset_id=str(d.relative_to(base)),
        norm_stats_provenance=_provenance(d) or None,
    )


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
            return _resolved(data_config, d, base, _normalize.load(d))

    # Nothing under this config's own directory. Statistics do not belong to a config, so look for
    # them wherever they were computed -- this is what lets a launcher stop naming an asset by hand.
    found = _search(pathlib.Path(config.assets_base_dir), data_config, key, config.model.action_horizon)
    if found is not None:
        strength, basis, d = found
        logger.info(
            "norm stats: reusing %s, matched on %s (%s). Nothing was recomputed, so this run shares "
            "the exact normalization of whatever produced those stats.",
            d,
            basis,
            "proven"
            if strength == 2
            else "declared -- the record describes its subset in prose, so "
            "the episode COUNT is all that could be checked",
        )
        # Copy it under THIS config's assets dir. asset_id is relative to assets_dirs everywhere it
        # is consumed -- the checkpoint saves <ckpt>/assets/<asset_id> and serving re-derives the id
        # from the config -- so returning a path that starts with another config's name produces a
        # checkpoint whose norm stats cannot be resolved at serving time. Copying is a few KB and
        # makes the checkpoint self-contained.
        local = base / f"{asset}__{key}"
        if not local.is_dir():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(d, local)
            (local / "provenance.json").write_text(
                json.dumps({"stats_key": key, "copied_from": str(d), **_provenance(d)}, indent=1)
            )
        return _resolved(data_config, local, base, _normalize.load(local))

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
                "transforms": _transform_fingerprint(data_config),
                "computed_on": {
                    "repo_id": data_config.repo_id,
                    "episodes_subset": sorted(data_config.episodes) if data_config.episodes else "all",
                    "action_horizon": int(config.model.action_horizon),
                },
            },
            indent=1,
        )
    )
    return _resolved(data_config, keyed, base, stats)


def _compute(config: _config.TrainConfig, data_config: _config.DataConfig, *, max_frames: int | None):
    from openpi.training import data_loader as _dl  # local: data_loader imports this module

    # skip_videos is what makes recompute-on-miss affordable at all -- the frames are never decoded.
    dataset = _dl.create_torch_dataset(data_config, config.model.action_horizon, config.model, skip_videos=True)
    dataset = _dl.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Without this, recompute-on-miss is dead on every prompt_from_task config -- which is all
            # of the YAM ones. YAMInputs passes `prompt` straight through, and the jax-framework
            # TorchDataLoader cannot collate a Python string, so the worker dies and takes the job with
            # it. scripts/compute_norm_stats.py has always applied RemoveStrings for exactly this
            # reason; the cache has to as well, and it is why the cache appeared to work -- every
            # verification so far happened to hit an existing asset and never reached this path.
            _RemoveStrings(),
        ],
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


def resolve_for_config(config_name: str, *, max_frames: int | None = None) -> pathlib.Path:
    """The norm_stats.json a TRAIN CONFIG resolves to, computing it on a genuine miss.

    This is the seam anything downstream of a policy should use instead of being handed a path.
    A critic scores a policy's actions, so it has to normalize them the way that policy does; naming
    the config says WHICH policy in a way a path does not. Hand-passed paths went stale three ways
    in practice: the asset key moved when a data condition changed (a config's stats live under
    `<repo_id>__<key>`, and the key is a function of the episode subset and transforms), the
    directory name did not say which condition it was, and an absolute path stopped existing when
    the workspace moved machines.

    Going through `ensure_norm_stats` also means a miss RECOMPUTES rather than failing, and a hit is
    matched on content, so the file returned provably matches what that config trains on.
    """
    import openpi.training.config as _cfg

    config = _cfg.get_config(config_name)
    data_config = ensure_norm_stats(config, config.data.create(config.assets_dirs, config.model), max_frames=max_frames)
    if data_config.asset_id is None:
        raise ValueError(f"{config_name} resolved no norm-stats asset; it may have no repo_id")
    path = pathlib.Path(config.assets_dirs) / data_config.asset_id / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"{config_name} resolved to {path}, which does not exist")
    return path


__all__ = ["ensure_norm_stats", "resolve_for_config", "stats_key"]
