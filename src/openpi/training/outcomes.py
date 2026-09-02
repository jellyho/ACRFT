"""Episode verdicts (success / fail / no verdict) of a LeRobot v3 dataset, from its own schema.

The i2rt recorder stores the verdict as two per-frame bool features -- the names LeRobot's own
datasets use -- and nowhere else:

    next.success   True on the episode's LAST frame  iff  it succeeded
    next.done      True on the episode's LAST frame  iff  it was judged at all

Because they are ordinary features, LeRobot keeps per-episode statistics for them in
``meta/episodes`` (``stats/next.success/max`` ...), so the episode-level verdict is readable from
the metadata alone, without opening a data file:

    success       stats/next.success/max > 0
    fail          stats/next.done/max > 0  and not success
    unknown       stats/next.done/max == 0      (kept without judging; do not treat as a failure)

Datasets recorded before this carried an ``outcomes.jsonl`` sidecar; the recorder migrates them in
place (``yam-data migrate-outcomes <dir>``), and a training run refuses to guess on one that has
not been migrated. The sidecar is still *accepted* here, when a caller names one explicitly, so the
research scripts keep working on data that lives elsewhere -- with a warning, and never by default.
"""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np
import pyarrow.dataset as pads

logger = logging.getLogger(__name__)

SUCCESS_KEY = "next.success"
DONE_KEY = "next.done"
SUCCESS, FAIL, UNKNOWN = "success", "fail", "unknown"
LEGACY_SIDECAR = "outcomes.jsonl"


class OutcomeColumnsMissingError(RuntimeError):
    """The dataset predates the outcome features (and no sidecar was named explicitly)."""

    def __init__(self, root: pathlib.Path):
        super().__init__(
            f"{root} has no '{SUCCESS_KEY}' / '{DONE_KEY}' features: it was recorded before the episode "
            "verdict moved into the LeRobot schema. Migrate it in place with the recorder's\n"
            f"    workstation/yam-data migrate-outcomes {root}\n"
            "(i2rt_rllab), or re-download it from the hub."
        )


def has_outcome_features(root: pathlib.Path) -> bool:
    """Whether ``meta/info.json`` declares both features."""
    try:
        feats = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text()).get("features", {})
    except (OSError, ValueError):
        return False
    return SUCCESS_KEY in feats and DONE_KEY in feats


def episode_outcomes(root: pathlib.Path) -> dict[int, str] | None:
    """``{episode_index: "success" | "fail" | "unknown"}`` from ``meta/episodes`` alone.

    Returns None when the dataset does not carry the features at all, so a caller can decide
    whether that is an error (a success-only training run) or simply "nothing to filter" (a
    demonstration dataset that is successful by construction).
    """
    root = pathlib.Path(root)
    if not has_outcome_features(root):
        return None
    files = sorted(root.glob("meta/episodes/**/*.parquet"))
    if not files:
        return {}
    s_col, d_col = f"stats/{SUCCESS_KEY}/max", f"stats/{DONE_KEY}/max"
    table = pads.dataset([str(f) for f in files], format="parquet").to_table(columns=["episode_index", s_col, d_col])
    out: dict[int, str] = {}
    for ep, s, d in zip(
        table["episode_index"].to_pylist(), table[s_col].to_pylist(), table[d_col].to_pylist(), strict=True
    ):
        succ = bool(np.asarray(s).reshape(-1)[0])
        done = bool(np.asarray(d).reshape(-1)[0])
        out[int(ep)] = SUCCESS if succ else (FAIL if done else UNKNOWN)
    return dict(sorted(out.items()))


def success_episode_indices(root: pathlib.Path) -> list[int] | None:
    """Sorted indices of the episodes that succeeded; None if the dataset carries no verdicts."""
    outcomes = episode_outcomes(root)
    if outcomes is None:
        return None
    return sorted(ep for ep, state in outcomes.items() if state == SUCCESS)


def episode_lengths(root: pathlib.Path) -> dict[int, int]:
    """``{episode_index: frame count}`` from ``meta/episodes``."""
    files = sorted(pathlib.Path(root).glob("meta/episodes/**/*.parquet"))
    if not files:
        return {}
    table = pads.dataset([str(f) for f in files], format="parquet").to_table(columns=["episode_index", "length"])
    return {
        int(e): int(n) for e, n in zip(table["episode_index"].to_pylist(), table["length"].to_pylist(), strict=True)
    }


def read_legacy_sidecar(path: pathlib.Path) -> dict[int, str]:
    """``{episode: outcome}`` from an ``outcomes.jsonl`` (values normalized to success / fail / unknown)."""
    out: dict[int, str] = {}
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        o = str(row.get("outcome", "")).strip().lower()
        out[int(row["episode"])] = (
            SUCCESS if o in ("success", "keep") else (FAIL if o in ("fail", "failure") else UNKNOWN)
        )
    return dict(sorted(out.items()))


def load_outcomes(root: pathlib.Path, *, legacy_jsonl: str | pathlib.Path | None = None) -> dict[int, str]:
    """The verdict table a script should use.

    The dataset's own schema, unless the caller explicitly names a legacy ``outcomes.jsonl`` --
    kept for research scripts whose data lives on machines the migration has not reached. Naming
    one is loud (a warning), and a dataset that has neither raises with the migration command.
    """
    root = pathlib.Path(root)
    if legacy_jsonl:
        logger.warning(
            "reading episode verdicts from the legacy sidecar %s; the dataset schema (%s / %s) is the source "
            "of truth now -- migrate the dataset and drop --outcomes",
            legacy_jsonl,
            SUCCESS_KEY,
            DONE_KEY,
        )
        return read_legacy_sidecar(legacy_jsonl)
    outcomes = episode_outcomes(root)
    if outcomes is None:
        raise OutcomeColumnsMissingError(root)
    return outcomes


def cache_outcomes(cache_meta: dict, *, legacy_jsonl: str | pathlib.Path | None = None) -> dict[int, str]:
    """Verdicts for a feature cache written by ``scripts/cache_patch_features.py``.

    The cache's ``meta.json`` records each episode's verdict when it is built (``outcome``; older
    caches only have the ``success`` bool, where "not success" has to be read as fail). Same
    explicit-sidecar escape hatch as :func:`load_outcomes`.
    """
    if legacy_jsonl:
        logger.warning(
            "reading episode verdicts from the legacy sidecar %s instead of the cache's meta.json", legacy_jsonl
        )
        return read_legacy_sidecar(legacy_jsonl)
    out: dict[int, str] = {}
    for e, info in cache_meta["episodes"].items():
        out[int(e)] = info.get("outcome") or (SUCCESS if info.get("success") else FAIL)
    return dict(sorted(out.items()))


def dataset_root(repo_id: str, root: str | pathlib.Path | None = None) -> pathlib.Path:
    """Where ``repo_id`` lives on disk: ``<root>/<repo_id>`` when a root is given, else wherever
    LeRobot resolves it (``HF_LEROBOT_HOME``)."""
    if root is not None:
        return pathlib.Path(root) / repo_id
    from lerobot.datasets import lerobot_dataset

    return pathlib.Path(lerobot_dataset.LeRobotDatasetMetadata(repo_id).root)
