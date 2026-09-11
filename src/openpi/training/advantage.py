"""Per-frame advantage labels for the policy-extraction arms.

`scripts/annotate_advantage.py` writes `q_data.npy` / `v_data.npy` over a dataset: one Q and one V
per frame, in LeRobot global-frame-index order (the same order the patch-feature cache uses, so row
i of either file is frame i of the dataset). This module turns that pair into the label the arms
actually consume and attaches it to a sample.

Normalization lives here, not in the model, because it is a statistic of the WHOLE dataset --
computing it per batch would make a sample's label depend on who it was batched with, exactly the
bug xbpeng/awr avoids by normalizing over the replay buffer (awr_agent.py:403).

Which normalization is not a detail the arms share, so the data config states it and the arm's
`with_*` transform sets it:

  ``zscore``  AWR. (A - mean) / (std + eps); the model exponentiates and clips it.
  ``raw``     CFGRL. A itself, because its label is the hard indicator 1{A > 0}
              (iql_diffusion.py:157) and a z-score would silently retarget that threshold to
              1{A > mean(A)} -- a different, larger set, with nothing to notice it by.
"""

import dataclasses
import logging
import pathlib

import numpy as np

import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def load_advantage(advantage_dir: str | pathlib.Path, *, normalize: str = "zscore", eps: float = 1e-5) -> np.ndarray:
    """Load `q_data.npy` / `v_data.npy` and return A = Q - V, one entry per frame.

    `normalize` is "zscore" (eps matches awr_agent.py:403, `(adv - mean) / (std + 1e-5)`) or "raw".
    See the module docstring for why the choice belongs to the arm.
    """
    if normalize not in ("zscore", "raw"):
        raise ValueError(f"normalize must be 'zscore' or 'raw', got {normalize!r}")
    d = pathlib.Path(advantage_dir)
    missing = [f for f in ("q_data.npy", "v_data.npy") if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{d} is missing {', '.join(missing)}. An extraction arm's config names this annotation, "
            "and it is produced by misc/scripts/annotate_advantage.py from a trained patch critic "
            "and its feature cache. If the workspace moved, repoint _ADVANTAGE_DIR in "
            "openpi/training/config_acrft.py rather than passing a flag at launch."
        )
    q = np.load(d / "q_data.npy")
    v = np.load(d / "v_data.npy")
    if q.shape != v.shape:
        raise ValueError(f"{d}: q_data {q.shape} and v_data {v.shape} disagree")
    adv = np.asarray(q - v, dtype=np.float32)
    out = adv if normalize == "raw" else (adv - adv.mean()) / (adv.std() + eps)
    logger.info(
        "advantage %s: n=%d  raw mean %.4f std %.4f  frac>0 %.3f  -> %s",
        d,
        adv.size,
        float(adv.mean()),
        float(adv.std()),
        float((adv > 0).mean()),
        normalize,
    )
    return out.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class AddAdvantage(_transforms.DataTransformFn):
    """Attach the frame's normalized advantage to a raw LeRobot item.

    Runs before the repack transform, while the raw LeRobot keys are still present. `index` is
    LeRobot's global frame index, which is the row the annotation files are keyed by. Like
    `AddProgress`, this is a TRAINING-only label: at inference the raw item carries no `index`, so
    the transform is a no-op and the same chain serves.
    """

    advantage: np.ndarray

    def __call__(self, data: dict) -> dict:
        if "index" not in data:
            return data
        idx = np.asarray(data["index"], dtype=np.int64)
        if idx.max(initial=0) >= self.advantage.shape[0]:
            raise IndexError(
                f"frame index {int(idx.max())} is past the {self.advantage.shape[0]} annotated frames. "
                "The annotation was computed over a different dataset (or a different episode subset) "
                "than this config trains on -- they have to be the same dataset in the same order."
            )
        return {**data, "advantage": self.advantage[idx]}
