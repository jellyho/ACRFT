"""Per-frame advantage labels for the policy-extraction arms.

`scripts/annotate_advantage.py` writes `q_data.npy` / `v_data.npy` over a dataset: one Q and one V
per frame, in LeRobot global-frame-index order (the same order the patch-feature cache uses, so row
i of either file is frame i of the dataset). This module turns that pair into the label the arms
actually consume and attaches it to a sample.

The z-score lives here, not in the model, because it is a statistic of the WHOLE dataset --
computing it per batch would make the weights depend on batch composition, which is exactly the bug
xbpeng/awr avoids by normalizing over the replay buffer (awr_agent.py:403). Temperature and clipping
stay on the model config, where they are swept.
"""

import dataclasses
import logging
import pathlib

import numpy as np

import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def normalized_advantage(advantage_dir: str | pathlib.Path, *, eps: float = 1e-5) -> np.ndarray:
    """Load `q_data.npy` / `v_data.npy` and return A = z-score(Q - V), one entry per frame.

    eps matches awr_agent.py:403 (`(adv - mean) / (std + 1e-5)`).
    """
    d = pathlib.Path(advantage_dir)
    q = np.load(d / "q_data.npy")
    v = np.load(d / "v_data.npy")
    if q.shape != v.shape:
        raise ValueError(f"{d}: q_data {q.shape} and v_data {v.shape} disagree")
    adv = np.asarray(q - v, dtype=np.float32)
    norm = (adv - adv.mean()) / (adv.std() + eps)
    logger.info(
        "advantage %s: n=%d  raw mean %.4f std %.4f -> normalized std %.4f",
        d,
        norm.size,
        float(adv.mean()),
        float(adv.std()),
        float(norm.std()),
    )
    return norm.astype(np.float32)


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
