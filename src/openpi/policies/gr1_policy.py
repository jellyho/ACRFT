"""GR1 tabletop (Teleop-Sim) policy transforms.

The dataset (nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim, LeRobot layout) carries ONE camera
(``observation.images.ego_view`` 256x256), a 44-d state and a 44-d action
(GR1ArmsAndWaistFourierHands). The repack transform in ``LeRobotGR1DataConfig`` maps the LeRobot
keys onto the ``observation/*`` keys read below.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.einsum("chw->hwc", image)
    return image


@dataclasses.dataclass(frozen=True)
class GR1Inputs(transforms.DataTransformFn):
    """GR1 sample -> model input. Single ego view: wrist slots are zero-filled and masked out."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # PI0_FAST attends masked slots anyway; standard pi0/pi05 masks them out.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        for k in ("progress", "episode_index"):
            if k in data:
                inputs[k] = data[k]
        return inputs


@dataclasses.dataclass(frozen=True)
class GR1Outputs(transforms.DataTransformFn):
    """Model output -> GR1 action space (truncate the padded action dim back to 44)."""

    action_dim: int = 44

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
