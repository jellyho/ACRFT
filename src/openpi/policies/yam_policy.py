"""Input/output transforms for the YAM bimanual dataset (jellyho/yam_lego_taxi, LeRobot v3).

YAM is a two-arm robot logged at 30 fps with three cameras (agentview, wrist_left, wrist_right), a
42-d proprioceptive state, and a 14-d action. The action is JOINT space: per arm, six joint targets
and one gripper (0..1), so [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]. The state's
first 14 dims are the two arms' joint positions in the same order (left.pos 0..6, right.pos 0..6 sit
at state indices 0..6 and 21..27), which is what a joint-space delta needs as its reference.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# State layout (see meta/info.json names): each arm block is pos(7), vel(7); left is 0..13,
# right is 21..34. Only the position part is a valid reference for a joint delta.
LEFT_JOINT_STATE = slice(0, 6)
RIGHT_JOINT_STATE = slice(21, 27)
# Action layout: left joints 0..5, left gripper 6, right joints 7..12, right gripper 13.
ACTION_DIM = 14
GRIPPER_ACTION_IDX = (6, 13)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def joint_delta_reference() -> np.ndarray:
    """A 14-d vector aligning each action dim to the state entry a joint delta subtracts.

    Joints subtract the current joint position; grippers subtract nothing (kept absolute). Returned
    as the state indices to read, with -1 meaning "no reference" (absolute).
    """
    ref = np.full(ACTION_DIM, -1, dtype=np.int64)
    ref[0:6] = np.arange(LEFT_JOINT_STATE.start, LEFT_JOINT_STATE.stop)
    ref[7:13] = np.arange(RIGHT_JOINT_STATE.start, RIGHT_JOINT_STATE.stop)
    return ref


@dataclasses.dataclass(frozen=True)
class YAMInputs(transforms.DataTransformFn):
    """Convert a YAM sample into the model input format (training and inference)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Pi0 exposes one third-person slot and two wrist slots; YAM fills all three with real views.
        inputs = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/image_right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "progress" in data:
            inputs["progress"] = data["progress"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YAMOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the 14-d YAM action space (drops the padding to model dim)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
