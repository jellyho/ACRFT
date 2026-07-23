import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the RoboCasa policy."""
    return {
        # 16-d proprioceptive state (base pose 7, relative EE pose 7, gripper qpos 2).
        "observation/state": np.random.rand(16),
        # Three 256x256 RGB cameras: one exterior (agentview) + one wrist + one more exterior.
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/image_right": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "close the lid of the blender",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Converts a RoboCasa 365 (PandaOmron) sample into the model input format.

    Used for both training and inference. RoboCasa provides three camera streams
    (``robot0_agentview_left``, ``robot0_eye_in_hand``, ``robot0_agentview_right``), a 16-d state,
    and a 12-d action; the repack transform in ``LeRobotRoboCasaDataConfig`` maps the LeRobot keys
    onto the ``observation/*`` keys read below.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot stores images as float32 (C, H, W); parse back to uint8 (H, W, C).
        # Pi0 exposes one third-person slot and two wrist slots. RoboCasa has two exterior
        # (agentview) views and one wrist (eye-in-hand) view, so we fill:
        #   base_0_rgb       <- agentview_left  (third person)
        #   left_wrist_0_rgb <- eye_in_hand     (wrist)
        #   right_wrist_0_rgb<- agentview_right  (second exterior; a real image, so it's unmasked)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_image = _parse_image(data["observation/image_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only present during training; padded to the model action dim downstream.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Natural-language instruction (populated from the LeRobot task when prompt_from_task=True).
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Auxiliary task-progress target for Pi0RLT; only present when the data config injects it.
        if "progress" in data:
            inputs["progress"] = data["progress"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the RoboCasa action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Return the first 12 actions: [base_motion(4), control_mode(1), ee_pos(3), ee_rot(3),
        # gripper(1)]. The rest is padding added to reach the model action dimension.
        return {"actions": np.asarray(data["actions"][..., :12])}
