"""Overlay the BC policy's action distribution on eval videos.

At every replan the policy is a distribution over action chunks (flow matching maps different noise
to different chunks). We sample N chunks, turn each into a predicted end-effector path, and draw all
N as translucent lines on the camera frame — so the video shows whether the policy is confident (a
tight bundle) or uncertain / multi-modal (a wide spray).

The 12-d action carries an EE-position delta (OSC_POSE) expressed in the robot BASE frame. To draw a
chunk we integrate those deltas from the current EE world position, rotating each into world with the
base orientation:

    world Δee_k = R(base_quat) @ (scale * ee_pos_delta_k)

The scale is chosen *adaptively* per replan so the executed chunk's path spans a fixed target world
length (``target_len``, ~0.25 m): raw action magnitudes are unknown and vary (a policy's unnormalized
OSC deltas can be ~1.0, which at a fixed scale would shoot the paths off-screen), so we normalize the
executed path to a legible length and apply that same scale to every candidate. Absolute magnitude is
thus not to scale, but the *relative* spread across candidates — the thing that shows confidence vs
multi-modality — is preserved. This is an approximate kinematic preview of *where the policy wants to
go*, frame-correct, and needs no online calibration (which was fragile: base motion contaminated it
and the spray flew off).

Robosuite/RoboCasa specifics: EE world position is ``obs["robot0_eef_pos"]``, base orientation is
``obs["robot0_base_quat"]`` (xyzw); the agentview camera is world-fixed so its projection matrix is
computed once. Frames are flipped vertically for display (rollout.image_from_obs), so projected rows
are flipped to match.
"""

import numpy as np

_EE_POS_DELTA = slice(5, 8)  # ee-position delta in the 12-d LeRobot action (see rollout.py)


class CameraProjector:
    """World 3-D points → pixel (row, col) in the *display* (vertically flipped) frame."""

    def __init__(self, sim, camera: str, height: int, width: int):
        from robosuite.utils import camera_utils as _cu

        self._cu = _cu
        self._w2p = _cu.get_camera_transform_matrix(sim, camera, height, width)
        self._h, self._w = height, width

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """[n, 3] world → [n, 2] (row, col) in the flipped display frame."""
        px = self._cu.project_points_from_world_to_camera(np.asarray(points_world), self._w2p, self._h, self._w)
        px = np.asarray(px, dtype=np.float64)
        px[:, 0] = self._h - 1 - px[:, 0]  # flip row to match the flipped display image
        return px


def _base_rotation(base_quat) -> np.ndarray:
    """Base->world rotation matrix from a robosuite (xyzw) quaternion."""
    from robosuite.utils import transform_utils as _tu

    return _tu.quat2mat(np.asarray(base_quat, dtype=np.float64))


def predict_path(ee_world, base_quat, chunk, scale: float) -> np.ndarray:
    """Predicted world EE path: integrate the chunk's base-frame ee-pos deltas from ee_world.

    chunk: [H, 12] raw LeRobot action. Returns [H+1, 3] including the start point.
    """
    r = _base_rotation(base_quat)
    p = np.asarray(ee_world, dtype=np.float64).copy()
    path = [p.copy()]
    for step in chunk:
        p = p + r @ (scale * np.asarray(step[_EE_POS_DELTA], dtype=np.float64))
        path.append(p.copy())
    return np.stack(path)


# Metres of end-effector motion per unit of normalised action, read off the controller the
# evaluation env actually runs: robosuite OSC_POSE for PandaOmron maps input [-1, 1] to
# output [-0.05, 0.05] m per step (default_pandaomron.json, body_parts.arms.right). With this the
# drawn path IS the predicted trajectory and lands where the gripper goes.
EE_METRES_PER_UNIT = 0.05


def _adaptive_scale(chunk, target_len: float) -> float:
    """Deprecated: rescaled the path to a fixed on-screen length instead of its true magnitude.

    It divided a target length by the chunk's own integrated displacement, so the drawn path was
    about 2x too short on average (measured: raw_len 4.73 against a 0.12 m target gives 0.025 against
    the true 0.05) and its length changed every replan - including a division by ~0 for a chunk that
    barely moves. Kept only so old call sites do not break; pass EE_METRES_PER_UNIT instead.
    """
    raw_len = float(np.abs(np.asarray(chunk)[:, _EE_POS_DELTA]).sum(axis=0).max())
    return target_len / max(raw_len, 1e-6)


def draw_overlay(
    frame: np.ndarray,
    projector: CameraProjector,
    ee_world,
    base_quat,
    candidate_chunks,
    executed_idx: int = 0,
    target_len: float = 0.25,
) -> np.ndarray:
    """Draw the N predicted EE paths on ``frame`` (uint8 HxWx3), anchored at ``ee_world``.

    Each candidate is a translucent line ending in a dot (so the spread reads even when the lines are
    short); the executed chunk is a bright opaque line. The draw scale is derived from the executed
    chunk so its path spans ~``target_len`` world metres (see module docstring); all candidates share
    it, so their relative spread is faithful while absolute magnitude is not to scale.
    """
    from PIL import Image
    from PIL import ImageDraw

    scale = _adaptive_scale(candidate_chunks[executed_idx], target_len)
    base = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, chunk in enumerate(candidate_chunks):
        path = predict_path(ee_world, base_quat, chunk, scale)
        px = projector.project(path)  # [H+1, 2] (row, col)
        pts = [(float(c), float(r)) for r, c in px]  # PIL wants (x=col, y=row)
        ex, ey = pts[-1]
        if i == executed_idx:
            d.line(pts, fill=(255, 210, 30, 255), width=3)  # executed: bright yellow, opaque
            d.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(255, 210, 30, 255))
        else:
            d.line(pts, fill=(80, 150, 255, 130), width=2)  # candidates: translucent blue
            d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(80, 150, 255, 200))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))
