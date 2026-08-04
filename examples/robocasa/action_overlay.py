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
``obs["robot0_base_quat"]`` (xyzw); ``robot0_agentview_left`` rides on the mobile base, so its
projection matrix is recomputed every frame.

robosuite's project_points_from_world_to_camera already returns rows in the DISPLAY convention (row 0
at the top), the same orientation rollout.image_from_obs produces. Flipping again here inverted the
overlay: measured on a reset scene, lifting the end-effector 15 cm moved the projected row from 137
to 171 — down the screen instead of up.
"""

import numpy as np

_EE_POS_DELTA = slice(5, 8)  # ee-position delta in the 12-d LeRobot action (see rollout.py)


class CameraProjector:
    """World 3-D points → pixel (row, col) in the display frame (row 0 = top)."""

    def __init__(self, sim, camera: str, height: int, width: int):
        from robosuite.utils import camera_utils as _cu

        self._cu = _cu
        # The matrix is recomputed on every projection, NOT cached. `robot0_agentview_left` is named
        # for the robot it is bolted to, and RoboCasa drives a mobile base, so the camera pose is a
        # function of time. Caching it froze the transform at the reset pose: as the base moved the
        # projected points drifted until they left the frame, and robosuite CLIPS out-of-view pixels
        # to the border rather than dropping them, so the paths did not disappear - they piled up
        # against the right edge, which is exactly how the bug presented.
        self._sim, self._cam = sim, camera
        self._h, self._w = height, width

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """[n, 3] world → [n, 2] (row, col), row 0 = top, matching the displayed frame."""
        w2p = self._cu.get_camera_transform_matrix(self._sim, self._cam, self._h, self._w)
        px = self._cu.project_points_from_world_to_camera(np.asarray(points_world), w2p, self._h, self._w)
        return np.asarray(px, dtype=np.float64)


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


# Metres of end-effector motion per unit of normalised action — MEASURED, not read off the
# controller config.
#
# The config says OSC_POSE maps [-1, 1] to [-0.05, 0.05] m per step, and this constant used to be
# that 0.05. But that is what the controller is COMMANDED to reach, not what the arm achieves: OSC is
# impedance control, so a single step lands well short of its target, and contacts and joint limits
# take more. Fitting actual EE displacement against commanded action over 600 steps
# (slurm/probes/scale_fit.py, least squares, PrepareCoffee seed 0) gives 0.0054 m/unit — the config
# value drew every path about 9x too long.
#
#   per-axis fit  0.0058 / 0.0053 / 0.0052   (close enough that one scalar is honest)
#   correlation   0.72                       (not 1.0: contacts and limits make it non-linear)
#
# The same file already carries the opposite mistake in _adaptive_scale, which drew paths ~2x too
# SHORT. Both came from assuming a number instead of measuring one.
EE_METRES_PER_UNIT = 0.0054

# Drawing magnification, applied in WORLD space (predict_path integrates scale*deltas from the live
# EE position, so the anchor is unaffected and the enlarged path still projects with correct
# perspective). A 16-step chunk really moves ~3 cm - about 5 px on this camera - so an unmagnified
# fan is smaller than its own endpoint dot. x6 makes the spread legible; the HUD prints the factor
# on every frame so the length is never mistaken for a real distance. The base constant already IS
# the control-frequency-corrected effective displacement (the config's 0.05 m is the COMMANDED
# offset per step; what one control period actually achieves is the measured 0.0054), so the gain is
# purely a legibility knob - and 2x proved enough once the fans were anchored and world-scaled.
PATH_GAIN = 2.0


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
