"""Compose the critic HUD at deploy time, from a CriticSelectPolicy response + live frames.

The serving policy returns, per replan (when the request carries ``critic_hud``):
``action_samples`` [N, H, A], ``critic_grid`` [N, P], ``critic_choice``, ``critic_best_prefix``,
``critic_macro``. That is everything ``hud.Dashboard`` needs to draw the same HUD we render in
the RoboCasa sim: candidate decision grid (which chunk was chosen), value/spread/commit trace,
and — given a projector — the candidate end-effector fan overlaid on the agent camera.

Robot host usage (sketch):

    rec = HudRecorder(mode="bon", horizon=H, discount=0.9998)
    for t, obs in enumerate(env_stream):
        resp = policy_client.infer({**obs, "critic_select": True, "critic_hud": True})
        rec.add(obs["agentview"], obs.get("wrist"), resp, step=t)   # every replan
        execute(resp["actions"])
    rec.save("deploy_hud.mp4", fps=15)

The on-frame path fan needs a projector ``chunks[N,H,A] -> list of [H,2] image-pixel paths``.
Supply a calibrated ``CameraProjector`` (intrinsics+extrinsics+FK) for a true projection; the
default ``SketchProjector`` integrates two action dims into a labeled corner minimap so the fan
is visible before calibration exists (clearly not a metric projection).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hud


class _Info:
    __slots__ = ("best_cand", "best_prefix", "cand", "n_exec", "q", "value")


def _first_step(array):
    """Serving lays its extras out PER STEP -- leading axis X, matching the actions.

    That is what the robot client records, and what lets the layout survive an adaptive chunk.
    Everything a HUD wants is decided once per replan and then broadcast across X, so taking
    row 0 recovers the per-replan value without assuming any particular X.
    """
    return np.asarray(array, np.float32)[0]


def info_from_response(resp):
    """Rebuild the Replan-like object hud.Dashboard consumes from a serving response."""
    q = _first_step(resp["critic_grid"])  # (X, N, P) -> [N, P]
    # (X, N, A) -> [N, H, A]: the candidates are per-step on the wire, chunk-major here.
    cand = np.swapaxes(np.asarray(resp["action_samples"], np.float32), 0, 1)
    best = int(_first_step(resp["critic_choice"])[0])
    pp = int(_first_step(resp["critic_best_prefix"])[0]) if "critic_best_prefix" in resp else q.shape[1] - 1
    macro = int(_first_step(resp["critic_macro"])[0]) if "critic_macro" in resp else 1
    info = _Info()
    info.q, info.cand, info.best_cand = q, cand, best
    info.best_prefix, info.n_exec, info.value = pp, (pp + 1) * macro, float(q[best, pp])
    return info


class SketchProjector:
    """Calibration-free fan: integrate two action dims into a corner minimap of the agent frame.

    Not a metric projection — a schematic of how the candidate chunks diverge, so the fan is
    visible before camera calibration/FK exist. Swap in a CameraProjector for the real thing.
    """

    def __init__(self, dims=(0, 1), box=(24, 24, 150), gain=1.0):
        self.dims, self.box, self.gain = dims, box, gain

    def __call__(self, chunks):  # chunks [N, H, A] -> list of [H, 2] pixel paths
        d0, d1 = self.dims
        x0, y0, size = self.box
        traj = np.cumsum(chunks[:, :, (d0, d1)], axis=1)  # [N, H, 2] integrated motion
        span = np.abs(traj).max() + 1e-6
        s = self.gain * size / (2 * span)
        cx, cy = x0 + size / 2, y0 + size / 2
        return [np.stack([cx + s * t[:, 0], cy + s * t[:, 1]], axis=1) for t in traj]


class HudRecorder:
    """Accumulate composed HUD frames across a deployment and write the video."""

    def __init__(self, *, mode="bon", horizon, discount=0.9998, camera_size=256, projector=None, hold=5):
        self._dash = hud.Dashboard(mode=mode, horizon=horizon, discount=discount, camera_size=camera_size)
        self._proj = projector
        self._hold = int(hold)  # frames to hold each replan on screen (BoN replans are sparse)
        self._frames = []
        self._step = 0

    def add(self, agent_rgb, wrist_rgb, response, *, step=None, success=False):
        if "critic_grid" not in response:
            raise KeyError("response has no critic_grid — send the request with critic_hud=True")
        info = info_from_response(response)
        paths = self._proj(info.cand) if self._proj is not None else None
        s = self._step if step is None else int(step)
        for _ in range(self._hold):
            self._frames.append(
                np.asarray(
                    self._dash.frame(agent_rgb, wrist_rgb, info, s, paths=paths, chosen=info.best_cand, success=success)
                )
            )
            s += 1
        self._step = s

    def save(self, path, *, fps=15):
        import imageio

        imageio.mimwrite(path, self._frames, fps=fps, quality=8, macro_block_size=1)
        return path
