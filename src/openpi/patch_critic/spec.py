"""The patch-critic's INPUT CONTRACT: what scale and layout of inputs a saved critic expects.

Why this exists. The critic is trained on inputs read straight out of the LeRobot dataset -- it never
passes through openpi's data transforms, so its state and actions are in RAW physical units (joint
angles in radians, gripper in [0,1], velocity/force channels with their own scales). The serving
wrapper honours that: it takes ``observation/state`` from BEFORE the input transform and pushes
candidate chunks through the OUTPUT transform to un-normalize them back to physical units.

Those two facts live in two unrelated files and agreed only by convention. Nothing in the checkpoint
recorded them, so a critic trained on normalized inputs would be served raw and would fail SILENTLY --
returning confident, meaningless values. This module writes the contract into the checkpoint and lets
the server check it at load time.
"""

import json
import pathlib

import numpy as np

# The critic eats raw dataset units. If a future variant normalizes, bump this and teach the server.
NORMALIZATION = "none"

PREPROCESS = "float[0,1] CHW -> uint8 HWC -> cv2.resize INTER_AREA -> DINOv2 (frozen), 2x2 pooled"


def input_spec(meta: dict, *, horizon: int) -> dict:
    """Build the input contract from a feature cache's meta.json."""
    return {
        "normalization": NORMALIZATION,
        "state_units": "raw LeRobot observation.state -- NOT openpi-normalized",
        "action_units": "raw LeRobot action -- NOT openpi-normalized",
        "state_dim": int(meta["sd"]),
        "action_dim": int(meta["ad"]),
        "horizon": int(horizon),
        "cameras": list(meta["cams"]),  # ORDER IS PART OF THE CONTRACT
        "num_cameras": len(meta["cams"]),
        "img_size": int(meta["img_size"]),
        # How the frames the cache was built from were mapped to img_size. `squash` does not preserve
        # aspect ratio, so a client that letterboxed with resize_with_pad supplies a DIFFERENT image
        # at the same shape -- which no shape check can catch. Absent on pre-2026-09 caches.
        "source_hw": meta.get("source_hw"),
        "resize_mode": meta.get("resize_mode", "squash"),
        "image_preprocess": PREPROCESS,
        "backbone": meta["backbone"],
        "num_patches": int(meta["npatch"]),
        "embed_dim": int(meta["emb"]),
    }


def norm_stats(cache: pathlib.Path, meta: dict, *, stride: int = 997) -> dict:
    """Empirical state/action statistics of the frames the critic was trained on.

    Not used to normalize anything (the critic is trained raw) -- they are the reference distribution
    the server compares incoming observations against, which is what turns a units mismatch from a
    silent failure into a loud one. Sampled with a stride; exact values are not needed for that job.
    """
    n, sd, ad = int(meta["N"]), int(meta["sd"]), int(meta["ad"])
    states = np.memmap(cache / "state.dat", np.float32, "r", shape=(n, sd))
    actions = np.memmap(cache / "action.dat", np.float32, "r", shape=(n, ad))
    idx = np.arange(0, n, stride)
    out = {"n_frames": n, "n_sampled": len(idx), "stride": stride}
    for name, arr in (("state", np.asarray(states[idx])), ("action", np.asarray(actions[idx]))):
        out[name] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return out


def check(spec: dict, *, model_action_dim: int, num_cameras: int, img_size: int) -> list[str]:
    """The ways a runtime cannot serve a critic (empty list = servable).

    Only genuine incompatibilities belong here. In particular the critic's action_dim is the ROBOT
    width (14 for YAM) while the server's is the model's PADDED width (32 for pi05); the wrapper
    slices, so those differing is normal and only the padded width being too narrow is a fault.
    """
    from openpi.patch_critic import preproc

    problems = []
    norm = spec.get("normalization", "raw")
    if norm not in ("raw", *preproc.MODES):
        problems.append(f"critic declares normalization={norm!r}, which this server does not implement")
    a_dim = spec.get("action_dim")
    if a_dim is not None and model_action_dim < a_dim:
        problems.append(f"model action width {model_action_dim} is narrower than the critic's {a_dim}")
    for what, got, want in (
        ("num_cameras", num_cameras, spec.get("num_cameras")),
        ("img_size", img_size, spec.get("img_size")),
    ):
        if want is not None and got != want:
            problems.append(f"{what}: critic trained with {want}, server supplies {got}")
    # resize_mode is deliberately NOT a problem here even when it disagrees, because this function
    # only sees server constants: what actually arrives from the client is a per-request fact and is
    # checked there (PatchCriticSelectPolicy._note_geometry). Recording it is what makes that check
    # possible at all -- see the comment on the field in input_spec.
    return problems


def out_of_range(stats: dict, state: np.ndarray, *, slack: float = 4.0) -> list[int]:
    """Indices of state channels that sit far outside the training range (a units-mismatch tripwire).

    Compares against the training [q01, q99] band widened by ``slack`` band-widths, so ordinary
    extrapolation is quiet and a wrong scale (or a permuted state vector) is not.
    """
    lo = np.asarray(stats["state"]["q01"], np.float64)
    hi = np.asarray(stats["state"]["q99"], np.float64)
    pad = slack * np.maximum(hi - lo, 1e-6)
    s = np.asarray(state, np.float64).reshape(-1)
    if s.shape != lo.shape:
        return list(range(len(s)))
    return np.nonzero((s < lo - pad) | (s > hi + pad))[0].tolist()


def load(critic_dir) -> tuple[dict, dict | None]:
    """(config, norm_stats-or-None) for a saved critic directory."""
    d = pathlib.Path(critic_dir)
    cfg = json.loads((d / "config.json").read_text())
    p = d / "norm_stats.json"
    return cfg, (json.loads(p.read_text()) if p.exists() else None)
