"""The squash-vs-pad contract, and why no shape check can enforce it.

The critic's feature cache was built by SQUASHING native 480x640 frames to 224x224 with
cv2.INTER_AREA -- aspect ratio deliberately not preserved. openpi's own documented client,
examples/droid/main.py and the lab's YAM bridge all pre-process with `resize_with_pad`, which
letterboxes. A client that does that hands over an already-224x224 frame, the server's resize
becomes a no-op, and the critic scores a letterboxed image it never saw in training. Nothing raises.

Measured cost of that convention on the deployed checkpoint (18 frames / 6 episodes): patch tokens
drift by 0.636 relative L2 and V moves by 222.5 mean / 894 max, positive on every frame -- the critic
reads the state as systematically closer to the goal -- against a V spread of 338 and a whole
arg-max selection effect of +100.6.
"""

import logging

import numpy as np

from openpi.policies.patch_critic_policy import _parse_image


def _native(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _resize_with_pad(img, size):
    """The convention the clients use: fit inside, then letterbox. Mirrors openpi's own helper."""
    import cv2

    h, w = img.shape[:2]
    s = min(size / h, size / w)
    r = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y : y + r.shape[0], x : x + r.shape[1]] = r
    return out


def test_native_frames_reach_the_cache_convention():
    """A native frame is squashed here, which is what the cache did, so the contract holds."""
    img = _native()
    out = _parse_image(img, 224)
    assert out.shape == (224, 224, 3)


def test_the_two_conventions_produce_different_images_at_the_same_shape():
    """The defect in one assertion: same shape, different pixels, no error anywhere.

    If this ever fails the two conventions have converged and the guard below is unnecessary.
    """
    img = _native()
    squashed = _parse_image(img, 224)
    padded = _parse_image(_resize_with_pad(img, 224), 224)
    assert squashed.shape == padded.shape
    drift = np.linalg.norm(squashed.astype(float) - padded.astype(float)) / np.linalg.norm(squashed.astype(float))
    assert drift > 0.1, f"conventions differ by only {drift:.3f}; the guard may be obsolete"


def test_parse_image_reports_the_geometry_it_received():
    """`arrived` is what makes the check possible at all -- after the resize the evidence is gone."""
    got: list = []
    _parse_image(_native(480, 640), 224, arrived=got)
    _parse_image(_resize_with_pad(_native(480, 640), 224), 224, arrived=got)
    assert got == [(480, 640), (224, 224)]


def test_an_already_square_arrival_warns(caplog):
    """The only signal available at serving time: squareness. A 224x224 arrival is ambiguous between
    the two conventions, so it cannot be decided -- only flagged."""
    from openpi.policies.patch_critic_policy import PatchCriticSelectPolicy

    obj = object.__new__(PatchCriticSelectPolicy)
    obj._arrived_hw = None
    obj._spec = {"source_hw": [480, 640]}
    with caplog.at_level(logging.WARNING):
        obj._note_geometry([(224, 224), (224, 224), (224, 224)])
    assert "ALREADY SQUARE" in caplog.text

    caplog.clear()
    obj._arrived_hw = None
    with caplog.at_level(logging.WARNING):
        obj._note_geometry([(480, 640)] * 3)
    assert caplog.text == "", "native frames are the contract; they must not warn"
