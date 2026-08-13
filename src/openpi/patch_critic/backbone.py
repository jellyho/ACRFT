"""Independent frozen DINOv2 (JAX/Flax) patch backbone for the patch-critic.

DINOv2 is self-supervised with dense objectives, so its patch tokens carry strong SPATIAL structure
(where each object is, contact geometry) -- unlike image-text SigLIP whose patches are globally
aligned. That spatial density is exactly what a value function scoring manipulation states needs.

Runs natively in JAX on B200 via ``transformers.FlaxDinov2Model`` (no PyTorch, which lacks sm_100
kernels in this env). The backbone is FROZEN: params live outside the trained critic and are applied
under stop_gradient.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# ImageNet stats DINOv2 was trained with.
_MEAN = jnp.asarray([0.485, 0.456, 0.406], jnp.float32)[:, None, None]
_STD = jnp.asarray([0.229, 0.224, 0.225], jnp.float32)[:, None, None]

_VARIANTS = {
    "small": "facebook/dinov2-small",  # 384-d, 22M
    "base": "facebook/dinov2-base",  # 768-d, 86M
    "large": "facebook/dinov2-large",  # 1024-d, 300M
}


class DinoV2Backbone:
    """Frozen DINOv2. ``__call__`` maps images [B, Ncam, 3, H, W] in [0,1] -> patches [B, Ncam*P, D]."""

    def __init__(self, variant: str = "small", dtype=jnp.float32):
        from transformers import FlaxDinov2Model

        self.model = FlaxDinov2Model.from_pretrained(_VARIANTS[variant], dtype=dtype)
        self.embed_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

    @functools.partial(jax.jit, static_argnums=0)
    def _encode_one(self, pixel_values):
        # pixel_values [b, 3, H, W], already ImageNet-normalized. Drop the CLS token (index 0).
        out = self.model(pixel_values=pixel_values, params=self.model.params)
        return jax.lax.stop_gradient(out.last_hidden_state[:, 1:])  # [b, P, D]

    def __call__(self, images: jax.Array) -> jax.Array:
        """images [B, Ncam, 3, H, W] in [0,1] -> patch tokens [B, Ncam*P, D]."""
        b, ncam = images.shape[:2]
        x = images.reshape(b * ncam, *images.shape[2:])
        x = (x - _MEAN) / _STD
        patches = self._encode_one(x)  # [B*Ncam, P, D]
        p = patches.shape[1]
        return patches.reshape(b, ncam * p, self.embed_dim)

    def num_patches(self, image_hw: int) -> int:
        return (image_hw // self.patch_size) ** 2


def to_nchw(imgs: np.ndarray) -> np.ndarray:
    """[.., H, W, 3] uint8/float -> [.., 3, H, W] float32 in [0,1]."""
    x = np.asarray(imgs)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    if x.shape[-1] == 3:
        x = np.moveaxis(x, -1, -3)
    return x.astype(np.float32)
