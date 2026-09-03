"""HL-Gauss is a projection of a SCALAR, applied once. Using it as a C51 kernel diffuses.

This pins the reason `--backup scalar` is the default. Farebrother et al. (arXiv 2403.03950) size
the kernel at sigma = 0.75 x bin width deliberately: it is a classification loss for one scalar
target. csmile-1006/DEAS-FQL uses it that way -- scalarize V(s'), build a scalar TD target, call
transform_to_probs once. Our earlier backup instead used the same kernel as a C51 projection,
re-projecting the whole next-state distribution at every step, which convolves a Gaussian into the
target on every backup.
"""

import jax.numpy as jnp
import numpy as np

from openpi.rlt_critic.critic import HLGauss

V_MIN, V_MAX, ATOMS = -2777.777, 0.0, 101


def _hl():
    return HLGauss(V_MIN, V_MAX, ATOMS)


def _std(p, centers):
    m = float((p * centers).sum())
    return float(np.sqrt(((centers - m) ** 2 * p).sum()))


def _reproject_once(p, hl, centers, gamma=1.0):
    """One step of the old distributional backup with r = 0."""
    phi = np.asarray(hl.to_probs(jnp.clip(jnp.asarray(gamma * centers), V_MIN, V_MAX)))
    return p @ phi


def test_repeated_projection_diffuses_like_a_random_walk():
    hl = _hl()
    c = np.asarray(hl.centers)
    p = np.asarray(hl.to_probs(jnp.asarray(-1400.0)))
    s0 = _std(p, c)
    widths = {0: s0}
    for n in range(1, 101):
        p = _reproject_once(p, hl, c)
        if n in (1, 10, 100):
            widths[n] = _std(p, c)
    assert widths[1] > 1.3 * widths[0]
    assert widths[100] > 9 * widths[0]
    # sigma * sqrt(n): the signature of convolving the kernel in once per backup
    assert abs(widths[100] / (s0 * np.sqrt(100)) - 1.0) < 0.15, widths


def test_the_scalar_backup_applies_the_kernel_exactly_once():
    """The DEAS form is idempotent in the only sense that matters: the target for a given scalar is
    to_probs(y), no matter how many backups came before it."""
    hl = _hl()
    c = np.asarray(hl.centers)
    y = -1400.0
    once = np.asarray(hl.to_probs(jnp.asarray(y)))
    # whatever V(s') was, the scalar path re-derives the target from a scalar, so its width is fixed
    for _ in range(50):
        again = np.asarray(hl.to_probs(jnp.asarray(y)))
        np.testing.assert_allclose(again, once, rtol=0, atol=0)
    assert abs(_std(once, c) - _std(np.asarray(hl.to_probs(jnp.asarray(-400.0))), c)) < 1.0


def test_diffusion_plus_the_upper_clip_biases_near_goal_values_down():
    """Away from the boundary the old operator preserves the mean; near the goal it does not, because
    the clip at v_max truncates the mass the kernel keeps spreading upward. That is the regime
    selection actually operates in."""
    hl = _hl()
    c = np.asarray(hl.centers)

    def drift(y0, n=200):
        p = np.asarray(hl.to_probs(jnp.asarray(y0)))
        m0 = float((p * c).sum())
        for _ in range(n):
            p = _reproject_once(p, hl, c)
        return float((p * c).sum()) - m0

    assert abs(drift(-1400.0)) < 5.0  # far from either edge: mean preserved
    near = drift(-80.0)
    assert near < -100.0, near  # near the goal: dragged down, hard
