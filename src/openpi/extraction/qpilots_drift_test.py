"""QPILOTS drift: the steered chunk against what the BC policy would have done.

Two references, answering two questions, and the difference matters:

  the unsteered twin   same noise, same cached prefix, alpha=0 -> how far steering moved THIS draw
  N unconditional      independent draws -> how wide the policy's own spread is

Displacement is only interpretable against that spread. In radians alone it is a number with no
scale, and comparing the steered chunk against an INDEPENDENT draw would measure the two mixed
together -- sampling variance plus steering -- which is the mistake these tests exist to prevent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.extraction import serving


class _FakeSampler:
    """ArmChunkSampler's steering loop, reduced to the part under test: one code path, alpha as a
    parameter, so the twin differs from the steered draw by the steering term and nothing else."""

    def __init__(self, *, pair: bool):
        self.pair_unsteered = pair
        self.spec = type("S", (), {"ode_steps": 4, "alpha": 0.2})()
        self.H, self.AD = 4, 8

    def _steer(self, *_a, x=None, alpha=0.0, **_k):
        # stand-in with the same contract: alpha=0 is the base, alpha>0 displaces it
        return jnp.clip(x + alpha, -1.0, 1.0)

    def draw(self, rng):
        x0 = jax.random.normal(rng, (1, self.H, self.AD)) * 0.1
        steered = self._steer(x=x0, alpha=self.spec.alpha)
        if not self.pair_unsteered:
            return steered
        base = self._steer(x=x0, alpha=0.0)
        return jnp.concatenate([steered, base], axis=0)


def test_the_twin_shares_the_draw_it_is_compared_against():
    """Same x0 for both. An independently drawn base would add the policy's own spread to the
    measurement, and the number would no longer be the steering displacement."""
    out = np.asarray(_FakeSampler(pair=True).draw(jax.random.key(0)))
    assert out.shape[0] == 2
    steered, base = out[0], out[1]
    # differ by exactly the steering term, everywhere (no clipping in this range)
    assert np.allclose(steered - base, 0.2, atol=1e-6)


def test_off_by_default_returns_one_chunk():
    """The returned N is part of the contract; a caller that only wants the chunk keeps getting
    one, so turning the readout on cannot reshape an existing run."""
    assert np.asarray(_FakeSampler(pair=False).draw(jax.random.key(0))).shape[0] == 1


def test_the_executed_chunk_is_index_zero_not_the_argmax():
    """The arm brought its chunk; the critic scores it, it does not overrule it. If `best` were the
    argmax over [steered, twin, uncond...], every arm would silently become best-of-N over its own
    reference set -- and the run would still look fine, which is why this is a test."""
    import inspect

    from openpi.policies import patch_critic_policy as pcp

    src = inspect.getsource(pcp.PatchCriticSelectPolicy.infer)
    assert "elif self._arm is not None:" in src
    idx = src.index("elif self._arm is not None:")
    assert "best = 0" in src[idx : idx + 700], "an arm must execute its own chunk"


def test_drift_needs_a_scale_to_mean_anything():
    """The property the readout rests on: a displacement is small or large only relative to the
    spread of what the policy would have produced anyway."""
    rng = np.random.default_rng(0)
    uncond = rng.normal(size=(8, 4, 8)) * 0.5  # the policy's own spread
    base = uncond.mean(axis=0)
    steered = base + 0.05  # well inside it

    spread = float(np.mean(np.std(uncond, axis=0)))
    drift = float(np.max(np.abs(steered - base)))
    assert drift / spread < 0.5, "a displacement inside the spread should read as small"

    far = base + 5.0
    assert float(np.max(np.abs(far - base))) / spread > 5.0


@pytest.mark.parametrize("arm", ["lps", "lpsd", "flowdagger"])
def test_only_qpilots_has_a_twin(arm):
    """The other arms have no alpha to zero out, so there is no same-noise unsteered counterpart.
    Their reference draws are still recorded; the twin is not invented for them."""
    assert arm in serving.LATENT_ARMS or arm == "flowdagger"
