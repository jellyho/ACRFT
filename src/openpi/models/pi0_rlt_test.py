"""Tests for the RLT bottleneck's objective terms.

These run on CPU in seconds because they exercise the loss functions directly rather than through a
PaliGemma forward. The end-to-end shape/pytree check (every objective string, real model) lives in
scripts/rlt_objective_smoke.py, which needs a GPU.
"""

import types

import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_rlt


def _behsim_stub(beta=0.1, tau=0.1):
    """A minimal object with just what `_behsim_loss` reads, with an identity projection.

    Calling the unbound method on this keeps the test off the 3B backbone; the identity projection
    means the test measures the LOSS, not whatever a randomly-initialized MLP happens to do.
    """

    class Ident:
        def __call__(self, x):
            return x

    return types.SimpleNamespace(
        _rlt_act_dim=12,
        rlt_behsim_beta=beta,
        rlt_behsim_tau=tau,
        rlt_behsim_proj_in=Ident(),
        rlt_behsim_proj_out=Ident(),
    )


def _clustered_actions(n_clusters=3, per_cluster=2, horizon=16, action_dim=32):
    """Action chunks with `n_clusters` distinct behaviours, repeated across "episodes"."""
    centers = jax.random.normal(jax.random.key(0), (n_clusters, horizon, action_dim))
    acts = jnp.repeat(centers, per_cluster, axis=0)
    acts = acts + 0.01 * jax.random.normal(jax.random.key(1), acts.shape)
    return acts, jnp.repeat(jnp.arange(n_clusters), per_cluster)


def test_behsim_prefers_behaviour_over_episode_identity():
    """The whole point of the term, in one assertion.

    RoboCasa's demos differ visually far more than they differ behaviourally, so a reconstruction-
    trained token ends up encoding "which episode" (measured: linear episode-ID probe at 100%).
    behsim has to make that the WORSE latent. With behaviour arranged in clusters that repeat across
    episodes, a latent encoding the cluster must beat one encoding the episode.
    """
    acts, cluster = _clustered_actions()
    stub = _behsim_stub()

    z_behaviour = jax.nn.one_hot(cluster, 8) * 5.0
    z_episode = jax.nn.one_hot(jnp.arange(acts.shape[0]), 8) * 5.0
    z_random = jax.random.normal(jax.random.key(2), (acts.shape[0], 8))

    def kl(z):
        return float(pi0_rlt.Pi0RLT._behsim_loss(stub, z, acts)[1]["rlt/behsim_kl"])

    # A latent that groups by behaviour matches the target distribution exactly.
    assert kl(z_behaviour) == pytest.approx(0.0, abs=1e-3)
    assert kl(z_behaviour) < kl(z_episode) < kl(z_random)
    # The reported KL is a divergence, so it can never go below zero however bad the latent is.
    assert kl(z_random) >= 0.0


def test_behsim_gradient_is_finite():
    """Both softmaxes mask the diagonal with dtype-min; a -inf would make the gradient NaN."""
    acts, _ = _clustered_actions()
    stub = _behsim_stub()
    z = jax.random.normal(jax.random.key(3), (acts.shape[0], 8))
    g = jax.grad(lambda zz: jnp.mean(pi0_rlt.Pi0RLT._behsim_loss(stub, zz, acts)[0]))(z)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_behsim_ignores_padded_action_dims():
    """Only the first `_rlt_act_dim` dims are real; the rest are constant padding.

    If the distance used all 32 dims, garbage in the padding would move the target distribution.
    """
    acts, _ = _clustered_actions()
    stub = _behsim_stub()
    z = jax.random.normal(jax.random.key(4), (acts.shape[0], 8))
    perturbed = acts.at[..., stub._rlt_act_dim :].add(100.0)
    base = pi0_rlt.Pi0RLT._behsim_loss(stub, z, acts)[0]
    same = pi0_rlt.Pi0RLT._behsim_loss(stub, z, perturbed)[0]
    assert jnp.allclose(base, same)


def test_epadv_gradient_is_reversed():
    """The whole mechanism is the sign flip: the token must get MINUS-lambda the classifier gradient.

    If the reversal were dropped, z_rl would be trained to HELP predict the episode — the exact
    opposite of the intent. This pins the gradient into z_rl to -lambda times the plain-CE gradient.
    """
    from flax import nnx

    rngs = nnx.Rngs(0)
    lam = 2.0
    stub = types.SimpleNamespace(
        rlt_epadv_lambda=lam,
        rlt_num_episodes=16,
        rlt_epadv_hidden=nnx.Linear(8, 32, rngs=rngs),
        rlt_epadv_out=nnx.Linear(32, 16, rngs=rngs),
    )
    z = jax.random.normal(jax.random.key(0), (5, 8))
    ep = jnp.arange(5)

    g_rev = jax.grad(lambda zz: jnp.mean(pi0_rlt.Pi0RLT._epadv_loss(stub, zz, ep)[0]))(z)

    def plain_ce(zz):  # same head, no gradient reversal
        logits = stub.rlt_epadv_out(jax.nn.gelu(stub.rlt_epadv_hidden(zz)))
        logp = jax.nn.log_softmax(logits, axis=-1)
        tgt = jnp.mod(ep, 16)
        return jnp.mean(-jnp.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0])

    g_plain = jax.grad(plain_ce)(z)
    assert jnp.allclose(g_rev, -lam * g_plain, atol=1e-4)


@pytest.mark.parametrize(
    "objective",
    [
        "reconstruction",
        "reconstruction+progress",
        "reconstruction+action",
        "reconstruction+behsim",
        "reconstruction+action+behsim",
        "reconstruction+progress+action",
        "reconstruction+epadv",
        "reconstruction+behsim+epadv",
    ],
)
def test_objective_accepted(objective):
    pi0_rlt.Pi0RLTConfig(pi05=True, rlt_objective=objective)


@pytest.mark.parametrize(
    "objective",
    [
        # Every addition is too low-dimensional to hold the bottleneck open on its own.
        "progress",
        "action",
        "behsim",
        "action+reconstruction",  # reconstruction must lead, so the tag mapping stays 1:1
        "reconstruction+bogus",
        "reconstruction+action+action",
    ],
)
def test_objective_rejected(objective):
    with pytest.raises(ValueError, match="rlt_objective"):
        pi0_rlt.Pi0RLTConfig(pi05=True, rlt_objective=objective)
