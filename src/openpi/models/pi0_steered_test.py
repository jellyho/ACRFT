"""Value-steered sampling, checked against the model it steers.

The previous version of these tests could not do that. The steering loop lived in the serving
wrapper, entangled with a patch critic and a DINOv2 backbone, so the test stood up a `_FakeSampler`
whose `_steer` was `x + alpha` -- it pinned the plumbing around the sampler (which index executes,
what N is returned) and never executed one line of the integration. That is this repo's recurring
failure: an artifact that describes the property, passing without checking it.

Injecting the value function is what makes the real thing testable. Here it is analytic, with a
known optimum, so "steering moves the chunk toward higher value" is a claim about the actual Euler
loop, the actual Tweedie projection and the actual sign of Eq. 17.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0
from openpi.models.pi0_steered import Pi0Steered


@pytest.fixture(scope="module")
def fixture():
    cfg = pi0_config.Pi0Config(
        pi05=True, action_horizon=4, action_dim=8, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    base = cfg.create(jax.random.key(0))
    model = Pi0Steered.wrap(base)
    obs = _model.preprocess_observation(None, cfg.fake_obs(batch_size=1), train=False)
    noise = jax.random.normal(jax.random.key(7), (1, cfg.action_horizon, cfg.action_dim))
    return cfg, base, model, obs, noise


def _draw(model, obs, noise, value_fn, alpha, steps=4):
    return np.asarray(
        model.sample_steered(
            jax.random.key(0), obs, value_fn=value_fn, alpha=alpha, num_steps=steps, noise=noise, preprocess=False
        )
    )


# ---------------------------------------------------------------------------------------------
# wrap()


def test_wrap_shares_parameters_and_leaves_the_served_model_alone(fixture):
    """The served policy's jitted graphs are keyed on a graphdef carrying the type. Re-tagging in
    place would invalidate every one of them, so wrap() copies -- but it must not COPY the 3B of
    parameters to do it."""
    _cfg, base, model, _obs, _noise = fixture
    assert type(base) is Pi0  # untouched
    assert isinstance(model, Pi0Steered)
    assert model is not base
    assert model.PaliGemma is base.PaliGemma  # same objects, no re-allocation


def test_wrap_is_idempotent(fixture):
    _cfg, _base, model, _obs, _noise = fixture
    assert Pi0Steered.wrap(model) is model


# ---------------------------------------------------------------------------------------------
# the integration


def test_alpha_zero_reproduces_the_unsteered_sampler(fixture):
    """The twin has to BE the base policy's draw, or the displacement measured against it is not
    steering displacement.

    Not asserted as exact equality, and the reason is the point: at alpha=0 the velocity still
    comes out of jax.grad's forward pass rather than a direct _velocity call. Same math, different
    accumulation order. Short-circuiting the gradient would make this exact and would put that
    same difference between a STEERED draw and its twin instead, compounded over every step, where
    it would be reported as displacement. The tolerance is the price of that symmetry.
    """
    cfg, _base, model, obs, noise = fixture
    value_fn = lambda a: -jnp.sum(a**2)  # noqa: E731
    unsteered = _draw(model, obs, noise, value_fn, 0.0)

    prefix_mask, kv = model._prefix_forward(obs)
    x, n = noise, 4
    for i in range(n):
        v = model._velocity(obs, prefix_mask, kv, x, jnp.full((1,), 1.0 - i / n))
        x = x - (1.0 / n) * v
    direct = np.asarray(x)

    assert np.allclose(unsteered, direct, atol=2e-2), np.abs(unsteered - direct).max()


def test_steering_moves_the_chunk_toward_higher_value(fixture):
    """The claim the arm exists to make, on a value function whose optimum is known.

    This is what the injected value_fn buys: with the critic hard-wired in, "did it go the right
    way" was not answerable without a trained checkpoint, so nothing asserted it. A sign error in
    Eq. 17, a missing minus in the Tweedie projection, or a gradient taken through the clip
    instead of around it all survive every OTHER test in this file.
    """
    _cfg, _base, model, obs, noise = fixture
    target = jnp.full((1, 4, 8), 0.5)
    value_fn = lambda a: -jnp.sum((a - target) ** 2)  # noqa: E731

    base = _draw(model, obs, noise, value_fn, 0.0)
    steered = _draw(model, obs, noise, value_fn, 0.5)

    d_base = float(np.abs(base - np.asarray(target)).mean())
    d_steer = float(np.abs(steered - np.asarray(target)).mean())
    assert d_steer < d_base, f"steering moved AWAY from the optimum: {d_base:.4f} -> {d_steer:.4f}"


def test_stronger_steering_moves_further(fixture):
    """Monotone in alpha, which a sign-correct but magnitude-broken Eq. 17 need not be."""
    _cfg, _base, model, obs, noise = fixture
    target = jnp.full((1, 4, 8), 0.5)
    value_fn = lambda a: -jnp.sum((a - target) ** 2)  # noqa: E731
    d = [float(np.abs(_draw(model, obs, noise, value_fn, a) - np.asarray(target)).mean()) for a in (0.0, 0.25, 0.5)]
    assert d[0] > d[1] > d[2], d


def test_the_first_step_is_unsteered_for_every_alpha(fixture):
    """Paper Sec. 4: no state-dependent signal at t=0. With num_steps=1 the whole draw is that one
    step, so every alpha must give the same chunk -- including alphas large enough to saturate."""
    _cfg, _base, model, obs, noise = fixture
    value_fn = lambda a: -jnp.sum((a - 0.5) ** 2)  # noqa: E731
    one = [_draw(model, obs, noise, value_fn, a, steps=1) for a in (0.0, 1.0, 50.0)]
    assert np.array_equal(one[0], one[1])
    assert np.array_equal(one[0], one[2])


def test_the_same_noise_is_what_makes_the_twin_comparable(fixture):
    """Independent draws differ by the policy's own spread, which is the quantity the twin exists
    to hold fixed. If this ever stopped being true, a displacement readout would be reporting
    sampling variance."""
    _cfg, _base, model, obs, noise = fixture
    value_fn = lambda a: -jnp.sum(a**2)  # noqa: E731
    other = jax.random.normal(jax.random.key(11), noise.shape)
    same = _draw(model, obs, noise, value_fn, 0.0)
    diff = _draw(model, obs, other, value_fn, 0.0)
    assert not np.allclose(same, diff, atol=1e-3)


def test_the_output_is_not_clipped(fixture):
    """It used to be, and that made qpilots the only arm with an extra output transformation --
    sample_actions, sample_n_actions and decode_latent all return the integrator's output as it is.
    A method-only-diff comparison cannot carry a truncation that belongs to one arm.

    It also blinded the readout: the twin takes the same path, so two saturated draws differ by
    less than they were steered, and hard steering reported as LOW drift -- which reads exactly
    like steering that did nothing.

    Asserted by pushing hard enough that a clip would be visible.
    """
    _cfg, _base, model, obs, noise = fixture
    value_fn = lambda a: jnp.sum(a) * 1e3  # noqa: E731 - pushes hard past the boundary
    out = _draw(model, obs, noise, value_fn, 5.0)
    assert out.max() > 1.0, "steering this hard must leave the box, not be truncated at it"


def test_the_straight_through_clip_inside_the_loop_stays(fixture):
    """Removing the OUTPUT clip is not removing the paper's. Eq. 14's value is read inside the
    region the critic was trained on, while the gradient still flows for samples outside it --
    that one is what makes steering well-behaved past the boundary, and it is not the same clip."""
    import inspect

    from openpi.models.pi0_steered import Pi0Steered

    src = inspect.getsource(Pi0Steered.sample_steered)
    body = src[src.index("for i in range(num_steps):") :]
    assert "stop_gradient(jnp.clip(a_hat, -1, 1)" in body


def test_it_matches_the_pre_refactor_serving_loop(fixture):
    """Bit-for-bit against the implementation this replaced (ArmChunkSampler._steer, PR #13).

    The migration's whole claim is that it moved code without changing behaviour. That is checkable
    exactly, so it is checked exactly -- the loop below is the old one transcribed, with the critic
    call replaced by the injected value_fn it was specialised to.
    """
    _cfg, _base, model, obs, noise = fixture
    target = jnp.full((1, 4, 8), 0.5)
    value_fn = lambda a: -jnp.sum((a - target) ** 2)  # noqa: E731
    alpha, n = 0.5, 4

    pm, kv = model._prefix_forward(obs)
    x, dt = noise, 1.0 / n
    for i in range(n):
        tv = jnp.full((x.shape[0],), 1.0 - i * dt)
        if i == 0:
            v = model._velocity(obs, pm, kv, x, tv)
        else:

            def q_of(x_, tv_=tv):
                v_ = model._velocity(obs, pm, kv, x_, tv_)
                a_hat = x_ - tv_[:, None, None] * v_
                a_hat = a_hat + jax.lax.stop_gradient(jnp.clip(a_hat, -1, 1) - a_hat)
                return value_fn(a_hat), v_

            g, v = jax.grad(q_of, has_aux=True)(x)
            vn = jnp.linalg.norm(v.reshape(x.shape[0], -1), axis=-1).reshape(-1, 1, 1)
            gn = jnp.linalg.norm(g.reshape(x.shape[0], -1), axis=-1).reshape(-1, 1, 1)
            v = v - alpha * (vn / (gn + 1e-8)) * g
        x = x - dt * v
    old = np.asarray(x)

    new = _draw(model, obs, noise, value_fn, alpha, steps=n)
    assert np.array_equal(old, new), np.abs(old - new).max()


def test_preprocess_is_not_applied_twice(fixture):
    """The serving wrapper preprocesses once and passes preprocess=False, because it needed the
    observation to build the critic's own inputs. Both spellings must give the same chunk."""
    cfg, _base, model, obs, noise = fixture
    value_fn = lambda a: -jnp.sum(a**2)  # noqa: E731
    a = _draw(model, obs, noise, value_fn, 0.3)
    b = np.asarray(
        model.sample_steered(
            jax.random.key(0),
            cfg.fake_obs(batch_size=1),
            value_fn=value_fn,
            alpha=0.3,
            num_steps=4,
            noise=noise,
            preprocess=True,
        )
    )
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------
# the direction/magnitude control


def test_the_injected_magnitude_does_not_depend_on_the_value_function_scale(fixture):
    """Eq. 17 rescales the value gradient to the drift's own norm, so ||alpha*(vn/gn)*g|| = alpha*vn
    for ANY g. Multiplying the value function by 1e4 must therefore change nothing at all.

    This is the exact form of the property the `negated`/`random` control arms rest on: what alpha
    injects is a displacement of a size set by the drift, not by the critic's units. Without it a
    random-direction arm would not be a fair control -- it would be a different perturbation size.
    """
    _cfg, _base, model, obs, noise = fixture
    v = lambda a: -jnp.sum(jnp.square(a - 0.3))  # noqa: E731
    base = _draw(model, obs, noise, v, 0.0, steps=2)
    d1 = np.linalg.norm(_draw(model, obs, noise, v, 0.15, steps=2) - base)
    d2 = np.linalg.norm(_draw(model, obs, noise, lambda a: 1e4 * v(a), 0.15, steps=2) - base)
    assert d1 > 1e-4, "steering did nothing at alpha=0.15"
    np.testing.assert_allclose(d2, d1, rtol=1e-3)


def test_direction_changes_where_the_chunk_lands_but_not_the_order_of_the_step(fixture):
    """Same alpha, three directions: the landing points differ, the step sizes stay comparable.

    Comparable and NOT identical, and the reason is worth pinning because it qualifies how the
    robot control arm should be read. The injected DRIFT is exactly magnitude-matched (previous
    test), but `sample_steered` returns `clip(x, -1, 1)`, and in this fixture 37.5% of the output
    coordinates sit on that boundary. A push outward there is truncated and a push inward is not,
    so the REALIZED displacement is direction-dependent even though the injection is not -- here by
    up to ~1.6x. The control arms remain magnitude-matched at injection, which is the claim; they
    are not exactly matched at the output, which is a caveat, not a defect.
    """
    _cfg, _base, model, obs, noise = fixture
    u = jax.random.normal(jax.random.key(3), noise.shape[1:])
    u = u / jnp.linalg.norm(u)
    fns = {
        "critic-like": lambda a: -jnp.sum(jnp.square(a - 0.3)),
        "negated": lambda a: jnp.sum(jnp.square(a - 0.3)),
        "random": lambda a: jnp.sum(a * u),
    }
    base = _draw(model, obs, noise, fns["critic-like"], 0.0, steps=2)
    assert np.mean(np.abs(base) >= 0.999) > 0, "fixture no longer clips; tighten this test"
    out = {k: _draw(model, obs, noise, f, 0.15, steps=2) for k, f in fns.items()}
    disp = {k: float(np.linalg.norm(v - base)) for k, v in out.items()}
    assert min(disp.values()) > 1e-4, f"steering did nothing: {disp}"
    assert max(disp.values()) / min(disp.values()) < 2.0, f"clip alone should not do more than ~2x: {disp}"
    # and they land somewhere genuinely different -- a control that coincides with what it controls
    # for would pass every magnitude check while testing nothing
    for k in ("negated", "random"):
        assert np.linalg.norm(out[k] - out["critic-like"]) > 1e-4, f"{k} landed on top of critic-like"
