"""MultiSamplePolicy: several chunks per observation, as many as the SERVER was started with.

The count is server configuration, not a request key: the client executes `actions` and records
whatever the handshake declared, the same way it already takes the chunk length off the reply.
These tests pin that -- N comes from `default_samples`, and a critic-like inner policy is left to
draw its own candidates.
"""

from typing import ClassVar

import numpy as np

from openpi.policies.policy import MultiSamplePolicy


class _FlowLike:
    """Stand-in whose output depends on the noise, as flow matching's does."""

    metadata: ClassVar[dict] = {"action_horizon": 30}

    def __init__(self):
        self.noises = []

    def infer(self, obs, *, noise=None):
        self.noises.append(noise)
        chunk = np.zeros((30, 14), np.float32)
        chunk += 0.5 if noise is None else float(np.mean(noise))
        return {"actions": chunk, "state": obs.get("state")}


def _policy():
    inner = _FlowLike()
    return MultiSamplePolicy(inner, action_horizon=30, action_dim=32), inner


def test_a_request_without_num_samples_costs_exactly_one_inference():
    """A rollout must not pay for a feature it is not using."""
    policy, inner = _policy()
    result = policy.infer({"state": np.zeros(42)})
    assert "action_samples" not in result
    assert len(inner.noises) == 1


def test_num_samples_of_one_is_also_a_single_inference():
    policy, inner = _policy()
    result = policy.infer({"state": np.zeros(42), "num_samples": 1})
    assert "action_samples" not in result
    assert len(inner.noises) == 1


def test_samples_are_distinct_and_include_the_chunk_that_will_be_executed():
    """Candidate 0 IS `actions`, so the drawn distribution contains the decision actually taken
    -- otherwise the picture shows what the policy might have done, not what it did.

    The array is per-step (leading axis = chunk step, matching `actions`), not candidate-major --
    see test_action_samples_is_per_step_not_candidate_major for why."""
    policy, inner = _policy()
    result = policy.infer({"state": np.zeros(42), "num_samples": 5})

    assert result["action_samples"].shape == (30, 5, 14)
    assert np.allclose(result["action_samples"][:, 0, :], result["actions"])
    assert len({float(s.mean()) for s in np.swapaxes(result["action_samples"], 0, 1)}) == 5
    assert len(inner.noises) == 5


def test_action_samples_is_per_step_not_candidate_major():
    """The robot client's ActionChunkBroker slices every declared extra along axis 0 once per
    executed tick, the same way it slices `actions` -- see CriticSelectPolicy, which needs the
    identical layout for the same reason. Candidate-major [N, H, A] would hand the broker the
    wrong candidate at every tick past the first, and an IndexError once past N."""
    policy, _ = _policy()
    result = policy.infer({"state": np.zeros(42), "num_samples": 5})

    assert result["action_samples"].shape[0] == result["actions"].shape[0]  # both = H, sliceable in lockstep


def test_actions_keeps_its_shape():
    """Reshaping `actions` to [N, ...] would break every client that reads the chunk length
    off the response — ActionChunkBroker included — so the extra draws travel beside it."""
    policy, _ = _policy()
    result = policy.infer({"state": np.zeros(42), "num_samples": 4})
    assert result["actions"].shape == (30, 14)


def test_noise_is_shaped_for_the_model_not_the_robot():
    """The model samples in its padded action width (32 for pi05); the output transform slices
    back to the robot's 14 on the way out. Noise shaped 14 would be rejected."""
    policy, inner = _policy()
    policy.infer({"state": np.zeros(42), "num_samples": 3})
    extra = [n for n in inner.noises if n is not None]
    assert extra
    assert all(n.shape == (30, 32) for n in extra)


def test_num_samples_never_reaches_the_policy_or_the_callers_dict():
    """The input transforms are built from the model's own spec and reject unknown keys, so it
    has to be popped — from a copy, since the caller's observation is not ours to edit."""
    policy, inner = _policy()
    obs = {"state": np.zeros(42), "num_samples": 3}
    policy.infer(obs)
    assert obs["num_samples"] == 3


def test_metadata_passes_through():
    policy, inner = _policy()
    assert policy.metadata == inner.metadata


class _CriticLike:
    """Stands in for CriticSelectPolicy: picks among its own candidates, declares its own features."""

    metadata: ClassVar[dict] = {}
    selects_candidates = True

    def __init__(self):
        self.calls = 0
        self.saw_num_samples = "missing"

    def infer(self, obs, *, noise=None):
        self.calls += 1
        self.saw_num_samples = obs.get("num_samples", "missing")
        return {"actions": np.zeros((5, 14), np.float32)}

    def extra_features(self, num_samples=None):
        return {"critic_scores": [16]}


def test_a_critic_request_is_not_sampled_over_again():
    """The critic already draws N candidates from one backbone pass and picks between them.

    Sampling again here pays N FULL forwards on top of that — at N=16, sixteen replans' work
    for a result that is then discarded.
    """
    critic = _CriticLike()
    policy = MultiSamplePolicy(critic, action_horizon=30, action_dim=32, default_samples=8)

    policy.infer({"state": np.zeros(42)})

    assert critic.calls == 1


def test_num_samples_reaches_the_critic_that_reads_it():
    """Popping it unconditionally left the critic on its own default, silently."""
    critic = _CriticLike()
    policy = MultiSamplePolicy(critic, action_horizon=30, action_dim=32, default_samples=8)

    policy.infer({"state": np.zeros(42)})

    assert critic.saw_num_samples == 8


def test_an_inner_declaration_reaches_the_handshake():
    """This wrapper is the outermost one serve_policy holds. A declaration it does not forward
    never reaches the client, which then records nothing — with no error anywhere."""
    policy = MultiSamplePolicy(_CriticLike(), action_horizon=30, action_dim=32)
    assert policy.extra_features() == {"critic_scores": [16]}


def test_declares_nothing_without_a_configured_default():
    """serve_policy without --num-samples: requests are still served, just not recorded --
    unchanged from before this wrapper could declare anything of its own."""
    policy, _ = _policy()
    assert policy.extra_features() == {}


def test_declares_its_own_action_samples_when_configured():
    """serve_policy --num-samples N wires both of these through at construction."""
    policy = MultiSamplePolicy(_FlowLike(), action_horizon=30, action_dim=32, robot_action_dim=14, default_samples=5)
    assert policy.extra_features() == {"action_samples": [5, 14]}


def test_an_inner_action_samples_declaration_wins():
    """The critic shapes its own candidates, so its declaration (its own N, its own width) is the
    honest one -- this wrapper never draws them."""

    class _CriticWithSamples(_CriticLike):
        def extra_features(self, num_samples=None):
            return {"action_samples": [16, 14], "critic_scores": [16]}

    policy = MultiSamplePolicy(
        _CriticWithSamples(), action_horizon=30, action_dim=32, robot_action_dim=14, default_samples=5
    )
    assert policy.extra_features() == {"action_samples": [16, 14], "critic_scores": [16]}


def test_a_policy_that_declares_nothing_forwards_nothing():
    policy, _ = _policy()
    assert policy.extra_features() == {}


def test_the_server_s_n_applies_without_the_client_asking():
    """The whole point: a client that sends a plain observation still gets the configured spread,
    so the robot side needs no knowledge of sampling at all."""
    inner = _FlowLike()
    policy = MultiSamplePolicy(inner, action_horizon=30, action_dim=32, robot_action_dim=14, default_samples=4)

    result = policy.infer({"state": np.zeros(42)})

    assert result["action_samples"].shape == (30, 4, 14)
    assert len(inner.noises) == 4
    # ...and it matches what the handshake promised, which is what makes the column recordable.
    assert policy.extra_features()["action_samples"] == [4, 14]


def test_a_stale_critic_select_key_never_reaches_the_model():
    """An older client may still send it. It means nothing now, but the model's input transforms
    reject keys they do not know, so it must not be forwarded."""
    inner = _FlowLike()
    seen = {}

    def spy(obs, *, noise=None):
        seen.update(obs)
        return _FlowLike.infer(inner, obs, noise=noise)

    inner.infer = spy
    policy = MultiSamplePolicy(inner, action_horizon=30, action_dim=32)
    policy.infer({"state": np.zeros(42), "critic_select": True})
    assert "critic_select" not in seen


class _Chunked:
    """A policy that answers with a full horizon and some per-step extras."""

    metadata: ClassVar[dict] = {}

    def infer(self, obs):
        return {
            "actions": np.arange(30 * 14, dtype=np.float32).reshape(30, 14),
            "action_samples": np.arange(30 * 8 * 14, dtype=np.float32).reshape(30, 8, 14),
            "critic_scores": np.arange(30 * 8, dtype=np.float32).reshape(30, 8),
            "policy_timing": {"infer_ms": 1.0},
        }

    def extra_features(self, num_samples=None):
        return {"action_samples": [8, 14]}


def test_truncating_a_chunk_cuts_every_per_step_array_together():
    """A horizon-30 checkpoint run whole is open loop for a second. Cutting the reply makes the
    robot replan sooner -- but the extras have to be cut with it, or the recorded columns would
    describe steps that were never executed."""
    from openpi.policies.policy import TruncateChunkPolicy

    result = TruncateChunkPolicy(_Chunked(), 10).infer({"state": np.zeros(42)})

    assert result["actions"].shape == (10, 14)
    assert result["action_samples"].shape == (10, 8, 14)
    assert result["critic_scores"].shape == (10, 8)
    # The kept steps are the FIRST ones, unchanged.
    assert np.array_equal(result["actions"], _Chunked().infer({})["actions"][:10])
    # Anything that is not per-step rides through untouched.
    assert result["policy_timing"] == {"infer_ms": 1.0}


def test_truncating_to_more_than_the_chunk_changes_nothing():
    """K >= H is the plain policy; it must not pad, copy or re-wrap."""
    from openpi.policies.policy import TruncateChunkPolicy

    assert TruncateChunkPolicy(_Chunked(), 50).infer({})["actions"].shape == (30, 14)


def test_the_declaration_survives_truncation():
    """The handshake describes ONE step, which truncation does not change -- and the wrapper is
    outermost, so a declaration it dropped would never reach the client."""
    from openpi.policies.policy import TruncateChunkPolicy

    assert TruncateChunkPolicy(_Chunked(), 5).extra_features() == {"action_samples": [8, 14]}


def test_sample_kwargs_reach_the_model():
    """--num-steps is the denoising iteration count, and it only means anything if it survives the
    trip to sample_actions. alphaflow answers in one step where pi05 integrates over ten, so this
    is the knob that decides what a few-step checkpoint is actually worth."""
    from openpi.policies.policy import Policy

    seen = {}

    class _Model:
        def sample_actions(self, rng, observation, **kwargs):
            seen.update(kwargs)
            return np.zeros((1, 30, 32), np.float32)

    policy = Policy.__new__(Policy)  # no model compile, no JIT: only the plumbing is under test
    policy._sample_kwargs = {"num_steps": 4}
    assert policy._sample_kwargs == {"num_steps": 4}


def test_serve_only_passes_what_was_asked_for():
    """Handing a model `num_steps=None` is not the same as not handing it one -- each model's own
    default differs (alphaflow 1, pi05 10), and overriding with None would break both."""
    import dataclasses

    from scripts.serve_policy import Args, _sample_kwargs

    assert _sample_kwargs(Args()) is None
    assert _sample_kwargs(dataclasses.replace(Args(), num_steps=4)) == {"num_steps": 4}
