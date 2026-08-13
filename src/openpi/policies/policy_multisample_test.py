"""MultiSamplePolicy: several chunks per observation, only when asked."""

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
    """Row 0 IS `actions`, so the drawn distribution contains the decision actually taken --
    otherwise the picture shows what the policy might have done, not what it did."""
    policy, inner = _policy()
    result = policy.infer({"state": np.zeros(42), "num_samples": 5})

    assert result["action_samples"].shape == (5, 30, 14)
    assert np.allclose(result["action_samples"][0], result["actions"])
    assert len({float(s.mean()) for s in result["action_samples"]}) == 5
    assert len(inner.noises) == 5


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
    """Stands in for CriticSelectPolicy: reads num_samples itself, declares its own features."""

    metadata: ClassVar[dict] = {}

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
    policy = MultiSamplePolicy(critic, action_horizon=30, action_dim=32)

    policy.infer({"state": np.zeros(42), "critic_select": True, "num_samples": 8})

    assert critic.calls == 1


def test_num_samples_reaches_the_critic_that_reads_it():
    """Popping it unconditionally left the critic on its own default, silently."""
    critic = _CriticLike()
    policy = MultiSamplePolicy(critic, action_horizon=30, action_dim=32)

    policy.infer({"state": np.zeros(42), "critic_select": True, "num_samples": 8})

    assert critic.saw_num_samples == 8


def test_an_inner_declaration_reaches_the_handshake():
    """This wrapper is the outermost one serve_policy holds. A declaration it does not forward
    never reaches the client, which then records nothing — with no error anywhere."""
    policy = MultiSamplePolicy(_CriticLike(), action_horizon=30, action_dim=32)
    assert policy.extra_features() == {"critic_scores": [16]}


def test_a_policy_that_declares_nothing_forwards_nothing():
    policy, _ = _policy()
    assert policy.extra_features() == {}
