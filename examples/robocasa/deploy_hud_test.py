"""The serving layout and the HUD have to agree about which axis is which."""

import numpy as np
import pytest

import deploy_hud

N, P, X, A = 6, 4, 9, 14


def _response(*, chunk=X):
    """A CriticSelectPolicy response, in the layout serving now sends.

    Everything but `actions` is per-step: leading axis X, matching the actions it accompanies.
    That is the robot client's recording contract, and what lets these survive an adaptive
    chunk. The values are decided once per replan and broadcast, so every row is identical.
    """
    q = np.arange(N * P, dtype=np.float32).reshape(N, P)
    candidates = np.arange(N * chunk * A, dtype=np.float32).reshape(N, chunk, A)
    best = int(np.argmax(q[:, -1]))
    return {
        "actions": candidates[best],
        "action_samples": np.swapaxes(candidates, 0, 1),
        "critic_scores": np.broadcast_to(q[:, -1], (chunk, N)).copy(),
        "critic_choice": np.full((chunk, 1), best, np.float32),
        "critic_grid": np.broadcast_to(q, (chunk, N, P)).copy(),
        "critic_best_prefix": np.full((chunk, 1), P - 1, np.float32),
        "critic_macro": np.full((chunk, 1), 2, np.float32),
    }, q, candidates, best


def test_every_extra_leads_with_the_chunk_axis():
    """The one rule the recording contract asks for."""
    response, _, _, _ = _response()
    for key in ("action_samples", "critic_scores", "critic_choice",
                "critic_grid", "critic_best_prefix", "critic_macro"):
        assert np.asarray(response[key]).shape[0] == X, key


def test_the_hud_recovers_its_own_layout():
    response, q, candidates, best = _response()
    info = deploy_hud.info_from_response(response)

    assert info.q.shape == (N, P)
    assert np.array_equal(info.q, q)
    assert info.cand.shape == (N, X, A), "the HUD wants candidates chunk-major"
    assert np.array_equal(info.cand, candidates)
    assert info.best_cand == best
    assert info.best_prefix == P - 1
    assert info.n_exec == P * 2


@pytest.mark.parametrize("chunk", [2, 9, 30])
def test_the_hud_does_not_care_how_long_the_chunk_is(chunk):
    """The chunk is adaptive; nothing here may assume a horizon."""
    response, q, candidates, best = _response(chunk=chunk)
    info = deploy_hud.info_from_response(response)
    assert info.q.shape == (N, P)
    assert info.cand.shape == (N, chunk, A)
    assert info.best_cand == best


def test_the_executed_chunk_is_the_chosen_candidate():
    """`actions` and candidate `critic_choice` must be the same trajectory, or the HUD marks
    one chunk as chosen while the robot runs another."""
    response, _, candidates, best = _response()
    info = deploy_hud.info_from_response(response)
    assert np.array_equal(np.asarray(response["actions"]), candidates[info.best_cand])
    assert info.best_cand == best
