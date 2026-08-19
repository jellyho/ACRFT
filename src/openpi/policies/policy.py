from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class MultiSamplePolicy(BasePolicy):
    """Draw several action chunks for one observation, as many as the server was started with.

    The policy is a distribution, not a single answer: flow matching maps different noise to
    different chunks. Drawing a handful and showing them together is the only cheap way to see
    whether it is confident (a tight bundle) or torn between options (a wide spray) -- a scalar
    loss averages exactly that away.

    Configured on the SERVER (``--num-samples N``), not asked for per request. What the policy
    returns is the policy's business: the client's job is to execute ``actions`` and record
    whatever the handshake declared, exactly as it already takes the chunk LENGTH from the reply
    rather than from a setting of its own. A count that lives on both sides is a count that can
    disagree -- and it did: the declared ``action_samples`` column came from the server's N while
    the array actually sent came from the request's, so a mismatch silently dropped the column
    from every frame. N is one number in one place now.

    A request may still override it (a viewer sampling more heavily than the rollout does), but
    nothing has to: with the server started at N <= 1 this costs exactly one inference, byte for
    byte the old behaviour.

    ``actions`` stays the single chunk to execute and keeps its shape. The extra draws ride
    along under ``action_samples``, PER STEP -- leading axis X (chunk step), matching
    ``actions`` -- with the executed chunk as candidate 0. Candidate-major [N, H, A] would read
    naturally in-process, but the robot client's ``ActionChunkBroker`` slices every declared
    extra along axis 0 once per executed tick (see :class:`CriticSelectPolicy`, which faces the
    same requirement for the same reason): candidate-major would hand it the wrong candidate at
    every tick past the first, and an IndexError once past N. Reshaping ``actions`` itself to
    [N, ...] instead would break every client that reads the chunk length off the response,
    ActionChunkBroker included -- so the extra draws travel beside it, not inside it.

    ``robot_action_dim`` and ``default_samples`` are only needed to declare ``action_samples`` at
    handshake (see :meth:`extra_features`); the sampling itself works without either.
    """

    def __init__(
        self,
        policy: BasePolicy,
        *,
        action_horizon: int,
        action_dim: int,
        seed: int = 0,
        robot_action_dim: int | None = None,
        default_samples: int = 0,
    ):
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)
        self._robot_action_dim = robot_action_dim
        self._default_samples = int(default_samples)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Pop before anything else touches it: the input transforms are built from the model's
        # own input spec and reject keys they do not know.
        obs = dict(obs)
        # The server's N unless the request overrides it (see the class docstring).
        num_samples = int(obs.pop("num_samples", 0) or self._default_samples)
        # Historic request key: selection is now decided by how the server was started, so this
        # says nothing. Popped rather than ignored -- the model's input transforms reject keys
        # they do not know.
        obs.pop("critic_select", None)

        # A policy that picks among its own candidates already samples N from ONE backbone pass
        # and returns them under `action_samples`. Drawing more here would pay N full forwards on
        # top of the N it already did -- at N=16, sixteen replans' work, thrown away.
        if getattr(self._policy, "selects_candidates", False):
            if num_samples:
                obs["num_samples"] = num_samples
            return self._policy.infer(obs)

        result = self._policy.infer(obs)
        if num_samples <= 1:
            return result

        samples = [np.asarray(result["actions"])]
        for _ in range(num_samples - 1):
            noise = self._rng.standard_normal((self._action_horizon, self._action_dim)).astype(np.float32)
            samples.append(np.asarray(self._policy.infer(obs, noise=noise)["actions"]))
        # [N, H, A] -> [H, N, A]: see the class docstring for why this has to be per-step.
        result["action_samples"] = np.swapaxes(np.stack(samples), 0, 1)
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def extra_features(self, *args, **kwargs) -> dict:
        """Whatever the wrapped policy declares, plus this wrapper's own ``action_samples``.

        This wrapper is the outermost one serve_policy holds, so a declaration made by an
        inner policy (a critic's, say) only reaches the handshake if it is forwarded. Without
        this it silently did not, and the client recorded nothing.

        Its own draws are declared only when constructed with ``robot_action_dim`` and
        ``default_samples`` (serve_policy does this when started with ``--num-samples``): the
        dataset schema is fixed at handshake, so the request has to commit to one N in advance,
        and a server started without either keeps today's behaviour -- served, but unrecorded.
        An inner declaration of the same key (a critic already declaring its own
        ``action_samples``) wins, since that key is only reachable via the ``critic_select``
        passthrough this wrapper never actually shapes itself.
        """
        declare = getattr(self._policy, "extra_features", None)
        inner = declare(*args, **kwargs) if callable(declare) else {}
        if self._default_samples and self._robot_action_dim and "action_samples" not in inner:
            inner = {**inner, "action_samples": [self._default_samples, self._robot_action_dim]}
        return inner


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


def _output_state_dim(output_transform: _transforms.DataTransformFn, fallback: int) -> int:
    """The width the output transform's ``state`` un-normalization expects.

    The real infer path feeds ``state`` at its NATIVE robot width (e.g. 42 for YAM) -- only the
    ACTIONS are padded to the model's width (32 for pi05). ``Unnormalize`` on quantile stats can
    pad/slice when the input is at least as wide as the stats, but not when it is narrower (see
    ``transforms._unnormalize_quantile``), so a probe that fed a 32-wide state would break on a
    42-wide state stat. Read the real width off the state norm stats instead of guessing it from
    the action width."""
    for t in getattr(output_transform, "transforms", [output_transform]):
        ns = getattr(t, "norm_stats", None)
        st = ns.get("state") if isinstance(ns, dict) else None
        if st is not None:
            for attr in ("q99", "q01", "mean", "std"):
                arr = getattr(st, attr, None)
                if arr is not None:
                    return int(np.asarray(arr).shape[-1])
    return fallback


def probe_robot_action_dim(policy: Policy, *, model_action_dim: int, action_horizon: int) -> int:
    """The output transform's real last-dim width, robot-space rather than the model's padded one.

    Decodes a zero chunk through the policy's own output transform and reads the shape back off
    it. Both :class:`CriticSelectPolicy` (whose critic is trained on robot-space chunks) and
    :class:`MultiSamplePolicy` (whose ``action_samples`` handshake declaration has to match what
    the dataset writer will reshape to) need this same recovery: the model's padded width (32 for
    pi05) is not what actually arrives on the wire (14 for YAM).

    ``state`` is fed at the width the output transform's own norm stats expect -- NOT the model
    action width, which differs from the state width on YAM (state 42 vs action 32) and would trip
    the state un-normalization (see ``_output_state_dim``).
    """
    state_dim = _output_state_dim(policy._output_transform, fallback=model_action_dim)
    probe = policy._output_transform(
        {
            "state": np.zeros((1, state_dim), np.float32),
            "actions": np.zeros((1, action_horizon, model_action_dim), np.float32),
        }
    )
    return int(np.asarray(probe["actions"]).shape[-1])


class CriticSelectPolicy(BasePolicy):
    """Best-of-N with a trained RLT critic, selected server-side.

    The client cannot run the critic itself: the critic reads the RL token, and the token never
    leaves the model in a plain infer. So selection has to happen where the token lives -- and
    it happens on every request, because a server started with a critic IS a critic-selected
    policy. It used to be per-request opt-in, which meant a server could load its critic, log it,
    and then be bypassed on every step by a client that did not know to ask; a plain rollout is a
    plain server instead. What the client sees is what it always sees: a chunk to execute, plus
    whatever the handshake declared.

    Sampling reuses ``extract_token_and_base_actions``: one backbone pass amortized over all N
    flow decodes, instead of MultiSamplePolicy's N full forwards - at N=16 that is the
    difference between one replan and sixteen.

    The critic's proprioception statistics ship in ``<critic_dir>/proprio_stats.json`` (written
    by ``scripts/export_critic_serving.py``); the annotation directory they came from is not a
    serving artifact and cannot be assumed to exist on the robot host.
    """

    def __init__(self, policy: Policy, critic_dir, *, flow_steps: int = 10, default_samples: int = 16, seed: int = 0):
        import json as _json
        import pathlib as _pathlib

        import openpi.rlt_critic.critic as _critic

        self._pol = policy
        self._flow_steps = int(flow_steps)
        self._default_samples = int(default_samples)
        self._rng = jax.random.key(seed)
        cdir = _pathlib.Path(critic_dir)
        model = policy._model
        self._extract = nnx_utils.module_jit(model.extract_token_and_base_actions)
        self._model_action_dim = int(model.action_dim)
        # The critic was trained on ROBOT-space chunks; recover that width the same way the
        # annotation did.
        raw_dim = probe_robot_action_dim(
            policy, model_action_dim=self._model_action_dim, action_horizon=int(model.action_horizon)
        )
        self._robot_action_dim = raw_dim
        score, _, self._macro = _critic.load_trained(
            cdir / "params.msgpack", action_dim=raw_dim, horizon=int(model.action_horizon)
        )
        self._score = jax.jit(score)
        stats_p = cdir / "proprio_stats.json"
        self._pro = None
        if stats_p.exists():
            st = _json.loads(stats_p.read_text())
            self._pro = (
                np.asarray(st["mean"], np.float32),
                np.asarray(st["std"], np.float32),
                st.get("key", "observation/state"),
            )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        # Historic opt-in key; selection is unconditional now (see the class docstring). Popped
        # because the model's input transforms reject keys they do not know.
        obs.pop("critic_select", None)
        want_hud = bool(obs.pop("critic_hud", False))
        num_samples = int(obs.pop("num_samples", 0) or self._default_samples)

        raw_state = np.asarray(obs.get(self._pro[2] if self._pro else "observation/state", ()), np.float32)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._pol._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        token, base = self._extract(sample_rng, observation, num_samples=num_samples, num_steps=self._flow_steps)
        chunks_model = np.asarray(base[0], np.float32)  # [N, H, model_dim]
        decoded = self._pol._output_transform(
            {"state": np.zeros((chunks_model.shape[0], self._model_action_dim), np.float32), "actions": chunks_model}
        )["actions"]
        decoded = np.asarray(decoded, np.float32)

        z = np.asarray(token, np.float32)  # [1, D]
        if self._pro is not None:
            mu, sd, _ = self._pro
            p = np.where(sd > 1e-6, (raw_state - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)
            z = np.concatenate([z, p[None, :].astype(np.float32)], axis=-1)
        zc = jnp.repeat(jnp.asarray(z)[:, None], decoded.shape[0], axis=1)  # [1, N, D(+P)]
        q = np.asarray(self._score(zc, jnp.asarray(decoded)[None]))
        q = np.min(q, axis=0)[0]  # ensemble-min -> [N, P]
        q_full = q[:, -1]
        best = int(np.argmax(q_full))
        chunk = decoded.shape[1]  # X — this replan's length, not a configured horizon
        # Everything but `actions` is laid out PER STEP: leading axis X, matching the actions
        # it accompanies. That is the recording contract (see the robot client's
        # `extra_features`), and it is what lets these survive an adaptive chunk -- the client
        # slices row i for step i without either side agreeing a horizon in advance. The values
        # are constant across the chunk, since one replan decides them all; broadcasting is
        # what makes them recordable, not what makes them meaningful.
        out = {
            "actions": decoded[best],  # (X, A)
            "action_samples": np.swapaxes(decoded, 0, 1),  # (X, N, A)
            "critic_scores": np.broadcast_to(q_full, (chunk, q_full.shape[0])).copy(),  # (X, N)
            "critic_choice": np.full((chunk, 1), best, np.float32),  # (X, 1)
        }
        # The full [N, P] grid is everything a HUD needs to reconstruct this replan on top of
        # the above (examples/robocasa/hud.py draws exactly these). It rides along only when the
        # client sets ``critic_hud`` so a plain critic_select response stays small.
        if want_hud:
            out["critic_grid"] = np.broadcast_to(q, (chunk, *q.shape)).copy()  # (X, N, P)
            out["critic_best_prefix"] = np.full((chunk, 1), q.shape[1] - 1, np.float32)
            out["critic_macro"] = np.full((chunk, 1), self._macro, np.float32)
        return out

    #: This policy draws and picks among its own candidates, so a MultiSamplePolicy wrapped
    #: around it must not draw more on top (N full forwards for a result it would discard).
    selects_candidates = True

    def extra_features(self, num_samples: int | None = None) -> dict:
        """Per-step shapes to advertise at handshake, for the client to record.

        One step's shape only. The chunk axis is deliberately absent: the chunk is adaptive, so
        no number either side could agree on in advance describes it.

        The HUD extras are not here. ``critic_grid`` is [N, P], and P falls out of the critic's
        own architecture rather than anything known at construction -- advertising a guessed
        width would put a wrong column in every episode. It stays a per-request value for a HUD
        to consume live, which is all it was ever for.
        """
        n = int(num_samples or self._default_samples)
        return {
            "action_samples": [n, self._robot_action_dim],
            "critic_scores": [n],
            "critic_choice": [1],
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._pol.metadata

    @property
    def robot_action_dim(self) -> int:
        """The robot-space action width recovered at construction (see ``probe_robot_action_dim``).

        Exposed so a wrapper built on top (MultiSamplePolicy, when stacked over a critic) can
        reuse this instead of probing the output transform a second time.
        """
        return self._robot_action_dim
