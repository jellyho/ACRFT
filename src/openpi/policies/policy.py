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
    """Draw several action chunks for one observation, when the client asks for them.

    The policy is a distribution, not a single answer: flow matching maps different noise to
    different chunks. Drawing a handful and showing them together is the only cheap way to see
    whether it is confident (a tight bundle) or torn between options (a wide spray) -- a scalar
    loss averages exactly that away.

    Opt-in per request, via a ``num_samples`` key on the observation, because N samples costs N
    forward passes. A rollout has no reason to pay that, so a client that never sends the key
    gets today's behaviour and today's latency, byte for byte.

    ``actions`` stays the single chunk to execute and keeps its shape. The extra draws ride
    along under ``action_samples`` as [N, horizon, dim], with the executed chunk as row 0 --
    reshaping ``actions`` to [N, ...] instead would break every client that reads the chunk
    length off the response, ActionChunkBroker included.
    """

    def __init__(self, policy: BasePolicy, *, action_horizon: int, action_dim: int, seed: int = 0):
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Pop before anything else touches it: the input transforms are built from the model's
        # own input spec and reject keys they do not know.
        obs = dict(obs)
        num_samples = int(obs.pop("num_samples", 0) or 0)

        result = self._policy.infer(obs)
        if num_samples <= 1:
            return result

        samples = [np.asarray(result["actions"])]
        for _ in range(num_samples - 1):
            noise = self._rng.standard_normal((self._action_horizon, self._action_dim)).astype(np.float32)
            samples.append(np.asarray(self._policy.infer(obs, noise=noise)["actions"]))
        result["action_samples"] = np.stack(samples)
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata


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


class CriticSelectPolicy(BasePolicy):
    """Best-of-N with a trained RLT critic, selected server-side.

    The client cannot run the critic itself: the critic reads the RL token, and the token never
    leaves the model in a plain infer. So selection has to happen where the token lives. Opt-in
    per request via a ``critic_select`` key (with an optional ``num_samples``, default 16) so a
    plain rollout keeps today's behaviour and latency.

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
        # annotation did - decode a zero chunk and read the shape off the output transform.
        probe = policy._output_transform(
            {
                "state": np.zeros((1, self._model_action_dim), np.float32),
                "actions": np.zeros((1, int(model.action_horizon), self._model_action_dim), np.float32),
            }
        )
        raw_dim = int(np.asarray(probe["actions"]).shape[-1])
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
        selected = bool(obs.pop("critic_select", False))
        want_hud = bool(obs.pop("critic_hud", False))
        num_samples = int(obs.pop("num_samples", 0) or self._default_samples)
        if not selected:
            return self._pol.infer(obs, noise=noise)

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
