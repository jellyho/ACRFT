"""Server-side value-guided selection with a trained standalone patch-critic.

Unlike ``CriticSelectPolicy`` (which reads the VLA's RL token), the patch-critic is VLA-INDEPENDENT:
it scores candidate action chunks from a FROZEN DINOv2 patch grid over the robot's camera images.
So selection can run anywhere the raw images are — but we still do it server-side because the base
VLA's shared-backbone sampler (``extract_token_and_base_actions``) is what draws N candidates in one
pass, and the critic weights live here.

Opt-in per request via ``critic_select`` (with optional ``num_samples``). Two modes:
  * ``bon``      execute the full chunk of the argmax-Q candidate (best-of-N).
  * ``adaptive`` execute only that candidate's highest-value commitment prefix K, then replan.

The critic dir must hold ``config.json`` + ``params.msgpack`` (written by train_patch_critic.py). The
camera keys default to YAM's (agentview, wrist_left, wrist_right) in the order the critic was trained.
"""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.policies.policy import BasePolicy
from openpi.policies.policy import Policy
from openpi.shared import nnx_utils

# YAM obs keys, in the SAME camera order the converter/critic used: agentview, wrist_left, wrist_right.
YAM_CAMERA_KEYS = ("observation/image", "observation/wrist_image", "observation/image_right")
YAM_STATE_KEY = "observation/state"


def _parse_image(image, size):
    from PIL import Image

    x = np.asarray(image)
    if np.issubdtype(x.dtype, np.floating):
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR))


class PatchCriticSelectPolicy(BasePolicy):
    def __init__(
        self,
        policy: Policy,
        critic_dir,
        *,
        mode: str = "bon",
        camera_keys=YAM_CAMERA_KEYS,
        state_key: str = YAM_STATE_KEY,
        img_size: int = 224,
        flow_steps: int = 10,
        default_samples: int = 8,
        seed: int = 0,
    ):
        from openpi.patch_critic.backbone import DinoV2Backbone
        from openpi.patch_critic.backbone import to_nchw
        from openpi.patch_critic.critic import HLGauss
        from openpi.patch_critic.critic import PatchCriticEnsemble

        self._pol = policy
        self._mode = mode
        self._camera_keys = tuple(camera_keys)
        self._state_key = state_key
        self._img_size = int(img_size)
        self._flow_steps = int(flow_steps)
        self._default_samples = int(default_samples)
        self._rng = jax.random.key(seed)
        self._to_nchw = to_nchw

        model = policy._model
        self._extract = nnx_utils.module_jit(model.extract_token_and_base_actions)
        self._model_action_dim = int(model.action_dim)
        self._action_horizon = int(model.action_horizon)

        cc = json.loads((pathlib.Path(critic_dir) / "config.json").read_text())
        self._macro = int(cc["macro_group_size"])
        atoms = int(cc["num_atoms"])
        import flax.serialization

        net = PatchCriticEnsemble(
            action_dim=cc.get("action_dim", 14),
            horizon=cc["horizon"],
            num_critics=cc["num_critics"],
            macro_group_size=self._macro,
            num_atoms=atoms,
        )
        params = flax.serialization.msgpack_restore((pathlib.Path(critic_dir) / "params.msgpack").read_bytes())
        centers = jnp.asarray(HLGauss(cc["v_min"], cc["v_max"], atoms).centers)
        bb = DinoV2Backbone(cc["backbone"])
        grid = int(bb.num_patches(self._img_size) ** 0.5)
        pooled = grid // 2
        ncam = len(self._camera_keys)
        npatch = ncam * pooled * pooled

        def pool(p):
            b, _, d = p.shape
            return (
                p.reshape(b, ncam, grid, grid, d)
                .reshape(b, ncam, pooled, 2, pooled, 2, d)
                .mean((3, 5))
                .reshape(b, npatch, d)
            )

        @jax.jit
        def patchify(imgs_nchw):  # [1, ncam, 3, S, S] -> [1, npatch, D]
            return pool(bb(imgs_nchw))

        @jax.jit
        def score(patches, state, cands):  # [P,D],[state],[N,H,A] -> [N, mh] ensemble-min per-prefix Q
            pc = jnp.repeat(patches[None], cands.shape[0], 0)
            st = jnp.repeat(state[None], cands.shape[0], 0)
            out = net.apply(params, pc, cands, st)  # [K,N,mh,atoms]
            q = jnp.sum(jax.nn.softmax(out, -1) * centers, -1)  # [K,N,mh]
            return jnp.min(q, 0)

        self._patchify = patchify
        self._score = score

    def _patches_of(self, obs):
        imgs = np.stack([_parse_image(obs[k], self._img_size) for k in self._camera_keys])  # [ncam,S,S,3]
        x = jnp.asarray(self._to_nchw(imgs), jnp.float32)[None]  # [1,ncam,3,S,S]
        return self._patchify(x)[0]  # [P,D]

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        selected = bool(obs.pop("critic_select", False))
        want_hud = bool(obs.pop("critic_hud", False))
        num_samples = int(obs.pop("num_samples", 0) or self._default_samples)
        if not selected:
            return self._pol.infer(obs, noise=noise)

        state = np.asarray(obs[self._state_key], np.float32).reshape(-1)
        patches = self._patches_of(obs)

        # N candidate chunks from ONE backbone pass (the token is unused by the patch-critic).
        inputs = self._pol._input_transform(dict(obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        _token, base = self._extract(sample_rng, observation, num_samples=num_samples, num_steps=self._flow_steps)
        chunks_model = np.asarray(base[0], np.float32)  # [N, H, model_dim]
        decoded = np.asarray(
            self._pol._output_transform(
                {
                    "state": np.zeros((chunks_model.shape[0], self._model_action_dim), np.float32),
                    "actions": chunks_model,
                }
            )["actions"],
            np.float32,
        )  # [N, H, A]

        pv = np.asarray(self._score(patches, jnp.asarray(state), jnp.asarray(decoded)))  # [N, mh]
        best = int(np.argmax(pv[:, -1]))  # argmax full-chunk value
        if self._mode == "adaptive":
            kbest = int(np.argmax(pv[best]))  # highest-value commitment prefix (macro-group index)
            n_exec = (kbest + 1) * self._macro
        else:
            n_exec = decoded.shape[1]
        chosen = decoded[best][: max(int(n_exec), 1)]  # (X, A)
        x = chosen.shape[0]

        out = {
            "actions": chosen,
            "action_samples": np.swapaxes(decoded, 0, 1),  # (H, N, A)
            "critic_scores": np.broadcast_to(pv[:, -1], (x, pv.shape[0])).copy(),  # (X, N)
            "critic_choice": np.full((x, 1), best, np.float32),  # (X, 1)
        }
        if want_hud:
            out["critic_grid"] = np.broadcast_to(pv, (x, *pv.shape)).copy()  # (X, N, mh)
            out["critic_best_prefix"] = np.full((x, 1), int(np.argmax(pv[best])), np.float32)
            out["critic_macro"] = np.full((x, 1), self._macro, np.float32)
        return out

    @property
    def metadata(self):
        return self._pol.metadata
