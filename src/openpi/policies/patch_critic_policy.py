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

import logging
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
    # INTER_AREA, not bilinear: the feature cache the critic trained on downsamples 480x640 -> 224 with
    # cv2.INTER_AREA, and bilinear downsampling aliases. Measured drift between the two was 5.5% mean /
    # 21% max relative L2 per patch token -- small, but free to remove.
    import cv2

    x = np.asarray(image)
    if np.issubdtype(x.dtype, np.floating):
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)


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

        from openpi.patch_critic import spec as critic_spec

        cc, self._norm_stats = critic_spec.load(critic_dir)
        # A critic's inputs are RAW dataset units (see openpi.patch_critic.spec). The wrapper honours that by
        # reading state before the input transform and un-normalizing candidates through the output
        # transform -- but only this check makes a mismatched critic fail loudly instead of returning
        # confident nonsense. Older checkpoints carry no spec; they are the raw-units generation.
        self._spec = cc.get("input_spec")
        if self._spec is not None:
            problems = critic_spec.check(
                self._spec,
                state_dim=int(self._spec["state_dim"]),  # checked against the live obs in infer()
                action_dim=self._model_action_dim,
                num_cameras=len(camera_keys),
                img_size=self._img_size,
            )
            if problems:
                raise ValueError("critic/server contract mismatch:\n  - " + "\n  - ".join(problems))
        else:
            logging.warning(
                "critic %s has no input_spec (pre-contract checkpoint); assuming raw dataset units. "
                "Re-save it with scripts/backfill_critic_spec.py to enable validation.",
                critic_dir,
            )
        self._warned_state = False
        self._critic_action_dim = int(cc.get("action_dim", 14))
        # A pi05-space critic eats the sampler's output as-is; a raw-space one needs the decode path.
        self._pre = None
        if (self._spec or {}).get("normalization") == "pi05":
            from openpi.patch_critic import preproc as critic_preproc

            self._pre = critic_preproc.Pi05Preproc(
                ref=np.asarray(self._spec["joint_delta_reference"], np.int64),
                stats=critic_preproc.load_norm_stats(self._spec["norm_stats"]),
                use_quantiles=bool(self._spec["use_quantiles"]),
                delta=self._spec["delta_mode"] == "joint",
            )
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
        if self._norm_stats is not None and self._pre is None and not self._warned_state:
            from openpi.patch_critic import spec as critic_spec

            bad = critic_spec.out_of_range(self._norm_stats, state)
            if bad:
                self._warned_state = True  # once per server; this is a diagnostic, not a rate-limiter
                logging.warning(
                    "state channels %s are far outside the critic's training range -- the critic "
                    "expects RAW dataset units; its values are not trustworthy here.",
                    bad[:12],
                )
        patches = self._patches_of(obs)

        # N candidate chunks from ONE backbone pass (the token is unused by the patch-critic).
        inputs = self._pol._input_transform(dict(obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        _token, base = self._extract(sample_rng, observation, num_samples=num_samples, num_steps=self._flow_steps)
        chunks_model = np.asarray(base[0], np.float32)  # [N, H, model_dim]
        if self._pre is not None:
            # Shared preprocessing: the sampler already emits normalized joint deltas, which is
            # precisely what the critic was trained on. No conversion, so nothing to get wrong.
            scored_actions = chunks_model[..., : self._critic_action_dim]
            scored_state = self._pre.state(state)
        else:
            # Legacy raw-units critic. The output transform un-normalizes AND undoes the joint-delta
            # parameterisation, and the latter needs the REAL state (JointAbsoluteActions does
            # actions[..., i] += state[..., ref[i]]). This passed zeros, which silently left the
            # chunks as DELTAS while the critic was trained on ABSOLUTE joint targets -- no error,
            # just meaningless values. Broadcast the live state over the N candidates.
            scored_actions = np.asarray(
                self._pol._output_transform(
                    {
                        "state": np.broadcast_to(state, (chunks_model.shape[0], state.shape[0])).copy(),
                        "actions": chunks_model,
                    }
                )["actions"],
                np.float32,
            )
            scored_state = state
        decoded = np.asarray(scored_actions, np.float32)  # [N, H, A] in the critic's own space

        pv = np.asarray(self._score(patches, jnp.asarray(scored_state), jnp.asarray(decoded)))  # [N, mh]
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
