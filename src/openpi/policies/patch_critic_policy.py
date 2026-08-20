"""Server-side value-guided selection with a trained standalone patch-critic.

Unlike ``CriticSelectPolicy`` (which reads the VLA's RL token), the patch-critic is VLA-INDEPENDENT:
it scores candidate action chunks from a FROZEN DINOv2 patch grid over the robot's camera images.
So selection can run anywhere the raw images are — but we still do it server-side because the base
VLA's shared-backbone sampler (``extract_token_and_base_actions``) is what draws N candidates in one
pass, and the critic weights live here.

Selection is unconditional: a server started with a critic IS a value-guided policy, so the
client sends a plain observation and gets back the chunk to execute (plus the candidates and their
scores, declared at handshake). Two modes:
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
from openpi.policies import policy as _policy_mod
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


def _policy_norm_stats(policy):
    """The norm stats the served policy actually uses, dug out of its input transform chain."""
    t = getattr(policy, "_input_transform", None)
    for sub in getattr(t, "transforms", [t] if t is not None else []):
        ns = getattr(sub, "norm_stats", None)
        if isinstance(ns, dict) and ns:
            return {
                k: {
                    a: np.asarray(getattr(v, a))
                    for a in ("mean", "std", "q01", "q99")
                    if getattr(v, a, None) is not None
                }
                for k, v in ns.items()
            }
    return None


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
        # num_samples/num_steps size the sampler's noise array, so they must be compile-time constants;
        # left traced, jax.random.normal rejects the shape. (This path had never actually been run.)
        self._extract = nnx_utils.module_jit(
            model.extract_token_and_base_actions, static_argnames=("num_samples", "num_steps")
        )
        self._model_action_dim = int(model.action_dim)
        self._action_horizon = int(model.action_horizon)
        # The width that actually arrives on the wire (14 for YAM), not the model's padded one
        # (32 for pi05) -- what the declared candidate columns have to match.
        self._robot_action_dim = _policy_mod.probe_robot_action_dim(
            policy, model_action_dim=self._model_action_dim, action_horizon=self._action_horizon
        )

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
                model_action_dim=self._model_action_dim,
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
        # The critic may have been trained on a SUBSET of proprio channels (positions only, matching
        # what ALOHA/Libero/DROID feed). Slice the served state the same way or the network sees
        # different quantities in different slots.
        pidx = (self._spec or {}).get("proprio_indices")
        self._proprio_idx = None if pidx is None else np.asarray(pidx, np.int64)
        self._critic_action_dim = int(cc.get("action_dim", 14))
        # A pi05-space critic eats the sampler's output as-is; a raw-space one needs the decode path.
        self._pre = None
        if (self._spec or {}).get("normalization") == "pi05":
            from openpi.patch_critic import preproc as critic_preproc

            # Prefer the copy INSIDE the checkpoint: the recorded path is provenance, and relying on
            # it means the critic silently picks up whatever norm_stats.json happens to sit there.
            embedded = pathlib.Path(critic_dir) / self._spec.get("norm_stats_file", "pi05_norm_stats.json")
            if embedded.exists():
                stats = critic_preproc.load_norm_stats(embedded)
            elif pathlib.Path(self._spec["norm_stats"]).exists():
                stats = critic_preproc.load_norm_stats(self._spec["norm_stats"])
                logging.warning(
                    "critic %s has no embedded %s; falling back to the recorded path %s. Re-save it so "
                    "the checkpoint is self-contained.",
                    critic_dir,
                    embedded.name,
                    self._spec["norm_stats"],
                )
            else:
                raise FileNotFoundError(
                    f"critic {critic_dir} declares pi05 preprocessing but neither its embedded "
                    f"{embedded.name} nor the recorded path {self._spec['norm_stats']} exists -- the "
                    "critic cannot be served without the stats it was trained against"
                )
            self._pre = critic_preproc.Pi05Preproc(
                ref=np.asarray(self._spec["joint_delta_reference"], np.int64),
                stats=stats,
                use_quantiles=bool(self._spec["use_quantiles"]),
                delta=self._spec["delta_mode"] == "joint",
            )
            # "I am using the same norm stats as pi05" is a claim worth checking, not assuming: read
            # the stats off the policy being served and compare the numbers, not the paths.
            served = _policy_norm_stats(policy)
            if served is not None:
                mismatch = critic_preproc.compare(stats, served)
                if mismatch:
                    raise ValueError(
                        "critic/policy norm-stats mismatch -- the critic was trained against different "
                        "statistics than the policy being served:\n  - " + "\n  - ".join(mismatch)
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
        # Historic opt-in key; a server started with a critic selects on every request now.
        # Popped because the model's input transforms reject keys they do not know.
        obs.pop("critic_select", None)
        want_hud = bool(obs.pop("critic_hud", False))
        num_samples = int(obs.pop("num_samples", 0) or self._default_samples)

        state = np.asarray(obs[self._state_key], np.float32).reshape(-1)
        want_sd = (self._spec or {}).get("state_dim")
        if want_sd is not None and state.shape[0] != want_sd:
            raise ValueError(
                f"state width {state.shape[0]} but the critic was trained on {want_sd} "
                f"({self._state_key}); the proprio channels would land in the wrong slots"
            )
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
        # A pi05-space critic takes the policy's own state, as-is. Read it off the transform rather
        # than recomputing it: one source of truth, and the load-time digest check is what guarantees
        # it matches the stats the critic trained against.
        norm_state = np.asarray(inputs["state"], np.float32).reshape(-1)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        _token, base = self._extract(sample_rng, observation, num_samples=num_samples, num_steps=self._flow_steps)
        chunks_model = np.asarray(base[0], np.float32)  # [N, H, model_dim]
        # ROBOT-space chunks: what gets executed, and what the client records. Always through the
        # policy's own output transform, fed the NORMALIZED state -- Unnormalize runs first in that
        # transform and converts it to real units before JointAbsoluteActions adds it
        # (actions[..., i] += state[..., ref[i]]). Handing it the raw state un-normalizes an
        # already-physical value a second time and displaces every candidate identically, which is
        # how a critic run ended up commanding 1.6 rad from the arm while a plain rollout of the
        # same checkpoint tracked it within 0.17.
        robot_actions = np.asarray(
            self._pol._output_transform(
                {
                    "state": np.broadcast_to(norm_state, (chunks_model.shape[0], norm_state.shape[0])).copy(),
                    "actions": chunks_model,
                }
            )["actions"],
            np.float32,
        )

        # What the CRITIC is scored on is a separate question from what the robot executes: the two
        # coincide only for a critic trained in robot units. Conflating them returned the critic's
        # own input as the action -- for a pi05-space critic that is a normalized joint DELTA, which
        # lands in a plausible numeric range and is meaningless as a joint target.
        if self._pre is not None:
            # Shared preprocessing: the sampler already emits normalized joint deltas, which is
            # precisely what the critic was trained on. No conversion, so nothing to get wrong.
            scored_actions = chunks_model[..., : self._critic_action_dim]
            scored_state = norm_state if self._proprio_idx is None else norm_state[self._proprio_idx]
        else:
            # Legacy raw-units critic: it was trained on absolute joint targets, so it scores the
            # same array the robot will execute, against the raw state those units live in.
            scored_actions = robot_actions
            scored_state = state if self._proprio_idx is None else state[self._proprio_idx]
        scored = np.asarray(scored_actions, np.float32)  # [N, H, *] in the CRITIC's space
        decoded = robot_actions  # [N, H, A] in ROBOT space -- executed and recorded

        pv = np.asarray(self._score(patches, jnp.asarray(scored_state), jnp.asarray(scored)))  # [N, mh]
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
            # (X, N, A), not (H, N, A): the broker slices an extra by the reply's OWN chunk length,
            # and in adaptive mode only X of the H steps are being executed. A full-horizon array
            # would have a leading axis nothing matches, so it would be passed through whole and
            # then dropped by the recorder as the wrong shape -- silently, every frame.
            "action_samples": np.swapaxes(decoded, 0, 1)[:x],
            "critic_scores": np.broadcast_to(pv[:, -1], (x, pv.shape[0])).copy(),  # (X, N)
            "critic_choice": np.full((x, 1), best, np.float32),  # (X, 1)
        }
        # How the commitment was carved up, always sent: one float each, and without them a
        # recording cannot say where the macro-group boundaries were or how much of the winning
        # chunk was actually committed. Both are per-REPLAN facts (the decision is made once);
        # they ride per-step because that is the only layout the broker slices and the recorder
        # records, so a replan's frames repeat them.
        out["critic_macro"] = np.full((x, 1), self._macro, np.float32)
        out["critic_best_prefix"] = np.full((x, 1), int(np.argmax(pv[best])), np.float32)
        if self._mode == "adaptive":
            # Only adaptive drops part of what the model proposed: `action_samples` above is the
            # EXECUTED prefix, so the un-executed tail exists nowhere else. In `bon` the executed
            # chunk IS the full horizon, so this would be a duplicate column.
            out["action_samples_full"] = np.broadcast_to(
                np.swapaxes(decoded, 0, 1)[None], (x, decoded.shape[1], decoded.shape[0], decoded.shape[2])
            ).copy()
        if want_hud:
            out["critic_grid"] = np.broadcast_to(pv, (x, *pv.shape)).copy()  # (X, N, mh)
        return out

    def extra_features(self, num_samples: int | None = None) -> dict:
        """The per-step arrays this policy sends, for the handshake.

        Nothing is recorded that is not declared here: the client turns exactly these into dataset
        columns and drops anything else. Without it a value-guided rollout looks completely
        ordinary on disk -- no candidates, no scores, no record of what the critic chose -- which
        is what happened to the first `bon8` runs.

        Shapes are PER STEP; the chunk axis is deliberately absent, because the chunk length is
        adaptive and is read off each reply (see the note on `action_samples` in `infer`).
        """
        n = int(num_samples or self._default_samples)
        declared = {
            "action_samples": [n, self._robot_action_dim],
            "critic_scores": [n],
            "critic_choice": [1],
            # Where the macro-group boundaries fell, and which group the commitment stopped at.
            "critic_macro": [1],
            "critic_best_prefix": [1],
        }
        if self._mode == "adaptive":
            # The full horizon the model proposed, of which only a prefix was executed (see infer).
            declared["action_samples_full"] = [self._action_horizon, n, self._robot_action_dim]
        return declared

    @property
    def robot_action_dim(self) -> int:
        """The robot-space width recovered at construction, so a caller need not re-probe it
        (this wrapper has no output transform of its own to probe through)."""
        return self._robot_action_dim

    #: Picks among its own candidates -- see CriticSelectPolicy.selects_candidates.
    selects_candidates = True

    @property
    def metadata(self):
        return self._pol.metadata
