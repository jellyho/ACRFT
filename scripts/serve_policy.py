import dataclasses
import enum
import json
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Which assets dir inside the checkpoint holds this run's norm stats, when it is not the
    # config's default (`asset_id or repo_id`). The data-scaling study needs this: each point
    # trains on a different subset, so it computes and trains with its own `--asset-id`, and
    # serving has to ask for the same one. Without it a scaling checkpoint fails to load with
    # "Norm stats file not found" while the stats sit right there under their own name.
    asset_id: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Directory of a trained critic. Serving it here selects on EVERY request: a server started
    # with a critic is a value-guided policy, and the client sends nothing about it.
    #
    # Either kind is accepted and the kind is read off the artifact, not passed in -- a directory
    # whose config.json has `num_patches` is a standalone patch critic (frozen DINOv2 over the
    # cameras, config.json + params.msgpack), anything else is an RLT-token critic (which also
    # wants proprio_stats.json, see scripts/export_critic_serving.py). One flag, because two
    # serving entry points was two things to keep in step and the difference is in the checkpoint.
    critic: str | None = None

    # Patch critics only: "bon" executes the whole winning chunk, "adaptive" executes just its
    # highest-value commitment prefix and replans -- so the chunk length varies per reply.
    #
    # Left unset, it follows the critic: one trained with `macro_group_size == horizon` has a
    # single commitment group and can only ever return the whole chunk, so adaptive would be bon
    # under another name; one trained with several groups was trained to be committed to a prefix.
    # Set it explicitly to run the other mode -- bon on a multi-group critic is a real comparison,
    # and the artifact cannot know which of the two you meant to run today.
    critic_mode: str | None = None

    # How many action chunks to draw per observation. This is what the server DOES, not what a
    # client may ask for: it is both what gets sampled and what the handshake declares the
    # `action_samples` column for, so the two cannot disagree. The robot client sends nothing about
    # it -- it executes `actions` and records whatever was declared. 0/1 (default) is a plain
    # rollout, one forward pass. Above that costs N forward passes (or one backbone pass with
    # --critic, which samples its own candidates).
    num_samples: int = 0

    # Execute only the first N steps of each reply, then replan. A checkpoint trained at horizon 30
    # is open loop for a second when its chunk is run whole; this trades inference cost for
    # reaction time without retraining, and the client needs no setting -- it reads the length off
    # the reply. 0 (default) leaves the chunk as the policy returned it.
    execute_steps: int = 0

    # Denoising iterations per chunk, passed to the model's sample_actions. This is where the
    # few-step models are worth what they cost: alphaflow is trained on mean velocity so it can
    # answer in one step (its own default), while pi05 integrates the flow over ten. Sweeping it
    # is how you find where quality stops paying for latency -- and the latency is per replan, so
    # it compounds with --execute-steps. Unset leaves each model's own default alone.
    num_steps: int | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _sample_kwargs(args: Args) -> "dict | None":
    """What to pass through to the model's own sample_actions, or None to leave its defaults.

    Only set what was asked for: handing a model `num_steps=None` is not the same as not handing
    it one, and each model's default is its own (alphaflow 1, pi05 10).
    """
    if args.num_steps is None:
        return None
    logging.info("sampling with num_steps=%d", args.num_steps)
    return {"num_steps": int(args.num_steps)}


def create_policy(args: Args) -> tuple[_policy.Policy, _config.TrainConfig]:
    """Create a policy from the given arguments, with the config it was built from.

    The config comes back too so the server can describe the policy to its client (see
    `spec_metadata`) without anyone hard-coding numbers per robot.
    """
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.policy.asset_id is not None:
                # Same override compute_norm_stats and train already take, so all three stages
                # can name the same assets dir.
                train_config = dataclasses.replace(
                    train_config,
                    data=dataclasses.replace(
                        train_config.data,
                        assets=dataclasses.replace(train_config.data.assets, asset_id=args.policy.asset_id),
                    ),
                )
            return (
                _policy_config.create_trained_policy(
                    train_config,
                    args.policy.dir,
                    default_prompt=args.default_prompt,
                    sample_kwargs=_sample_kwargs(args),
                ),
                train_config,
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT.get(args.env)
            if checkpoint is None:
                raise ValueError(f"Unsupported environment mode: {args.env}")
            train_config = _config.get_config(checkpoint.config)
            return (
                _policy_config.create_trained_policy(
                    train_config, checkpoint.dir, default_prompt=args.default_prompt, sample_kwargs=_sample_kwargs(args)
                ),
                train_config,
            )


def spec_metadata(train_config: _config.TrainConfig) -> dict:
    """The part of the obs/action spec a client can act on, read off the train config.

    Every model config carries `action_horizon`, so this is the same code for RoboCasa,
    YAM, or anything added later — no per-robot branch. Without it the chunk size has to
    reach the client out of band, and a checkpoint trained at 30 served to a client
    assuming 16 raises nothing: it silently discards half of every chunk.

    `model.action_dim` is deliberately not advertised. It is the model's padded width (32
    for pi05), while the output transform slices the chunk back to the robot's real action
    size on the way out — 14 for YAM. Publishing 32 would describe something no client ever
    receives, which is worse than publishing nothing.
    """
    return {
        "action_horizon": int(train_config.model.action_horizon),
        # Tells a client it may ask for several chunks per observation (see MultiSamplePolicy).
        # Without it a viewer has to try and interpret the absence of `action_samples` in the
        # reply, which is indistinguishable from a server that simply ignored the request.
        "supports_multi_sample": True,
        # Which stack is behind the port, and which checkpoint. Several serve this same wire --
        # upstream openpi, this fork, a LeRobot checkpoint behind an adapter on the robot side --
        # and they do not take interchangeable observations. Serving the wrong one still returns a
        # well-formed chunk of the right shape, so the handshake is the only place to catch it.
        "framework": "acrft",
        "policy_name": str(train_config.name),
    }


def _build_critic_policy(policy, args: Args):
    """Wrap the base policy in whichever critic `args.critic` holds.

    The kind is read off the critic's own config.json rather than selected by the caller: a
    directory carrying `num_patches` was trained as a standalone patch critic, anything else is an
    RLT-token critic. Getting that from the artifact is what lets one server serve both.

    N is taken from --num-samples when given, so the count the critic samples and the count its
    handshake declares are one number instead of two that can disagree.
    """
    critic_dir = pathlib.Path(args.critic).expanduser()
    cfg = json.loads((critic_dir / "config.json").read_text())
    samples = {"default_samples": args.num_samples} if args.num_samples > 1 else {}
    if "num_patches" in cfg:
        from openpi.policies import patch_critic_policy as _pcp

        groups = max(1, int(cfg.get("horizon", 0)) // max(1, int(cfg.get("macro_group_size", 0) or 1)))
        mode = args.critic_mode or ("adaptive" if groups > 1 else "bon")
        if args.critic_mode == "adaptive" and groups <= 1:
            logging.warning(
                "critic-mode=adaptive but %s has one commitment group (macro_group_size == horizon)"
                " -- every reply will be the whole chunk, which is what bon does",
                critic_dir.name,
            )
        logging.info(
            "critic: patch-critic (%s), mode=%s%s, %d commitment group(s)",
            critic_dir.name,
            mode,
            "" if args.critic_mode else " (from the critic)",
            groups,
        )
        return _pcp.PatchCriticSelectPolicy(policy, str(critic_dir), mode=mode, **samples)
    logging.info("critic: RLT-token critic (%s)", critic_dir.name)
    return _policy.CriticSelectPolicy(policy, str(critic_dir), **samples)


def main(args: Args) -> None:
    policy, train_config = create_policy(args)
    # Wrapped unconditionally: it is inert unless a request carries `num_samples`, so a plain
    # rollout pays nothing, and there is no server-side mode to remember to turn on before
    # looking at the action distribution.
    if args.critic is not None:
        # Selection must wrap the BARE policy: it drives the model's own shared-backbone
        # sampler, and stacking it over MultiSamplePolicy would pay N full forwards instead.
        policy = _build_critic_policy(policy, args)
    robot_action_dim = None
    if args.num_samples > 1:
        # A critic already probed this at its own construction; reuse it rather than decode a
        # second zero chunk through the same output transform.
        # Ask the policy first: a critic wrapper already recovered this at construction, and has no
        # output transform of its own to probe through. Duck-typed, so it holds for either critic.
        declared = getattr(policy, "robot_action_dim", None)
        logging.info("robot action width: %s", f"{declared} (from the policy)" if declared else "probing")
        robot_action_dim = declared or _policy.probe_robot_action_dim(
            policy,
            model_action_dim=int(train_config.model.action_dim),
            action_horizon=int(train_config.model.action_horizon),
        )
    policy = _policy.MultiSamplePolicy(
        policy,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
        robot_action_dim=robot_action_dim,
        default_samples=args.num_samples,
    )
    if args.execute_steps > 0:
        # Outermost, so it cuts the actions and every per-step extra together -- see
        # TruncateChunkPolicy. Wrapping inside MultiSamplePolicy would leave the candidates
        # describing steps that never ran.
        logging.info("executing %d of each chunk, then replanning", args.execute_steps)
        policy = _policy.TruncateChunkPolicy(policy, args.execute_steps)

    # Config-derived spec first, so an explicit policy_metadata entry can still override it.
    policy_metadata = {**spec_metadata(train_config), **(policy.metadata or {})}
    # What this policy sends alongside its actions, so the robot client records it without
    # either side hard-coding a name. Per-step shapes only -- the chunk axis is adaptive.
    declare = getattr(policy, "extra_features", None)
    if callable(declare):
        policy_metadata["extra_features"] = declare()
    logging.info("Serving %s: %s", train_config.name, policy_metadata)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
