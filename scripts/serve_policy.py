import dataclasses
import enum
import logging
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


def create_policy(args: Args) -> tuple[_policy.Policy, _config.TrainConfig]:
    """Create a policy from the given arguments, with the config it was built from.

    The config comes back too so the server can describe the policy to its client (see
    `spec_metadata`) without anyone hard-coding numbers per robot.
    """
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            return (
                _policy_config.create_trained_policy(train_config, args.policy.dir, default_prompt=args.default_prompt),
                train_config,
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT.get(args.env)
            if checkpoint is None:
                raise ValueError(f"Unsupported environment mode: {args.env}")
            train_config = _config.get_config(checkpoint.config)
            return (
                _policy_config.create_trained_policy(train_config, checkpoint.dir, default_prompt=args.default_prompt),
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
    return {"action_horizon": int(train_config.model.action_horizon)}


def main(args: Args) -> None:
    policy, train_config = create_policy(args)
    # Config-derived spec first, so an explicit policy_metadata entry can still override it.
    policy_metadata = {**spec_metadata(train_config), **(policy.metadata or {})}
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
