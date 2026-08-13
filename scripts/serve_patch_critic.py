"""Serve a base VLA behind the standalone patch-critic for value-guided deployment.

Builds the same stack serve_policy.py does, but the value-guided wrapper is the PATCH-critic
(DINOv2 patches over the robot cameras) instead of the RLT-token critic. A robot client opts in per
request with ``critic_select`` (+ optional ``num_samples``); the server samples N chunks from the
base VLA in one backbone pass, scores them with the trained patch-critic, and returns the best chunk
(``bon``) or its highest-value commitment prefix (``adaptive``).

    # yam: base pi05 + the cost_to_goal patch-critic, adaptive chunking
    uv run python scripts/serve_patch_critic.py \
        --config pi05_yam_lego_taxi_rlt \
        --checkpoint checkpoints/pi05_yam_lego_taxi_rlt/yam_lego_taxi_rlt_s200_successonly/200000 \
        --critic .scratch/patch_critic_yam_cgfloor --mode adaptive --port 8000
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import patch_critic_policy as _pcp
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    critic: str  # trained patch-critic dir (config.json + params.msgpack)
    config: str = "pi05_yam_lego_taxi_rlt"  # base VLA train config
    checkpoint: str = "checkpoints/pi05_yam_lego_taxi_rlt/yam_lego_taxi_rlt_s200_successonly/200000"
    mode: str = "bon"  # "bon" (full best chunk) | "adaptive" (best commitment prefix, then replan)
    num_samples: int = 8  # candidate chunks scored per decision
    img_size: int = 224
    port: int = 8000
    default_prompt: str | None = None
    asset_id: str | None = None


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    train_config = _config.get_config(args.config)
    if args.asset_id is not None:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(
                train_config.data,
                assets=dataclasses.replace(train_config.data.assets, asset_id=args.asset_id),
            ),
        )
    base = _policy_config.create_trained_policy(train_config, args.checkpoint, default_prompt=args.default_prompt)

    # value-guided selection with the patch-critic (must wrap the BARE policy: it drives the model's
    # own shared-backbone sampler), then MultiSamplePolicy for plain multi-sample requests.
    policy = _pcp.PatchCriticSelectPolicy(
        base, args.critic, mode=args.mode, img_size=args.img_size, default_samples=args.num_samples
    )
    policy = _policy.MultiSamplePolicy(
        policy,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
    )

    policy_metadata = {
        "action_horizon": int(train_config.model.action_horizon),
        "supports_multi_sample": True,
        "patch_critic": True,
        "patch_critic_mode": args.mode,
        **(policy.metadata or {}),
    }
    hostname = socket.gethostname()
    logging.info("Serving %s + patch-critic (%s) on :%d as %s", train_config.name, args.mode, args.port, hostname)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy_metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
