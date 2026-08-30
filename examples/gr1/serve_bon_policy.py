"""Serve a VLA(+critic BoN) policy over the standard openpi websocket interface — GR1 harness server.

Runs in the MAIN venv (JAX VLA + critic). The .venv-gr1 rollout client sends the raw element
(observation/image, observation/state, prompt) and receives {"actions": chunk[:n_exec]} — BoN
selection and the commit length both live here, so the client stays a dumb executor and the server
protocol needs no extension.

    uv run examples/gr1/serve_bon_policy.py --config pi05_gr1_rlt --checkpoint <ckpt_dir> \\
        --mode vla                    # baseline server
    uv run examples/gr1/serve_bon_policy.py ... --critic <params.msgpack> --mode critic   # BoN server
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "robocasa"))

from openpi_client import base_policy as _base_policy

from openpi.serving import websocket_policy_server


class BoNServePolicy(_base_policy.BasePolicy):
    """Wraps eval_critic's policy_fn in the infer() contract; chunk length carries the commit."""

    def __init__(self, policy_fn):
        self._fn = policy_fn

    def infer(self, obs: dict) -> dict:
        out = self._fn(obs)
        if isinstance(out, tuple):
            chunk, n_exec, *_ = out
        else:
            chunk, n_exec = out, len(out)
        chunk = np.asarray(chunk)[: max(int(n_exec), 1)]
        return {"actions": chunk}

    def reset(self) -> None:
        if hasattr(self._fn, "reset"):
            self._fn.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_gr1_rlt")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--critic", default=None, help="critic params.msgpack; omit for the vla baseline server")
    ap.add_argument("--mode", default="vla", help="vla / critic / bon / prefix ... (eval_critic modes)")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from eval_critic import build_policy  # examples/robocasa/eval_critic.py (task-agnostic machinery)

    policy_fn, _, _ = build_policy(
        args.config,
        pathlib.Path(args.checkpoint),
        args.critic,
        mode=args.mode,
        num_samples=args.num_samples,
        flow_steps=args.flow_steps,
        seed=args.seed,
    )
    policy = BoNServePolicy(policy_fn)

    # Warm up: the first infer triggers minutes of JIT compilation, which would starve the websocket
    # event loop past the keepalive ping timeout (the client's connection dies with 1011). Compile
    # before serving so live requests stay near real-time.
    warmup = {
        "observation/image": np.zeros((256, 256, 3), np.uint8),
        "observation/state": np.zeros(44, np.float32),
        "prompt": "PnPCanToDrawerClose",
    }
    print("warming up (jit compile)...", flush=True)
    policy.infer(warmup)
    policy.reset()

    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port)
    print(f"serving {args.mode} policy on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
