"""Serve (or smoke-test) one policy-extraction arm.

    # websocket server the robot client connects to, exactly like scripts/serve_policy.py
    uv run python scripts/serve_extraction_arm.py --arm qpilots --alpha 0.2 --port 8000

    # no robot: run one inference on a dataset frame and print the chunk's stats
    uv run python scripts/serve_extraction_arm.py --arm dql --self-test

Arms: bc | awr | cfgrl | flowdpg | qam | dql | lps | lpsd | flowdagger | qpilots | idql | bon
(see src/openpi/extraction/serving.py for what each one swaps in the sampler).
"""

# ruff: noqa: PLC0415

import argparse
import dataclasses
import logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default=None, help="default prompt injected when the client omits one")
    ap.add_argument("--self-test", action="store_true", help="one inference on a dataset frame, then exit")
    ap.add_argument("--cfg-w", type=float, default=None, help="cfgrl guidance weight")
    ap.add_argument("--alpha", type=float, default=None, help="qpilots steering scale")
    ap.add_argument("--n-samples", type=int, default=None, help="idql/bon draws")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.extraction import serving

    over = {k: v for k, v in (("cfg_w", a.cfg_w), ("alpha", a.alpha), ("n_samples", a.n_samples)) if v is not None}
    policy = serving.load_arm(a.arm, step=a.step, default_prompt=a.prompt, **over)
    logging.info("arm spec: %s", dataclasses.asdict(policy._spec))

    if a.self_test:
        import numpy as np

        from openpi.extraction import data as exdata

        dataset, _cfg = exdata.make_bc_dataset(str(serving.BC_CKPT / "assets"))
        # the raw (pre-transform) frame the server would receive from the robot client
        raw = dataset._dataset[0] if hasattr(dataset, "_dataset") else dataset[0]
        obs = {k: np.asarray(v) for k, v in raw.items() if k != "actions"}
        out = policy.infer(obs)
        act = np.asarray(out["actions"])
        print(f"chunk {act.shape}  |a| mean {np.abs(act).mean():.4f}  min {act.min():.3f}  max {act.max():.3f}")
        jerk = np.mean(np.diff(act, n=2, axis=0) ** 2)
        print(f"jerk {jerk:.5f}   (BC reference ~0.88 in normalized units)")
        return

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=a.port, metadata=policy.metadata
    )
    logging.info("serving arm %s on :%d", a.arm, a.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
