"""End-to-end smoke test of the critic-select deploy pipeline (no robot, no websocket).

Builds the policy exactly as serve_policy does — create_trained_policy -> CriticSelectPolicy ->
MultiSamplePolicy — then calls infer() on a correctly-shaped dummy observation, once plainly and
once with critic_select+critic_hud, and checks the response carries the chosen chunk plus the HUD
provenance. Finally composes one HUD frame through deploy_hud so the whole path is exercised.

    uv run python scripts/smoke_deploy_hud.py \
        --config pi05_yam_lego_taxi_rlt \
        --checkpoint checkpoints/pi05_yam_lego_taxi_rlt/yam_lego_taxi_rlt_s200_successonly/200000 \
        --critic .scratch/critic_yam_s200_iql
"""

import argparse
import pathlib
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--prompt", default="pick up the lego and put it in the taxi")
    ap.add_argument("--hud-out", type=pathlib.Path, default=pathlib.Path(".scratch/deploy_smoke_hud.mp4"))
    a = ap.parse_args()

    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(a.config)
    policy = _policy_config.create_trained_policy(train_config, a.checkpoint, default_prompt=a.prompt)
    policy = _policy.CriticSelectPolicy(policy, a.critic)
    policy = _policy.MultiSamplePolicy(
        policy,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
    )
    print("policy built (create_trained -> CriticSelect -> MultiSample)", flush=True)

    def obs():
        img = lambda: (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)  # noqa: E731
        return {
            "observation/state": np.zeros(42, np.float32),
            "observation/image": img(),
            "observation/wrist_image": img(),
            "observation/image_right": img(),
            "prompt": a.prompt,
        }

    # 1) plain infer — critic path OFF, behaves like a normal server
    plain = policy.infer(obs())
    assert "actions" in plain, "plain infer returned no actions"
    print("plain infer OK: actions", np.asarray(plain["actions"]).shape, flush=True)

    # 2) critic-select + HUD provenance
    resp = policy.infer({**obs(), "critic_select": True, "critic_hud": True})
    for k in (
        "actions",
        "action_samples",
        "critic_scores",
        "critic_choice",
        "critic_grid",
        "critic_best_prefix",
        "critic_macro",
    ):
        assert k in resp, f"critic_hud response missing '{k}'"
    cand = np.asarray(resp["action_samples"])
    grid = np.asarray(resp["critic_grid"])
    print(
        f"critic infer OK: chosen #{int(resp['critic_choice'])}  cand {cand.shape}  grid {grid.shape}  "
        f"spread {float(grid.max() - grid.min()):.4f}",
        flush=True,
    )

    # 3) compose one HUD frame through the deploy module
    sys.path.insert(0, "examples/robocasa")
    import deploy_hud

    rec = deploy_hud.HudRecorder(
        mode="bon",
        horizon=grid.shape[1] * int(resp["critic_macro"]),
        discount=0.9998,
        projector=deploy_hud.SketchProjector(gain=6.0),
        hold=2,
    )
    frame = obs()["observation/image"]
    rec.add(frame, frame, resp, step=0)
    rec.save(a.hud_out, fps=5)
    print(f"HUD composed via deploy_hud -> {a.hud_out}", flush=True)
    print("MARKER_DEPLOY_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
