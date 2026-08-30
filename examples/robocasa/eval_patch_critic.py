"""Deploy-time evaluation of the standalone patch-critic: does value-based chunk selection beat the
raw pi05 actor?

At each replan the pi05 actor proposes N candidate chunks (stochastic flow-matching samples). The
frozen-DINOv2 patch-critic scores every candidate; the policy then either:
  * ``vla``      execute pi05's first sample (no selection)         -> reproduces the SR baseline
  * ``bon``      execute the argmax-Q candidate's full chunk        -> Best-of-N
  * ``adaptive`` execute the argmax-Q candidate, but only its best commitment horizon K (the prefix
                 with the highest per-prefix value), then replan    -> AC-RFT adaptive chunking

Runs in the RoboCasa sim in env/HDF5 action order (the official checkpoint's convention), so it is
directly comparable to ``run_eval.sh --env-action-order`` numbers.

    uv run --group eval examples/robocasa/eval_patch_critic.py \
        --task OpenDrawer --critic .scratch/patch_critic_v1 --mode bon --num-cand 8 --num-trials 25
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--mode", choices=["vla", "bon", "adaptive"], default="bon")
    ap.add_argument("--num-cand", type=int, default=8)
    ap.add_argument("--num-trials", type=int, default=25)
    ap.add_argument("--config", default="pi05_robocasa_pretrained")
    ap.add_argument(
        "--ckpt",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pi05_robocasa_pretrained/human300_pretrain/75000"),
    )
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    import sys

    import flax.serialization
    import jax
    import jax.numpy as jnp
    from PIL import Image

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.patch_critic.backbone import to_nchw
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import rollout as R  # noqa: N812

    cc = json.loads((a.critic / "config.json").read_text())
    H, gsz, atoms, nc = cc["horizon"], cc["macro_group_size"], cc["num_atoms"], cc["num_critics"]
    net = PatchCriticEnsemble(action_dim=12, horizon=H, num_critics=nc, macro_group_size=gsz, num_atoms=atoms)
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(cc["v_min"], cc["v_max"], atoms)
    centers = jnp.asarray(hl.centers)
    bb = DinoV2Backbone(cc["backbone"])
    grid = int(bb.num_patches(224) ** 0.5)
    pooled = grid // 2
    npatch = 3 * pooled * pooled

    def pool(p):
        b, _, d = p.shape
        return p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d).mean((3, 5)).reshape(b, npatch, d)

    @jax.jit
    def score(patches, state, cands):
        # patches [P,D], state [16], cands [N,H,12] -> per-candidate per-prefix value [N, mh]
        pc = jnp.repeat(patches[None], cands.shape[0], 0)  # [N,P,D]
        st = jnp.repeat(state[None], cands.shape[0], 0)
        out = net.apply(params, pc, cands, st)  # [K,N,mh,atoms]
        q = jnp.sum(jax.nn.softmax(out, -1) * centers, -1)  # [K,N,mh]
        return jnp.min(q, 0)  # ensemble-min -> [N, mh]

    def patches_of(obs):
        imgs = np.stack(
            [
                np.asarray(Image.fromarray(R.image_from_obs(obs, R.CAMERAS[k])).resize((224, 224), Image.BILINEAR))
                for k in ("observation/image", "observation/wrist_image", "observation/image_right")
            ]
        )  # [3,224,224,3]
        x = jnp.asarray(to_nchw(imgs))[None]  # [1,3,3,224,224]... to_nchw handles [3,H,W,3]->[3,3,H,W]
        return pool(bb(x.reshape(1, 3, 3, 224, 224)))[0]  # [P,D]

    policy = _policy_config.create_trained_policy(_config.get_config(a.config), a.ckpt)
    env = R.make_env(a.task, camera_size=256, seed=a.seed)
    max_steps = a.max_steps or int(getattr(env, "horizon", 500))
    successes, trials = 0, []

    for trial in range(a.num_trials):
        env.rng = np.random.default_rng(a.seed + trial)
        np.random.seed(a.seed + trial)
        obs = env.reset()
        prompt = env.get_ep_meta().get("lang", a.task)
        step, success = 0, False
        while step < max_steps and not success:
            element = R.obs_to_element(obs, prompt)
            cands = np.stack([np.asarray(policy.infer(element)["actions"], np.float32) for _ in range(a.num_cand)])
            if a.mode == "vla":
                chunk, n_exec = cands[0], H
            else:
                pv = np.asarray(
                    score(patches_of(obs), jnp.asarray(R.state_from_obs(obs), jnp.float32), jnp.asarray(cands))
                )  # [N, mh]
                best = int(np.argmax(pv[:, -1]))  # argmax full-chunk value
                chunk = cands[best]
                if a.mode == "adaptive":
                    kbest = int(np.argmax(pv[best]))  # best commitment prefix (macro-group index)
                    n_exec = (kbest + 1) * gsz
                else:
                    n_exec = H
            for act in chunk[: max(int(n_exec), 1)]:
                obs, _, _, _ = env.step(np.asarray(act, np.float32)[:12])  # env-action-order
                step += 1
                if env._check_success():
                    success = True
                    break
                if step >= max_steps:
                    break
        successes += int(success)
        trials.append({"trial": trial, "success": bool(success), "steps": step})
        print(f"[{a.mode} {trial + 1}/{a.num_trials}] {'SUCCESS' if success else 'fail'} ({step})", flush=True)

    rate = successes / a.num_trials
    print(f"\n{a.task} [{a.mode}, N={a.num_cand}]: {successes}/{a.num_trials} = {rate:.1%}", flush=True)
    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(
            json.dumps(
                {
                    "task": a.task,
                    "mode": a.mode,
                    "num_cand": a.num_cand,
                    "successes": successes,
                    "num_trials": a.num_trials,
                    "success_rate": rate,
                    "trials": trials,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
