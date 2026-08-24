"""Roll a trained MimicGen policy in the simulator and report success and time.

Runs in .venv-mimicgen: MimicGen's environments target robosuite 1.4.x, while the main environment
carries 1.5.2 for RoboCasa, so the two cannot share an interpreter. Set MUJOCO_GL=osmesa; the
low-dimensional evaluation needs no rendering, but robosuite still initializes a context.

Commitment is a flag rather than a constant baked into the policy, so the same checkpoint can be
measured at every fixed length and under the critic's own choice, which is the comparison the
fixed-k sweep taught us to make.

    MUJOCO_GL=osmesa PYTHONPATH=third_party/mimicgen .venv-mimicgen/bin/python \
        slurm/probes/mimicgen_eval.py --ckpt <run>/ckpt.pt --commit 8 --episodes 50
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mimicgen_models import ChunkPolicy
from mimicgen_models import PrefixCritic
from mimicgen_models import select_k

LOWDIM_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def make_env(env_name, dataset_path):
    import json as _json

    import h5py
    import mimicgen  # noqa: F401  (registers the environments)
    import robosuite

    with h5py.File(dataset_path, "r") as f:
        ea = _json.loads(f["data"].attrs["env_args"])
    kw = dict(ea["env_kwargs"])
    kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    return robosuite.make(env_name=env_name, **kw)


# robosuite emits "object-state" while the recorded datasets call the same vector "object"
ENV_ALIAS = {"object": ("object", "object-state")}


def flat_obs(o):
    parts = []
    for k in LOWDIM_KEYS:
        for name in ENV_ALIAS.get(k, (k,)):
            if name in o:
                parts.append(np.asarray(o[name], np.float32).reshape(-1))
                break
        else:
            raise KeyError(f"{k} not in observation (have {sorted(o)[:8]})")
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, default=None, help="hdf5 the env args come from")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--commit", type=int, default=0, help="fixed commitment length; 0 uses the critic")
    ap.add_argument("--flow-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, norm = blob["cfg"], blob["norm"]
    o_mu = np.asarray(norm["o_mu"], np.float32)
    o_sd = np.asarray(norm["o_sd"], np.float32)

    pi = ChunkPolicy(cfg["obs_dim"], cfg["act_dim"], cfg["horizon"])
    pi.load_state_dict(blob["policy"])
    pi.eval()
    critic = None
    if a.commit == 0:
        if blob.get("critic") is None:
            raise SystemExit("this checkpoint has no critic; pass --commit to fix the length")
        critic = PrefixCritic(cfg["obs_dim"], cfg["act_dim"], cfg["horizon"], hist_len=cfg["hist_len"])
        critic.load_state_dict(blob["critic"])
        critic.eval()

    ds = a.dataset or pathlib.Path(f"/scratch/jellyho/acrft/mimicgen/core/{cfg['task']}.hdf5")
    env = make_env(cfg["env_name"], ds)

    succ, steps_used, ks, succ_steps = 0, [], [], []
    for ep in range(a.episodes):
        np.random.seed(a.seed + ep)
        env.reset()
        o = flat_obs(env._get_observations())
        hist = np.zeros((cfg["hist_len"], cfg["act_dim"] + 1), np.float32)
        done_ok, t = False, 0
        while t < a.max_steps:
            ot = torch.as_tensor((o - o_mu) / o_sd, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                chunk = pi.sample(ot, steps=a.flow_steps)[0].numpy()
                if critic is None:
                    k = a.commit
                else:
                    ht = torch.as_tensor(hist.reshape(-1), dtype=torch.float32).unsqueeze(0)
                    q = critic(ot, ht, torch.as_tensor(chunk, dtype=torch.float32).unsqueeze(0))
                    k = int(select_k(q)[0].item())
            ks.append(k)
            for j in range(min(k, cfg["horizon"], a.max_steps - t)):
                act = np.clip(chunk[j], -1, 1)
                env.step(act)
                hist = np.roll(hist, 1, axis=0)
                hist[0, :-1] = act
                hist[0, -1] = 1.0
                t += 1
                if env._check_success():
                    done_ok = True
                    break
            o = flat_obs(env._get_observations())
            if done_ok:
                break
        succ += done_ok
        steps_used.append(t)
        if done_ok:
            succ_steps.append(t)
        if (ep + 1) % 10 == 0:
            print(f"  {ep + 1}/{a.episodes}: success {succ}/{ep + 1}", flush=True)

    res = {
        "ckpt": str(a.ckpt),
        "task": cfg["task"],
        "arm": cfg["arm"],
        "commit": a.commit if a.commit else "critic",
        "episodes": a.episodes,
        "successes": int(succ),
        "success_rate": succ / a.episodes,
        "mean_steps": float(np.mean(steps_used)),
        "mean_steps_success_only": float(np.mean(succ_steps)) if succ_steps else None,
        "mean_k": float(np.mean(ks)),
        "seed": a.seed,
    }
    print(json.dumps(res, indent=1))
    out = a.out or (a.ckpt.parent / f"eval_commit{a.commit or 'critic'}.json")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
