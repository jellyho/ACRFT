"""Turn a MimicGen HDF5 into flat tensors for chunk-level training.

MimicGen datasets are all successful, which does not make them optimal: the episodes in one task
range over a wide span of lengths, so "how efficiently" varies even when "whether" does not. That
is the suboptimality this benchmark offers, and a discounted value ranks it without any extra
labelling.

Low-dimensional observations first (fast to iterate); the same files also carry camera images, so
the image version needs no new download.

    uv run python slurm/probes/mimicgen_data.py --task three_piece_assembly_d0
"""

import argparse
import json
import pathlib

import h5py
import numpy as np

# the robomimic low-dim convention for these environments
LOWDIM_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def build(path: pathlib.Path, out: pathlib.Path, max_demos=None, gamma=0.99):
    f = h5py.File(path, "r")
    d = f["data"]
    env_args = json.loads(d.attrs["env_args"])
    keys = sorted(d.keys(), key=lambda k: int(k.split("_")[-1]))
    if max_demos:
        keys = keys[:max_demos]

    obs, act, rew, done, ep_id, ret = [], [], [], [], [], []
    lengths = []
    for i, k in enumerate(keys):
        g = d[k]
        o = np.concatenate([np.asarray(g["obs"][j], np.float32) for j in LOWDIM_KEYS], axis=-1)
        a = np.asarray(g["actions"], np.float32)
        r = np.asarray(g["rewards"], np.float32)
        n = len(a)
        lengths.append(n)
        # discounted return-to-go: with a terminal success reward this ranks faster episodes higher,
        # which is exactly the quality signal an all-successful dataset still contains
        rtg = np.zeros(n, np.float32)
        acc = 0.0
        for t in range(n - 1, -1, -1):
            acc = r[t] + gamma * acc
            rtg[t] = acc
        obs.append(o[:n])
        act.append(a)
        rew.append(r)
        dn = np.zeros(n, np.float32)
        dn[-1] = 1.0
        done.append(dn)
        ep_id.append(np.full(n, i, np.int64))
        ret.append(rtg)

    payload = {
        "obs": np.concatenate(obs).astype(np.float32),
        "action": np.concatenate(act).astype(np.float32),
        "reward": np.concatenate(rew).astype(np.float32),
        "done": np.concatenate(done).astype(np.float32),
        "episode": np.concatenate(ep_id),
        "rtg": np.concatenate(ret).astype(np.float32),
        "ep_lengths": np.asarray(lengths, np.int64),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    meta = {
        "task": path.stem,
        "env_name": env_args["env_name"],
        "demos": len(keys),
        "frames": int(len(payload["obs"])),
        "obs_dim": int(payload["obs"].shape[1]),
        "action_dim": int(payload["action"].shape[1]),
        "ep_len_mean": float(np.mean(lengths)),
        "ep_len_min": int(np.min(lengths)),
        "ep_len_max": int(np.max(lengths)),
        "lowdim_keys": LOWDIM_KEYS,
        "gamma": gamma,
    }
    (out.parent / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="three_piece_assembly_d0")
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/mimicgen/core"))
    ap.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/mimicgen/lowdim"))
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args()
    build(a.root / f"{a.task}.hdf5", a.out_root / f"{a.task}.npz", a.max_demos, a.gamma)


if __name__ == "__main__":
    main()
