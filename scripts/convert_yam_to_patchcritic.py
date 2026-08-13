"""Convert the yam_lego_taxi LeRobot dataset into the per-step patch-critic rollout format.

The patch-critic reads DINOv2 patches from raw camera images, but the yam RLT annotation only saved
VLA tokens. So we build a patch-critic dataset straight from the LeRobot demos: 3 cameras
(agentview + wrist L/R) resized to 224, observation.state[42], action[14], and a sparse terminal
reward from outcomes.jsonl (success episode -> reward 1 on its last frame). The cost_to_goal relabel
+ the value support / discount are applied at TRAIN time (train_patch_critic.py --reward-scheme).

Episodes are selected up to --max-frames, all FAIL episodes first (they are the scarce negatives the
cost_to_goal critic needs) then SUCCESS episodes, keeping whole episodes contiguous.

    uv run python scripts/convert_yam_to_patchcritic.py \
        --repo-id jellyho/yam_lego_taxi --root /data5/jellyho/yam_v2/lerobot \
        --outcomes /data5/jellyho/yam_v2/lerobot/jellyho/yam_lego_taxi/outcomes.jsonl \
        --max-frames 160000 --out /data5/jellyho/pc_rollouts_yam/lego_taxi
"""

import argparse
import json
import pathlib

import numpy as np

CAMS = ["observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right"]


def to_hwc_uint8(t, size):
    """A LeRobot image frame (CHW or HWC, float[0,1] or uint8) -> HWC uint8 resized to size x size."""
    from PIL import Image

    x = np.asarray(t)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--outcomes", required=True, help="outcomes.jsonl with per-episode success/fail")
    ap.add_argument("--max-frames", type=int, default=160000)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit-frames-test", type=int, default=0, help="convert only the first k frames (smoke)")
    a = ap.parse_args()

    import lerobot.datasets.lerobot_dataset as lrd

    ds = lrd.LeRobotDataset(a.repo_id, root=a.root)
    n_total = ds.num_frames
    # episode -> [from, to) GLOBAL frame index, from the episodes metadata
    epds = ds.meta.episodes
    starts = {int(e): int(f) for e, f in zip(epds["episode_index"], epds["dataset_from_index"], strict=True)}
    ends = {int(e): int(t) for e, t in zip(epds["episode_index"], epds["dataset_to_index"], strict=True)}
    print(f"dataset {a.repo_id}: {n_total} frames, {ds.num_episodes} episodes", flush=True)

    # only episodes that carry an outcome label (success/fail) can be used for cost_to_goal
    outc = {}
    for line in pathlib.Path(a.outcomes).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outc[int(r["episode"])] = r["outcome"]
    labeled = [e for e in outc if e in starts]
    fails = [e for e in labeled if outc[e] != "success"]
    succ = [e for e in labeled if outc[e] == "success"]
    print(f"labeled episodes: {len(labeled)} ({len(fails)} fail / {len(succ)} success)", flush=True)
    # all fails first (scarce negatives), then successes, until the frame budget is hit
    order, kept, tot = fails + succ, [], 0
    for e in order:
        length = int(ends[e] - starts[e])
        kept.append(e)
        tot += length
        if tot >= a.max_frames:
            break
    n_kept = sum(int(ends[e] - starts[e]) for e in kept)
    nfail = sum(1 for e in kept if e in set(fails))
    print(f"keeping {len(kept)} episodes ({nfail} fail / {len(kept) - nfail} success) = {n_kept} frames", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    S = a.img_size
    # probe dims from the first frame
    f0 = ds[int(starts[kept[0]])]
    state_dim = int(np.asarray(f0["observation.state"]).reshape(-1).shape[0])
    action_dim = int(np.asarray(f0["action"]).reshape(-1).shape[0])
    print(f"state_dim={state_dim} action_dim={action_dim} cams={len(CAMS)} img={S}", flush=True)

    images = np.memmap(a.out / "images.dat", np.uint8, "w+", shape=(n_kept, 3, S, S, 3))
    state = np.memmap(a.out / "state.dat", np.float32, "w+", shape=(n_kept, state_dim))
    action = np.memmap(a.out / "action.dat", np.float32, "w+", shape=(n_kept, action_dim))
    reward = np.zeros(n_kept, np.float32)
    done = np.zeros(n_kept, np.int8)
    epidx = np.zeros(n_kept, np.int32)

    w = 0
    for new_e, e in enumerate(kept):
        s, t = int(starts[e]), int(ends[e])
        is_succ = outc.get(e) == "success"
        for i in range(s, t):
            fr = ds[i]
            for c, cam in enumerate(CAMS):
                images[w, c] = to_hwc_uint8(fr[cam], S)
            state[w] = np.asarray(fr["observation.state"], np.float32).reshape(-1)
            action[w] = np.asarray(fr["action"], np.float32).reshape(-1)
            epidx[w] = new_e
            last = i == t - 1
            done[w] = 1 if last else 0
            reward[w] = 1.0 if (last and is_succ) else 0.0  # sparse terminal success; relabel at train time
            w += 1
            if a.limit_frames_test and w >= a.limit_frames_test:
                break
        if a.limit_frames_test and w >= a.limit_frames_test:
            break
        if (new_e + 1) % 5 == 0 or new_e == len(kept) - 1:
            print(f"  episode {new_e + 1}/{len(kept)}  frames {w}/{n_kept}", flush=True)

    images.flush()
    state.flush()
    action.flush()
    np.memmap(a.out / "reward.dat", np.float32, "w+", shape=(w,))[:] = reward[:w]
    np.memmap(a.out / "done.dat", np.int8, "w+", shape=(w,))[:] = done[:w]
    np.memmap(a.out / "episode_index.dat", np.int32, "w+", shape=(w,))[:] = epidx[:w]
    meta = {
        "num_steps": int(w),
        "num_episodes": len(kept),
        "num_success": len(kept) - nfail,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "img_size": S,
        "task": "yam_lego_taxi",
        "source": a.repo_id,
        "fps": 30,
        "shapes": {"images": [int(w), 3, S, S, 3], "state": [int(w), state_dim], "action": [int(w), action_dim]},
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {w} steps -> {a.out}  ({len(kept)} eps, {nfail} fail)", flush=True)


if __name__ == "__main__":
    main()
