"""Direct LeRobot -> patch-critic PATCHES converter (skips images.dat entirely).

The normal path writes a raw images.dat (300+ GB for the full yam set), then train_patch_critic.py
runs frozen DINOv2 over it once and caches the pooled patches. DINOv2 is frozen+deterministic, so we
can fold that step INTO conversion: decode each LeRobot frame, run DINOv2 on the 3 cameras right away,
and stream only the pooled patches to disk. Result: no 300 GB images.dat, ~138 GB of patches instead,
and the trainer loads them via --prebuilt-patches (no recompute).

Writes, into --out:  patches_<backbone>_p<npatch>_n<N>.npy (float16 [N, npatch, D]),
                     state.dat action.dat reward.dat done.dat episode_index.dat, meta.json.

    uv run python scripts/convert_yam_to_patches.py \
        --repo-id jellyho/yam_lego_taxi --root /data5/jellyho/yam_v2/lerobot \
        --outcomes <fresh outcomes.jsonl> --backbone small \
        --max-frames 10000000 --out /data5/jellyho/pc_rollouts_yam/lego_taxi_s300_patches
"""

import argparse
import json
import pathlib
import time

import numpy as np

CAMS = ["observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right"]


def to_hwc_uint8(t, size):
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
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--max-frames", type=int, default=10_000_000)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--backbone", default="small")
    ap.add_argument("--batch", type=int, default=64, help="frames per DINOv2 forward")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit-frames-test", type=int, default=0)
    a = ap.parse_args()

    import jax.numpy as jnp
    import lerobot.datasets.lerobot_dataset as lrd

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.patch_critic.backbone import to_nchw

    # --- backbone + pooling, IDENTICAL to train_patch_critic.py's precompute ---
    bb = DinoV2Backbone(a.backbone)
    grid = int(bb.num_patches(224) ** 0.5)  # 16
    pooled = grid // 2  # 8
    npatch = 3 * pooled * pooled  # 192

    def pool(p):  # [b, 3cam*256, D] -> [b, 3*64, D]
        b, _, d = p.shape
        p = p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d)
        return p.mean((3, 5)).reshape(b, npatch, d)

    ds = lrd.LeRobotDataset(a.repo_id, root=a.root, tolerance_s=0.05)
    epds = ds.meta.episodes
    starts = {int(e): int(f) for e, f in zip(epds["episode_index"], epds["dataset_from_index"], strict=True)}
    ends = {int(e): int(t) for e, t in zip(epds["episode_index"], epds["dataset_to_index"], strict=True)}
    print(f"dataset {a.repo_id}: {ds.num_frames} frames, {ds.num_episodes} episodes", flush=True)

    outc = {}
    for line in pathlib.Path(a.outcomes).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outc[int(r["episode"])] = r["outcome"]
    labeled = [e for e in outc if e in starts]
    fails = [e for e in labeled if outc[e] != "success"]
    succ = [e for e in labeled if outc[e] == "success"]
    print(f"labeled episodes: {len(labeled)} ({len(fails)} fail / {len(succ)} success)", flush=True)
    # fails first (scarce negatives), then successes, whole episodes, until the frame budget is hit
    order, kept, tot = fails + succ, [], 0
    for e in order:
        kept.append(e)
        tot += int(ends[e] - starts[e])
        if tot >= a.max_frames:
            break
    n_kept = sum(int(ends[e] - starts[e]) for e in kept)
    nfail = sum(1 for e in kept if e in set(fails))
    if a.limit_frames_test:
        n_kept = min(n_kept, a.limit_frames_test)
    print(f"keeping {len(kept)} episodes ({nfail} fail / {len(kept) - nfail} success) = {n_kept} frames", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    S = a.img_size
    f0 = ds[int(starts[kept[0]])]
    state_dim = int(np.asarray(f0["observation.state"]).reshape(-1).shape[0])
    action_dim = int(np.asarray(f0["action"]).reshape(-1).shape[0])
    print(f"state_dim={state_dim} action_dim={action_dim} npatch={npatch} embed={bb.embed_dim}", flush=True)

    cache = a.out / f"patches_{a.backbone}_p{npatch}_n{n_kept}.npy"
    patches = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16, shape=(n_kept, npatch, bb.embed_dim))
    state = np.memmap(a.out / "state.dat", np.float32, "w+", shape=(n_kept, state_dim))
    action = np.memmap(a.out / "action.dat", np.float32, "w+", shape=(n_kept, action_dim))
    reward = np.zeros(n_kept, np.float32)
    done = np.zeros(n_kept, np.int8)
    epidx = np.zeros(n_kept, np.int32)

    buf_hwc, buf_w = [], []  # pending frames [3cam,S,S,3] uint8 and their write indices
    prev_cams = None
    w = 0
    t0 = time.time()

    def flush_batch():
        nonlocal buf_hwc, buf_w
        if not buf_hwc:
            return
        arr = np.stack(buf_hwc)  # [b, 3cam, S, S, 3] uint8
        imgs = jnp.asarray(to_nchw(arr))  # [b, 3cam, 3, S, S] float32 in [0,1]
        out = np.asarray(pool(bb(imgs)), np.float16)  # [b, npatch, D]
        for k, wi in enumerate(buf_w):
            patches[wi] = out[k]
        buf_hwc, buf_w = [], []

    done_flag = False
    for new_e, e in enumerate(kept):
        if done_flag:
            break
        s, t = int(starts[e]), int(ends[e])
        is_succ = outc.get(e) == "success"
        for i in range(s, t):
            try:
                fr = ds[i]
                cams = np.stack([to_hwc_uint8(fr[cam], S) for cam in CAMS])  # [3cam,S,S,3]
                st = np.asarray(fr["observation.state"], np.float32).reshape(-1)
                ac = np.asarray(fr["action"], np.float32).reshape(-1)
                prev_cams = (cams, st, ac)
            except Exception as ex:
                if prev_cams is None:
                    continue
                print(f"  skip frame {i} (reuse prev): {type(ex).__name__}", flush=True)
                cams, st, ac = prev_cams
            buf_hwc.append(cams)
            buf_w.append(w)
            state[w] = st
            action[w] = ac
            epidx[w] = new_e
            last = i == t - 1
            done[w] = 1 if last else 0
            reward[w] = 1.0 if (last and is_succ) else 0.0
            w += 1
            if len(buf_hwc) >= a.batch:
                flush_batch()
            if a.limit_frames_test and w >= a.limit_frames_test:
                done_flag = True
                break
        if (new_e + 1) % 5 == 0 or new_e == len(kept) - 1:
            fps = w / max(1e-6, time.time() - t0)
            print(f"  episode {new_e + 1}/{len(kept)}  frames {w}/{n_kept}  ({fps:.1f} frame/s)", flush=True)
    flush_batch()

    patches.flush()
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
        "patches_file": cache.name,
        "backbone": a.backbone,
        "npatch": npatch,
        "no_images": True,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    dt = time.time() - t0
    print(
        f"wrote {w} steps -> {a.out}  ({len(kept)} eps, {nfail} fail) in {dt / 60:.1f} min ({w / dt:.1f} frame/s)",
        flush=True,
    )
    print(f"patches: {cache.name}  ({patches.nbytes / 1e9:.1f} GB, no images.dat)", flush=True)


if __name__ == "__main__":
    main()
