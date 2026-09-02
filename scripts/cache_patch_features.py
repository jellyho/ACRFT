"""Precompute FROZEN DINOv2 pooled patch features for every frame of the yam dataset, once.

The patch backbone is frozen, so its per-frame output never changes during critic training -- yet the
clip trainer re-runs DINOv2 on ~1440 images per step, which is the throughput wall (~0.68 it/s, GPU
bound). Caching the pooled tokens turns critic training into a tiny-transformer-only loop that reads
features from a memmap (train_patch_critic_cached.py), ~20-40x faster, with the SAME model and inputs.

Layout (self-contained bundle at --out, one memmap each, frames laid out per-episode contiguously):
  features.dat  [N, npatch, emb]  float16   pooled DINOv2 tokens (3 cams x pooled x pooled)
  state.dat     [N, sd]           float32
  action.dat    [N, ad]           float32   per-frame action (chunk = action[g:g+H] at train time)
  meta.json     per-episode {offset, full_len, success}, npatch, emb, sd, ad, backbone, cams, img_size

FULL episodes are cached (no homing truncation): truncation is a train-time mask over eff_len, so the
same cache serves truncated and untruncated experiments. Cache lives on /data1 (NFS, ~2T free); node
RAM page-cache keeps it hot after the first epoch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import openpi.training.outcomes as _outcomes

CAMS = ["observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument(
        "--outcomes",
        default=None,
        help="legacy outcomes.jsonl (deprecated: the verdict is read from the dataset's next.success / next.done)",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True, help="cache dir (put on /data1)")
    ap.add_argument("--backbone", default="small")
    ap.add_argument("--dino-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--clip-len", type=int, default=100, help="tile size for decode (non-overlapping)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--encode-batch", type=int, default=4, help="clips per DINOv2 forward")
    a = ap.parse_args()

    import os

    os.environ.setdefault("LEROBOT_VIDEO_BACKEND", "pyav")
    import jax
    import jax.numpy as jnp
    import torch

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.training.critic_data_loader import CriticClipDataset

    outc = _outcomes.load_outcomes(_outcomes.dataset_root(a.repo_id, a.root), legacy_jsonl=a.outcomes)
    episodes = list(outc)

    # Cache FULL episodes: stride == clip_len (non-overlapping tiles), homing_onsets=None (no truncation).
    ds = CriticClipDataset(
        a.repo_id,
        root=a.root,
        episodes=episodes,
        image_keys=CAMS,
        horizon=a.horizon,
        clip_len=a.clip_len,
        stride=a.clip_len,
        outcomes=outc,
        img_size=a.img_size,
        homing_onsets=None,
    )
    # Per-episode FULL lengths + contiguous offsets into the cache.
    epds = ds.ds.meta.episodes
    starts = {int(e): int(f) for e, f in zip(epds["episode_index"], epds["dataset_from_index"], strict=True)}
    ends = {int(e): int(t) for e, t in zip(epds["episode_index"], epds["dataset_to_index"], strict=True)}
    full_len = {int(e): ends[int(e)] - starts[int(e)] for e in episodes}
    offset, N = {}, 0
    for e in episodes:
        offset[e] = N
        N += full_len[e]
    print(f"{len(episodes)} episodes, N={N} frames", flush=True)

    bb = DinoV2Backbone(a.backbone, dtype=getattr(jnp, a.dino_dtype))
    grid = int(bb.num_patches(a.img_size) ** 0.5)
    pooled = grid // 2
    npatch = 3 * pooled * pooled
    emb = bb.embed_dim
    ncam = len(CAMS)

    def pool(p):
        b, _, d = p.shape
        p = p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d)
        return p.mean((3, 5)).reshape(b, npatch, d)

    @jax.jit
    def encode(imgs_u8):  # [M, ncam, S, S, 3] u8 -> [M, npatch, emb] f16
        x = imgs_u8.astype(jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 1, 4, 2, 3))
        return pool(bb(x)).astype(jnp.float16)

    sd = int(np.asarray(ds[0]["state"]).shape[1])
    ad = int(np.asarray(ds[0]["action"]).shape[1])
    a.out.mkdir(parents=True, exist_ok=True)
    feats = np.memmap(a.out / "features.dat", np.float16, "w+", shape=(N, npatch, emb))
    states = np.memmap(a.out / "state.dat", np.float32, "w+", shape=(N, sd))
    actions = np.memmap(a.out / "action.dat", np.float32, "w+", shape=(N, ad))
    written = np.zeros(N, bool)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=a.encode_batch,
        num_workers=a.num_workers,
        shuffle=False,
        collate_fn=CriticClipDataset.collate,
        drop_last=False,
        persistent_workers=a.num_workers > 0,
        prefetch_factor=2 if a.num_workers > 0 else None,
    )
    t0 = time.time()
    done_frames = 0
    for bi, clip in enumerate(dl):
        imgs = clip["images"].numpy()  # [B, cl, ncam, S, S, 3] u8
        B, cl = imgs.shape[0], imgs.shape[1]
        patches = np.asarray(encode(jnp.asarray(imgs.reshape(B * cl, ncam, a.img_size, a.img_size, 3))))
        patches = patches.reshape(B, cl, npatch, emb)
        state = clip["state"].numpy()
        action = clip["action"].numpy()  # [B, cl+H, ad] -> per-frame action is action[:, :cl]
        pad = clip["img_pad"].numpy()  # [B, cl]
        pos0 = clip["clip_pos0"].numpy()  # [B] frame-in-episode of clip start
        # clip carries only ep_id via is_success/outcomes? recover ep via clip_starts order is fragile;
        # instead map through global frame: gstart = clip_starts[idx]. Use the pos0 + episode lookup.
        for b in range(B):
            # shuffle=False + drop_last=False -> global clip index is sequential; recover its episode.
            ep = ds.clip_starts[bi * a.encode_batch + b][1]
            p0 = int(pos0[b])
            good = np.flatnonzero(~pad[b])
            if len(good) == 0:
                continue
            gidx = offset[ep] + p0 + good
            feats[gidx] = patches[b, good]
            states[gidx] = state[b, good]
            actions[gidx] = action[b, good]
            written[gidx] = True
            done_frames += len(good)
        if bi % 50 == 0:
            rate = done_frames / (time.time() - t0 + 1e-9)
            print(f"  batch {bi} frames {done_frames}/{N} ({rate:.0f} f/s)", flush=True)
    feats.flush()
    states.flush()
    actions.flush()
    meta = {
        "N": N,
        "npatch": npatch,
        "emb": emb,
        "sd": sd,
        "ad": ad,
        "backbone": a.backbone,
        "cams": CAMS,
        "img_size": a.img_size,
        "horizon": a.horizon,
        "episodes": {
            str(e): {
                "offset": int(offset[e]),
                "full_len": int(full_len[e]),
                "success": outc[e] == "success",
                "outcome": outc[e],  # success / fail / unknown -- "not success" is not always "fail"
            }
            for e in episodes
        },
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    miss = int((~written).sum())
    print(f"DONE N={N} written_missing={miss}  ({done_frames} frames, {time.time()-t0:.0f}s)", flush=True)
    if miss:
        print(f"WARNING: {miss} frames never written (tail padding gaps?) -- check", flush=True)


if __name__ == "__main__":
    main()
