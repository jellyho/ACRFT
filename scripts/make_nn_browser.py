"""Cross-episode nearest-neighbour browser: is a frame's neighbour actually the same situation?

Every representation argument in this project so far has rested on retrieval statistics -- kNN
purity, cross-episode action agreement, episode-identity probes -- and those are summaries. This
extracts the underlying pairs as IMAGES so the premise can be checked by looking: pick a frame,
see the frames another episode offers as its nearest match, with a random cross-episode frame in
the same row as the control.

Neighbours are computed on the pooled DINOv2 features the critic's backbone produces
(pc_cache/yam_s347/features_pooled_f32.npy), restricted to OTHER episodes, by cosine similarity.
Two feature variants are extracted so the page can toggle between them: raw, and with each
episode's mean feature subtracted (the per-episode constant is 10.8% of feature variance, so if
episode identity is a fixed background/lighting offset, centring should remove it).

Images are decoded from the LeRobot dataset on the LOGIN node -- compute nodes have no ffmpeg.
"""

# ruff: noqa: PLC0415

import argparse
import base64
import io
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=["observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right"],
        help="views to EXTRACT. Note this does not change the matching: the cached 192 patch tokens already "
        "span all three cameras (64 each), so features_pooled_f32.npy is pooled ACROSS views and the "
        "neighbours were always 3-view. The wrist views are what carry gripper-object relative geometry, "
        "which is the thing a 'same situation' judgement actually rests on.",
    )
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--neighbours", type=int, default=4)
    ap.add_argument("--ref-stride", type=int, default=30, help="subsample the reference pool (1 s at 30 Hz)")
    ap.add_argument("--width", type=int, default=176)
    ap.add_argument("--quality", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k",
        help="score every extracted frame with the DEPLOYED critic, so the page can ask the question the "
        "images only pose: if two frames look like the same situation, does the critic agree? Three reads "
        "per frame -- V(s); Q(s, its own demonstrated chunk); and Q(s, the QUERY's chunk), the last "
        "re-normalized against THIS frame's state so it means 'run the query's motion from here'.",
    )
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/nn_browser.json")
    a = ap.parse_args()

    meta = json.loads((a.cache / "meta.json").read_text())
    N = meta["N"]
    eps = list(meta["episodes"].values())
    epid = np.empty(N, np.int32)
    ep_start = np.empty(N, np.int64)
    ep_len = np.empty(N, np.int64)
    for i, e in enumerate(eps):
        lo, hi = e["offset"], e["offset"] + e["full_len"]
        epid[lo:hi], ep_start[lo:hi], ep_len[lo:hi] = i, lo, e["full_len"]
    succ = np.array([bool(e["success"]) for e in eps])

    patch_f = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, meta["npatch"], meta["emb"]))
    npatch, emb = meta["npatch"], meta["emb"]
    per_cam = npatch // len(meta["cams"])  # cache_patch_features.py pools as [b, 3, g, g, d] -> camera-major
    rng = np.random.default_rng(a.seed)

    ref = np.arange(0, N, a.ref_stride)
    ref = ref[succ[epid[ref]]]  # neighbours drawn from successful episodes only

    # queries: spread over episodes AND over normalized progress, so the page is not all one phase
    qeps = rng.choice(np.flatnonzero(succ), size=min(a.queries, int(succ.sum())), replace=False)
    fracs = np.linspace(0.08, 0.92, len(qeps))
    rng.shuffle(fracs)
    first = {e: int(np.flatnonzero(epid == e)[0]) for e in qeps}
    q = np.array([ep_start[first[e]] + int(f * ep_len[first[e]]) for e, f in zip(qeps, fracs, strict=True)])

    # ---- three retrieval spaces --------------------------------------------------------------
    # The point of this comparison. `pooled` is the mean over all 192 patch tokens, which is what the
    # earlier version of this page searched in -- and mean-pooling is exactly where a small, local
    # fact like "how many blocks are already assembled" goes to die, because it occupies a handful of
    # patches out of 192. `percam` keeps the three cameras separate. `patch` keeps every token in its
    # own position, which is the closest thing to what the critic itself reads: the critic consumes
    # all 192 tokens, never their mean.
    print(
        f"loading patch features for {len(ref)} reference frames ({len(ref) * npatch * emb * 2 / 1e9:.1f} GB)",
        flush=True,
    )
    pr = np.asarray(patch_f[ref], np.float32)
    pq = np.asarray(patch_f[q], np.float32)

    def spaces(x):
        return {
            "pooled": x.mean(1),
            "percam": x.reshape(len(x), len(meta["cams"]), per_cam, emb).mean(2).reshape(len(x), -1),
            "patch": x.reshape(len(x), -1),
        }

    sp_r, sp_q = spaces(pr), spaces(pq)
    del pr, pq

    def topk(qmat, rmat, qep):
        qn = qmat / (np.linalg.norm(qmat, axis=1, keepdims=True) + 1e-9)
        rn = rmat / (np.linalg.norm(rmat, axis=1, keepdims=True) + 1e-9)
        sim = qn @ rn.T
        sim[epid[ref][None, :] == qep[:, None]] = -9  # cross-episode only
        order = np.argsort(-sim, axis=1)[:, : a.neighbours]
        return ref[order], np.take_along_axis(sim, order, 1)

    SPACES = ("pooled", "percam", "patch")
    nn, cs = {}, {}
    for k in SPACES:
        nn[k], cs[k] = topk(sp_q[k], sp_r[k], epid[q])
        print(f"  {k:7s} dim {sp_q[k].shape[1]:6d}  median cos {np.median(cs[k]):.4f}", flush=True)
    ctrl = np.array([rng.choice(ref[epid[ref] != epid[i]]) for i in q])

    need = sorted({int(x) for x in np.concatenate([q, *[nn[k].ravel() for k in SPACES], ctrl])})
    print(f"{len(q)} queries, {len(need)} frames to decode", flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    ds = LeRobotDataset(a.repo_id)
    if ds.num_frames != N:
        raise SystemExit(f"index contract broken: dataset {ds.num_frames} vs cache {N}")
    imgs = {}
    for n, g in enumerate(need):
        item = ds[g]
        views = []
        for cam in a.cameras:
            arr = (np.asarray(item[cam]).transpose(1, 2, 0) * 255).astype(np.uint8)
            im = Image.fromarray(arr)
            im = im.resize((a.width, int(a.width * im.height / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=a.quality)
            views.append(base64.b64encode(buf.getvalue()).decode())
        imgs[g] = views
        if n % 25 == 0:
            print(f"  {n}/{len(need)}", flush=True)

    # ---- score every extracted frame with the DEPLOYED critic ---------------------------------
    # The images only pose the question; this answers it. If two frames look like the same situation,
    # does the critic agree? Q(s, the QUERY's chunk) is the sharpest of the three: the query's raw
    # joint targets re-normalized against THIS frame's state, i.e. "run the query's motion from here".
    import jax
    import jax.numpy as jnp

    from openpi.extraction import critic_q as cq

    critic = cq.load(a.critic)
    view = cq.CacheView(a.cache)
    hc, adc = critic.config["horizon"], critic.config["action_dim"]
    ep_last = ep_start + ep_len - 1
    rows = np.array(need, np.int64)
    f_all, raw_all, prop_all = view.rows(rows, critic)
    gch = np.clip(rows[:, None] + np.arange(hc)[None], 0, ep_last[rows][:, None])
    raw_chunk = np.asarray(view.actions[gch.reshape(-1)]).reshape(len(rows), hc, adc)
    own = critic.pre.actions(raw_chunk, raw_all)[..., :adc]
    vf, qf = jax.jit(critic.v), jax.jit(critic.q_mean)
    fj, pj = jnp.asarray(f_all), jnp.asarray(prop_all)
    val_v = np.asarray(vf(fj, pj))
    val_qown = np.asarray(qf(fj, jnp.asarray(own), pj))
    pos_of = {int(g): k for k, g in enumerate(rows)}
    q_xfer = {}
    for i in range(len(q)):
        qi = int(q[i])
        tgt = [qi] + [int(x) for k in SPACES for x in nn[k][i]] + [int(ctrl[i])]
        ks = np.array([pos_of[t] for t in tgt])
        ch = critic.pre.actions(np.repeat(raw_chunk[pos_of[qi]][None], len(ks), 0), raw_all[ks])[..., :adc]
        vals = np.asarray(qf(fj[ks], jnp.asarray(ch), pj[ks]))
        for t, v in zip(tgt, vals, strict=True):
            q_xfer[(qi, t)] = float(v)
    print(f"critic {a.critic.name}: scored {len(rows)} frames", flush=True)

    def info(g, qi=None):
        i = int(g)
        k = pos_of[i]
        d = {
            "g": i,
            "ep": int(epid[i]),
            "t": int(i - ep_start[i]),
            "T": int(ep_len[i]),
            "prog": round(float((i - ep_start[i]) / ep_len[i]), 3),
            "ttg": int(ep_last[i] - i),
            "V": round(float(val_v[k]), 1),
            "Qown": round(float(val_qown[k]), 1),
        }
        if qi is not None:
            d["Qxfer"] = round(q_xfer[(int(qi), i)], 1)
        return d

    out = {
        "cameras": [c.split(".")[-1] for c in a.cameras],
        "critic": a.critic.name,
        "spaces": list(SPACES),
        "ref_stride": a.ref_stride,
        "n_ref": len(ref),
        "queries": [
            {
                "q": info(q[i], q[i]),
                **{
                    k: [info(nn[k][i, j], q[i]) | {"cos": round(float(cs[k][i, j]), 4)} for j in range(a.neighbours)]
                    for k in SPACES
                },
                "control": info(ctrl[i], q[i]),
            }
            for i in range(len(q))
        ],
        "images": imgs,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    print(f"wrote {a.out}  ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
