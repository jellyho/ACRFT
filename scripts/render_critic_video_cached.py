"""Animate a patch-critic's value along one logged YAM episode, from the feature cache.

The older renderer (render_patch_value_video.py) reads the RoboCasa pc_rollouts layout and feeds the
critic raw state and absolute actions. A pi05-space critic does not eat those, so it would produce
confident nonsense -- the same failure the scorer had. This one takes the input pipeline from the
critic's own ``input_spec`` (pi05 joint delta + quantile norm, the proprio slice, the clamp-and-hold
chunk convention) and reads features from the cache, decoding the LeRobot video only for the picture.

Three panels per frame: camera | value distribution over the HL-Gauss atoms | the value arc so far,
with the per-prefix values overlaid. What to look for: a SUCCESS should climb toward zero as the goal
nears; a FAILURE should track the success early (the frames are visually alike) and only fall once the
failure becomes visible. A failure pinned at the floor from its first frame is the pathology.

    uv run python scripts/render_critic_video_cached.py \
        --critic .scratch/patch_critic_yam_s347_g5_tau9_mean \
        --cache /data1/jellyho/pc_cache/yam_s347 --out .scratch/critic_video
"""

import argparse
import json
import pathlib

import numpy as np

CAMS = ["observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--outcomes", default=".scratch/yam_outcomes_347.jsonl")
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=pathlib.Path(".scratch/yam_homing_onsets.json"))
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="explicit episode ids; default one of each")
    ap.add_argument("--stride", type=int, default=10, help="frames between rendered video frames")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/critic_video"))
    a = ap.parse_args()

    import os

    os.environ.setdefault("LEROBOT_VIDEO_BACKEND", "pyav")
    import flax.serialization
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import matplotlib as mpl

    mpl.use("Agg")
    from lerobot.datasets import lerobot_dataset
    import matplotlib.pyplot as plt

    from openpi.patch_critic import preproc as critic_preproc
    from openpi.patch_critic import spec as critic_spec
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    cc, _ = critic_spec.load(a.critic)
    isp = cc.get("input_spec", {})
    H, gsz, atoms = cc["horizon"], cc["macro_group_size"], cc["num_atoms"]
    prefixes = list(range(gsz, H + 1, gsz))

    pre = None
    if isp.get("normalization") == "pi05":
        ns = a.critic / isp.get("norm_stats_file", "pi05_norm_stats.json")
        pre = critic_preproc.Pi05Preproc(
            ref=np.asarray(isp["joint_delta_reference"], np.int64),
            stats=critic_preproc.load_norm_stats(ns if ns.exists() else isp["norm_stats"]),
            use_quantiles=bool(isp["use_quantiles"]),
            delta=isp["delta_mode"] == "joint",
        )
    pidx = isp.get("proprio_indices")
    pidx = None if pidx is None else np.asarray(pidx, np.int64)
    print(
        f"critic: {isp.get('normalization', 'raw')} proprio={isp.get('proprio_dims', 'all')} "
        f"steps={cc.get('steps')} prefixes={len(prefixes)}",
        flush=True,
    )

    meta = json.loads((a.cache / "meta.json").read_text())
    N, npatch, emb, sd, ad = meta["N"], meta["npatch"], meta["emb"], meta["sd"], meta["ad"]
    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, npatch, emb))
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, sd))
    actions = np.memmap(a.cache / "action.dat", np.float32, "r", shape=(N, ad))

    outc = {}
    for line in pathlib.Path(a.outcomes).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outc[int(r["episode"])] = r["outcome"]
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets.exists() else {}

    eps = a.episodes
    if not eps:  # one success and one failure, both present in the cache
        s = next(e for e in sorted(outc) if outc[e] == "success" and str(e) in meta["episodes"])
        f = next(e for e in sorted(outc) if outc[e] != "success" and str(e) in meta["episodes"])
        eps = [s, f]
    print(f"episodes: {eps}", flush=True)

    net = PatchCriticEnsemble(
        action_dim=ad, horizon=H, num_critics=cc["num_critics"], macro_group_size=gsz, num_atoms=atoms
    )
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(cc["v_min"], cc["v_max"], atoms)
    centers = jnp.asarray(hl.centers)

    @jax.jit
    def read(p, chunk, st):
        out = net.apply(params, p.astype(jnp.float32), chunk, st)  # [K,B,mh,atoms]
        prob = jax.nn.softmax(out, -1)
        q = jnp.sum(prob * centers, -1)  # [K,B,mh]
        return jnp.mean(prob[:, :, -1, :], 0), jnp.mean(q, 0)  # dist at full H, per-prefix values

    ds = lerobot_dataset.LeRobotDataset(a.repo_id, root=a.root, episodes=eps, video_backend="pyav")
    ep_rows = {
        int(e): (int(f), int(t))
        for e, f, t in zip(
            ds.meta.episodes["episode_index"],
            ds.meta.episodes["dataset_from_index"],
            ds.meta.episodes["dataset_to_index"],
            strict=True,
        )
    }

    a.out.mkdir(parents=True, exist_ok=True)
    for e in eps:
        info = meta["episodes"][str(e)]
        off, full = info["offset"], info["full_len"]
        eff = int(homing[str(e)]["homing_onset"]) if str(e) in homing else full
        succ = outc[e] == "success"
        pos = np.arange(0, eff, a.stride)

        ch = np.asarray(actions[off + np.clip(pos[:, None] + np.arange(H)[None], 0, eff - 1).reshape(-1)])
        ch = ch.reshape(len(pos), H, ad)
        st = np.asarray(states[off + pos])
        if pre is not None:
            ch, st = pre.actions(ch, st), pre.state(st)
        if pidx is not None:
            st = st[..., pidx]
        dist, qpref = [], []
        for i in range(0, len(pos), 256):
            d, q = read(
                jnp.asarray(np.asarray(feats[off + pos[i : i + 256]])),
                jnp.asarray(ch[i : i + 256]),
                jnp.asarray(st[i : i + 256]),
            )
            dist.append(np.asarray(d))
            qpref.append(np.asarray(q))
        dist, qpref = np.concatenate(dist), np.concatenate(qpref)
        vfull = qpref[:, -1]

        row0 = ep_rows[int(e)][0]
        cen = np.asarray(hl.centers)
        frames = []
        for j, p in enumerate(pos):
            sample = ds[row0 + int(p)]
            img = np.asarray(sample[CAMS[0]], np.float32)
            img = (np.clip(np.transpose(img, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)

            fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw={"width_ratios": [1, 1.2, 1.35]})
            ax[0].imshow(img)
            ax[0].axis("off")
            ax[0].set_title(f"ep {e} ({'success' if succ else 'failure'})  frame {p}/{eff}", fontsize=10)

            ax[1].fill_between(cen, dist[j], color="#4c72b0", alpha=0.75, lw=0)
            ax[1].axvline(vfull[j], color="#c44e52", lw=1.4, label=f"mean {vfull[j]:.0f}")
            ax[1].set_xlim(cc["v_min"], cc["v_max"])
            ax[1].set_xlabel("value")
            ax[1].set_ylabel("probability")
            ax[1].set_title("value distribution at k = H", fontsize=10)
            ax[1].legend(frameon=False, fontsize=8)

            ax[2].plot(pos[: j + 1], vfull[: j + 1], color="#4c72b0", lw=1.8, label="Q at k = H")
            for pi_, k in enumerate(prefixes[:-1]):
                ax[2].plot(pos[: j + 1], qpref[: j + 1, pi_], lw=0.8, alpha=0.5, label=f"k={k}" if j == 0 else None)
            ax[2].set_xlim(0, eff)
            ax[2].set_ylim(cc["v_min"] * 1.02, 60)
            ax[2].axhline(0, color="#999", lw=0.8, ls=":")
            ax[2].axhline(cc["v_min"], color="#999", lw=0.8, ls=":")
            ax[2].set_xlabel("episode frame")
            ax[2].set_ylabel("value")
            ax[2].set_title("value along the episode (thin = shorter prefixes)", fontsize=10)
            ax[2].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
            for s_ in ("top", "right"):
                ax[1].spines[s_].set_visible(False)
                ax[2].spines[s_].set_visible(False)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)

        outp = a.out / f"{a.critic.name}_ep{e}_{'success' if succ else 'failure'}.mp4"
        imageio.mimwrite(outp, frames, fps=a.fps, quality=8, macro_block_size=1)
        print(f"wrote {outp}  ({len(frames)} frames, v[first]={vfull[0]:.0f} v[last]={vfull[-1]:.0f})", flush=True)


if __name__ == "__main__":
    main()
