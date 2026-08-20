"""Animate a cached-feature patch-critic's value along YAM episodes (values from the cache, frames
from the LeRobot videos).

The yam_s347 critics were trained on the feature CACHE (pooled frozen DINOv2 patches, pi05-normalized
state/action -- see the checkpoint's input_spec), which stores no images. So this renderer splits the
two sources: per-frame values come straight from the cache memmaps (no DINOv2, no GPU needed), and
the camera frames for the SELECTED episodes only are decoded from the LeRobot dataset with pyav
(login node -- compute nodes have no ffmpeg).

Panels per frame, matching render_patch_value_video.py: camera | HL-Gauss value distribution with the
per-prefix Q_h dots (the adaptive-K signal) | the value-vs-step arc against the analytic cost-to-goal.

    uv run python scripts/render_yam_value_video.py \
        --critic .scratch/patch_critic_yam_s347_g5_pi05 --cache /data1/jellyho/pc_cache/yam_s347 \
        --pick both --out .scratch/yam_value_videos
"""

import argparse
import json
import pathlib

import numpy as np


def cost_to_goal(n, success, h_goal, discount):
    r = -np.ones(n, np.float32)
    if success:
        r[-h_goal:] = 0.0
    mc = np.zeros(n, np.float32)
    g = 0.0
    for t in range(n - 1, -1, -1):
        g = r[t] + discount * g
        mc[t] = g
    return mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--cam", default="observation.images.agentview")
    ap.add_argument("--pick", choices=["success", "fail", "both"], default="both")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="explicit episode ids (override --pick)")
    ap.add_argument("--stride", type=int, default=8, help="score/draw every k-th frame (YAM eps are ~2-4k frames)")
    ap.add_argument("--h-goal", type=int, default=3)
    ap.add_argument(
        "--homing-onsets",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json"),
        help="trim rendering at each episode's homing onset (the return-home tail is not task "
        "behavior and the newer critic arms do not even train on it); missing file disables",
    )
    ap.add_argument("--no-trim-homing", action="store_true", help="render the full episode incl. homing")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/yam_value_videos"))
    a = ap.parse_args()

    import flax.serialization
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import report_style as rs

    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    rs.use()
    cc = json.loads((a.critic / "config.json").read_text())
    H, gsz, atoms, nc = cc["horizon"], cc["macro_group_size"], cc["num_atoms"], cc["num_critics"]
    vmin, vmax, disc = cc["v_min"], cc["v_max"], cc.get("discount", 0.99964)
    m = json.loads((a.cache / "meta.json").read_text())
    npatch, emb, sd, ad, n_total = m["npatch"], m["emb"], m["sd"], m["ad"], m["N"]
    eps = {int(k): v for k, v in m["episodes"].items()}

    net = PatchCriticEnsemble(action_dim=ad, horizon=H, num_critics=nc, macro_group_size=gsz, num_atoms=atoms)
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(vmin, vmax, atoms)
    centers = np.asarray(hl.centers)
    prefixes = list(range(gsz, H + 1, gsz))

    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(n_total, npatch, emb))
    state = np.asarray(np.memmap(a.cache / "state.dat", np.float32, "r", shape=(n_total, sd)))
    action = np.asarray(np.memmap(a.cache / "action.dat", np.float32, "r", shape=(n_total, ad)))

    @jax.jit
    def value_heads(patches, chunk, st):
        out = net.apply(params, patches, chunk, st)  # [K,B,mh,atoms]
        sm = jax.nn.softmax(out, -1)
        dist_full = jnp.mean(sm[:, :, -1, :], 0)
        qpref = jnp.min(jnp.sum(sm * jnp.asarray(centers), -1), 0)
        return dist_full, qpref

    if a.episodes:
        picks = [(e, bool(eps[e]["success"])) for e in a.episodes]
    else:
        want = {"success": [True], "fail": [False], "both": [True, False]}[a.pick]
        picks = []
        for w in want:
            cand = [e for e, v in eps.items() if bool(v["success"]) == w]
            # median length of its class: short failures are instant aborts, long successes drag
            cand.sort(key=lambda e: eps[e]["full_len"])
            if cand:
                picks.append((cand[len(cand) // 2], w))

    # ---- frames for the picked episodes only (pyav via LeRobot, login node) ----
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    a.out.mkdir(parents=True, exist_ok=True)
    written = []
    onsets = {}
    if not a.no_trim_homing and a.homing_onsets.exists():
        onsets = {int(k): v["homing_onset"] for k, v in json.loads(a.homing_onsets.read_text()).items()}

    for episode, is_succ in picks:
        off, n = eps[episode]["offset"], eps[episode]["full_len"]
        # render only up to the homing onset: the tail is the arms returning home, not the task.
        # values/mc keep using the FULL episode geometry (that is what training saw); we just stop
        # drawing where the task ends.
        n_draw = min(n, onsets.get(episode, n)) if onsets else n
        sel = np.arange(0, n_draw, a.stride)
        T = len(sel)
        chunks = np.zeros((T, H, ad), np.float32)
        for i, t in enumerate(sel):
            k = min(H, n - t)
            chunks[i, :k] = action[off + t : off + t + k]
        dists, qprefs = [], []
        B = 256
        for s in range(0, T, B):
            fr = off + sel[s : s + B]
            d, q = value_heads(
                jnp.asarray(np.asarray(feats[fr]), jnp.float32),
                jnp.asarray(chunks[s : s + B]),
                jnp.asarray(state[fr]),
            )
            dists.append(np.asarray(d))
            qprefs.append(np.asarray(q))
        dist = np.concatenate(dists)
        qpref = np.concatenate(qprefs)
        qfull = qpref[:, -1]
        mc = cost_to_goal(n, is_succ, a.h_goal, disc)[: n_draw : a.stride]

        # with episodes=[e] the dataset is exactly that episode, indexed from 0
        ds = lerobot_dataset.LeRobotDataset(a.repo_id, root=a.root, episodes=[episode], video_backend="pyav")
        assert len(ds) >= n_draw, f"dataset has {len(ds)} frames, need {n_draw}"
        frames_np = []
        for i in range(T):
            img = ds[int(sel[i])][a.cam]  # [3,h,w] float tensor 0..1
            frames_np.append((np.asarray(img).transpose(1, 2, 0) * 255).astype(np.uint8))

        frames = []
        for i in range(T):
            fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
            ax[0].imshow(frames_np[i])
            ax[0].axis("off")
            ax[0].set_title(
                f"yam ep{episode} {'SUCCESS' if is_succ else 'FAILURE'}  frame {int(sel[i])}/{n}", loc="left"
            )
            ax[1].bar(centers, dist[i], width=(vmax - vmin) / atoms * 0.9, color=rs.BLUE, alpha=0.85)
            ax[1].axvline(qfull[i], color=rs.ORANGE, lw=2, label=f"mean Q={qfull[i]:.0f}")
            ax[1].axvline(mc[i], color=rs.GRAY, ls=":", lw=1.6, label=f"cost-to-goal={mc[i]:.0f}")
            ax[1].scatter(
                qpref[i],
                np.full(len(prefixes), dist[i].max() * 0.5),
                s=14,
                color=rs.GREEN,
                zorder=5,
                label="per-prefix Q_h",
            )
            ax[1].set_xlim(vmin * 1.02, vmax + abs(vmin) * 0.02)
            ax[1].set_xlabel("value")
            ax[1].set_ylabel("prob")
            ax[1].legend(fontsize=7, loc="upper left")
            ax[1].set_title("value distribution", loc="left")
            ax[2].plot(sel[: i + 1], qfull[: i + 1], color=rs.ORANGE, lw=2, label="mean Q")
            ax[2].plot(sel, mc, color=rs.GRAY, ls=":", lw=1.4, label="cost-to-goal")
            ax[2].scatter([sel[i]], [qfull[i]], s=30, color=rs.ORANGE, zorder=5)
            ax[2].set_xlim(0, n_draw)
            lo = min(qfull.min(), mc.min())
            ax[2].set_ylim(lo * 1.05, vmax + abs(lo) * 0.05)
            ax[2].set_xlabel("frame")
            ax[2].set_ylabel("value")
            ax[2].legend(fontsize=7, loc="lower right")
            ax[2].set_title("value along the episode", loc="left")
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(buf[..., :3].copy())
            plt.close(fig)
        outp = a.out / f"yam_{a.critic.name}_ep{episode}_{'success' if is_succ else 'fail'}.mp4"
        imageio.mimwrite(outp, frames, fps=a.fps, quality=8, macro_block_size=1)
        written.append(outp)
        print(f"wrote {outp}  ({T} frames, Q range [{qfull.min():.0f},{qfull.max():.0f}])", flush=True)
    print("VIDEOS:", " ".join(str(p) for p in written), flush=True)


if __name__ == "__main__":
    main()
