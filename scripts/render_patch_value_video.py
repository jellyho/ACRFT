"""Animate the patch-critic's value along one logged rollout episode.

Replays a success (and a failure) episode from a pc_rollouts dir and, at each frame, reads the trained
patch-critic on the state + the executed chunk:
  * the HL-Gauss value DISTRIBUTION p(value | s, executed chunk) at the full commitment horizon
  * the ensemble-min mean value Q(s, chunk)_H (what deployment reads to rank chunks)
  * the per-prefix values Q_h (the adaptive-K signal)
and overlays the behaviour cost-to-goal (mc_return) for reference.

Three panels per frame: camera | value distribution (where the mass sits + per-prefix dots) | the
value-vs-step arc (does it climb toward 0 as the goal nears, or sit flat = collapsed / action-blind?).

    uv run python scripts/render_patch_value_video.py \
        --critic .scratch/patch_critic_cgfloor --data /data5/jellyho/pc_rollouts/OpenDrawer \
        --pick both --out .scratch/patch_value
"""

import argparse
import json
import pathlib

import numpy as np


def episode_slices(ep):
    out = {}
    for e in np.unique(ep):
        out[int(e)] = np.flatnonzero(ep == e)
    return out


def cost_to_goal(idx_len, success, h_goal, discount):
    """mc return-to-go for one episode under the cost_to_goal reward (for the reference line)."""
    r = -np.ones(idx_len, np.float32)
    if success:
        r[-h_goal:] = 0.0
    mc = np.zeros(idx_len, np.float32)
    g = 0.0
    for t in range(idx_len - 1, -1, -1):
        g = r[t] + discount * g
        mc[t] = g
    return mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True, help="a pc_rollouts task dir")
    ap.add_argument("--pick", choices=["success", "fail", "both"], default="both")
    ap.add_argument("--episode", type=int, default=-1, help="explicit episode index (overrides --pick)")
    ap.add_argument("--stride", type=int, default=2, help="score every k-th frame")
    ap.add_argument("--h-goal", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/patch_value"))
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import matplotlib as mpl

    mpl.use("Agg")
    import flax.serialization
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import report_style as rs

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.patch_critic.backbone import to_nchw
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    rs.use()
    cc = json.loads((a.critic / "config.json").read_text())
    H, gsz, atoms, nc = cc["horizon"], cc["macro_group_size"], cc["num_atoms"], cc["num_critics"]
    vmin, vmax, disc = cc["v_min"], cc["v_max"], cc.get("discount", 0.98)
    scheme = cc.get("reward_scheme", "sparse")
    net = PatchCriticEnsemble(action_dim=12, horizon=H, num_critics=nc, macro_group_size=gsz, num_atoms=atoms)
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(vmin, vmax, atoms)
    centers = np.asarray(hl.centers)
    prefixes = list(range(gsz, H + 1, gsz))  # commitment horizons the per-prefix heads score
    bb = DinoV2Backbone(cc["backbone"])
    grid = int(bb.num_patches(224) ** 0.5)
    pooled = grid // 2
    npatch = 3 * pooled * pooled

    def pool(p):
        b, _, d = p.shape
        return p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d).mean((3, 5)).reshape(b, npatch, d)

    @jax.jit
    def patchify(imgs_nchw):  # [B,3,3,224,224] -> [B, npatch, D]
        return pool(bb(imgs_nchw))

    @jax.jit
    def value_heads(patches, chunk, state):  # [B,P,D],[B,H,12],[B,16] -> dist[B,atoms], qpref[B,mh]
        out = net.apply(params, patches, chunk, state)  # [K,B,mh,atoms]
        sm = jax.nn.softmax(out, -1)
        dist_full = jnp.mean(sm[:, :, -1, :], 0)  # ensemble-mean full-prefix distribution
        qpref = jnp.min(jnp.sum(sm * jnp.asarray(centers), -1), 0)  # ensemble-min per-prefix mean
        return dist_full, qpref

    # ---- data ----
    m = json.loads((a.data / "meta.json").read_text())
    n = m["num_steps"]
    images = np.memmap(a.data / "images.dat", np.uint8, "r", shape=(n, 3, 224, 224, 3))
    state = np.asarray(np.memmap(a.data / "state.dat", np.float32, "r", shape=(n, 16)))
    action = np.asarray(np.memmap(a.data / "action.dat", np.float32, "r", shape=(n, 12)))
    reward = np.asarray(np.memmap(a.data / "reward.dat", np.float32, "r", shape=(n,)))
    ep = np.asarray(np.memmap(a.data / "episode_index.dat", np.int32, "r", shape=(n,)))
    sl = episode_slices(ep)
    succ = {e: reward[ix].max() > 0.5 for e, ix in sl.items()}

    if a.episode >= 0:
        picks = [(a.episode, succ.get(a.episode, False))]
    else:
        want = {"success": [True], "fail": [False], "both": [True, False]}[a.pick]
        picks = []
        for w in want:
            cand = [e for e in sl if bool(succ[e]) == w]
            if cand:
                picks.append((min(cand, key=lambda e: len(sl[e])), w))  # shortest such episode (crisp arc)

    a.out.mkdir(parents=True, exist_ok=True)
    task = m.get("task", a.data.name)
    written = []
    for episode, is_succ in picks:
        idx = sl[episode]
        sel = idx[:: a.stride]
        T = len(sel)
        # chunk[t] = action[idx_t : idx_t+H] within the episode (zero-pad past the end)
        chunks = np.zeros((T, H, 12), np.float32)
        for i, t in enumerate(sel):
            loc = t - idx[0]
            k = min(H, len(idx) - loc)
            chunks[i, :k] = action[idx[0] + loc : idx[0] + loc + k]
        # DINOv2 patches for all 3 cams, batched
        dists, qprefs = [], []
        B = 64
        for s in range(0, T, B):
            fr = sel[s : s + B]
            imgs = to_nchw(np.asarray(images[fr]))  # [b,3,3,224,224]
            pat = patchify(jnp.asarray(imgs, jnp.float32))
            d, q = value_heads(pat, jnp.asarray(chunks[s : s + B]), jnp.asarray(state[fr]))
            dists.append(np.asarray(d))
            qprefs.append(np.asarray(q))
        dist = np.concatenate(dists)  # [T, atoms]
        qpref = np.concatenate(qprefs)  # [T, mh]
        qfull = qpref[:, -1]  # deployed full-chunk value
        mc = cost_to_goal(len(idx), bool(is_succ), a.h_goal, disc)[:: a.stride]  # reference

        frames = []
        steps = np.arange(len(idx))[:: a.stride]
        for i in range(T):
            fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
            # (1) camera (cam0)
            ax[0].imshow(np.asarray(images[sel[i], 0]))
            ax[0].axis("off")
            ax[0].set_title(f"{task}  ep{episode} {'SUCCESS' if is_succ else 'fail'}  step {int(steps[i])}", loc="left")
            # (2) value distribution at this frame
            ax[1].bar(centers, dist[i], width=(vmax - vmin) / atoms * 0.9, color=rs.BLUE, alpha=0.85)
            ax[1].axvline(qfull[i], color=rs.ORANGE, lw=2, label=f"mean Q={qfull[i]:.1f}")
            ax[1].axvline(mc[i], color=rs.GRAY, ls=":", lw=1.6, label=f"cost-to-goal={mc[i]:.1f}")
            ax[1].scatter(
                qpref[i],
                np.full(len(prefixes), dist[i].max() * 0.5),
                s=14,
                color=rs.GREEN,
                zorder=5,
                label="per-prefix Q_h",
            )
            ax[1].set_xlim(vmin - 1, vmax + 1)
            ax[1].set_xlabel("value")
            ax[1].set_ylabel("prob")
            ax[1].legend(fontsize=7, loc="upper left")
            ax[1].set_title("value distribution", loc="left")
            # (3) value-vs-step arc
            ax[2].plot(steps[: i + 1], qfull[: i + 1], color=rs.ORANGE, lw=2, label="mean Q")
            ax[2].plot(steps, mc, color=rs.GRAY, ls=":", lw=1.4, label="cost-to-goal")
            ax[2].scatter([steps[i]], [qfull[i]], s=30, color=rs.ORANGE, zorder=5)
            ax[2].set_xlim(steps[0], steps[-1] + 1)
            ax[2].set_ylim(vmin - 1, vmax + 1)
            ax[2].set_xlabel("step")
            ax[2].set_ylabel("value")
            ax[2].legend(fontsize=7, loc="lower right")
            ax[2].set_title("value along the rollout", loc="left")
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(buf[..., :3].copy())
            plt.close(fig)
        outp = a.out / f"{task}_ep{episode}_{'success' if is_succ else 'fail'}_{scheme}.mp4"
        imageio.mimwrite(outp, frames, fps=a.fps, quality=8, macro_block_size=1)
        written.append(outp)
        print(f"wrote {outp}  ({T} frames, value range [{qfull.min():.1f},{qfull.max():.1f}])", flush=True)
    print("VIDEOS:", " ".join(str(p) for p in written), flush=True)


if __name__ == "__main__":
    main()
