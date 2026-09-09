"""Animate the distributional critic's value distribution along one trajectory.

For a chosen episode, at each frame the HL-Gauss head gives p(value | state, executed chunk) over
the value bins. This renders that distribution frame by frame into a video: a single mode means the
state's value is pinned to one outcome (its own trajectory), multiple modes mean the backup pulls
from several outcomes (branching / cross-trajectory). Overlays the running mean and the return the
behaviour policy actually collected (mc_return).

    uv run python scripts/render_value_dist_video.py \
        --critic .scratch/critic_yam_dist101 --annot .scratch/annot_yam_s200 --episode 3
"""

import argparse
import json
import pathlib

import numpy as np


def load_bins(critic_dir, *, action_dim, horizon):
    """Like critic.load_trained but returns the FULL softmax over value bins, not just the mean."""
    import flax.serialization
    import jax

    import openpi.rlt_critic.critic as critic_mod

    cfg = json.loads((critic_dir / "config.json").read_text())
    num_atoms = cfg.get("num_atoms", 1)
    if num_atoms <= 1:
        raise SystemExit("critic is scalar (num_atoms=1) — nothing to visualize; retrain with --num-atoms>1")
    arch = {
        "macro_group_size": cfg.get("macro_group_size", 2),
        "num_layers": cfg.get("num_layers", 3),
        "num_heads": cfg.get("num_heads", 8),
        "head_dim": cfg.get("head_dim", 48),
        "mlp_dim": cfg.get("mlp_dim", 1024),
    }
    net = critic_mod.Ensemble(
        make_critic=lambda: critic_mod.make_critic(
            "arq", action_dim=action_dim, horizon=horizon, num_atoms=num_atoms, **arch
        ),
        num_critics=cfg.get("num_critics", 2),
    )
    hl = critic_mod.HLGauss(v_min=cfg.get("v_min", 0.0), v_max=cfg.get("v_max", 1.0), num_atoms=num_atoms)
    params = flax.serialization.msgpack_restore((critic_dir / "params.msgpack").read_bytes())

    @jax.jit
    def probs(obs, actions):
        out = net.apply(params, obs, actions)  # [K, S, M, P, atoms]
        return jax.nn.softmax(out, axis=-1)

    return probs, np.asarray(hl.centers), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, default=pathlib.Path(".scratch/critic_yam_dist101"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_yam_s200"))
    ap.add_argument("--episode", type=int, default=-1, help="-1 = auto-pick the most multimodal episode")
    ap.add_argument("--stride", type=int, default=3, help="score every k-th frame")
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi", help="LeRobot source dataset for the camera")
    ap.add_argument("--no-camera", action="store_true", help="render the distribution alone")
    ap.add_argument(
        "--candidates",
        action="store_true",
        help="overlay the value distribution of every VLA candidate chunk (spread = do they separate?)",
    )
    ap.add_argument("--zoom", action="store_true", help="magnify the x-axis to where the mass lives per frame")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/value_dist_video.mp4"))
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()

    import sys

    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    sys.path.insert(0, "scripts")
    import report_style

    report_style.use()

    meta = json.loads((a.annot / "meta.json").read_text())
    n, H, A, D = meta["num_frames"], meta["horizon"], meta["action_dim"], meta["token_dim"]
    probs, centers, cfg = load_bins(a.critic, action_dim=A, horizon=H)

    tok = np.memmap(a.annot / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, D))
    chunk = np.memmap(a.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))
    ep = np.asarray(np.memmap(a.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    fr = np.asarray(np.memmap(a.annot / "frame_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    mc = np.asarray(np.memmap(a.annot / "mc_return.dat", dtype=np.float32, mode="r", shape=(n,)))
    use_prop = cfg.get("proprio_mode") == "concat" and (a.annot / "proprio.dat").exists()
    if use_prop:
        pd_ = meta["proprio_dim"]
        prop = np.memmap(a.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, pd_))
        # match training z-scoring: stats over the (capped) training frames
        cap = min(n, 400000)
        mu, sd = prop[:cap].mean(0), prop[:cap].std(0)

    N = meta["num_samples"]
    cand = np.memmap(a.annot / "base_action.dat", dtype=np.float32, mode="r", shape=(n, N, H, A))

    def obs_z(r):
        z = tok[r].astype(np.float32)
        if use_prop:
            p = np.where(sd > 1e-6, (prop[r] - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0).astype(np.float32)
            z = np.concatenate([z, p])
        return z

    def dist_at(r):
        z = obs_z(r)
        # critic expects obs [S, M, D] and actions [S, M, H, A]; here S=M=1
        pr = np.asarray(probs(jnp.asarray(z)[None, None], jnp.asarray(chunk[r])[None, None]))  # [K,1,1,P,atoms]
        return pr.mean(0)[0, 0, -1]  # ensemble-mean, full prefix -> [atoms]

    def dist_cands(r):
        # value distribution of every VLA candidate chunk at this state -> [N, atoms]
        z = obs_z(r)
        zc = jnp.repeat(jnp.asarray(z)[None, None], N, axis=1)  # [1, N, D]
        pr = np.asarray(probs(zc, jnp.asarray(cand[r])[None]))  # [K,1,N,P,atoms]
        return pr.mean(0)[0, :, -1]  # [N, atoms]

    # auto-pick: episode whose frames are most often multimodal (mass outside the dominant mode)
    eps = np.unique(ep[: min(n, 400000)])
    if a.episode < 0:
        best, best_score = eps[0], -1.0
        for e in eps[:: max(1, len(eps) // 40)]:
            rows = np.flatnonzero(ep == e)
            s = [multimodality(dist_at(int(r))) for r in rows[:: max(1, len(rows) // 12)]]
            if np.mean(s) > best_score:
                best, best_score = e, float(np.mean(s))
        a.episode = int(best)
        print(f"auto-picked episode {a.episode} (mean multimodality {best_score:.3f})", flush=True)

    sel = np.flatnonzero(ep == a.episode)[:: a.stride]
    fidx = fr[sel]  # frame index within the source episode
    dists = np.stack([dist_at(int(r)) for r in sel])  # [T, atoms] (demo chunk)
    means = (dists * centers).sum(-1)
    mcs = mc[sel]
    cdists = np.stack([dist_cands(int(r)) for r in sel]) if a.candidates else None  # [T, N, atoms]

    # camera: the source LeRobot episode aligns 1:1 with the annotation (same episode/frame index)
    cam = None
    if not a.no_camera:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        try:
            cam = LeRobotDataset(a.repo_id, episodes=[a.episode])
            print(f"camera loaded: {a.repo_id} ep {a.episode} ({cam.num_frames} frames)", flush=True)
        except Exception as e:
            print(f"camera unavailable ({type(e).__name__}: {e}); rendering distribution only", flush=True)

    def cam_frame(k):
        item = cam[int(fidx[k])]
        img = np.asarray(item["observation.images.agentview"])  # [3,H,W] float [0,1]
        return (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)

    import imageio
    from PIL import Image

    frames = []
    ymax = float(max(dists.max(), cdists.max() if cdists is not None else 0)) * 1.15
    for i, (d, m, mcr) in enumerate(zip(dists, means, mcs, strict=True)):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        if cdists is not None:
            cd = cdists[i]  # [N, atoms]
            cmeans = (cd * centers).sum(-1)
            best = int(np.argmax(cmeans))
            for j in range(cd.shape[0]):  # every candidate's value distribution, faint
                ax.plot(centers, cd[j], color=report_style.PALETTE[0], alpha=0.30, lw=0.9)
            ax.plot(centers, cd[best], color=report_style.PALETTE[1], lw=2.0, label=f"argmax cand #{best}")
            ax.set_title(
                f"per-candidate value   frame {int(fidx[i])}   spread {cmeans.std():.3f}   ({i + 1}/{len(sel)})"
            )
        else:
            ax.fill_between(centers, d, color=report_style.PALETTE[0], alpha=0.85, step="mid")
            ax.set_title(f"value distribution   frame {int(fidx[i])}   ({i + 1}/{len(sel)})")
        ax.axvline(m, color=report_style.PALETTE[3], lw=1.4, label=f"demo mean {m:.2f}")
        ax.axvline(mcr, color="#555", lw=1.2, ls="--", label=f"mc_return {mcr:.2f}")
        if a.zoom:  # magnify the x-window to where the mass actually lives (candidates are tight)
            src = cdists[i] if cdists is not None else d[None]
            active = np.flatnonzero(src.max(0) > src.max() * 0.02)
            lo, hi = centers[active.min()], centers[active.max()]
            pad = max((hi - lo) * 0.35, 0.008)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(0, float(src.max()) * 1.15)
        else:
            ax.set_xlim(cfg.get("v_min", 0.0), cfg.get("v_max", 1.0))
            ax.set_ylim(0, ymax)
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        if cam is not None:
            ph = plot.shape[0]
            c = cam_frame(i)
            cw = int(c.shape[1] * ph / c.shape[0])
            c = np.asarray(Image.fromarray(c).resize((cw, ph), Image.LANCZOS))
            frames.append(np.concatenate([c, plot], axis=1))
        else:
            frames.append(plot)

    imageio.mimwrite(a.out, frames, fps=a.fps, quality=8, macro_block_size=1)
    print(
        f"wrote {a.out}  ({len(frames)} frames, episode {a.episode}, camera={'yes' if cam is not None else 'no'})",
        flush=True,
    )


def multimodality(p):
    """Fraction of probability mass NOT in the dominant contiguous mode (0 = unimodal)."""
    p = np.asarray(p)
    k = int(np.argmax(p))
    lo = hi = k
    while lo > 0 and p[lo - 1] <= p[lo]:
        lo -= 1
    while hi < len(p) - 1 and p[hi + 1] <= p[hi]:
        hi += 1
    return float(1.0 - p[lo : hi + 1].sum())


if __name__ == "__main__":
    main()
