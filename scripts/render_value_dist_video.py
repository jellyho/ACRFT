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
    mc = np.asarray(np.memmap(a.annot / "mc_return.dat", dtype=np.float32, mode="r", shape=(n,)))
    use_prop = cfg.get("proprio_mode") == "concat" and (a.annot / "proprio.dat").exists()
    if use_prop:
        pd_ = meta["proprio_dim"]
        prop = np.memmap(a.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, pd_))
        # match training z-scoring: stats over the (capped) training frames
        cap = min(n, 400000)
        mu, sd = prop[:cap].mean(0), prop[:cap].std(0)

    def dist_at(r):
        z = tok[r].astype(np.float32)
        if use_prop:
            p = np.where(sd > 1e-6, (prop[r] - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0).astype(np.float32)
            z = np.concatenate([z, p])
        pr = np.asarray(probs(jnp.asarray(z)[None], jnp.asarray(chunk[r])[None, None]))  # [K,1,1,P,atoms]
        return pr.mean(0)[0, 0, -1]  # ensemble-mean, full prefix -> [atoms]

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

    rows = np.flatnonzero(ep == a.episode)[:: a.stride]
    dists = np.stack([dist_at(int(r)) for r in rows])  # [T, atoms]
    means = (dists * centers).sum(-1)
    mcs = mc[rows]

    import imageio

    frames = []
    ymax = float(dists.max()) * 1.15
    for i, (d, m, mcr) in enumerate(zip(dists, means, mcs, strict=True)):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.fill_between(centers, d, color=report_style.PALETTE[0], alpha=0.85, step="mid")
        ax.axvline(m, color=report_style.PALETTE[3], lw=1.6, label=f"mean {m:.2f}")
        ax.axvline(mcr, color="#555", lw=1.4, ls="--", label=f"mc_return {mcr:.2f}")
        ax.set_xlim(cfg.get("v_min", 0.0), cfg.get("v_max", 1.0))
        ax.set_ylim(0, ymax)
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.set_title(f"ep {a.episode}   frame {int(rows[i])}   ({i + 1}/{len(rows)})")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    imageio.mimwrite(a.out, frames, fps=a.fps, quality=8, macro_block_size=1)
    print(f"wrote {a.out}  ({len(frames)} frames, episode {a.episode})", flush=True)


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
