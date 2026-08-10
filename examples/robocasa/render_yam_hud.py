"""Render the RoboCasa-style critic HUD for YAM, from real critic decisions.

The HUD renderer (hud.Dashboard) is camera-agnostic: it composes an agent frame, a wrist frame,
the critic's candidate decision grid (Q[cand, commit-steps] with the chosen cell outlined), and
the value/spread/commit trace. This script drives it with a YAM critic scoring the stored 16
candidates per frame, so the panels are the real decision surface. The camera panels take live
robot frames at deploy time; offline (no aligned video for this annotation snapshot) they fall
back to a labeled placeholder so the layout and decision panels can be reviewed.

    uv run python examples/robocasa/render_yam_hud.py --episode 0 --replans 12
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hud


class _Info:
    __slots__ = ("best_cand", "best_prefix", "cand", "n_exec", "q", "value")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_yam_s200"))
    ap.add_argument("--critic", type=pathlib.Path, default=pathlib.Path(".scratch/critic_yam_s200_iql"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--replans", type=int, default=12, help="number of BoN replans to render")
    ap.add_argument("--mode", default="bon", choices=["bon", "full"])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/yam_hud_demo.mp4"))
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    import jax.numpy as jnp

    import openpi.rlt_critic.critic as critic_mod

    meta = json.loads((a.annot / "meta.json").read_text())
    n, H, A, N, D = (meta["num_frames"], meta["horizon"], meta["action_dim"], meta["num_samples"], meta["token_dim"])
    pdim = meta["proprio_dim"]
    score, _params, macro = critic_mod.load_trained(a.critic / "params.msgpack", action_dim=A, horizon=H)
    st = json.loads((a.critic / "proprio_stats.json").read_text())
    mu, sd = np.asarray(st["mean"], np.float32), np.asarray(st["std"], np.float32)

    tok = np.memmap(a.annot / "rl_token.dat", dtype=np.float32, mode="r", shape=(n, D))
    cand = np.memmap(a.annot / "base_action.dat", dtype=np.float32, mode="r", shape=(n, N, H, A))
    prop = np.memmap(a.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, pdim))
    ep = np.memmap(a.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,))
    rows = np.flatnonzero(np.asarray(ep) == a.episode)
    if not len(rows):
        raise SystemExit(f"episode {a.episode} not found")

    def score_row(r):
        z = np.concatenate(
            [
                tok[r].astype(np.float32),
                np.where(sd > 1e-6, (prop[r] - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0).astype(np.float32),
            ]
        )[None]
        zc = jnp.repeat(jnp.asarray(z)[:, None], N, axis=1)
        q = np.asarray(score(zc, jnp.asarray(cand[r])[None]))
        return np.min(q, axis=0)[0]  # [N, P]

    dash = hud.Dashboard(mode=a.mode, horizon=H, discount=meta["discount"], camera_size=256)
    # placeholder camera panels — replaced by live robot frames (agentview / wrist) at deploy
    ph = np.full((480, 640, 3), 24, np.uint8)
    ph[::40] = 40
    from PIL import Image
    from PIL import ImageDraw

    def placeholder(label):
        im = Image.fromarray(ph.copy())
        ImageDraw.Draw(im).text((14, 14), label, fill=(150, 175, 200))
        return np.asarray(im)

    agent_ph = placeholder("agentview — live at deploy")
    wrist_ph = placeholder("wrist")

    import imageio

    frames = []
    step = 0
    n_macro = H  # BoN commits the full chunk each replan
    for k in range(min(a.replans, len(rows) // n_macro)):
        r = int(rows[k * n_macro])
        q = score_row(r)
        best = int(np.argmax(q[:, -1]))
        info = _Info()
        info.q, info.cand, info.best_cand = q, np.asarray(cand[r]), best
        info.best_prefix, info.n_exec, info.value = q.shape[1] - 1, H, float(q[best, -1])
        # hold each decision on screen for a few frames so the video is readable
        for _ in range(max(1, a.fps // 3)):
            frames.append(np.asarray(dash.frame(agent_ph, wrist_ph, info, step)))
            step += 1

    imageio.mimwrite(a.out, frames, fps=a.fps, quality=8, macro_block_size=1)
    print(f"wrote {a.out}  ({len(frames)} frames, {a.replans} replans, spread~{float(q.max() - q.min()):.4f})")


if __name__ == "__main__":
    main()
