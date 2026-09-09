"""Video clips of the sustained short-lag-dominant regions (login node: pyav decode).

Regions: within the 30-frame chunk window, SHORT lags {1,3,5,10} beat LONG lags {15,20,30} by
>15%p (per-frame, smoothed) for at least --min-sec seconds. For each region of the chosen
episodes, decode the agentview frames (with +-0.7s context margin) and write one mp4 with a
minimal overlay: episode, real frame counter, and a live short-vs-long gap readout.
"""

# ruff: noqa: E402, ICN001, PLC0415

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "slurm"))

D = R / ".scratch/nonmarkov_yam_lags"
EXTRAS = [(R / ".scratch/nonmarkov_yam_lags2", (10, 20)), (R / ".scratch/nonmarkov_yam_lags3", (2, 4, 6, 8, 12))]
KMAX, STRIDE, FPS = 150, 2, 30
SHORT, LONG = (1, 3, 5, 10), (15, 20, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[81, 159, 185, 204])
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--margin-sec", type=float, default=0.7)
    ap.add_argument("--gap-thresh", type=float, default=15.0)
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--cam", default="observation.images.agentview")
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/nm_region_clips")
    a = ap.parse_args()

    import imageio
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    import matplotlib.pyplot as plt

    idx = np.load(D / "perframe_idx.npy")
    ERR = {0: np.load(D / "perframe_err_k0.npy")}
    for k in (1, 3, 5, 15, 30, 60, 150):
        ERR[k] = np.load(D / f"perframe_err_k{k}.npy")
    for extra, ks in EXTRAS:
        f = extra / "perframe_idx.npy"
        if not f.exists():
            continue
        idx2 = np.load(f)
        pos = np.searchsorted(idx2, idx)
        ok = (pos < len(idx2)) & (idx2[np.minimum(pos, len(idx2) - 1)] == idx)
        for k in ks:
            g = extra / f"perframe_err_k{k}.npy"
            if g.exists() and ok.all():
                ERR[k] = np.load(g)[pos]

    meta = json.loads(pathlib.Path("/data1/jellyho/pc_cache/yam_s347/meta.json").read_text())
    eps_meta = {int(k): v for k, v in meta["episodes"].items()}

    def smooth(v, w=31):
        return np.convolve(v, np.ones(w) / w, mode="same")

    a.out.mkdir(parents=True, exist_ok=True)
    written = []
    min_val_frames = int(a.min_sec * FPS / STRIDE)
    for e in a.episodes:
        off, n = eps_meta[e]["offset"], eps_meta[e]["full_len"]
        m = (idx >= off) & (idx < off + n)
        if m.sum() < 50:
            print(f"ep{e}: not in the val split, skipping")
            continue
        base = max(float(ERR[0][m].mean()), 1e-6)
        g_s = np.mean([100 * smooth(ERR[0][m] - ERR[k][m]) / base for k in SHORT], axis=0)
        g_l = np.mean([100 * smooth(ERR[0][m] - ERR[k][m]) / base for k in LONG], axis=0)
        diff = g_s - g_l
        cls = diff > a.gap_thresh
        real_pos = idx[m] - off  # real episode frame of each val point
        edges = np.flatnonzero(np.diff(np.concatenate([[0], cls.astype(int), [0]])))
        regions = [(s, t) for s, t in zip(edges[::2], edges[1::2], strict=True) if t - s >= min_val_frames]
        if not regions:
            print(f"ep{e}: no sustained (> {a.min_sec}s) short-dominant region")
            continue
        ds = lerobot_dataset.LeRobotDataset(a.repo_id, root=a.root, episodes=[e], video_backend="pyav")
        for r, (s0, s1) in enumerate(regions):
            f0 = max(int(real_pos[s0]) - int(a.margin_sec * FPS), 0)
            f1 = min(int(real_pos[min(s1, len(real_pos) - 1)]) + int(a.margin_sec * FPS), n - 1)
            frames = []
            for t in range(f0, f1):
                img = ds[t][a.cam]
                arr = (np.asarray(img).transpose(1, 2, 0) * 255).astype(np.uint8)
                fig, ax = plt.subplots(figsize=(5.4, 4.6))
                ax.imshow(arr)
                ax.axis("off")
                in_region = real_pos[s0] <= t < real_pos[min(s1, len(real_pos) - 1)]
                vi = int(np.clip(np.searchsorted(real_pos, t), 0, len(diff) - 1))
                ax.set_title(
                    f"ep{e}  frame {t}/{n}  ({t / FPS:.1f}s)   short-long gap {diff[vi]:+.0f}%p"
                    + ("   [SHORT-DOMINANT]" if in_region else ""),
                    loc="left",
                    fontsize=9,
                    color="#B03030" if in_region else "#444444",
                )
                fig.tight_layout()
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                rgb = buf[..., :3]
                rgb = rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2]  # even dims for yuv420p
                frames.append(rgb.copy())
                plt.close(fig)
            outp = a.out / f"nm_short_ep{e}_r{r}_f{f0}-{f1}.mp4"
            imageio.mimwrite(outp, frames, fps=FPS, quality=8, macro_block_size=1)
            written.append(outp)
            print(f"wrote {outp} ({f1 - f0} frames, {(f1 - f0) / FPS:.1f}s)", flush=True)
    print("CLIPS:", " ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
