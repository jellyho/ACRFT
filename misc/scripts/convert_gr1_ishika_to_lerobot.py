"""Convert the downloaded ishika RoboCasa-GR1 PnPCanToDrawerClose episodes (LeRobot v2.1,
one parquet + one ego_view mp4 per episode) into a fresh LeRobot v3.0 dataset that openpi's
pi05 trainer reads.

Source layout (already downloaded, 115 episodes 2120..2234):
    <src>/data/chunk-002/episode_0021xx.parquet          state(43) action(24) condition(14)
    <src>/videos/chunk-002/observation.images.egoview/episode_0021xx.mp4   256x256x3 @20fps

We recreate it via LeRobotDataset.create + add_frame/save_episode so the output is a real v3
dataset (chunk/file packing, meta/stats/tasks) under HF_LEROBOT_HOME.

    uv run python scripts/convert_gr1_ishika_to_lerobot.py \
        --src /data5/jellyho/gr1_data/ishika_PnPCanToDrawerClose \
        --repo-id jellyho/gr1_pnp_can_to_drawer_close --task PnPCanToDrawerClose
"""

import argparse
import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd


def read_video(path: pathlib.Path) -> np.ndarray:
    """Decode an mp4 to [T, H, W, 3] uint8. Try torchcodec, then imageio."""
    try:
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(str(path))
        frames = dec[:].data  # [T, 3, H, W] uint8 tensor
        return np.transpose(np.asarray(frames), (0, 2, 3, 1))
    except Exception:
        import imageio.v3 as iio

        return np.asarray(iio.imread(path, plugin="pyav"))  # [T, H, W, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, required=True)
    ap.add_argument("--repo-id", default="jellyho/gr1_pnp_can_to_drawer_close")
    ap.add_argument("--task", default="PnPCanToDrawerClose")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cam-key", default="observation.images.egoview")
    a = ap.parse_args()

    parquets = sorted((a.src / "data").rglob("episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquet under {a.src}/data")
    print(f"{len(parquets)} episodes to convert", flush=True)

    df0 = pd.read_parquet(parquets[0])
    state_dim = np.asarray(df0["observation.state"].iloc[0]).shape[0]
    action_dim = np.asarray(df0["action"].iloc[0]).shape[0]
    cond_dim = np.asarray(df0["observation.condition"].iloc[0]).shape[0]
    # peek one frame for the image resolution
    vid0 = read_video(next((a.src / "videos").rglob("episode_*.mp4")))
    h, w = vid0.shape[1], vid0.shape[2]
    print(f"state {state_dim}  action {action_dim}  condition {cond_dim}  image {h}x{w}", flush=True)

    features = {
        a.cam_key: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "observation.condition": {"dtype": "float32", "shape": (cond_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }

    ds = LeRobotDataset.create(
        repo_id=a.repo_id,
        fps=a.fps,
        features=features,
        robot_type="gr1",
        use_videos=True,
    )

    for pi, pq in enumerate(parquets):
        df = pd.read_parquet(pq).sort_values("frame_index").reset_index(drop=True)  # noqa: PD901
        ep = int(df["episode_index"].iloc[0])
        vid_path = a.src / "videos" / f"chunk-{ep // 1000:03d}" / a.cam_key / f"episode_{ep:06d}.mp4"
        vid = read_video(vid_path)
        if len(vid) != len(df):
            n = min(len(vid), len(df))
            print(f"  ep {ep}: video {len(vid)} vs rows {len(df)} -> truncating to {n}", flush=True)
            df, vid = df.iloc[:n].reset_index(drop=True), vid[:n]
        for i in range(len(df)):
            ds.add_frame(
                {
                    a.cam_key: vid[i],
                    "observation.state": np.asarray(df["observation.state"].iloc[i], np.float32),
                    "observation.condition": np.asarray(df["observation.condition"].iloc[i], np.float32),
                    "action": np.asarray(df["action"].iloc[i], np.float32),
                    "task": a.task,
                }
            )
        ds.save_episode()
        if (pi + 1) % 20 == 0:
            print(f"  {pi + 1}/{len(parquets)} episodes", flush=True)

    print(f"done: {ds.num_episodes} episodes, {ds.num_frames} frames -> {ds.root}", flush=True)


if __name__ == "__main__":
    main()
