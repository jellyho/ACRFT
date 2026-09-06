"""Did the critic's input actually violate the squash-vs-pad contract on the robot?

The contract: the feature cache was built by SQUASHING native 480x640 frames to 224 (cv2.INTER_AREA,
aspect ratio not preserved). A client that pre-processes with resize_with_pad instead sends an
already-224x224 LETTERBOXED frame, the server's resize becomes a no-op, and the critic scores an
image it never saw in training -- with no error raised. Nothing logged what arrived, so the question
could not be answered from the server side.

It can be answered from the recording. /data5/jellyho/pc_rollouts_yam/lego_taxi retains the images
that actually went through the pipeline. A letterbox of 480x640 into 224 leaves 224x168 of content
with 28 rows of exact black at the top and the bottom; a squash fills all 224. This draws the row
profile that decides it, next to a real frame and the letterboxed frame it would have been.
"""

# ruff: noqa: PLC0415, ICN001

import argparse
import json
import pathlib
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=pathlib.Path, default=pathlib.Path("/data5/jellyho/pc_rollouts_yam/lego_taxi"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/fig_image_contract.png")
    ap.add_argument("--json", type=pathlib.Path, default=R / ".scratch/extraction/image_contract.json")
    a = ap.parse_args()

    meta = json.loads((a.rollouts / "meta.json").read_text())
    shapes = meta["shapes"] if isinstance(meta["shapes"], dict) else json.loads(meta["shapes"].replace("'", '"'))
    sh = tuple(shapes["images"])
    im = np.memmap(a.rollouts / "images.dat", np.uint8, "r", shape=sh)
    n, ncam, size = sh[0], sh[1], sh[2]
    # 480x640 fitted inside 224 gives 224x168 -> (224-168)/2 = 28 rows of bar at each edge
    bar = (size - round(size * 480 / 640)) // 2

    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(n, min(a.frames, n), replace=False))
    X = np.asarray(im[rows], np.float32)

    res = {
        "rollouts": str(a.rollouts),
        "episodes": int(meta["num_episodes"]),
        "success": int(meta["num_success"]),
        "frames_sampled": len(rows),
        "img_size": size,
        "bar_rows_if_padded": bar,
        "cameras": [],
    }
    for c in range(ncam):
        v = X[:, c]
        res["cameras"].append(
            {
                "row_mean": v.mean(axis=(0, 2, 3)).tolist(),
                "top_band_max": float(v[:, :bar].max()),
                "bottom_band_max": float(v[:, -bar:].max()),
                "top_band_mean": float(v[:, :bar].mean()),
                "middle_mean": float(v[:, bar:-bar].mean()),
            }
        )
    res["verdict"] = "squash" if min(c["top_band_max"] for c in res["cameras"]) >= 5 else "pad"
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1))

    sys.path.insert(0, str(R / "slurm"))
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    PAL = plot_style.PALETTE

    real = np.asarray(im[rows[0], 0], np.uint8)
    # what the padded convention would have produced from the same native frame
    up = cv2.resize(real, (640, 480), interpolation=cv2.INTER_LINEAR)
    fit = cv2.resize(up, (size, round(size * 480 / 640)), interpolation=cv2.INTER_AREA)
    padded = np.zeros_like(real)
    padded[bar : bar + fit.shape[0]] = fit

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), gridspec_kw={"width_ratios": [1.55, 1, 1]})
    ax = axes[0]
    for c in range(ncam):
        ax.plot(res["cameras"][c]["row_mean"], lw=1.5, color=PAL[c], label=f"camera {c}")
    ax.axvspan(0, bar, color="0.55", alpha=0.22, zorder=0)
    ax.axvspan(size - bar, size, color="0.55", alpha=0.22, zorder=0)
    ax.set_xlim(0, size - 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("image row")
    ax.set_ylabel("mean brightness")
    ax.set_title("content where a letterbox bar would be")
    ax.legend(loc="lower center", fontsize=9, ncol=3)
    for x in (bar / 2, size - bar / 2):
        ax.text(x, ax.get_ylim()[1] * 0.955, "bar", ha="center", va="top", fontsize=8.5, color="0.4")

    for k, (img, ttl) in enumerate(((real, "as recorded"), (padded, "if it had been letterboxed"))):
        axes[k + 1].imshow(img)
        axes[k + 1].set_title(ttl)
        axes[k + 1].set_xticks([])
        axes[k + 1].set_yticks([])
        for s in axes[k + 1].spines.values():
            s.set_visible(False)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"verdict: {res['verdict']}  (bar would be {bar} rows)")
    for c, d in enumerate(res["cameras"]):
        print(f"  cam{c}: top-band max {d['top_band_max']:.0f}, bottom-band max {d['bottom_band_max']:.0f}")
    print(f"wrote {a.out}\nwrote {a.json}")


if __name__ == "__main__":
    main()
