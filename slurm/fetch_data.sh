#!/usr/bin/env bash
# ============================================================================================
#  Pull the annotated dataset, or the VLA checkpoint, from the Hub. Run on the LOGIN node (network).
#
#      slurm/fetch_data.sh                                   # dataset -> $ANNOT_ROOT/noprop
#      slurm/fetch_data.sh jellyho/acrft-annot-other         # another dataset repo
#      slurm/fetch_data.sh --ckpt                            # VLA checkpoint -> $VLA_CKPT
#      slurm/fetch_data.sh --ckpt some/other-ckpt-repo       # another checkpoint repo
#      NAME=noprop_v2 slurm/fetch_data.sh                    # override the local dataset dir name
#
#  Resumable: re-running skips files that are already complete, so a dropped transfer just needs
#  the same command again.
#
#  Env: REPO (positional wins) · NAME (dataset dir under $ANNOT_ROOT) · ANNOT_ROOT · VLA_CKPT · HF_TOKEN
# ============================================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source slurm/env.sh
acrft_mkdirs

KIND=dataset
MODE=annot
case "${1:-}" in
    --ckpt) KIND=model; MODE=ckpt; shift ;;
    --lerobot) MODE=lerobot; shift ;;
esac

if [[ "$MODE" == "lerobot" ]]; then
    # The source LeRobot dataset. Needed only by the ROLLOUT=1 path: building the VLA calls
    # data.create(), which computes progress labels, which reads the reward column out of the
    # dataset's parquet files. The annotation alone does not carry it. Small (0.6 GB) because the
    # videos are never touched here.
    #
    # It has to land at $HF_LEROBOT_HOME/<repo_id> — that exact path is what lerobot resolves.
    REPO="${1:-${REPO:-$(python3 -c "import json;print(json.load(open('$ANNOT_ROOT/${DATA_NAME:-noprop}/meta.json'))['repo_id'])" 2>/dev/null)}}"
    [[ -n "$REPO" ]] || { echo "ERROR: could not read repo_id from the annotation; pass it explicitly"; exit 1; }
    DEST="$HF_LEROBOT_HOME/$REPO"
elif [[ "$MODE" == "ckpt" ]]; then
    # openpi loads params from <ckpt>/params and norm stats from <ckpt>/assets, so the repo root maps
    # straight onto $VLA_CKPT — no extra nesting.
    REPO="${1:-${REPO:-jellyho/pi05-robocasa-prepcoffee-rlt-pardec-noprop-70k}}"
    DEST="$VLA_CKPT"
else
    REPO="${1:-${REPO:-jellyho/acrft-annot-noprop}}"
    NAME="${NAME:-$(basename "$REPO" | sed 's/^acrft-annot-//')}"
    DEST="$ANNOT_ROOT/$NAME"
fi

# Before setup.sh has run there is no .venv; the conda python on this box already carries
# huggingface_hub, so either interpreter works and the download does not have to wait on uv sync.
PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" -c "import huggingface_hub" 2>/dev/null \
    || { echo "ERROR: no huggingface_hub in $PY — run slurm/setup.sh first"; exit 1; }
# hf_transfer is worth ~20x on this link, but it is an optional dep: only ask for it if it is there.
"$PY" -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0

echo "repo   : $REPO ($KIND)"
echo "dest   : $DEST"
echo "accel  : hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER:-1}"

REPO="$REPO" DEST="$DEST" KIND="$KIND" "$PY" - <<'PY'
import os, pathlib
from huggingface_hub import snapshot_download

dest = pathlib.Path(os.environ["DEST"])
# local_dir writes the real files into dest (no blob-cache second copy), and keeps its own
# resume metadata under dest/.cache so an interrupted pull picks up where it stopped.
snapshot_download(os.environ["REPO"], repo_type=os.environ["KIND"], local_dir=str(dest))
print(f"downloaded to {dest}")
PY

# --- verify -------------------------------------------------------------------------------------
if [[ "$KIND" == "model" ]]; then
    # What openpi actually opens: restore_params(<ckpt>/params) and load_norm_stats(<ckpt>/assets).
    [[ -d "$DEST/params" ]] || { echo "ERROR: no params/ under $DEST"; exit 1; }
    [[ -f "$DEST/params/manifest.ocdbt" ]] || { echo "ERROR: $DEST/params is not an orbax ocdbt checkpoint"; exit 1; }
    ls "$DEST/assets"/*/norm_stats.json >/dev/null 2>&1 \
        || echo "WARNING: no assets/*/norm_stats.json — the VLA will need norm stats from the config"
    echo
    echo "checkpoint OK: $(du -sh "$DEST" | cut -f1) at $DEST"
    echo "  params/  $(du -sh "$DEST/params" | cut -f1)"
    ls -d "$DEST/assets"/* 2>/dev/null | sed 's|^|  assets/  |'
    echo
    echo "Next: ROLLOUT=1 slurm/sweep.sh   (needs 'ROLLOUT=1 slurm/setup.sh' for the sim deps)"
    exit 0
fi

DEST="$DEST" "$PY" - <<'PY'
import json, os, pathlib, sys

d = pathlib.Path(os.environ["DEST"])
meta = json.loads((d / "meta.json").read_text())
T, D, H, A, N = (meta[k] for k in ("num_frames", "token_dim", "horizon", "action_dim", "num_samples"))
nh = meta.get("num_heldout", 0)
itemsize = {"float32": 4, "float16": 2, "bfloat16": 2}[meta.get("dtype", "float32")]

# Every array the critic memmaps at a fixed shape: a short file is a truncated transfer, and it
# would surface much later as a silently wrong reshape rather than as a missing-file error.
expect = {
    "rl_token": T * D * itemsize,
    "action_chunk": T * H * A * itemsize,
    "base_action": T * N * H * A * itemsize,
    "reward": T * 4,
    "mc_return": T * 4,
    "done": T * 1,
    "episode_index": T * 4,
}
if nh:
    expect["base_action_heldout"] = T * nh * H * A * itemsize
bad = []
for name, want in expect.items():
    p = d / f"{name}.dat"
    got = p.stat().st_size if p.exists() else -1
    if got != want:
        bad.append(f"  {name}.dat: {got} bytes, expected {want}")
if meta.get("stride", 1) != 1:
    bad.append(f"  stride={meta['stride']} — the critic needs stride 1")

resident = (T * D + T * H * A + T * N * H * A) * itemsize / 1e9
print(f"\nframes {T}  token {D}  chunk {H}x{A}  candidates {N} (+{nh} held-out)  {meta.get('dtype')}")
print(f"scheme {meta.get('reward_scheme', 'raw')!r}  discount {meta.get('discount')}  support {meta.get('value_support')}")
print(f"GPU-resident during training: {resident:.2f} GB")
if bad:
    print("\nINCOMPLETE — re-run this script to resume:", *bad, sep="\n")
    sys.exit(1)
print("all arrays complete")
PY

echo
echo "Next:"
echo "  # discount axis (optional):"
echo "  uv run slurm/make_discount_variant.py --data $DEST --discount 0.999"
echo "  uv run slurm/make_discount_variant.py --data $DEST --discount 0.9995"
echo "  # submit (DATA_NAME is the dir here; TASK stays the RoboCasa env for the rollout eval):"
echo "  DATA_NAME=$NAME DRYRUN=1 slurm/sweep.sh"
