#!/usr/bin/env bash
# One-time, reproducible setup for RoboCasa 365 evaluation (sim rollout client + assets).
# Idempotent — safe to re-run. After this, use examples/robocasa/run_eval.sh.
#
#   examples/robocasa/setup_eval_env.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/third_party/robocasa${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1/4  submodule =="
git submodule update --init third_party/robocasa

echo "== 2/4  sim deps into .venv (uv sync --group eval) =="
uv sync --group eval

echo "== 3/4  kitchen assets =="
# Official packs whose upstream URLs still work (textures + objaverse/aigen objects). The official
# script always prompts (~10 GB) and checks per-pack whether the folder already exists, so re-runs
# are cheap. `yes` may exit via SIGPIPE — that's fine.
yes y | "$PY" third_party/robocasa/robocasa/scripts/download_kitchen_assets.py \
    --type tex tex_generative objs_objaverse objs_aigen || true
# Lightwheel objects/fixtures: NVIDIA renamed + restructured their HF repo (single zip -> per-object
# zips), so robocasa365's official download 404s. This fetches the same assets from the new layout.
"$PY" examples/robocasa/download_lightwheel_assets.py

echo "== 4/4  robosuite private macros (silences warnings; optional) =="
"$PY" third_party/robocasa/robocasa/scripts/setup_macros.py || true

echo
echo "Done. Verify:"
echo "  MUJOCO_GL=egl PYTHONPATH=third_party/robocasa:packages/openpi-client/src \\"
echo "    .venv/bin/python -c \"from robocasa.utils.env_utils import create_env; create_env(env_name='PrepareCoffee', robots='PandaOmron', split='target').reset(); print('OK')\""
echo "Then:  examples/robocasa/run_eval.sh PrepareCoffee"
