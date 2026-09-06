#!/usr/bin/env bash
# 2-step smoke that a trainer's checkpoint is servable AS WRITTEN (BC layout, no export step):
# train, then open <out>/<step> with create_trained_policy -- the exact path serve_policy.py takes.
#SBATCH --job-name=servable_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --chdir=/data5/jellyho/ACRFT/openpi
#SBATCH --output=/data5/jellyho/ACRFT/openpi/servable_smoke.slurm.log
export LEROBOT_VIDEO_BACKEND=pyav
export PYTHONPATH=/data5/jellyho/ACRFT/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/data5/jellyho/ACRFT/openpi/.venv/bin/python   # venv python directly: `uv run` hangs on node100
OUT=/data1/jellyho/acrft_ckpts/extraction/_smoke_gpu
rm -rf "$OUT"
for arm in awr qam; do
  echo "==== $arm ===="
  if timeout 2400 $PY -u scripts/train_$arm.py --steps 2 --batch 2 --save-every 1 --out $OUT/$arm; then
    $PY - <<PYEOF && echo "VERDICT $arm PASS" || echo "VERDICT $arm FAIL(load)"
import pathlib, jax
import openpi.training.config as C
from openpi.policies import policy_config
root = pathlib.Path("$OUT/$arm"); steps = sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())
d = root / str(steps[-1]); print("saved steps:", steps, "layout:", sorted(p.name for p in d.iterdir()))
cfg = C.get_config("pi05_yam_lego_taxi_cfgrl" if "$arm" == "cfgrl" else "pi05_yam_lego_taxi")
pol = policy_config.create_trained_policy(cfg, d); print("create_trained_policy OK on", d)
PYEOF
  else echo "VERDICT $arm FAIL(train $?)"; fi
done
rm -rf "$OUT"; echo "==== DONE ===="
