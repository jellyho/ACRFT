#!/usr/bin/env bash
# ============================================================================================
#  One-time setup for the critic jobs on this cluster. Run it on the LOGIN node — it needs
#  network access, and the compute nodes may not have it. Idempotent.
#
#      slurm/setup.sh              # venv + scratch dirs
#
#  There is no ROLLOUT=1 mode any more: it installed robosuite/mujoco and the RoboCasa asset packs
#  for the in-job sim rollout eval, and RoboCasa was retired on 2026-09-09. The sweep runs the
#  offline diagnostics only.
#
#  Doing this once, here, is also what lets the array jobs run `uv run --no-sync`: fourteen
#  members starting at the same moment would otherwise race to sync the same .venv.
# ============================================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source slurm/env.sh

echo "== 1/4  scratch directories =="
acrft_mkdirs
mkdir -p "$CACHE_DIR/checkpoints"
printf '  %s\n' "$CACHE_DIR" "$ANNOT_ROOT" "$CRITIC_RUNS" "$SLURM_LOGS" "$JAX_COMPILATION_CACHE_DIR"

echo "== 2/4  python env (uv sync) =="
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "== 3/4  sim deps — not needed (the RoboCasa sim rollout eval was retired 2026-09-09) =="

echo "== 4/4  verify =="
# JAX_PLATFORMS=cpu is required, not tidiness: on this GPU-less login node jax.devices() does not
# fall back to CPU, it RAISES on the cuda plugin, which under `set -e` would abort setup right
# before it prints the next steps. The GPU is checked where it matters — in train_critic.sbatch.
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
import hashlib  # the libcrypto conflict env.sh works around — fail here, not inside a job
import jax, flax, optax
import openpi.rlt_critic.critic  # noqa: F401  the repo itself is importable
print(f"  jax {jax.__version__}  flax {flax.__version__}  optax {optax.__version__}  hashlib OK")
print("  openpi.rlt_critic imports; GPU is verified per-job, not here (login node has none)")
PY

cat <<EOF

Setup done. Next:

  1. Pull the annotated dataset (7.7 GB, resumable):
       slurm/fetch_data.sh              # jellyho/acrft-annot-noprop -> $ANNOT_ROOT/noprop

  2. For the discount axis, build the re-labelled copies (token/candidate arrays are hardlinked,
     so each copy costs only the small reward/return columns):
       uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.999
       uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.9995

  3. Submit (DATA_NAME picks the dataset dir):
       DRYRUN=1 slurm/sweep.sh            # inspect the manifest
       slurm/sweep.sh
EOF
