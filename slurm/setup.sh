#!/usr/bin/env bash
# ============================================================================================
#  One-time setup for the critic jobs on this cluster. Run it on the LOGIN node — it needs
#  network access, and the compute nodes may not have it. Idempotent.
#
#      slurm/setup.sh              # venv + scratch dirs only (offline diagnostics)
#      ROLLOUT=1 slurm/setup.sh    # + sim deps and RoboCasa assets (~10 GB) for rollout eval
#
#  Doing this once, here, is also what lets the array jobs run `uv run --no-sync`: fourteen
#  members starting at the same moment would otherwise race to sync the same .venv.
# ============================================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source slurm/env.sh

ROLLOUT="${ROLLOUT:-0}"

echo "== 1/4  scratch directories =="
acrft_mkdirs
mkdir -p "$CACHE_DIR/checkpoints"
printf '  %s\n' "$CACHE_DIR" "$ANNOT_ROOT" "$CRITIC_RUNS" "$SLURM_LOGS" "$JAX_COMPILATION_CACHE_DIR"

echo "== 2/4  python env (uv sync) =="
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

if [[ "$ROLLOUT" == "1" ]]; then
    echo "== 3/4  sim deps + RoboCasa assets =="
    git submodule update --init third_party/robocasa

    # RoboCasa resolves its asset dir as robocasa.__path__[0]/models/assets — inside this checkout,
    # on /home, which is a 200 G lustre share at 91% (~19 G free) against ~11 G of compressed asset
    # packs. Redirect the three subdirectories the downloads create onto $CACHE_DIR first; together
    # they are all but ~0.5 G of the total (textures 0.5, generative_textures 1.2, objects 8.7).
    # models/assets itself is left alone — it holds 208 git-tracked files. The packs that land here
    # all pass check_folder_exists=False, so a pre-created directory does not suppress the download.
    # The zips themselves cache under HF_HOME, which env.sh already points at $CACHE_DIR.
    assets="third_party/robocasa/robocasa/models/assets"
    for sub in textures generative_textures objects; do
        if [[ ! -e "$assets/$sub" ]]; then
            mkdir -p "$CACHE_DIR/robocasa_assets/$sub"
            ln -s "$CACHE_DIR/robocasa_assets/$sub" "$assets/$sub"
            echo "  $assets/$sub -> $CACHE_DIR/robocasa_assets/$sub"
        fi
    done

    # setup_eval_env.sh does `uv sync --group eval`, the kitchen asset packs, the lightwheel
    # workaround and robosuite's macros. Same path examples/robocasa/run_eval.sh expects.
    examples/robocasa/setup_eval_env.sh
else
    echo "== 3/4  sim deps — skipped (ROLLOUT=0) =="
fi

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
if [[ "$ROLLOUT" == "1" ]]; then
    uv run --no-sync python -c "import robosuite, mujoco; print('  robosuite', robosuite.__version__, 'mujoco', mujoco.__version__)"
fi

cat <<EOF

Setup done. Next:

  1. Pull the annotated dataset (7.7 GB, resumable):
       slurm/fetch_data.sh              # jellyho/acrft-annot-noprop -> $ANNOT_ROOT/noprop

  2. For the discount axis, build the re-labelled copies (token/candidate arrays are hardlinked,
     so each copy costs only the small reward/return columns):
       uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.999
       uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.9995

  3. For rollout eval, put the pi05 checkpoint the tokens were annotated from at
       \$VLA_CKPT = $VLA_CKPT
     Until then the sweep still runs offline-only with ROLLOUT=0.

  4. Submit (DATA_NAME picks the dataset dir; TASK stays the RoboCasa env for the rollout eval):
       DRYRUN=1 slurm/sweep.sh            # inspect the manifest
       slurm/sweep.sh                     # or ROLLOUT=0 slurm/sweep.sh until the ckpt is up
EOF
