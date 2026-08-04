#!/usr/bin/env bash
# Cluster-local environment for the ACRFT critic jobs. Sourced by everything in slurm/.
# Nothing here is repo logic — it is the mapping from this SLURM cluster to the paths and
# flags the stage-3 scripts expect. Every value is overridable from the submitting shell.

ACRFT_REPO="${ACRFT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ACRFT_REPO

# $HOME is quota'd and /data5 (the other box's cache root) does not exist here. /scratch is an NFS
# share (powerscale4, ~10 T free) mounted on every node, so caches, datasets, run outputs and job
# logs all live under it. /lustre/jellyho is the other shared option if /scratch ever fills up.
export CACHE_DIR="${CACHE_DIR:-/scratch/jellyho/acrft}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR/.cache/openpi}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Persistent XLA cache. The critic's scanned update is a slow compile, and most sweep variants
# differ only in a scalar that does not change the graph — they hit this cache instead of
# recompiling. Keyed by (graph, shapes, flags, jax version), so a shape change just adds an entry.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$CACHE_DIR/.cache/jax_compile}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# Pipeline artefacts on this cluster.
#   ANNOT_ROOT   annotate_rlt.py output dirs, copied here from the annotating machine
#   CRITIC_RUNS  train_rlt_critic.py --out  (one dir per sweep variant)
#   SLURM_LOGS   job stdout/stderr
#   VLA_CKPT     the VLA checkpoint the tokens were annotated from — needed ONLY for the
#                in-process rollout eval (--rollout-every), which keeps the 3 B model resident
export ANNOT_ROOT="${ANNOT_ROOT:-$CACHE_DIR/annot}"
export CRITIC_RUNS="${CRITIC_RUNS:-$CACHE_DIR/critic_runs}"
export SLURM_LOGS="${SLURM_LOGS:-$CACHE_DIR/logs}"
# The exp dir the annotation's meta.json records as its source: pi05_robocasa_PrepareCoffee_rlt,
# exp PrepareCoffee_rlt5_pardec_noprop, step 70000.
export VLA_CKPT="${VLA_CKPT:-$CACHE_DIR/checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_noprop/70000}"

# HF datasets/models. The token lives in the default user cache, which moving HF_HOME hides — carry
# it over explicitly so a private repo still resolves. (acrft-annot-noprop itself is public.)
if [[ -z "${HF_TOKEN:-}" && ! -f "$HF_HOME/token" && -f "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi

# The annotation is loaded onto the device once and stays there — token 2.29 GB + candidates
# 3.44 GB + executed chunks 0.21 GB = 5.9 GB for the noprop set (T=279,534, N=16, H=16, float32;
# base_action_heldout is read only by eval_rlt_critic.py, not by training). The allocator must
# therefore be allowed to grow around it rather than pre-grabbing 75%.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

# In-process sim rollout: headless offscreen GL, and robocasa imported from the submodule
# (it is not installed into the venv — examples/robocasa/setup_eval_env.sh uses the same path).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
case ":${PYTHONPATH:-}:" in
    *":$ACRFT_REPO/third_party/robocasa:"*) ;;
    *) export PYTHONPATH="$ACRFT_REPO/third_party/robocasa${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

export WANDB_PROJECT="${WANDB_PROJECT:-acrft_critic}"
export WANDB_ENTITY="${WANDB_ENTITY:-RSS-PFT_RLLAB}"
# wandb writes a local run dir per job (media, logs, and the full history it replays on resume).
# Default is <cwd>/wandb, i.e. the checkout on the quota'd /home; a sweep of 15 runs would put it
# there. Keep it on scratch with everything else.
export WANDB_DIR="${WANDB_DIR:-$CACHE_DIR/wandb}"
mkdir -p "$WANDB_DIR" 2>/dev/null || true

export PATH="$HOME/.local/bin:$PATH"   # uv

# ~/.bashrc puts $HOME/miniconda3/lib on LD_LIBRARY_PATH, and its libcrypto.so.3 (2024) is older
# than the OPENSSL_3.4.0 that the system python3.11 links _hashlib against. uv builds .venv from
# that interpreter, so with miniconda's lib in front every `import hashlib` dies — it breaks the
# uv build backend at setup time and would break every job at run time. Drop exactly that entry;
# the CUDA and MuJoCo entries next to it are needed.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _acrft_ldp=""
    while IFS= read -r _p; do
        [[ -z "$_p" || "$_p" == "$HOME/miniconda3/lib" || "$_p" == "$HOME/miniconda3/lib/" ]] && continue
        _acrft_ldp="${_acrft_ldp:+$_acrft_ldp:}$_p"
    done < <(printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}")
    export LD_LIBRARY_PATH="$_acrft_ldp"
    unset _acrft_ldp _p
fi

# tier -> "<partition list>|<qos>".
#
# A job carries exactly one QOS, so a tier may only span partitions that share one. big_qos
# covers every consumer-GPU partition on this cluster, which is what makes a wide fallback list
# possible at all; the pro6000 and a100 boxes each sit behind their own QOS and cannot be mixed in.
#
# Sizing for the noprop annotation: 5.9 GB resident, plus the target forward (B*P*N = 1024*8*16
# chunks per step), plus — with rollout eval on — a resident 3 B VLA in bfloat16 (~6 GB). 48 GB is
# comfortable either way; 24 GB is workable for ROLLOUT=0 but leaves little room once the VLA is
# also resident, so prefer `a6000`/`pro6000` whenever ROLLOUT=1.
#
# QOS: base_qos is NON-PREEMPTABLE but caps concurrent jobs (~8); big_qos goes wider but its jobs
# can be preempted. train_rlt_critic.py has no resume path, so a preempted 200k-step run restarts
# from zero — base_qos is therefore the default and MAXPAR is set to match its cap. Only
# big_suma_rtx3090 refuses base_qos, so it appears solely in the `big` tier. Override with QOS=.
acrft_tier() {
    local q="${QOS:-base_qos}"
    case "${1:-a6000}" in
        # 48 GB: A6000 x6 nodes + gigabyte A6000 x6 + tyan A6000 + RTX6000Ada — ~20 nodes.
        a6000)   echo "suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada|$q" ;;
        # 24 GB: 3090/4090/A5000. Most nodes, tightest memory.
        ampere)  echo "base_suma_rtx3090,suma_rtx4090,dell_rtx3090,gigabyte_a5000,asus_a5000|$q" ;;
        # Everything base_qos reaches, 48 GB partitions first. Slurm picks by partition priority
        # rather than list order, so this trades a guaranteed 48 GB for the shortest wait.
        wide)    echo "suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada,base_suma_rtx3090,suma_rtx4090,dell_rtx3090,gigabyte_a5000|$q" ;;
        # Adds big_suma_rtx3090 (12 nodes, big_qos only). Use when more than base_qos' cap of
        # concurrent jobs is worth the preemption risk — i.e. short runs, not the 200k sweep.
        big)     echo "big_suma_rtx3090,suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada,base_suma_rtx3090,suma_rtx4090,dell_rtx3090|big_qos" ;;
        # 96 GB RTX PRO 6000, 4 nodes — most headroom, longest queue.
        pro6000) echo "asus_pro6000,gigabyte_pro6000|pro6000_qos" ;;
        a100)    echo "suma_a100|a100_qos" ;;
        *) echo "acrft_tier: unknown tier '$1' (a6000|ampere|wide|big|pro6000|a100)" >&2; return 1 ;;
    esac
}

# Nodes that took a job and could not run it — typically they advertise a GPU but hand jax
# "No visible GPU devices". The preflight in train_critic.sbatch catches that and fails fast rather
# than silently training on CPU, but the slot is still lost, so a node that has failed once is kept
# out of every later allocation.
#
# The list is a FILE, not a constant here: the jobs themselves append to it (acrft_mark_bad_node),
# so a node that breaks tomorrow is excluded from the next submission without anyone editing this
# script. One node per line; anything after '#' is the reason and is ignored when building the list.
export ACRFT_BAD_NODES="${ACRFT_BAD_NODES:-$CACHE_DIR/bad_nodes.txt}"

# Record a node as unusable. Safe to call from many jobs at once: a single short append on a POSIX
# filesystem is atomic, and duplicates are collapsed when the list is read.
acrft_mark_bad_node() {
    local node="${1:-$(hostname -s)}" reason="${2:-unspecified}"
    mkdir -p "$(dirname "$ACRFT_BAD_NODES")" 2>/dev/null || true
    printf '%s  # %s  (job %s, %s)\n' "$node" "$reason" "${SLURM_JOB_ID:-?}" "$(date '+%Y-%m-%d %H:%M')" \
        >> "$ACRFT_BAD_NODES"
}

# Comma-separated node list for --exclude, deduplicated, comments stripped.
acrft_exclude_list() {
    [[ -f "$ACRFT_BAD_NODES" ]] || return 0
    sed 's/#.*//' "$ACRFT_BAD_NODES" | tr -d ' \t' | grep -v '^$' | sort -u | paste -sd, -
}

acrft_mkdirs() { mkdir -p "$ANNOT_ROOT" "$CRITIC_RUNS" "$SLURM_LOGS" "$JAX_COMPILATION_CACHE_DIR"; }
