# Stage-3 critic ablation on the SLURM cluster

Everything here is cluster glue, not repo logic: it maps this SLURM cluster (partitions, QOS,
`/scratch`) onto `scripts/train_rlt_critic.py` + `scripts/eval_rlt_critic.py`. Stages 1–2 (VLA
training and annotation) happened elsewhere — the annotated arrays are pulled from the Hub and the
critic is trained here.

| File | What it is |
|---|---|
| `env.sh` | paths, caches, wandb, and the tier → partition/QOS map. Sourced by everything. |
| `setup.sh` | one-time, on the login node: venv, scratch dirs, (optionally) sim deps + assets |
| `fetch_data.sh` | pull an annotated dataset from the Hub into `$ANNOT_ROOT` and verify it |
| `make_discount_variant.py` | builds a gamma-swept copy of an annotation, hardlinking the heavy arrays |
| `train_critic.sbatch` | one critic variant: train → offline diagnostics. Array-aware. |
| `sweep.sh` | writes the ablation manifest and submits it as one array job |
| `collect.py` | one table from all the `diag.json` files + the rollout lines in the job logs |

---

## Once

```bash
ROLLOUT=1 slurm/setup.sh      # drop ROLLOUT=1 if you do not want in-training sim rollouts
```

`uv sync`, the scratch directories, and — with `ROLLOUT=1` — `uv sync --group eval`, the robocasa
submodule and ~10 GB of kitchen assets. Run it on the **login node**: it needs network, and doing it
once here is what lets the array members run `uv run --no-sync` instead of fourteen jobs racing to
sync the same `.venv`.

Everything lands under `/scratch/jellyho/acrft` (`CACHE_DIR`). `/scratch` is an NFS share mounted on
every node with ~10 T free, so a login-node download is visible to the jobs.

## Every time: get the data here

```bash
slurm/fetch_data.sh          # jellyho/acrft-annot-noprop -> $ANNOT_ROOT/noprop
```

Resumable, and it verifies every array against the shape `meta.json` declares before you queue
anything — a truncated transfer would otherwise surface much later as a wrong reshape rather than a
missing file. **`meta.json` must say `"stride": 1`**; the per-prefix reward sums need every frame,
and both the fetch check and `train_critic.sbatch` refuse anything else.

The `acrft-annot-noprop` set, for reference:

| | |
|---|---|
| frames | 279,534 (stride 1) |
| token / chunk | 2048-d · 16×12 |
| candidates | **16** + 8 held-out |
| reward | `sparse`, terminal success, gamma 0.99, support [0, 1] — already re-labelled |
| on disk | 7.7 GB float32 |
| **GPU-resident during training** | **5.94 GB** (token + candidates + executed chunks; the held-out set is read only by `eval_rlt_critic.py`) |
| source VLA | `pi05_robocasa_PrepareCoffee_rlt`, exp `PrepareCoffee_rlt5_pardec_noprop`, step 70000 |

N=16 matters for the sweep: `train_rlt_critic.py` only subsamples when `0 < --bootstrap-candidates <
N`, so that arm is generated as N/2 and N/4 rather than hardcoded.

For the **rollout eval** you also need the pi05 checkpoint above at `$VLA_CKPT`
(`$CACHE_DIR/checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_noprop/70000`).
It is a 3 B model kept resident next to the annotation, which is the other reason the default tier is
48 GB. Until it is there, run with `ROLLOUT=0` — everything else works.

## Discount axis

Gamma is not a training flag — it also sets `mc_return` and the value support, and
`train_rlt_critic.py` reads all three from `meta.json` on purpose, so a re-labelled dataset can never
be trained against the previous scheme's constants. A gamma sweep is a sweep over datasets:

```bash
uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.999
uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.9995
```

`rl_token`/`action_chunk`/`base_action` are hardlinked (they are ~99.9% of the bytes and identical
across gammas); only the return columns are real copies, so each variant costs **3.5 MB** against the
source's 7.2 GB. The return is re-accumulated backwards from the stored per-frame reward, exactly as
`annotate_rlt.py` built it, so this needs **no LeRobot dataset on this machine**.

Both variants are already built. Why the axis is worth running at all — episodes here have a median
length of 521 steps:

| dataset | gamma | effective horizon | mean `mc_return` over live frames |
|---|---|---|---|
| `noprop` | 0.99 | ~100 steps | 0.19 |
| `noprop_g999` | 0.999 | ~1000 steps | 0.77 |
| `noprop_g9995` | 0.9995 | ~2000 steps | 0.87 |

At 0.99 the horizon is a fifth of a typical episode, so from most states the goal is nearly invisible
in the target — which is exactly the concern commit `5649cc4` raised when it prepped this sweep.

Changing the reward *scheme* (`sparse` ↔ `v3`) is a different operation — that goes back to the raw
success column, so it needs `scripts/relabel_reward.py` and the source LeRobot dataset present.

## Submit

```bash
DRYRUN=1 slurm/sweep.sh      # print the manifest, submit nothing
slurm/sweep.sh               # 14 runs, 8 at a time, 48 GB tier
```

Two separate knobs, deliberately: `DATA_NAME` (default `noprop`) is the dataset dir under
`$ANNOT_ROOT`, `TASK` (default `PrepareCoffee`) is the RoboCasa env the rollout eval builds. They are
not the same string and sharing one variable would hand `make_env()` a dataset name.

The sweep is **one-factor-at-a-time from one baseline**, not a cross product — with rollout eval on,
each run is expensive, and each flag answers "does this knob move the ranking signal", which a grid
does not answer better.

| Axis (`AXES=`) | Runs | Knob |
|---|---|---|
| — | `base` | ARQ, hard max over all N, min over the ensemble, scalar Q, macro group 2 |
| `agg` | `topm` `soft` | reduce over candidates: average the best 3 / softmax instead of a hard max |
| | `lcb` | reduce over the ensemble: mean−std instead of a hard min |
| | `bc8` `bc4` | narrow the arg-max to a resampled subset of N/2, N/4 (also the cheapest step speedup) |
| | `tn03` `tn10` | temporally coherent target smoothing around each candidate |
| `discount` | `g999` `g9995` | the re-labelled datasets above |
| `structure` | `qc` | flat Q per chunk — no per-prefix head, no adaptive commit |
| | `mg4` `mg8` | coarser prefix granularity |
| | `hlg` | HL-Gauss distributional instead of scalar regression |

Common variations:

```bash
SWEEP=abl2 AXES=agg slurm/sweep.sh          # one axis
SEEDS="0 1 2" AXES=agg slurm/sweep.sh       # three seeds per variant
TIER=pro6000 MAXPAR=4 slurm/sweep.sh        # 96 GB nodes, fewer at a time
ROLLOUT=0 STEPS=50000 slurm/sweep.sh        # quick offline-only pass
```

One variant on its own, no array:

```bash
RUN=probe DATA=$ANNOT_ROOT/PrepareCoffee FLAGS="--v-agg soft --soft-tau 0.05" \
    sbatch -p suma_a6000 -q big_qos slurm/train_critic.sbatch
```

## Tiers

A job carries one QOS, so a partition list may only span partitions that share one. `big_qos` covers
every consumer-GPU partition here, which is what makes a wide fallback list possible; pro6000 and
a100 sit behind their own.

| `TIER=` | Partitions | GPU | Batch | Note |
|---|---|---|---|---|
| `a6000` | suma_a6000, gigabyte_a6000, tyan_a6000, asus_6000ada | 48 GB | 1024 | ~20 nodes; the default when `ROLLOUT=1` |
| `wide` | the above + 3090/4090 | 48/24 GB | **256** | the default when `ROLLOUT=0`; 379 GPUs instead of 106 |
| `ampere` | 3090 / 4090 / A5000 | 24 GB | **256** | most nodes, shortest queue |
| `pro6000` | asus_pro6000, gigabyte_pro6000 | 96 GB | 1024 | most headroom, only 4 nodes |
| `a100` | suma_a100 | 80 GB | 1024 | own QOS |

`TIER` defaults to `a6000` when `ROLLOUT=1` (a resident 3 B VLA does not fit beside the data on
24 GB) and `wide` otherwise. `BATCH` follows the tier and is fixed for the **whole sweep** — never
per node. That matters: a run's arithmetic has to be settled when it is submitted, or the same
manifest line means different things depending on where it landed, and the ablation ends up
comparing runs that were never the same experiment. Override either explicitly.

### Batch size vs. the card (measured, RTX 3090 24576 MiB, ARQ baseline, `ROLLOUT=0`)

| batch | outcome |
|---|---|
| 1024 | **OOM** — one 9.00 GiB tensor on top of 5.94 GB resident data |
| 512 | **OOM** — asks for **10.69 GiB**, *larger* than at 1024 |
| 256 | peak **22344** of 24576 MiB, ~3 it/s, completes |

Peak memory is **not monotonic in batch size**: XLA's rematerialization pass fires at 1024 and not at
512, so the smaller batch requests the bigger buffer. Do not interpolate — measure the setting you
intend to run. Every job records its own peak to `gpu.json`, so the table extends itself.

The ARQ baseline (`mg2`, all 16 candidates) is the memory ceiling of the sweep — `qc`, `mg4`, `mg8`
and the `bc*` arms all shrink `B·P·N`. If the baseline fits, the rest do. But 22344 of 24576 MiB is
only ~1.4 GB of slack after the 0.92 `MEM_FRACTION` cap, so if a `wide` run OOMs, drop to
`BATCH=128` for the whole sweep rather than for one variant.

## Watch and collect

```bash
squeue -u $USER
tail -f /lustre/jellyho/acrft/logs/critic_abl_<jobid>_1.out
uv run slurm/collect.py critic_abl --csv abl.csv
```

`collect.py` prints `critic%` / `vla%` (the last rollout eval) next to the within-state diagnostics.
Read `act_sens` and `rank_cand` first: a critic that has collapsed to `Q(z,a) = V(z)` still scores
well on TD loss and on Q-vs-return while ranking candidates at chance, so a good-looking training
curve means nothing on its own.

## Known rough edges

- **No resume.** `train_rlt_critic.py` has no resume path, so jobs are submitted *without*
  `--requeue` — a preempted job would silently restart at step 0. If a node dies, resubmit that
  variant by hand (the manifest line is in `$CRITIC_RUNS/<sweep>/manifest.tsv`).
- **Rollout eval cost.** Each one runs `ROLLOUT_TRIALS` trials twice (critic and VLA) in MuJoCo with
  the VLA in the loop. At the default `ROLLOUT_EVERY=50000` that is 4 evals per run; lower it only if
  you have the wall-clock.
- **`LD_LIBRARY_PATH` surgery in `env.sh`.** `~/.bashrc` puts `~/miniconda3/lib` on the path, and its
  `libcrypto.so.3` is older than the `OPENSSL_3.4.0` the system python3.11 links `_hashlib` against.
  `.venv` is built from that interpreter, so leaving it in front breaks `import hashlib` — at
  `uv sync` time and in every job. `env.sh` drops that one entry and keeps CUDA/MuJoCo. If you run
  the stage-3 scripts by hand without sourcing `env.sh`, you will hit this.
- **All 514 episodes in `acrft-annot-noprop` succeed.** There are no failure episodes, so `mc_return`
  is never 0 on a live frame and every discrimination the critic can make is about *timing*. Combined
  with the discount table above, that is worth keeping in mind when reading the diagnostics.
