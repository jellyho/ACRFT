# Policy-extraction arms — how to run each one

Ten ways to turn a frozen π₀.₅ BC policy plus a frozen patch critic into a better policy, and the
command for each. They exist to be compared, so the point of this page is that the commands differ
only where the *method* differs.

Two families, and which one an arm belongs to decides how it is served:

| | how it is served |
|---|---|
| **weight-only** — `awr`, `flowdpg`, `qam`, `dql`, `fqlx`, `cfgrl` | the trainer produced weights; export them and serve an ordinary checkpoint |
| **critic-consuming** — `bon`, `implicit`, `qpilots`, `lps`/`lpsd`, `flowdagger` | no policy of their own; they are *modes of the critic wrapper* and act at inference |

There is **one** serving entry point, `scripts/serve_policy.py`. There is no
`serve_extraction_arm.py` — it existed briefly and was reverted (`8889930`) precisely so that
there would be one.

### Where an arm's code lives

An arm that changes **how a chunk is drawn** is a property of the policy, so it lives with the
policy — `qpilots` is `Pi0Steered(Pi0)` in `src/openpi/models/pi0_steered.py`, which is `Pi0` plus
one sampler and no parameters of its own, so it can wrap a checkpoint trained as a plain `Pi0`.

The critic is **injected** into it as `value_fn(a_hat) -> scalar`. That is what keeps the model
from learning that a critic exists, and it is why one critic can score a BC, an α-Flow or an RLT
base without any of them knowing about it. It is also what makes the sampler testable: with the
critic hard-wired, "did steering move the chunk the right way" was not answerable without a trained
checkpoint and a DINOv2 backbone, so nothing asserted it.

The layers, and what each is allowed to know:

| layer | knows |
|---|---|
| `models/pi0_steered.py` | how to integrate a flow with a value gradient. Not what the value is. |
| `extraction/serving.py` | which arms exist, what each one's value is, how to load their heads |
| `policies/patch_critic_policy.py` | features, scoring, selection, robot-space decode — **no arm names** |

That last row is enforced by a test: `patch_critic_policy.py` contains no arm name at all. It asks
the arm registry (`offers_unsteered_twin`, `SAMPLER_ARMS`) instead, so a second steering arm is a
line in `serving.py` rather than an edit spread across the serving stack.

---

## Weight-only arms

```bash
# 1. export the trainer's {"expert": ...} subtree onto the BC parameters
uv run python scripts/export_extraction_checkpoint.py --arm dql        # awr | flowdpg | qam | dql | fqlx

# 2. serve it like any other checkpoint — the critic is not involved at inference
uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi \
    --policy.dir <exported checkpoint>
```

**`cfgrl` is the exception**: its guidance weight lives in the *model config*, not in a serving
flag, so it needs its own config.

```bash
uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_cfgrl \
    --policy.dir <exported checkpoint>
```

`with_cfgrl(base, cfg_w=...)` builds that variant for any π₀.₅ task config, so the weight is a
property of the run rather than of the command that serves it.

---

## Critic-consuming arms

All of these take `--critic <critic dir>` and differ only in `--critic-mode`.

```bash
CRITIC=~/hf_utils_downloads/acrft-yam-critics/patch_critic_yam_s347_fixed_tau9_min_200k
BASE="policy:checkpoint --policy.config pi05_yam_lego_taxi --policy.dir <bc checkpoint>"
```

```bash
# best-of-N — draw N chunks, execute the argmax of min-ensemble Q.
# This is ALSO IDQL's argmax rule, so an IDQL run is bon labelled by its N.
uv run scripts/serve_policy.py --port 8000 --critic $CRITIC --critic-mode bon --num-samples 8 $BASE

# IDQL implicit policy — draw one with probability proportional to the expectile weights of its
# advantage instead of taking the best. Trades value for less critic exploitation.
# The expectile is read off the critic artifact; it is not a serving flag.
uv run scripts/serve_policy.py --port 8000 --critic $CRITIC --critic-mode implicit --num-samples 64 $BASE

# QPILOTS-U — no weights at all: steer the BC sampler with the critic gradient at every Euler step.
uv run scripts/serve_policy.py --port 8000 --critic $CRITIC --critic-mode qpilots --steer-alpha 0.2 $BASE

# latent-actor arms — a small MLP picks the sampler's latent
uv run scripts/serve_policy.py --port 8000 --critic $CRITIC --critic-mode lpsd \
    --extraction-head <latent_actor_*.msgpack> $BASE

# flowdagger — a head predicts the DCT coefficients of the sampler's initial noise.
# --extraction-head is a DIRECTORY here, not a file: the head is meaningless without the DCT basis
# it was fitted against, so the two travel together or the coefficients decode into a different
# chunk. (lps/lpsd point at a single file for the same reason inverted -- they store one array set.)
uv run scripts/serve_policy.py --port 8000 --critic $CRITIC --critic-mode flowdagger \
    --extraction-head <run dir with steering_head.msgpack + dct_basis.npy> $BASE
```

### Measuring how far steering moved the policy

```bash
uv run scripts/serve_policy.py --port 8000 --critic $CRITIC \
    --critic-mode qpilots --steer-alpha 0.2 --drift-samples 8 $BASE
```

`--drift-samples N` records, alongside the chunk that actually executes, **the unsteered twin**
(same noise, same observation, alpha = 0) and **N unconditional draws**. They are references,
never candidates — the arm's chunk is what runs, and the critic scores the set rather than
choosing within it.

Two questions, two references, and the second is what makes the first mean anything:

| | |
|---|---|
| executed vs twin | how far steering displaced *this* draw — same noise, so the difference is the steering term and nothing else |
| the unconditional spread | how wide the policy's own distribution is at this state |

A displacement of 0.1 rad is small inside a spread of 0.5 and enormous inside 0.02. `misc/yam-misc
stats` reports both, plus `drift/spread`, and the renderer draws the twin and the fan so the
question "did steering leave the distribution or move inside it" is visible rather than inferred.

Comparing the steered chunk against an *independently drawn* sample instead would measure sampling
variance and steering mixed together, which is why the twin shares the draw.

`--num-steps` sets the denoising iterations, and with a critic attached it is charged **per
candidate** — the difference between BoN-8 costing 8 suffix passes and 80.

---

## Three things that are easy to get wrong

**1. `adaptive` needs a critic with `macro_group_size < horizon`.** The ring currently trains
against `patch_critic_yam_s347_fixed_tau9_min_200k`, whose `macro_group_size` is 30 — one
commitment group, so `adaptive` there *is* `bon` under another name. The server warns, but the run
still produces perfectly plausible numbers, which is how an "adaptive vs bon" comparison ends up
being bon twice. Adaptive needs a multi-group critic such as `..._g5_tau9_min`.

**2. There is no `idql` mode.** It was an alias for `bon` and was removed: two names for one code
path is how they drift apart. An IDQL argmax run is `--critic-mode bon --num-samples 64`.

**3. The arms optimize the LAST prefix only** (full-chunk Q). Serving a weight-only arm
*adaptively* therefore executes it under a commitment rule its objective never saw. That is a
legitimate experiment, but it is not a default and a report should say when it was done.

---

## What belongs in the report

**Which critic a run used, not just the command.** Two critics that differ only in how their
proprio was normalized give `grad_a Q` directions that disagree by cosine 0.85 on average and
−0.69 at worst — and Q stays in range the whole time, so nothing in a rollout video shows it. That
provenance is what let this ring tell a real bug from noise; it cost nine trained arms to learn.

For reference, the fixed critic's advantage distribution over the annotation set is
mean −6.79, std 61.0, fraction positive **0.204**. Under the raw-proprio bug it was −15.09 and
**27.7%**. A deploy run whose advantage fraction comes back near 0.28 is a signal that the
raw-proprio path has crept back in somewhere, not a result.

## Offline evaluation

`scripts/eval_extraction.py --arm <name>` runs the paired offline comparison — same states, same
noise seeds, base sampling vs the arm — which is what makes its deltas comparable across arms.
`ArmSpec` (`src/openpi/extraction/serving.py`) is the record of which checkpoint and which
hyperparameters make an arm reproducible; keep it accurate or the comparison is between two things
you can no longer name.
