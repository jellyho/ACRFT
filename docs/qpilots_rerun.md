# QPILOTS re-run: is the collapse the critic's direction, or the injection's size?

The original sweep cannot answer that, for two independent reasons.

**1. It differentiated a two-sample standard deviation.** The steering follows `_q(reduce="pess")`,
which was `mean − 0.5·std` over **K=2** members. At K=2, `std` is `|q1−q2|/2` — a single-sample
estimate with ~75% relative sampling error. Measured on 64 on-manifold states, that term carried
**59.6** of gradient magnitude against **123.5** from the mean, in a direction **near-orthogonal**
to it (cos −0.078). A third of every steering step was noise. The reduction and `rho=0.5` came from
QAM (`agents/qam.py:33`), which sizes its ensemble for it at `num_qs=10` (`qam.py:424`).

**2. It was a magnitude sweep, and the result was a threshold, not a dose-response.** At
α = 0, .005, .01, .025, .05, .1 the means were 1.80, 1.10, 1.30, 1.00, 1.20, 0.30 — two rank
inversions, the middle four mutually indistinguishable (Kruskal p=0.45) across an 8× range of
injected displacement (2/3/8/17% of ‖a‖), and only the largest (34%) collapsing. **A threshold is
direction-agnostic.**

## What changed in the code

- `serving.py` now **refuses** the `mean − rho·std` read below `MIN_MEMBERS_FOR_PESSIMISM = 10` and
  falls back to the ensemble **mean**, warning once and naming K. So this re-run steers along a
  different value function than the original sweep — that is the point, not a confound.
  The ranking arms are untouched: `bon` reduces by `min`.
- `--steer-value {critic,negated,random}` selects **what** is ascended. Eq. 17 rescales every
  gradient to the drift norm, so all three inject the **same displacement magnitude** at the same α
  and differ only in direction (pinned by test: scaling the value function by 1e4 moves the
  displacement <0.1%).
- `random` is one fixed unit direction per chunk, supported on exactly the sub-array the critic
  reads. Coherent, not redrawn per Euler step: a per-step redraw random-walks and partly cancels,
  which would understate a systematically wrong direction.

## The runs — 3 conditions × 10 episodes = 30

Serve from this branch (it carries both the gate and the control arms):

```bash
uv run scripts/serve_policy.py \
  --policy.config pi05_yam_lego_taxi \
  --policy.dir /data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/100000 \
  --critic /data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k \
  --critic-mode qpilots --steer-alpha 0.1 --steer-value critic     # then: random, then negated
```

α is fixed at **0.1** — the only point that collapsed, so it is the only one where there is an
effect to attribute. The existing α=0 result (**1.80**, same critic, same policy, same block) is the
control; it does not need re-running, because at α=0 the steering term is multiplied by zero and the
`rho` gate cannot change it.

**Check the first log line of each server start.** It now prints the arriving image geometry.
`[(480, 640), ...]` is the contract; an already-square arrival warns and should be reported.

## Pre-registered readout

| observation at α=0.1 | conclusion | what follows |
|---|---|---|
| `critic` recovers toward 1.80 | the std term was the defect | steering is viable; sweep α again on the clean read |
| `critic` ≈ 0.30 **and** `random` ≈ 0.30 | **magnitude**, direction-agnostic | bound the displacement; the critic's gradient is not implicated |
| `critic` ≈ 0.30, `random` unharmed | the critic's **gradient direction** is bad | fix the critic before any gradient arm |
| `negated` **beats** `critic` | the gradient is **anti-correlated** with quality | worst case, most informative |

Also record, per condition, the realized displacement from the unsteered twin (`--drift-samples`).
The arms are magnitude-matched at **injection**, not at the output: `sample_steered` returns
`clip(x, −1, 1)`, and where a coordinate sits on that boundary an outward push is truncated and an
inward one is not — ~1.6× across directions in the test fixture. Do not assume α fixes it; measure.

## Not part of this run

Nothing else needs redoing. The BC execution-length sweep, the selection-rule block, the adaptive
commitment block and every offline measurement stand. The image contract was checked against the
rollout recording and found to match the cache (`image-contract` entry).
