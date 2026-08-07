# Model-based adaptive chunking (MB-AC) — working notes

Overnight design thread, started 2026-08-08 ~01:40. Hourly iterations until morning.
Goal: turn "model based adaptive chunking RL" from a phrase into a mechanism with
measured justification, using artifacts we already have.

## What adaptive chunking actually decides

At each replan the policy produces a 16-action chunk. Two coupled decisions:

1. **Selection** — which chunk to execute (BoN over N samples).
2. **Commitment** — how many actions k of it to execute before replanning.

Everything so far (critic BoN, prefix critic) attacks selection. Commitment is
still fixed-k. The model-based angle is that a dynamics model is the natural
authority on *commitment*: it knows how far into the future its own predictions
(and hence the plan's preconditions) remain trustworthy.

## Grounding: what we've measured

| Fact | Source | Design consequence |
|---|---|---|
| Dyn v1 beats copy-forward only from k≥8 (R² .30/.43/.52 @8/12/16) | cheapz_dyn_v1/report.json | model-based scores are informative only at LONG commit horizons; don't use the model for 4-step decisions |
| AUSE .09–.11 → uncertainty ranks its own error well | same | σ-triggered cutting is viable |
| Ensemble disagreement separates demo vs perturbed chunks (AUC .58), V(d(z,a)) flat | eval_modelbased_scorer | disagreement is a *behaviour-consistency* signal, not a value signal — use it as a veto/commit signal, not for ranking |
| Critic is action-blind under plain IQL; fixed by Cal-QL+swap (binding .995) | critic_swap eval | selection can stay with the critic; the model doesn't need to replace Q |
| φ = local ruler (Spearman .885 ≤50 steps) + far-field compass | value-accuracy anatomy | V(φ)-based scores are meaningful over exactly the 16–50-step range a chunk covers |
| worker-B: 23 critic configs, only mild IQL non-harmful in rollouts | Space master report | authority handed to learned components must be *conservative*; default to VLA, intervene on confident signals only |

## Proposal v0: "commit while the model agrees"

At each replan with candidates {a_i}:

- **Selection** (unchanged): Cal-QL+swap critic picks a* = argmax Q(φ(z), a_i).
- **Commitment** (new): roll the φ-space dynamics ensemble along a*;
  k* = largest macro-step k with cumulative ensemble disagreement Σσ ≤ τ.
  Execute k*·stride actions, then replan.
- Conservative floor/ceiling: k* ∈ [4, 16]; τ calibrated offline so the median
  k* ≈ 8 (i.e., adaptivity redistributes replans, doesn't add compute).

Why this split: selection needs a *value* signal (critic's job, now that binding
works); commitment needs a *trust-region* signal (dynamics' job, per AUSE). The
AUC-.58 disagreement that failed as a ranker is exactly a "the data has never
continued this way from here" alarm — the right trigger for replanning.

## Proposal v1 (RL): commitment as a learned policy

Frame k as an action: π_k(k | φ(z), a*, σ_{1..4}) trained with offline RL where
reward = Δprogress − c·1[replan]. The offline dataset already contains the
counterfactuals: for every state we know the demo's future, so the advantage of
cutting at k vs k' is computable from d_φ progress. This is small-scale (input
is 128-d φ + 4 σ's), trainable in minutes. But v0 must be measured first — if
σ-threshold cutting already captures most of the oracle gain, RL adds risk for
nothing (worker-B's table is a warning about exactly this).

## Overnight experiment ladder

- **E0 (running, job 34635)**: train dyn-v1 architecture on φ space
  (`.scratch/phi_dyn_v1`). 128-d targets instead of 256-d DINO; check R²/AUSE.
  Everything below runs in φ space so the ruler/compass anatomy applies.
- **E1 offline cut-point value**: along held-out episodes, at each anchor roll
  the ensemble down the *demo's own* chunk; compute σ_k and true error e_k.
  Questions: (a) does e_k grow superlinearly past the σ-cut? (b) do σ-cut
  points coincide with progress plateaus / junctions (grasp, place moments)?
  (c) oracle: progress-per-step of adaptive-k vs fixed-k ∈ {4,8,16}.
- **E2 binding through the model**: demo chunk vs jnp.roll'd other-state demo
  chunk (same negative as --cql-swap), scored by V_φ(ẑ_k*) = −d_φ(ẑ, goal-set)
  and by disagreement. Does the *model* achieve binding without any CQL? If
  yes, model-based scoring is a CQL-free alternative selection path; if no,
  the selection/commitment split above is confirmed.
- **E3 horizon of visibility**: separation of good-vs-perturbed chunk scores as
  a function of rollout depth k — the minimum commit horizon at which
  model-based ranking is informative. Expect ≥2 macro-steps given the R² curve.

## Open questions for later iterations

- Replan cost model: on L40S, one VLA replan ≈ 10 flow steps ≈ ~0.4 s vs 16
  control steps ≈ 0.8 s sim time. What's the actual wall-clock/latency budget
  argument for adaptive k on real robots (YAM)?
- Interaction with prefix-argmax concentration: if commitment comes from the
  dynamics, the prefix mode's known 0.88-shortest-chunk bias becomes moot —
  adaptive commitment *replaces* prefix selection rather than fixing it.
- Distribution shift: σ thresholds calibrated on demos will fire constantly
  off-distribution (that's desired — replan more when lost) but must not
  thrash at k=4 permanently; hysteresis or a σ-EMA may be needed.
- Where this meets AC-RFT: the RFT loop can fine-tune the VLA on its own
  rollouts *with* adaptive commitment in the loop, making chunk-length a
  first-class part of the behaviour being reinforced.
