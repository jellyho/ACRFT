# Delayed-observation history-aware critic (RCV on a frozen short-context VLA)

Design for putting the Reactive-Commitment Value framework
(`docs/reactive_commitment_value.md`) onto the YAM patch-critic, under the fixed premise that the VLA
policy is **short-context** (Zeng/Tedrake 2608.15938, Lazzati/Metelli/Levine 2608.02547 both prescribe
a long-context policy; we cannot pay that in a deployed generative VLA). The move: memory lives in the
**critic**, which is offline and latency-free, not in the policy.

## What changes vs the current critic

The current stack has two problems the theory names:

1. **`Q` is chunk-conditioned and fit by (near-)return regression** → the belief-shift leak (Lemma 1).
   Measured: `V(s0)` success/failure gap of 986 at a frame where the observation is identical.
2. **`V` (PatchV) is fit to that leaky `Q`** by expectile → it inherits the leak, so it is not an
   honest observation value.

The redesign is three heads, each with the corrections that make its belief honest:

| head | conditions on | trained by | role |
|---|---|---|---|
| `V_react(o)` | current observation only, **no chunk** | 1-step synthetic backup `V(o_t) ← r_t + γ V(o_{t+1})` on cache transitions | honest observation-belief value (Theorem 4) |
| `Q_commit(o, o_{t−Δ}, a_{1:k})` | current + **delayed** observation(s) + chunk | window built by 1-step synthetic backups through observed intermediate frames, bootstrap `V_react` at the boundary | the value of committing, with `I_plan` carried by the delayed obs |
| (selector, no head) | — | — | `A(k) = Q_commit − V_react`; commit the longest `k` with `A ≥ −ε` |

`V_react` is the load-bearing new object: a **chunk-free** value trained by **1-step** obs-TD, which
Theorem 4 proves is unbiased for `E_{b(s|o)}[·]`. The current PatchV cannot be reused — it is fit to the
leaky Q.

## The delayed observation

`o_{t−Δ}` carries the information the current frame lost — the cheap history of Lazzati et al.
(`π(a_t | o_{t−Δ})`), here on the critic side. From the feature cache it is free: frames are stored
per-episode contiguous, so `o_{t−Δ} = features[offset + max(0, pos − Δ)]`; no re-caching. Architecture:
the critic already takes patch tokens as leading context (`PatchARQCritic`), so a delayed frame is just
another block of `192` leading tokens with a distinct "delay" type-embedding. A small stack
`Δ ∈ {δ, 2δ, …}` approximates a short history; `Δ = 0` recovers the Markov critic (ablation).

Only `Q_commit` sees the delay. `V_react` stays observation-only on purpose — it is the value of what
re-planning actually has (the current observation), and its impoverishment relative to `Q_commit` is
the reactive advantage.

## Training

Both by 1-step synthetic backup on the cache, cost_to_goal reward analytic (as now), homing truncated,
pi05-normalized inputs. Disjoint from the current per-prefix ARQ trainer — `V_react` is a single scalar
(distributional) value net; `Q_commit` is per-prefix but its window backup marginalizes the chunk
(never regresses the chunk-return). Ensemble `K` for the epistemic split (`uncertainty-split`).

## Diagnostic first (measure_reactive_map)

Before training the honest heads, `scripts/measure_reactive_map.py` reports the leak on the CURRENT
critic as `Q_reg − Q_syn` per frame, where `Q_reg` is the trained chunk-conditioned `Q(o, a_{1:k})` and
`Q_syn` is the synthetic reconstruction `Σ γ^j r_j + γ^k V(o_{t+k})` from the analytic reward and the
critic's boundary `V`. This is a **window-level** leak proxy (the boundary `V` is still the leaky one),
honestly labelled as such, and it is computable now with no new training. It answers ACRFT-D's question
— is the measured `k*` success/failure split a real reactive-map or the leak? — by checking whether the
`k*` map overlaps the `Q_reg − Q_syn` map. Predictions: the leak is large at contact/alignment
(aleatoric peak) and at the first frame (intent hidden), small in free-space transport; and it should
correlate with paper 2's policy-side `ΔL_val` computed on the same episodes.

The rigorous version (v2) uses the trained chunk-free `V_react` as the boundary, isolating the full leak
rather than the window leak.
