# Reactive-Commitment Value (RCV): making value evaluation credit commitment

**Problem.** Plain temporal-difference value learning structurally prefers short execution
(re-plan every step). We want a value-evaluation framework in which longer commitment can be *strictly*
preferred when it should be — carrying information through an occlusion, or letting a contraction absorb
policy error — instead of TD's silent bias toward reacting.

This note states *why* Markov TD prefers short (an impossibility theorem), *when* longer strictly wins
(a sign-reversal theorem that requires non-Markov value), and *the value-evaluation object* whose four
corrections make the estimated advantage's sign track the true one. The corridor experiment
`slurm/probes/toy_tunnel_fork.py` is the pre-registered test.

---

## 1. Setup

Base MDP `M = (S, A, T, r, γ)`. The agent sees observations `o = O(s)` through a channel `O` that may
be many-to-one (occlusion ⇒ non-Markov). A **chunk policy** `π` proposes `H` actions at a decision
point; a **selector** `κ` chooses a commitment length `k ∈ {1..H}`; after `k` open-loop steps the
policy is re-queried. `h_t = (o_0, a_0, …, o_t)` is the history.

Two value objects on the same decision point `(s, h)`:

- **Reactive value** `V_react(s,h)` — value of the deployment process that re-queries every step
  (`κ ≡ 1`).
- **Committed value** for length `k`, executing the *proposed* chunk `a_{1:k}` open-loop:

  ```
  V_commit^k(s,h,a_{1:H}) = Σ_{j=0}^{k-1} γ^j r(s_j, a_j)  +  γ^k · V_react(s_k, h_k)
  ```

- **Reactive advantage** `A(s,h,k) = V_commit^k(s,h) − V_react(s,h)` — the selection signal. The
  deployment rule is the event-triggered / lexicographic one: **commit the longest `k` with
  `A(s,h,k) ≥ −ε`**.

---

## 2. Theorem 1 — Markov impossibility (why plain TD prefers short)

**Hypotheses.** (i) full observability: `O` is injective on the reachable support, so `V_react`,
`V_commit`, `π` may be taken as functions of `s` alone; (ii) the value is the true value of the
deployment process.

**Claim.** For every reachable `s` and every `k`, `A(s,k) ≤ 0`, with equality iff the committed
actions coincide with what re-querying would have chosen along the rollout.

**Proof.** Re-querying at every step is the *interruption* of the committed option: at each intermediate
state the agent may switch to a weakly better continuation. Under full observability the re-query has
access to at least the information the commitment used, so by the options interruption theorem
(Sutton–Precup–Singh 1999) interrupting is weakly better: `V_react(s) ≥ V_commit^k(s)`, i.e.
`A ≤ 0`. Equality holds exactly when interruption never triggers — the committed chunk already equals
the re-queried one (a consistent optimal policy). ∎

**Corollary (short-preference).** Under Theorem 1 the value-maximizing length is `k*=1` weakly, and
strictly whenever either the policy is imperfect (`ε_π > 0`, so the committed actions differ from the
re-queried ones — the interruption strictly fires) or the bootstrap carries free-requery optimism
`δ > 0` (`V_react` overstates the value of landing at a re-query point). **No `k > 1` is ever strictly
preferred.** A Markov value function can, at best, tie on long horizons.

This is exactly the empirical picture measured on the YAM critic: with the true value all `k` tie to
`5·10⁻¹³`; the observed short-preference (`k* = 8.8`, `Vslope = 0.72`) is the corollary's `ε_π + δ`
terms, and the `V(s₀)` success/failure gap of `986` is `δ` (free-requery optimism) made visible.

**Consequence for the framework.** To make a longer horizon *strictly* preferred, the value evaluation
**must leave the Markov class** — no amount of re-weighting a state-value can do it. History
conditioning is therefore not a convenience; it is necessary.

---

## 3. Theorem 2 — non-Markov sign reversal (when longer strictly wins)

**Hypotheses.** `O` is non-injective: there are reachable `s_A ≠ s_B` with `O(s_A)=O(s_B)=o` whose
optimal continuations differ. There is a pre-occlusion state `s_0`, reachable before the collision,
at which `O(s_0)` determines the branch.

**Claim.** There exist a length `k` and history `h` with `A(s_0,h,k) > 0` — committing strictly beats
reacting.

**Proof.** The chunk proposed at `s_0` (branch observed) carries the correct continuation open-loop
through the occluded region `[j,k)`. The reactive arm, upon reaching `o`, must decide on the coarser
σ-algebra `σ(O)`; by Blackwell's theorem a decision on a garbled channel is weakly worse, strictly when
the two branches' optimal actions differ. The committed arm forfeits reaction but keeps the
pre-occlusion information; when the occlusion spans the window, the committed value exceeds the
reactive one. ∎

Room B of `nonmarkov-longer`. The imitation-learning name for the reactive failure is
copycat/causal-confusion: per-step re-inference on partial observations collapses to shortcuts, which
committing removes.

**Theorem 3 (contraction reversal, stated).** If the closed-loop dynamics under commitment are
`(C,ρ)`-EISS (`ρ<1`) and per-step policy error `ε_π` re-enters at each re-query, there is a length
`ℓ > log poly(L_π,C)/log(1/ρ)` above which `A>0` (Zhang et al. 2507.09061, Prop 3.1) — frequent
re-planning re-injects error the contraction would otherwise absorb. This holds under Markov
*dynamics*, but the value must see the accumulated error; a stationary `V(s)` with approximation error
does not, which is why the estimator, not just the environment, must change.

---

## 4. The framework: four corrections, each killing one bias

`A` is only useful if the *estimated* `Â` has the same sign as the true `A`. Four corrections make it
so; each is necessary for one failure, and Theorem 1 says the first is load-bearing.

| correction | removes | without it |
|---|---|---|
| **H — history / latent conditioning** `V(s,h)` | Markov collapse | Theorem 1 forces `Â ≤ 0` everywhere → selector can never choose long → degenerates to plain TD |
| **O — open-loop committed target** (roll the *proposed* chunk, bootstrap at the boundary) | "committed value" that isn't about committing | the window is scored as if reacting; `V_commit` = `V_react` by construction |
| **S — synthetic backup** (build the in-window value from 1-step backups through *observed* intermediate states, not from chunk-conditioned returns) | chunk-regression confound = DQC hindsight leakage | events revealed inside the window that jointly cause the demo action and the outcome make long spuriously optimistic |
| **P — policy-expectation bootstrap** `V_react(s,h) = E_{a∼π_deploy}[Q(s,h,a,κ)]` | free-requery optimism `δ` | landing at a re-query point is scored by demo returns (which had the latent) not by what the *deployment* proposer can do there → short spuriously optimistic |

**H, O make long representable; S, P remove the two spurious biases.** What remains in `Â` is the true
tradeoff: branching/aleatoric pushes short (Thm 1 regime), occlusion/contraction pushes long
(Thm 2/3). This is CFAC's four clauses re-derived from the impossibility theorem, with the new content
being (a) Theorem 1 as the *reason* H is mandatory, (b) `A(s,h,k)` promoted to a first-class learned
signal, and (c) its sign law unifying the two "longer wins" rooms.

---

## 5. Falsifiable predictions (pre-registered for the toy)

Corridor with straights `S`, occluded tunnels `T` (a plan seen only at the entrance must be carried
blind), a delayed-response cell `R` (act on a signal seen earlier, now hidden), and a stochastic fork
`F` (branch revealed only on arrival). `H = 8`; tunnels of length 5 and 3 so the optimal commit is
*graded*, not just "max". Oracle `κ*`: commit each tunnel to its exit, stop at the fork.

- **P1 — Markov impossibility, measured.** For every Markov arm (input = observation), the estimated
  advantage `Â(s,·)` at every tunnel-entrance state is `≤ 0`: it cannot strictly prefer the long
  commit. Only history arms produce `Â > 0` there. *Reject if a Markov arm yields `Â > 0` at an
  entrance.*
- **P2 — necessity.** No single correction recovers the oracle; the full RCV arm (H+O+S+P) does.
  The factorial says which corrections are load-bearing in this environment.
- **P3 — graded commit.** The full arm commits ≈ each tunnel's own length (5 and 3), not a constant.
- **P4 — fixed-k is non-monotone.** No fixed commitment length is within `ε` of RCV on all zones;
  the best fixed `k` differs by zone. *Reject if some fixed `k` ties RCV everywhere.*
- **P5 — stopping.** RCV stops at the fork (mean commit ≈ 1 there) while committing the tunnels;
  a long-biased arm over-commits through the fork and pays the branch.

Classification is programmatic; 8 seeds; the table is recomputed from the run's JSON.
