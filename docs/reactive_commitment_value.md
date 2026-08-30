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

---

## 6. Why it is learnable: the leak is a posterior shift, the fix pins the belief

The re-plan "cost" is not a cost. It is the value of the information re-planning discards by
re-conditioning on the current observation. This section proves (a) that the observed critic leak is a
posterior shift caused by conditioning the value on the executed chunk, (b) that a synthetic (1-step,
marginalized) backup pins the belief and removes it, and (c) that the reactive advantage learned from
two such honest values is exactly a value-of-information functional whose sign is Blackwell's.

**Setup.** POMDP with latent `s`, observation `o = O(s)` (many-to-one), behavior policy `β(a_{1:H} | s)`
that conditions on the latent (a closed-loop teleoperator). `b(s | o)` is the belief induced by the data
visitation. `V_exec(s, a_{1:k})` is the true return of executing the chunk `a_{1:k}` open-loop from `s`.

### Lemma 1 (the leak is a posterior shift by β)

The chunk-return regression estimator
```
Q_reg(o, a_{1:k}) = E_data[ G | o, chunk = a_{1:k} ] = E_{s ~ b(s | o, chunk=a)} [ V_exec(s, a_{1:k}) ]
```
uses the Bayes-shifted belief
```
b(s | o, chunk=a) ∝ b(s | o) · β(a | s).
```
When `β(a | s)` depends on `s` (a closed-loop demo chooses the chunk from the state), the belief is
reweighted toward states where `a` was likely — typically states where `a` succeeds — so `Q_reg` is
optimistic by the mutual information `I(s ; chunk | o)`. This is the identity behind the measured
`V(s0)` gap of 986: at the first frame success and failure share the observation, but the demo chunk's
style correlates with the outcome, so conditioning on it moves the belief and splits the value.

### Theorem 4 (synthetic backup pins the belief at b(s|o))

Define the synthetic estimator by the 1-step recursion, marginalizing the next observation over the
current `(o, a_1)` only:
```
Q_syn(o, a_{1:k}) = E[r | o, a_1] + γ · E_{o' ~ P(o' | o, a_1)} [ Q_syn(o', a_{2:k}) ],   base Q_syn(o, ∅)=V(o).
```
Then for all `k`,
```
Q_syn(o, a_{1:k}) = E_{s ~ b(s | o)} [ V_exec(s, a_{1:k}) ]   — the belief is b(s|o), NOT reweighted by the chunk.
```

*Proof (induction on k).* Base `k=0`: `Q_syn(o,∅) = V(o) = E_{s~b(s|o)}[V_exec(s,∅)]` by definition of
the observation value. Inductive step: assume it holds for `k-1`. The data transition kernel out of
`(o, a_1)` marginalizes the latent over the belief:
```
P(o' | o, a_1) = Σ_s b(s | o) · P(o' | s, a_1),
```
because the data at observation `o` visits latents `s ~ b(·|o)` and the successor observation depends
only on `(s, a_1)` — crucially NOT on `a_{2:k}`, since those actions have not been taken yet and cannot
enter the one-step kernel. Hence
```
Q_syn(o, a_{1:k}) = E[r|o,a_1] + γ Σ_{o'} P(o'|o,a_1) Q_syn(o', a_{2:k})
                 = Σ_s b(s|o) [ E[r|s,a_1] + γ Σ_{o'} P(o'|s,a_1) · E_{s'~b(·|o')}[V_exec(s', a_{2:k})] ].
```
The inner bracket is `E_{s'~b(·|o')}[...]`; but the successor state `s'` reached from `s` under `a_1` is
distributed `P(s'|s,a_1)`, and averaging `V_exec(s', a_{2:k})` over the successor is exactly
`V_exec(s, a_{1:k})` unrolled one step. Because every backup conditions on the CURRENT `(o, a_j)` and
never on the remaining chunk, no step reweights the belief by `β(future chunk | s)`. Collecting terms,
`Q_syn(o, a_{1:k}) = E_{s~b(s|o)}[V_exec(s, a_{1:k})]`. ∎

**Corollary (no leak).** `Q_syn − Q_reg = E_{b(s|o)}[V_exec] − E_{b(s|o,chunk)}[V_exec]`, the exact bias
of Lemma 1. Synthetic backup is unbiased for the observation-belief value; chunk-return regression is
biased by `I(s ; chunk | o)`. The leak is removed not by TD-vs-MC but by refusing to condition the tail
on the future chunk.

### Theorem 5 (the reactive advantage is a value-of-information, signed by Blackwell)

Evaluate two values, both by the honest synthetic backup, differing only in the ACTION SOURCE at the
decision point:
```
V_react(o)         = Q_syn(o, a^π),   a^π ~ π(· | o)         — the observation-conditioned proposal
Q_commit(o,a_{1:k}) = Q_syn(o, a_{1:k})                       — a specific plan drawn earlier with h_{t0}
```
Then the reactive advantage is a value-of-information functional over the observation belief:
```
A(k) = Q_commit − V_react = E_{s ~ b(s|o)} [ V_exec(s, a_{1:k}^plan) − V_exec(s, a^π(o)) ].
```
Both terms integrate the SAME belief `b(s|o)`; they differ only in whether the action was chosen with
the plan's information `I_plan` (the chunk drawn when the branch was observable) or the current
observation's `I_now`. Let `I_now(t) = σ(o_t)` be the information the re-planning policy conditions on
along the window, and `I_plan = σ(h_{t0})` the information the plan was drawn with. Then:

- If `I_now(t) ⊇ I_plan` for all `t` in the window (full observability / a sufficient current
  observation), the observation-conditioned proposal can reproduce the plan's action, so
  `V_exec(s, a^π) ≥ V_exec(s, a_plan)` in expectation and `A(k) ≤ 0` — reacting weakly dominates
  (Theorem 1).
- If `I_now(t) ⊊ I_plan` on part of the window (occlusion), the proposal decides on the coarser
  σ-algebra; by Blackwell's theorem a decision on a garbled channel is weakly worse, strictly when the
  branches' optimal actions differ, so `A(k) > 0` — committing strictly wins (Theorem 2). ∎

**Reading.** `A` is not a shaped reward and not a cost. It is the gap between two values learned by the
same honest backup that condition on different information sets. Conditioning the critic on the history
injects `I_plan`; a state value sees only `I_now`; their difference is the value of what re-planning
would discard. The learning recipe is therefore: a state value `V(o)` and a committed value
`Q_syn(o, a_{1:k})`, both by 1-step synthetic backup (never chunk-return regression), and the selector
reads `A = Q_commit − V_react`.

---

## 7. Positioning vs concurrent work, and the short-context prescription

Two concurrent papers (Aug 2026) make the non-Markovian-expert thesis that our force decomposition
rests on, so we cite rather than claim it, and sharpen where our contribution actually is.

- **Zeng, Agarwal, Bati, Lee, Ancha, Tedrake, "Revisiting Open-Loop Execution in Robotics" (2608.15938).**
  Central claim, matching ours: long open-loop execution primarily compensates a *short-context policy*
  imitating a *non-Markovian expert*; expert non-Markovianity shapes the success-horizon curve more
  strongly than compounding errors. Prescription, opposite to adaptive chunking: **increase the
  policy's context length** and the benefit of long execution vanishes — the most reactive closed-loop
  policy wins.
- **Lazzati, Stachowicz, Chen, Metelli, Wagenmaker, Levine, "Why Does Action Chunking Improve BC
  Performance?" (2608.02547).** Rejects temporal-consistency / horizon-reduction / representation
  hypotheses. Chunking's benefit = non-Markovian delays + reduced compounding error, and these are
  captured by a **delayed policy** `π(a_t | o_{t-k})` (condition on a past observation) and by an
  **implicit ensemble** of `k` delayed policies `{π(a_t | o_{t-i})}`. Explicit ensembles of delayed
  policies exceed chunking.

**What is preempted, and what is not.** The proposition "non-Markovianity sets the optimal chunk
length" is now published, concurrent — we cite it. Both papers are about the **policy** (how a BC model
imitates a non-Markovian expert). Neither touches **value estimation**: the belief-shift leak
(Lemma 1), the synthetic backup that pins it (Theorem 4), and the reactive advantage as a
value-of-information (Theorem 5) are about the **critic**, and are absent from both. Our
`I(s;chunk|o) = Q_reg − Q_syn = reactive-map` is a property of the *value estimator's* bias, not of the
policy's expressivity. Paper 2's `ΔL_val` (the validation-loss gap between a chunked and a Markovian
policy, its non-Markovianity metric) is the **policy-side twin** of our critic-side `Q_reg − Q_syn`;
cross-checking the two maps is a concrete validation.

**Empirical anchor (Park, "The Behavioral Cloning Mystery", seohong.me/blog, 2026).** A controlled
sim (scripted-spline demos, *infinite* data) grounds the non-Markovian story on the policy side and,
usefully, warns us. (i) Even in a fully Markovian environment the *dataset* is non-Markovian from
temporal correlations, so a closed-loop Markovian policy learns a "Markovianized" behavior and suffers
severe test-time distribution shift -- the policy-side twin of our belief-shift leak, and infinite data
does **not** fix it (so it is expressivity, not scarcity). (ii) A history-conditioned policy
`π(a_t | s_{t-24:t})` gets **lower** training loss but **worse** rollout than a no-history one -- direct
elimination evidence against "just make the policy long-context," which supports putting memory in the
critic instead. (iii) Validation flow loss fails to predict rollout while action MSE tracks it better,
matching our demo-MSE gate.

But (ii) is also a **caveat for us**, stated honestly. The history-conditioned policy fails because it
overfits *spurious* temporal correlations (the copycat / causal-confusion shortcut: conditioning on
recent history lets the policy copy its own past, a correlation that breaks under its self-induced
rollout distribution). A delayed-observation *critic* is not automatically immune: if its `Q_commit`
leans on temporal correlation rather than on genuine hidden **state** (the plan's `z` carried through an
occlusion), then `A` overfits the demo distribution and mis-selects at deployment. The framework's
defense is that the delayed observation must carry the *latent the observation lost* (Theorem 2's
occlusion), not an arbitrary action-history shortcut -- and this is testable: the reactive-map should
track genuine occlusion/aleatoric structure (uncertainty-split), not arbitrary temporal correlation. So
Park's result both motivates the critic-side design and names its failure mode.

**The short-context prescription (why adaptive chunking survives the context-length result).** Take
the VLA as short-context by fiat (compute and latency make long context impractical for a deployed
generative policy). Then the papers' fix — a long-context reactive policy — is off the table, and
carrying information forward (commitment) is forced. The move that keeps this principled is an
**asymmetry the papers do not use**: the *critic* need not share the policy's context budget. The
policy (a frozen VLA) proposes chunks short-context; the *critic*, trained offline with no latency
constraint, may condition on history (or, cheaply, on a delayed observation `o_{t-Δ}`) and decide the
commitment length from `A = Q_commit − V_react`. Memory lives in the critic; acting stays in the
policy. A long-context policy still leaves the value function's belief-shift leak unaddressed
(Theorem 4 is about the estimator, not the policy), so the critic-side contribution is orthogonal to
the context-length result. This is the design of Section 8.

## 8. Learning it on a frozen short-context VLA (delayed-observation critic)

Concretely, for the YAM patch-critic:

1. **`V_react(o)` — a chunk-free state value by 1-step synthetic backup.** No chunk conditioning, no MC
   return; `V(o_t) ← r_t + γ V(o_{t+1})` on the cache's real transitions. Theorem 4 makes this the
   honest observation-belief value; it is what our current PatchV is NOT (it is fit to the chunk-
   conditioned, leaky Q, so it inherits the leak — the `V(s0)` gap of 986).
2. **`Q_commit(o, h_Δ, a_{1:k})` — history-aware, synthetic backup.** Condition the committed value on a
   *delayed observation* `o_{t-Δ}` (or a short stack), which carries `I_plan` the current frame lacks —
   the cheap history of Lazzati et al. Build the window by 1-step synthetic backups through the
   observed intermediate frames, bootstrap `V_react` at the boundary.
3. **Selector** `A(k) = Q_commit − V_react`, commit the longest `k` with `A ≥ −ε`. No re-plan cost.
4. **Diagnostic (measure_reactive_map).** `Q_reg − Q_syn` per frame is the leak = `I(s;chunk|o)` =
   reactive-map. Predict: large at contact/alignment (aleatoric peak, from uncertainty-split) and at
   the first frame (intent hidden), small in free-space transport. Cross-check against paper 2's
   `ΔL_val` computed on the same episodes.
