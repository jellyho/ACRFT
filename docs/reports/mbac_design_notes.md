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

## Iteration 2 (03:50) — replan cost, hysteresis, and the sim integration path

### Replan cost, measured rather than assumed

From tonight's rollout_control log: ~1 min per 1000-step trial with a replan
every 16 steps → ≈62 replans/trial ≈ 1 s per 16-step block, VLA inference
(10 flow steps, N=16 candidates) plus sim inside that. Two consequences:

- **Latency, not throughput, is the argument.** On a real robot (YAM) the sim
  cost vanishes but inference latency stays; at k=16 the arm runs open-loop for
  16 control steps between corrections. Adaptive k is not about saving compute
  (worst case it replans MORE) — it is about *spending* the replan budget where
  the model says the plan decays fastest, and coasting where it doesn't.
- So the objective for π_k is not `Δprogress − c·replans` with a made-up c;
  the honest offline objective is **terminal prediction error at matched mean
  commitment** (E1's metric). If adaptive-k dominates the fixed-k frontier
  (lower error at the same mean k), it buys reactivity for free; c only decides
  the operating point on that frontier.

### σ-hysteresis: preventing the k=4 doom loop

Off-distribution, σ is high everywhere (desired: replan often when lost). But a
pure threshold can lock at k_min forever — replanning cannot *reduce* σ if the
state itself is OOD; each replan re-derives the same alarm. Design:

- Cut rule uses **relative** disagreement: τ_t = τ · median(σ over the last W
  replans). Persistent OOD inflates the baseline, so the rule re-normalizes and
  k recovers; only *spikes* relative to the recent regime trigger early cuts.
  This mirrors how AUSE was the right calibration metric (ranking, not scale).
- Floor stays k=4 (one macro-step): even in full alarm the arm executes 4 steps
  — thrash-free by construction, since replanning below the model's stride
  gives the model no new information anyway.

### Sim integration is a small delta, not a new harness

`eval_critic` already supports per-replan variable `n_exec` (the prefix mode
returns `(chunk, n_exec, Replan)`). A new mode `mbac`:

1. selection: Cal-QL+swap critic argmax over N candidates at full horizon
   (unchanged from `bon`);
2. commitment: run the φ-dyn ensemble on the winner (one forward, 4 macro
   slots, numpy or torch-CPU — 5 tiny models, negligible next to the VLA);
   n_exec = 4·k*, k* from the hysteresis rule.

This reuses the existing HUD/trace fields (`n_exec`, `best_prefix`) so the
per-trial traces stay comparable with tonight's fixed-k runs — the paired
comparison extends to a 3rd arm without touching the harness.

### Where this meets AC-RFT (the RL part, sharpened)

The RFT loop fine-tunes the VLA on its own successful rollouts. If those
rollouts run under MB-AC commitment, the *data distribution itself* becomes
adaptive-chunked: chunks that survived long commits appear as long coherent
segments, alarm regions appear as dense replan boundaries. Fine-tuning on that
data teaches the VLA to produce chunks that are *stable where the model can
verify them* — a virtuous cycle where the dynamics model shapes the policy
without ever backpropagating through it. π_k (proposal v1) then only needs to
be learned if E1 shows the σ rule leaves oracle gain on the table.

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

## Iteration 3 (04:50) — first rollout evidence + mbac implementation landed

Control rollout (action-blind plain-IQL critic, 30 paired trials, seed 0):
prefix .800 > vla .700 > bon .600 > critic .567 (no pair significant at n=30;
prefix-vs-critic p=.065 comes closest). Two readings:

- bon ≈ critic ≈ vla: the offline action-sensitivity ~0 verdict survives contact
  with the simulator — an action-blind critic's BoN is a no-op minus noise. The
  negative control behaved exactly as the diagnostics predicted.
- prefix (+.10, n.s.) is the only mode above baseline, and offline it commits
  the SHORTest prefix 88% of the time — i.e. it approximates "replan every 2
  steps". Weak but directionally consistent with commitment mattering more than
  selection for this checkpoint. The mbacv ablation (VLA sample + sigma-rule
  commitment, no selection at all) is now the decisive experiment: if mbacv
  recovers prefix's gain with ~8-step mean commits, adaptive commitment is real
  and cheap; if not, prefix's edge was frequent-replanning noise.

Implementation: `eval_critic --modes mbac mbacv --dyn <ensemble> --dyn-tau 2.0`
landed (DynCommit, CPU torch, relative-median hysteresis, per-trial reset).
Blocked on phi_dyn_v1 (job 34635, running) / the hist=1 deployment variant
(34641, queued).

## Iteration 4 (05:55) — the dynamics belongs in phi space

phi_dyn_v1 (job 34635, same architecture/steps as the DINO-space run):

| horizon | R2 phi | R2 DINO | AUSE phi | AUSE DINO |
|---|---|---|---|---|
| +4  | .286 | .064 | .080 | .114 |
| +8  | .512 | .305 | .072 | .106 |
| +12 | .606 | .429 | .072 | .096 |
| +16 | .659 | .519 | .075 | .089 |

OOD action-swap disagreement ratio: **1.41** (DINO: 1.09).

Reading: phi's TD training already threw away the appearance nuisance that the
DINO dynamics wasted capacity on; what remains is the reachability geometry,
which is exactly what actions move. Three consequences:

- The R2(+4)>0.6 acceptance gate stays unmet (.286) — the 4-step slot is still
  mostly copy-forward. The sigma rule's k floor of one macro-step remains the
  right guard: the model has little to say about the first 4 steps.
- Swap ratio 1.41 upgrades E2's prior: wrong-action chunks now visibly inflate
  disagreement, so binding-by-disagreement has a real chance where the DINO
  model (1.09) had almost none.
- tau sweep design for the mbac rollout: with sigma ratios ~1.4 between
  in/out-of-distribution chunks, tau_mult=2.0 (cut on 2x the running median)
  may be too permissive — plan a {1.3, 2.0, 3.0} sweep AFTER the first 30-trial
  run rather than burning trials on a grid now; the first run's sigma trace
  (Replan.value logs sig[k]) calibrates the sweep for free.

Interim from the phi+calswap rollout (34633): critic mode opened 4/4 (control
was .567 over 30). Too early to celebrate at n=4; the full paired table lands
this hour.

## Iteration 5 (06:30) — E1-E3 results: disagreement binds, value-through-model breaks

Offline battery on held-out episodes (n=8000 anchors each):

**E2 — binding without CQL.** phi space: `binding_by_disagreement = .817`
(sig ratio 1.61); v4b: .780 (1.10). The ensemble alone tells the demo chunk
from another state's demo chunk four times out of five — no CQL anywhere.
But `binding_by_goal_distance = .371` — *below* chance: ranking by predicted
terminal-state closeness to goal actively prefers the WRONG chunk. Value
estimated through the model is not just weak, it is inverted; the model's
prediction under an inconsistent action drifts in a way that mimics progress.
The selection/commitment split is confirmed from both sides: sigma is a real
signal, model-predicted value is a trap.

**E3 — visibility is immediate.** Disagreement binding at k=4/8/12/16:
.837/.815/.807/.798. It PEAKS at one macro-step and decays slightly — the
trust signal does not need depth (revises iteration 2's "only long horizons"
assumption, which was about R2, i.e. mean accuracy, not about sigma).

**E1 — adaptive cutting beats the fixed frontier, narrowly.** phi space,
quantile-tau on cumulative sigma: q0.3 gives mean_k 14.2 with terminal error
4.13 vs 4.19 interpolated on the fixed-k frontier at the same k — a real but
thin margin. Cuts land where they should: anchors cut early show error slope
3.01 vs 2.00 for late-cut (the rule finds genuinely unpredictable states), and
cut positions pile up in the late-progress bins (881/934 of cuts in the last
60% — the grasp/place/button phases). v4b sigma almost never cuts (97-100%
full commit): yet another way the DINO-space model is blunt where phi is sharp.

**Design consequence — sigma-filtered BoN.** .817 binding suggests sigma's
best role may be *selection veto*, not just commitment: among N candidates,
drop those whose one-step disagreement spikes (behaviourally inconsistent),
critic-argmax the survivors. This directly targets tonight's live failure -
the phi+calswap critic in full-authority `critic` mode is collapsing in sim
(7/25 after a 4/4 start vs .70 vla) exactly as worker-B's table predicted;
a sigma veto bounds how far the critic can wander into exploitable chunks.
Implementation is three lines in the mbac branch (score sigma per candidate,
mask top-half sigma, argmax Q on the rest) - queue as `mbacf` after the first
mbac numbers land.

## Iteration 6 (07:10) — the full paired table: authority, not accuracy, is the variable

All 60 paired trials in (seed 0, scene-paired, shared vla baseline 21/30):

| arm | rate | vs vla | p (McNemar) |
|---|---|---|---|
| control prefix (raw token, plain IQL) | .800 | +6/-3 | .51 |
| phi bon / phi prefix / vla | .700 | +5/-5 | 1.0 |
| control bon | .600 | +3/-6 | .51 |
| control critic | .567 | +5/-9 | .42 |
| **phi critic (full authority)** | **.300** | **+2/-14** | **.004** |

The night's only significant result is a NEGATIVE one, and it is instructive:
the *better* critic (action-sensitivity .524 vs .001, binding .996) is
CATASTROPHIC when given full authority (candidate x prefix arg-max), while the
action-blind one was merely useless. Offline binding fixed what the critic
knows; it did not fix what the arg-max DOES with 128 (16x8) options against an
imperfect Q. The action-blind critic couldn't act on its opinions; the calswap
critic acts on them boldly and is wrong at the tails. worker-B's harm table,
reproduced with a sharper instrument.

phi bon exactly matches vla (5 swaps each way) - selection-only authority is
now *safe* (control bon was mildly harmful) but not yet profitable: the
candidates at N=16 are too similar for ranking to matter on PrepareCoffee, or
Q's tail noise cancels its median signal.

This makes tonight's remaining arms (34652: mbacv / mbac / mbacf) precisely
the right experiment: they test whether BOUNDED authority - commitment-only
(mbacv), sigma-vetoed selection (mbacf), or both (mbac) - is where the gain
lives. If mbacv > vla while phi-critic collapsed, the lesson generalizes:
give learned components narrow, verifiable authority.

## Iteration 7 (10:40) — replication: the positives were seed noise

rep2 (fresh scenes, seed 30-59, its own vla baseline 19/30 = .633):
mbacv tau1.3 .567, mbacf tau1.3 .500, prefix .500 — every overnight positive
reversed sign on new scenes. Pooled at 60 config-matched paired trials:

| arm | pooled | vs vla .667 | p |
|---|---|---|---|
| prefix | .650 (39/60) | +10/-11 | 1.00 |
| mbacv tau1.3 | .667 (40/60) | +8/-8 | 1.00 |

Exactly even. The honest ledger of the whole night, at adequate power:
- ONE robust effect: full-authority critic HARMS (-.40, p=.004). Negative.
- Everything else — prefix's +.10, mbacv tau1.3's +.067 — is within-seed noise
  exactly as worker-B's 16-seed CI runs foretold. 30-trial arms have ±.1 noise;
  never celebrate a +.1 again without a replication.
- Adaptive commitment is at least SAFE (no arm harmed), which the full-authority
  mode is not. Bounded authority buys robustness, not (yet) success rate.

Why no gain? The sim rollouts run at ~replan-every-16-steps against a policy
whose failure modes (wrong grasp choice, missed button) are decided WITHIN a
chunk, not between chunks; cutting a chunk early cannot fix a chunk that was
never going to work, and N=16 candidates from the same flow are too correlated
for selection to matter. The leverage adaptive chunking has on THIS task at
THIS checkpoint is small by construction. Where it should matter instead:
(a) weaker/earlier checkpoints (more divergent candidates, more recoverable
mid-chunk failures), (b) tasks with contact-rich junctures (GarnishPancake),
(c) the AC-RFT loop where commitment shapes the TRAINING distribution, not
just deployment selection.

## Iteration 8 — MAC x ACSAC: model-based action-chunking RL, unified design

Prompt: can Q-chunking-style chunk-level RL (ACSAC: the chunk is ONE macro-action,
SAC/RLPD-style TD over the chunk MDP) be combined with our model-based stack?
Answer: yes, and the combination is more principled than either half alone. The
dynamics model is exactly the missing piece that makes chunk-level offline RL
work, and chunk-level RL is exactly the frame that makes our model useful
without falling into the model-value trap.

### The unifying object: Q(phi, a_chunk, k)

Extend the macro-action to include the COMMITMENT: a macro-action is (chunk
a_{1:16}, commit length k in {4,8,12,16}). One critic learns

    Q(phi_t, a, k)  <-  r_{t:t+k} + gamma^k * V(phi_{t+k})     (real data, executed k)
    Q(phi_t, a, k)  <-  gamma^k * [ V(phihat_k(phi_t, a)) - lambda * sum_j sigma_j(phi_t, a) ]
                                                                (model-backed, ANY a, ANY k)

The second line is the synthesis. Offline demo data only contains the demo's
action at the demo's pace - that is WHY critics go action-blind (no
counterfactuals) and why commitment had to be hand-tuned (no variable-k data).
The phi-dynamics fills both gaps at once: its 4 macro-slots produce phihat at
every k for every stored candidate (the 16 candidates per frame we already
annotate), so the critic sees counterfactual (a, k) pairs with model-backed
targets instead of CQL's uniform push-down.

### Why the measured failure modes are answered, not ignored

- Model-value inversion (E2: ranking by d(phihat, goal) = .371, WORSE than
  chance): the trap was a raw distance heuristic on the model's output. Here
  the model output is fed to the LEARNED V (Cal-QL-calibrated), and the
  lambda*sigma penalty subtracts exactly where phihat is untrustworthy - and E2
  measured sigma to be 1.6x larger on wrong-action rollouts. The penalty is
  large precisely where the inversion lives. (MOPO's pessimism, at chunk level.)
- sigma-rule tau tuning (iteration 5-7: tau 2.0 inert, tau 1.3 nulled): the
  hand rule dissolves. Commitment becomes argmax_k Q(phi, a, k), where longer k
  accumulates more sigma penalty AND more gamma discount - "commit while the
  model agrees" now falls out of the Bellman objective with lambda the single
  pessimism knob, learned against real returns instead of hand-set.
- Full-authority collapse (.300, p=.004): authority here is bounded by
  construction - the argmax runs over 16 candidates x 4 commit lengths, but
  every option's value carries its own model-risk discount; the exploitable
  tail (options the data never continued) is exactly the high-sigma tail.
- Q-chunking's own result (unbiased n-step backups, better exploration) is
  preserved: line 1 IS Q-chunking's backup; we only add line 2.

### ACSAC correspondence

| ACSAC piece | here |
|---|---|
| actor (chunk proposal) | frozen VLA flow (N=16 samples) - no actor training |
| chunk-level Q | Q(phi, a, k) above |
| SAC entropy/exploration | candidate diversity of the flow + (online) softmax over Q |
| replay | annotation frames + model-backed (a, k) branches (macro-Dyna) |
| pessimism (RLPD/Cal-QL) | Cal-QL floor on real data + lambda*sigma on model branches |

### Offline validation ladder (all runnable on existing artifacts)

- M1 model-backed counterfactual targets vs CQL-swap: retrain the phi critic
  replacing --cql-swap with target line 2 on the stored 16 candidates
  (k=4 only). Gate: binding >= .95 AND action-sensitivity >= .3 without any
  CQL term. If it passes, pessimism-by-model replaces pessimism-by-copy.
- M2 variable-k head: extend the critic to Q(phi, a, k) with the model filling
  the k-grid. Diagnostics: does argmax_k reproduce a sane commit distribution
  (compare to ctl-prefix's bimodal, which correlated with the only positive
  arm)? Does the implied commit anti-correlate with sigma-spikes?
- M3 GP protocol rollout (3 seed sets x 50, CI): arms = vla / bon(calswap) /
  mac-k (Q-learned commitment) / mac-full (selection + commitment from Q).
  MAC beats the sigma-rule iff the learned lambda/gamma trade-off finds the
  cut points the hand rule missed.

### Where RFT enters (the AC-RFT endgame)

Online, the same Q(phi, a, k) is the natural advantage estimator for chunk-level
RFT: A(s, a, k) = Q - V weights the VLA's flow-matching update per CHUNK (Q-VGM
style, never PPO through the flow), and k* labels how much of each rollout
segment was policy-coherent. The dynamics model keeps training on the fresh
rollouts (its data need not be success-only), so the pessimism boundary expands
exactly where the robot has actually been - the model-based half converts
failures into information the critic half can trust.

## Iteration 8b — correction: REAL model-based RL, not critic patching

User pushback, correct: iteration 8 used the model as a target-manufacturing
gadget for the critic. Original MB-RL uses the model as a WORLD - you roll in
it, learn in it, plan in it. The real MAC + ACSAC:

### The world model (upgrade phi-dynamics into an MDP model)

phi-dynamics today predicts phi' from (phi, a_chunk) - a transition model only.
A world model needs reward and termination:
- reward head r_hat(phi, a, j): per-macro-step progress gain. Trainable NOW -
  annotation has per-frame progress labels; r = Delta progress (or gamma-return
  regression). No new data needed.
- success/termination head from the terminal-success labels.
- imagination ACTOR pi_hat(a_chunk | phi): the KEY missing piece. The VLA
  cannot act in imagination (it needs images; imagined states are phi-only).
  Distill: we hold (phi_t, 16 VLA-sampled chunks) for 280k frames - train a
  small flow/MLP proposal in phi space to mimic the VLA's chunk distribution.
  Same scale as the phi readout (minutes to train). This unlocks MULTI-CHUNK
  imagination: at an imagined phi, propose chunks without the VLA.

### Learning: chunk-level Dyna/MBPO (this is the ACSAC half)

ACSAC = SAC where the action is a chunk. Its replay is exactly what MBPO
augments:
- from real annotation states, branch k=2-3 macro-step imagined rollouts:
  a ~ pi_hat, phi' ~ dynamics, r = r_hat - lambda*sigma  (MOPO penalty on the
  IMAGINED REWARD - its proper home, not a target patch).
- depth cap 2-3 macro-steps comes from the measured R2 curve (model beats
  copy-forward only at 8-16 steps and compounding beyond is unvalidated).
- train chunk-level Q (and optionally soft-update pi_hat toward high-Q chunks,
  SAC-style with entropy against the distilled prior) on real + imagined
  transitions. This is genuine policy improvement beyond the data: the agent
  tries chunks the demos never executed, in states the demos never reached
  (imagined branches), and values propagate back through Bellman.

### Acting: receding-horizon planning (TD-MPC at chunk scale)

Deployment is no longer "score one chunk myopically":
1. VLA proposes N=16 real chunks (as now) - the proposal prior at the REAL state.
2. For each, imagine D=2-3 chunks ahead: rollout dynamics, propose continuations
   with pi_hat, score  sum gamma^j (r_hat_j - lambda*sigma_j) + gamma^D Q(phi_D, .).
3. Execute the first chunk of the best sequence (receding horizon); commit
   length = where the imagined plan's sigma stays low (adaptive chunking falls
   out of the plan, again).
This sees PAST the current chunk - "this chunk looks fine but lands where
nothing continues well" - the exact blind spot of every myopic mode we ran, and
a capability no amount of offline Q calibration provides.

### What exists vs what to build

| piece | status |
|---|---|
| transition model (phi, 5-ens, calibrated) | DONE (phi_dyn_v1_h1) |
| reward head | small: progress labels already in annotation |
| distilled proposal pi_hat(a|phi) | small: (phi, candidates) pairs on disk |
| chunk-level Q on real+imagined | new training script, existing critic arch |
| planner (imagination MPC) | numpy/torch loop, cheap next to VLA inference |

### Validation ladder

- W1: reward head quality - r_hat vs true Delta-progress on held-out episodes.
- W2: pi_hat fidelity - do distilled chunks match VLA candidate distribution
  (MMD / coverage on held-out states)? If pi_hat is bad, imagination is bad.
- W3: imagination sanity - imagined 2-3 chunk rollouts from held-out states:
  does imagined progress track real episode progress? (the MBPO validity check)
- W4: offline RL gain - chunk-Q trained real+imagined vs real-only (calswap):
  binding, sensitivity, AND a new probe - value of demo continuation vs value
  of pi_hat continuation (does imagination discover better-than-demo chunks?).
- W5: GP protocol rollout (3x50 CI): vla / bon / mpc-D1 (myopic, ablation) /
  mpc-D3 (plans ahead). The D1-vs-D3 gap isolates the model-based claim.
