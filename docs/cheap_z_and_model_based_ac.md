# Cheap-z + Model-Based Adaptive Chunking — design document

*2026-08-07, overnight research session. Numbers herein are measured on PrepareCoffee
(514 episodes / 279,534 frames) unless cited.*

## 1. Why we are replacing the VLA-extracted token

The RLT approach reads z from the 3B VLA's prefix hidden states. Three real advantages
(free at rollout, language-conditioned, policy-aligned) — and three measured costs:

1. **Annotation tax**: every critic experiment needs a full-dataset 3B forward pass
   (hours of GPU per checkpoint; the mae0.5 annotation crashed twice before finishing).
2. **Appearance inheritance**: SigLIP features are a function of what the frame looks
   like; every RoboCasa demo is a visually distinct kitchen, so episode identity is
   linearly decodable from z_rl at ~100% and dominates its neighbourhood geometry
   (kNN same-episode purity 0.42 vs 0.002 chance). MAE masking does not fix this
   (measured across mask 0 / 0.5 / 0.75); it is the objective, not the difficulty.
3. **Iteration speed**: a representation change = retrain the VLA arm (a day of B200).

The probing literature (2605.28527) says frozen DINOv2 sits ~0.04 R² behind the best
VLA features for value prediction; Q-VGM (2606.08015) warns a *scratch* ResNet loses
10pts. So the design target was: **frozen pretrained backbone + small trained head**.

## 2. The cheap-z recipe (validated)

```
frozen DINOv2-small (22M), 3 cams, patch-MEAN tokens (not CLS), cached once for the
whole dataset (25 min; video decode disappears from every later experiment)
  -> shared per-camera MLP head 384->512->512->256 (LayerNorm+GELU), frame z = mean over cams
  -> loss = VIP-I + 3.0*multi-pos InfoNCE + 1.0*view-InfoNCE + 0.5*VICReg + 0.5..1.0*epadv
```

- **VIP-I** (2210.00030) with the *code-true* sign — the paper's printed Eq. 6 is
  sign-flipped and rewards collapse. γ=0.98.
- **Cross-episode positives**: 4 frames from *other* episodes within ±0.025 progress,
  τ=0.07. This is the term that attacks the measured pathology; actions are never the
  pairing key (measured motor bias of PSE-style pairing).
- **View positives**: same timestep, other cameras (TCN, validated at 133 sequences).
- **VICReg var+cov**: blocks the 1-D progress collapse (neural regression collapse,
  2409.04180). RankMe stayed ~150/256 throughout — no collapse observed.
- **epadv**: DANN gradient-reversal episode adversary, same construction as the
  `+epadv` RLT objective in `pi0_rlt.py`.

**Leaderboard** (identical 20k frames, episodes held out; probe = ridge):

| repr | mc_return R² | progress R² | episode acc | kNN purity |
|---|---|---|---|---|
| VLA z_rl (rlt5@70k, 2048d) | 0.734 | 0.926 | 0.993 | 0.420 |
| raw frozen DINOv2 (1152d) | 0.610 | 0.823 | 1.000 | 0.698 |
| cheap-z v4b (256d, 15 min) | **0.740** | 0.926 | **0.910** | **0.145** |

cheap-z **dominates the VLA token on every probe** at ~1/130 the encoder size, with no
annotation step. Critic-level A/B (same labels/candidates/seed IQL, only z differs) is
running; that is the decision-grade test.

Files: `scripts/probe_cheap_z.py` (harness), `scripts/train_cheap_z.py` (recipe),
`scripts/make_cheapz_annot.py` (packages z as an annotate_rlt-compatible dir so the
whole critic stack runs unmodified), `.scratch/dino_cache_PrepareCoffee/` (features).

## 3. Model-based Adaptive Chunking — the design

### 3.1 What the literature licenses (and forbids)

- **Forbidden**: long imagined rollouts / Dreamer-style imagination on narrow demos
  (V-D4RL: offline DreamerV2 scores 4.8 where plain BC scores 91.5; FOWM zero-shot
  pick 0%). Model-based value expansion for a *drifting* policy is the configuration
  every offline-MBRL paper warns about (MOPO/MOReL narrow-data failures; edge-of-reach
  2402.12527; oracle-model expansion gains ≈ nil, 2412.20537).
- **Licensed**: (i) a **one-step, chunk-conditioned** latent dynamics head as a critic
  auxiliary — WCM (2607.29613) shows exactly this beating a VLM-feature critic for RL
  fine-tuning of π0-class policies; (ii) **prediction-reality consistency** checks that
  re-anchor on every real observation — CheckVLA (2607.26789), +8.5pts over periodic
  replanning; (iii) 1-step lookahead ranking.

### 3.2 Architecture

All of this trains on the cached z (no video, minutes per run):

```
dynamics ensemble (M=5, independent inits):
  d_i(z_t, chunk emb) -> ẑ_{t+k}         k = macro-prefix grid {2,4,...,16}
  chunk embedding: 1D-conv or MLP over the 16x12 chunk (per-prefix truncated+padded)
value/reward heads (shared trunk with the critic, IQL as today):
  V(z), Q(z, chunk-prefix)  — unchanged ARQ recipe, value support clamped to [0,1]
```

### 3.3 Three uses, in deployment order

1. **Critic auxiliary (offline, now)** — add `‖d(z_t, a_{t:t+k}) − sg(z_{t+k})‖²`
   to critic training. Costs nothing at rollout; WCM's ablations attribute their OOD
   gains to it. Risk: none identified.
2. **1-step lookahead ranking (drop-in for eval_critic's `bon`/`critic` modes)** —
   `score(candidate, prefix) = r̂ + γ^prefix · V(d(z, candidate_prefix))`, computed for
   all N×P pairs in one batched forward. Never unroll d twice: compounding error and
   model exploitation start at step 2 (MBPO's own regime analysis).
3. **Uncertainty-gated commit (the AC upgrade)** — commit further into the chunk while
   the ensemble agrees: `commit = max{ p : disagreement(z, chunk, p) < ε }`, where
   disagreement = mean pairwise ‖d_i − d_j‖ at prefix p, and ε is calibrated on
   held-out demos (disagreement-vs-error sparsification). At execution, CheckVLA-style:
   after each macro step compare ẑ against the realized z; if consistency breaks early,
   replan immediately. "Commit as far as you can predict" becomes the principled
   replacement for the argmax-over-prefix rule.

### 3.4 Why this composes with the AC 2x2

The existing eval (`vla`/`bon`/`prefix`/`critic`) isolates chunk-choice vs
commit-length gains. The dynamics model upgrades both axes independently, so the same
2x2 attributes its contribution: model-based ranking should move `bon`, uncertainty
commit should move `prefix`, and their product moves `critic`.

### 3.5 Failure modes + tripwires

| risk | tripwire |
|---|---|
| dynamics memorizes demos, blind to OOD chunks | disagreement-vs-error Spearman on held-out candidates (`base_action_heldout`) — must be > 0 and calibrated |
| ranking exploits model error | compare rank-correlation of model-based vs model-free scores on held-out demonstrated chunks (the demo chunk should rank top-quartile) |
| ensemble agrees-and-wrong off-distribution | evaluate disagreement on shuffled (z, chunk) pairs — must exceed on-distribution disagreement |
| value bootstrap blowup at short prefixes | keep the [0,1] value-support clamp (measured 49x amplification without it) |

## 4. Open questions (deep-dive agents in flight)

1. Exact dynamics hyperparameters at our scale (DINO-WM / V-JEPA2-AC / TD-MPC2 / WCM
   recipes, chunk-level vs per-step prediction, SimNorm placement).
2. Online-stage mechanics (V-GPS/RoboMonkey/PA-RL/ConRFT/πRL): candidate generation,
   conservatism (Cal-QL vs IQL at large N), when/how the VLA itself updates.
3. Offline instruments: stitching probes constructible from single-task demos,
   OPE estimator choice, uncertainty calibration protocol (conformal bounds for the
   commit rule), and rollout-budget statistics for the final comparison.
