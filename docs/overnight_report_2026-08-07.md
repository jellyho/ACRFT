# Overnight report — cheap representations & model-based AC (2026-08-07)

*Everything below was measured tonight on PrepareCoffee (514 eps / 279,534 frames) unless cited.
Code: `scripts/{probe_cheap_z,train_cheap_z,make_cheapz_annot,train_cheapz_dynamics,
eval_modelbased_scorer,probe_stitching}.py`; design doc: `docs/cheap_z_and_model_based_ac.md`.*

## The recommendation (one paragraph)

Replace the VLA-extracted token with **cheap-z v5b** (frozen DINOv2-small + 15-min head:
VIP-I + cross-view/cross-episode InfoNCE + VICReg + episode-adversary + HILP-TD): it now beats
the 3B VLA token on every value probe at 1/130 the encoder cost with no annotation pass. But the
night's bigger finding is that the *critic*, not the representation, is the binding constraint:
with all-success demos, ANY IQL critic is action-blind (measured for both z's), and value-through-
dynamics does not fix it — while **dynamics-ensemble disagreement does** (the only action-
discriminating offline signal we found). So the best method = cheap-z v5b + one-jump dynamics
ensemble, ranking by behaviour-consistency (disagreement) offline, with value taking over as
failure data arrives online — then Q-VGM-style distillation into the VLA. Nobody fine-tuning a
flow VLA on a real robot does policy gradients through the sampler; we shouldn't either.

## 1. Representation leaderboard (identical 20k frames, episodes held out)

| repr | mc R² | prog R² | ep acc ↓ | purity ↓ | stitch ρ ↑ |
|---|---|---|---|---|---|
| VLA z_rl (3B, rlt5@70k) | 0.734 | 0.926 | 0.993 | 0.420 | **0.465** |
| raw frozen DINOv2 | 0.610 | 0.823 | 1.000 | 0.698 | — |
| v4b (60k + epadv) | 0.740 | 0.926 | **0.910** | **0.145** | 0.236 |
| **v5b (v4b + HILP-TD 3.0)** | **0.762** | **0.941** | 0.919 | 0.154 | 0.251 |

- Champion on value axes: **v5b** (+0.028 mc R² over the VLA token). The episode adversary sits at
  chance (CE = ln 514) — episode identity is gone from the trainable head's reach.
- **Honest caveat — stitching**: the half-split compositional probe (train a distance on A-first-
  halves ∪ B-second-halves, query across) still favors VLA-z (0.47 "moderate" vs our 0.24–0.27
  "fail"). Exactly what the stitching taxonomy (2401.11237) predicts for MC/alignment losses; our
  first HILP-TD arm raised value probes but did not close this. Open item with concrete next steps
  (§5). Note no representation reaches "strong" (0.7) here.

## 2. The critic discovery (changes the plan more than any recipe)

Offline critic battery (`eval_rlt_critic.py`), identical IQL recipe, only z differs:

> **FAIL — action sensitivity ≈ 0: the critic ignores the action; best-of-N is a no-op.**
> (both VLA-z and cheap-z; within-state candidate spread ~0.0013 in [0,1] value units)

Cause: all 514 demos succeed → every demonstrated action leads to success → the data contains **no
action contrast** → Q(z, a) degenerates to V(z). This is precisely why V-GPS found IQL ranking
collapses beyond K≈10 candidates while Cal-QL survives to K≈50 (2410.13816 App. H), why RoboMonkey
manufactures synthetic negatives (2506.17811), and why WCM grounds its critic with a dynamics head
(2607.29613).

**Model-based rescue, tested tonight** (`eval_modelbased_scorer.py`):

| scoring rule | within-state spread | demo-vs-shuffled AUC |
|---|---|---|
| Q(z, a) (model-free) | 0.0013 | FAIL |
| V(d(z, a)) (value through dynamics) | 0.0011 | 0.501 |
| **ensemble disagreement of d(z, a)** | **0.0253–0.0281 (20×)** | **0.582** |

Value-through-dynamics does NOT restore sensitivity (V projects out the future differences —
consistent across both z's). **Disagreement does**: offline demos can teach *"is this chunk
demonstrated-like from this state"* (behaviour-consistency), not *"which chunk succeeds more"*.
This is DREAM-Chunk's selection rule (2606.18589), discovered independently by measurement.

## 3. Dynamics model v0 — calibrated

5-member chunk-conditioned ensemble on cheap-z (minutes to train on cached features):
beats the copy-forward baseline at every prefix (31% lower at p=16); error grows smoothly with
prefix; **disagreement→error Spearman 0.70–0.78 at every prefix** — the "commit as far as you can
predict" rule has its prerequisite. OOD ratio (shuffled/matched) 1.09 — direction right, needs
strengthening (random-prior nets, §5).

## 4. What three deep-dive surveys settled (all verified against primary sources)

1. **Dynamics recipe** (DINO-WM/V-JEPA-2-AC/TD-MPC2/WCM/PLDM, code-level): predict in z (not raw
   features); macro-stride 4 block-causal micro-transformer, history 3, Gaussian NLL + bounded
   logvar, teacher-forcing + 2-step rollout loss 1:1, NO VICReg/EMA (frozen targets = collapse-
   free), 5 independent inits, ensemble-VARIANCE as the disagreement metric (ρ=0.82 best-
   calibrated, 2110.04135). WCM integration: joint value+dynamics loss, λ=0.4.
2. **Critic-guided VLA mechanics**: the proven stack is Cal-QL-calibrated conservative chunk-critic
   (`max(Q, MC)` — we HAVE the MC returns) + LayerNorm everywhere (RLPD: collapse without it on
   exactly our data type) + WSRL frozen warmup + 50/50 buffers + UTD≤4; VLA improvement via
   **Q-VGM residual velocity matching** (π0.5 79→92.5% from ~500 logged episodes) or advantage-
   conditioned flow-BC (π*0.6/RECAP) — never PPO through denoising on a real robot.
3. **Offline instruments**: RankQ bands (healthy critics rank demo>noise at 95–99%), Cal-QL sign
   test |Q−MC|≲0.1, VOC bands for V-monotonicity, AUSE (not Spearman) per horizon for uncertainty,
   split-conformal per-horizon thresholds for the commit rule (n≥159 segments for jointly valid
   H=16, δ=0.1 — we have thousands), and paired McNemar/STEP for rollout budgets (50 paired trials
   detect ~16–20pp; unpaired would need 170/arm).

## 5. Roadmap

**Phase 0 (this week, offline)**
- Critic: add Cal-QL calibration + LayerNorm; retrain on v5b. Acceptance: RankQ permuted-pair
  ranking ≥85%, |Q−MC| ≲ 0.1.
- Dynamics v1 per the verified recipe (macro-stride 4, NLL, rollout loss, random-prior ablation
  for the OOD ratio). Acceptance: R²(1 macro-step) > 0.6 vs copy-forward, AUSE ≲ 0.2.
- Stitch gap: try (a) HILP φ-space as the probe distance directly, (b) larger hilp-dim/weight with
  the value-probe guard, (c) quasimetric (MRN/IQE) head — TMD-lite from the survey.
- mae0.5 embedding comparison (z-only annotation running; 3-way probe when it lands).

**Phase 1 (online, policy frozen)** — WSRL warmup → 50/50 buffers → online critic refinement;
failures finally provide action contrast, activating value-based ranking. Dynamics refreshed on
online transitions. Commit rule goes conformal.

**Phase 2 (VLA improvement)** — Q-VGM-style: Q-gradient ascent on 1-step-Euler chunks (≤10 steps,
anchored, accept-if-improves) → residual velocity matching into the action expert (LoRA). Budget
discipline per PA-RL; frozen-VLA arm kept as the rollback baseline.

## Sidebars

- GarnishPancake mae0.5 (task #2): training normally, 13k/100k at report time.
- mae0.5 full annotation: failed 4× (B200 cudnn-autotune hang; L40S OOM at N=24 samples/frame) —
  bypassed with a z-only annotation (N=1) for the embedding comparison; full candidate annotation
  only needed if we critic-train on mae0.5, which the cheap-z result makes moot.
- All torch jobs must exclude node200 (no sm_100 kernels in torch 2.7.1+cu126).
