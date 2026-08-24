# pi05 + FQL: one-step actor + critic expert (in-VLA offline RL)

**Goal.** Better *offline RL for VLAs*. IQL/DEAS decouple policy from critic (trainable separately) but
underperform actor-critic; actor-critic on a flow VLA needs real VLA samples (slow, sample-inefficient)
and its Q-gradient must backprop through the iterative denoising ODE (the wall). FQL (Park et al.,
ICML'25, [2502.02538](https://arxiv.org/abs/2502.02538)) bypasses the wall: distill the flow policy into
a **one-step** actor and put the Q-gradient through that single forward.

## Architecture — 4-expert MoT over ONE shared VLM prefix

`gemma` already runs a mixture-of-experts (`configs=[paligemma, action_expert]`); we extend the list.
The VLM (paligemma 2b) prefix (image + prompt) is computed once; every expert attends to its KV.

| expert | role | trainable | input tokens | output |
|---|---|---|---|---|
| paligemma 2b | VLM backbone | (frozen by default) | image + prompt | prefix KV |
| flow action expert (gemma 300m) | BC policy μ_θ | **FROZEN** | noisy action x_t + timestep t + state | flow velocity |
| **one-step actor** μ_ω (gemma 300m) | fast RL actor | train | **noise z** + state (no timestep) | action chunk (1 forward) |
| **critic** Q_φ (gemma 300m) | in-VLA critic | train | action chunk + state | distributional Q (HL-Gauss) |

μ_θ stays frozen: it is the distillation target that keeps μ_ω's actions in the BC manifold (the leash),
exactly like a frozen decoder in the latent-actor design.

## Objectives (FQL, offline)

- **Critic** (Eq. 1): `L_Q(φ) = E[(Q_φ(s,a) − r − γ Q_φ̄(s', a'))²]`, `a' ~ μ_ω(s', z')` (one-step).
- **Distillation** (Eq. 7): `L_distill(ω) = E‖μ_ω(s,z) − μ_θ(s,z)‖²`, where `μ_θ(s,z)` is the frozen flow
  expert's **ODE output at t=1** for the same noise `z` (direct L2 regression, not flow-matching).
- **Actor** (Eq. 9): `L_π(ω) = E[−Q_φ(s, μ_ω(s,z))] + α · L_distill(ω)`. `α` is the behavioural coefficient.

The reparameterised `∇_ω Q_φ(s, μ_ω(s,z))` flows through **one** actor forward — no BPTT through the
denoising ODE, no VLA rollout — so it is off-policy and sample-efficient.

## Reward / data (RoboCasa, DEAS protocol)

Experiment 1 follows DEAS on RoboCasa; baselines CO-RFT / DEAS / QC / AQC. Critic needs a reward signal:
DEAS uses demos + reward-labelled rollouts. Start from the Human300 BC pretrain (frozen μ_θ), then offline
RL on reward-labelled data (sparse success or cost_to_goal). Distillation target from the frozen flow
expert; Q from the critic expert.

## Plan

1. `Pi0FQLConfig` (subclass `Pi0Config`): add the one-step actor + critic experts to the gemma MoT +
   their in/out projections; load the frozen BC flow expert from a pretrained pi05 checkpoint.
2. Forward: `embed_suffix_onestep` (noise z + state) and `embed_suffix_critic` (action + state); one joint
   attention over [VLM prefix | expert suffix].
3. `scripts/train_fql.py`: distillation-target rollout of the frozen flow ODE, critic TD, actor distill+Q.
4. Experiments on RoboCasa (DEAS protocol) vs baselines.

Worker D implements the LPS / mean-flow route in parallel; this is the FQL / one-step route.
