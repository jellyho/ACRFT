"""Offline FQL training for Pi0FQL: distill the frozen flow policy into a one-step actor + learn an
in-VLA critic expert, then improve the actor by maximizing Q (Flow Q-Learning, Park et al. ICML'25).

Frozen: paligemma VLM (expert 0) + flow BC policy mu_theta (expert 1) + img + flow projections -- loaded
from a pretrained pi05 checkpoint (CheckpointWeightLoaderKeepMissing: base loads, new experts keep init).
Trainable: one-step actor mu_omega (expert 2) + critic Q_phi (expert 3) + their projections.

Objectives (per FQL, offline transitions (s, a, r, s', done)):
  L_critic = CE( HL-Gauss[ max(r + gamma (1-done) Qbar(s', mu_omega(s',z')),  MC) ],  Q_phi(s, a) )
             # Eq.1 + the patch-critic's MC floor (--no-mc-floor disables): the bootstrapped target is
             # floored by the transition's observed Monte-Carlo return, Cal-QL's max(Q, V^mu) in
             # target form -- with sparse rewards this stops an uncalibrated early critic free-fall.
  L_distill = || mu_omega(s, z) - mu_theta_ODE(s, z) ||^2                                     # Eq.7
  L_actor  = -E[ Q_phi(s, mu_omega(s, z)) ] + alpha * L_distill                               # Eq.9

Critic and actor params are DISJOINT, so we take two filtered grads (separate optimizers): L_critic ->
critic params (the actor's a' is stop-gradient), L_actor -> actor params (the critic Q is read but not
updated by the actor step). Qbar is an EMA target critic.

Staged recipe (QC-FQL):
  stage 0  pi05 BC pretraining -- the frozen base loaded via --init-base.
  stage 1  --critic-warmup-steps N: the critic regresses the PURE MC return (y = MC, no bootstrap,
           no actor involved) so it is calibrated before anything depends on it; the actor meanwhile
           trains on distillation ONLY (no Q term) -- the two are fully decoupled in this stage.
  stage 2  actor-critic proper: critic TD (with the MC floor), actor distill + Q-max.
--critic-backbone {never,warmup,always} decides whether the CRITIC's gradient also flows into the
VLM backbone (prefix KV not stop-gradient'd, backbone params join the critic optimizer). never keeps
the backbone a frozen feature extractor; warmup lets the value signal shape it only while the target
is the trustworthy MC return; always is full value backprop (B200-sized memory).

Data: --synthetic runs a shape/consistency smoke on random transitions (no data needed). Without it,
trains on the real 347-episode YAM chunk-transition set (scripts/yam_fql_data.py: house patch-critic
reward conventions -- cost_to_goal, gamma 0.99964, failure homing truncation + terminal anchor). The
backup over a chunk is SMDP-style: y = R_chunk + gamma^H (1-done) Qbar(s_{t+H}, a'), so the discount
applied to the bootstrap is gamma**action_horizon, not gamma.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import pathlib
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-base", type=str, default=None, help="pretrained pi05 params path (frozen base)")
    ap.add_argument("--synthetic", action="store_true", help="smoke on random transitions (no dataset)")
    ap.add_argument("--action-dim", type=int, default=14)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--num-atoms", type=int, default=101)
    ap.add_argument("--v-min", type=float, default=-100.0)
    ap.add_argument("--v-max", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=10.0, help="distillation coefficient in L_actor")
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument(
        "--mc-floor",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="floor the TD target with the transition's MC return (house patch-critic recipe)",
    )
    ap.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=0,
        help="stage-1 length: critic regresses pure MC returns, actor trains distill-only",
    )
    ap.add_argument(
        "--critic-backbone",
        choices=["never", "warmup", "always"],
        default="never",
        help="when the critic gradient may flow into the VLM backbone",
    )
    ap.add_argument("--flow-ode-steps", type=int, default=10)
    # expert variants: real by default; 'dummy' everywhere gives a CPU-runnable trainer smoke
    ap.add_argument("--paligemma-variant", default="gemma_2b")
    ap.add_argument("--flow-variant", default="gemma_300m")
    ap.add_argument("--actor-variant", default="gemma_300m")
    ap.add_argument("--critic-variant", default="gemma_150m")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/fql"))
    # ---- real-data (YAM) path: scripts/yam_fql_data.py conventions ----
    ap.add_argument("--yam-repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--yam-root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--outcomes", default="/data5/jellyho/ACRFT/openpi/.scratch/yam_outcomes_347.jsonl")
    ap.add_argument("--homing-onsets", default="/data5/jellyho/ACRFT/openpi/.scratch/yam_homing_onsets.json")
    ap.add_argument("--h-goal", type=int, default=3)
    ap.add_argument("--failure-reward", type=float, default=None, help="failure terminal anchor; default v_min")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="yam-rlt")
    ap.add_argument("--wandb-entity", default="RSS-PFT_RLLAB")
    ap.add_argument("--wandb-name", default=None)
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    import openpi.models.model as _model
    from openpi.models.pi0_fql import Pi0FQLConfig
    from openpi.rlt_critic.critic import HLGauss
    import openpi.shared.array_typing as at
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    cfg = Pi0FQLConfig(
        action_dim=a.action_dim,
        action_horizon=a.action_horizon,
        pi05=True,
        paligemma_variant=a.paligemma_variant,
        action_expert_variant=a.flow_variant,
        actor_expert_variant=a.actor_variant,
        critic_expert_variant=a.critic_variant,
        fql_num_atoms=a.num_atoms,
        fql_v_min=a.v_min,
        fql_v_max=a.v_max,
        fql_alpha=a.alpha,
        fql_discount=a.discount,
        fql_target_tau=a.target_tau,
        fql_flow_ode_steps=a.flow_ode_steps,
        fql_mc_floor=a.mc_floor,
    )
    hl = HLGauss(a.v_min, a.v_max, a.num_atoms)

    # trainable = actor expert (gemma "_2") + critic expert (gemma "_3") + their in/out projections.
    actor_filter = nnx.Any(nnx_utils.PathRegex(".*llm.*_2.*"), nnx_utils.PathRegex(".*actor_(in|out)_proj.*"))
    critic_filter = nnx.Any(nnx_utils.PathRegex(".*llm.*_3.*"), nnx_utils.PathRegex(".*critic_(in|out)_proj.*"))
    # the VLM backbone (paligemma + siglip), excluding every expert suffix -- what the critic ALSO
    # updates when --critic-backbone allows it. The EMA target tracks only critic_filter either way:
    # the target's job is a slow-moving Q head, not a slow-moving backbone.
    backbone_filter = nnx.All(nnx_utils.PathRegex(".*PaliGemma.*"), nnx.Not(nnx_utils.PathRegex(".*_[123].*")))
    critic_bb_filter = nnx.Any(critic_filter, backbone_filter)

    print("building Pi0FQL...", flush=True)
    model = cfg.create(jax.random.key(0))

    # ---- load the frozen base from a pretrained pi05 checkpoint (new experts keep their init) ----
    if a.init_base is not None:
        graphdef, state = nnx.split(model)
        loaded = CheckpointWeightLoaderKeepMissing(a.init_base).load(state.to_pure_dict())
        state.replace_by_pure_dict(loaded)
        model = nnx.merge(graphdef, state)
        print(f"loaded frozen base from {a.init_base}", flush=True)

    graphdef = nnx.graphdef(model)
    params = nnx.state(model)
    # EMA target: only the CRITIC subset. Copying the full 3.4B-param state (frozen VLM included)
    # would burn ~13.6GB for weights that never move; the target model is the online model with just
    # its critic params swapped for the EMA copy.
    target_critic = jax.tree.map(jnp.copy, params.filter(critic_filter))
    tx_c = optax.adam(a.lr)
    tx_a = optax.adam(a.lr)
    critic_train_filter = critic_bb_filter if a.critic_backbone != "never" else critic_filter
    opt_c = tx_c.init(params.filter(critic_train_filter))
    opt_a = tx_a.init(params.filter(actor_filter))

    # The VLM prefix (SigLIP x3 cams + 2b attention) is by far the most expensive forward, and it is
    # FROZEN -- so each loss computes it ONCE per observation (stop-gradient) and every expert call
    # reuses the KV. The first smoke recomputed it six times per training step and OOM'd an L40S.
    def _kv(model, obs, *, grad=False):
        kv = model._prefix_kv(obs)
        return kv if grad else jax.tree.map(jax.lax.stop_gradient, kv)

    def warmup_critic_loss_fn(model, batch, bb_grad):
        """Stage 1: pure MC regression -- no bootstrap, no actor, no target network."""
        obs, act, _rew, _nobs, _done, mc = batch
        kv, pm = _kv(model, obs, grad=bb_grad)
        tgt = jax.lax.stop_gradient(hl.to_probs(jnp.clip(mc, a.v_min, a.v_max)))
        q_logits = model.critic_logits(obs, act, kv, pm)
        loss = jnp.mean(-jnp.sum(tgt * jax.nn.log_softmax(q_logits, -1), -1))
        return loss, {"q_mean": jnp.mean(hl.from_logits(q_logits)), "td_target": jnp.mean(mc)}

    def make_critic_loss_fn(bb_grad):  # closure: nnx transforms cannot resolve keyword-only args
        def critic_loss_fn(model, target_model, batch, rng):
            obs, act, rew, nobs, done, mc = batch
            kv, pm = _kv(model, obs, grad=bb_grad)
            kv_n, pm_n = _kv(model, nobs)
            zr = jax.random.normal(rng, act.shape)
            a_next = jax.lax.stop_gradient(model.actor(nobs, zr, kv_n, pm_n))  # a' ~ mu_omega(s', z')
            qn = hl.from_logits(jax.lax.stop_gradient(target_model.critic_logits(nobs, a_next, kv_n, pm_n)))
            # SMDP chunk backup: rew is the discounted H-step sum, the successor is H steps away,
            # so the bootstrap carries gamma**H (a plain gamma here would over-bootstrap ~H/gamma x).
            y = rew + (a.discount**a.action_horizon) * (1.0 - done) * qn
            floor_frac = jnp.zeros(())
            if a.mc_floor:
                # the observed discounted return from s is a certificate: a bootstrapped target below it
                # is provably too low, so lift it (and log how often the floor is doing work).
                floor_frac = jnp.mean((mc > y).astype(jnp.float32))
                y = jnp.maximum(y, mc)
            tgt = jax.lax.stop_gradient(hl.to_probs(jnp.clip(y, a.v_min, a.v_max)))
            q_logits = model.critic_logits(obs, act, kv, pm)
            loss = jnp.mean(-jnp.sum(tgt * jax.nn.log_softmax(q_logits, -1), -1))
            return loss, {
                "q_mean": jnp.mean(hl.from_logits(q_logits)),
                "td_target": jnp.mean(y),
                "mc_floor_frac": floor_frac,
            }

        return critic_loss_fn

    def actor_loss_fn(model, batch, rng):
        obs = batch[0]
        kv, pm = _kv(model, obs)
        z = jax.random.normal(rng, (obs.state.shape[0], a.action_horizon, a.action_dim))
        a_theta = jax.lax.stop_gradient(model.flow_ode(obs, z, kv_cache=kv, prefix_mask=pm))
        a_omega = model.actor(obs, z, kv, pm)
        l_distill = jnp.mean(jnp.square(a_omega - a_theta))
        qpi = hl.from_logits(model.critic_logits(obs, a_omega, kv, pm))  # grad only to actor (filter)
        l_q = -jnp.mean(qpi)
        return l_q + a.alpha * l_distill, {"l_distill": l_distill, "q_pi": -l_q}

    def warmup_actor_loss_fn(model, batch, rng):
        """Stage 1 actor: distillation ONLY (the critic is not trustworthy yet, so no Q term)."""
        obs = batch[0]
        kv, pm = _kv(model, obs)
        z = jax.random.normal(rng, (obs.state.shape[0], a.action_horizon, a.action_dim))
        a_theta = jax.lax.stop_gradient(model.flow_ode(obs, z, kv_cache=kv, prefix_mask=pm))
        a_omega = model.actor(obs, z, kv, pm)
        l_distill = jnp.mean(jnp.square(a_omega - a_theta))
        return a.alpha * l_distill, {"l_distill": l_distill, "q_pi": jnp.zeros(())}

    # donate params/targets/opt states: without donation the jit holds input AND output copies of
    # the 3.15B-param state (~25GB just in weights), which is what OOM'd the L40S even after the
    # prefix-KV sharing fix. Donation makes the update in-place.
    #
    # Both phases differentiate over critic_train_filter so the gradient tree always matches the
    # critic optimizer state; whether the backbone part of that gradient is nonzero is decided by
    # bb_grad (stop-gradient on the prefix KV), not by reshaping trees between phases.
    def _make_step(*, warmup: bool, bb_grad: bool):
        @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3))
        def step(params, target_critic, opt_c, opt_a, batch, rng):
            rc, ra = jax.random.split(rng)
            model = nnx.merge(graphdef, params)
            if warmup:
                (lc, ci), gc = nnx.value_and_grad(
                    warmup_critic_loss_fn, argnums=nnx.DiffState(0, critic_train_filter), has_aux=True
                )(model, batch, bb_grad)
                (la, ai), ga = nnx.value_and_grad(
                    warmup_actor_loss_fn, argnums=nnx.DiffState(0, actor_filter), has_aux=True
                )(model, batch, ra)
                ci = {**ci, "mc_floor_frac": jnp.ones(())}  # stage 1 IS the MC target
            else:
                target_model = nnx.merge(graphdef, params)
                nnx.update(target_model, target_critic)  # online everywhere, EMA critic
                (lc, ci), gc = nnx.value_and_grad(
                    make_critic_loss_fn(bb_grad), argnums=nnx.DiffState(0, critic_train_filter), has_aux=True
                )(model, target_model, batch, rc)
                (la, ai), ga = nnx.value_and_grad(actor_loss_fn, argnums=nnx.DiffState(0, actor_filter), has_aux=True)(
                    model, batch, ra
                )
            pc = params.filter(critic_train_filter)
            uc, opt_c_new = tx_c.update(gc, opt_c, pc)
            nnx.update(model, optax.apply_updates(pc, uc))
            pa = params.filter(actor_filter)
            ua, opt_a_new = tx_a.update(ga, opt_a, pa)
            nnx.update(model, optax.apply_updates(pa, ua))
            params_new = nnx.state(model)
            target_critic_new = jax.tree.map(
                lambda t, p: a.target_tau * p + (1 - a.target_tau) * t,
                target_critic,
                params_new.filter(critic_filter),
            )
            return params_new, target_critic_new, opt_c_new, opt_a_new, {"l_critic": lc, "l_actor": la, **ci, **ai}

        return step

    bb_warm = a.critic_backbone in ("warmup", "always")
    bb_ac = a.critic_backbone == "always"
    step_warmup = _make_step(warmup=True, bb_grad=bb_warm) if a.critic_warmup_steps > 0 else None
    step_ac = _make_step(warmup=False, bb_grad=bb_ac)

    # ---- data ----
    def synthetic_batch(rng):
        k = jax.random.split(rng, 4)
        res = _model.IMAGE_RESOLUTION
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        with at.disable_typechecking():
            obs = _model.Observation(
                images={n: jax.random.uniform(k[i], (a.batch, *res, 3)) * 2 - 1 for i, n in enumerate(names)},
                image_masks={n: jnp.ones((a.batch,), bool) for n in names},
                state=jnp.zeros((a.batch, a.action_dim), jnp.float32),
                tokenized_prompt=jnp.zeros((a.batch, cfg.max_token_len), jnp.int32),
                tokenized_prompt_mask=jnp.ones((a.batch, cfg.max_token_len), bool),
            )
            nobs = dataclasses.replace(obs)
        act = jax.random.normal(k[3], (a.batch, a.action_horizon, a.action_dim))
        rew = -jnp.ones((a.batch,), jnp.float32)
        done = jnp.zeros((a.batch,), jnp.float32)
        # synthetic MC returns spread over the value support so the floor branch actually fires for
        # a fraction of the batch (a constant below every TD target would silently test nothing).
        mc = jax.random.uniform(k[2], (a.batch,), minval=a.v_min, maxval=a.v_max)
        return obs, act, rew, nobs, done, mc

    data_iter = None
    if not a.synthetic:
        if a.init_base is None:
            raise ValueError("real-data training needs --init-base (frozen flow expert + its norm stats)")
        from yam_fql_data import YamFQLTransitions  # scripts/ is on sys.path when run as a script
        from yam_fql_data import make_loader  # scripts/ is on sys.path when run as a script

        ds = YamFQLTransitions(
            repo_id=a.yam_repo_id,
            root=a.yam_root,
            horizon=a.action_horizon,
            bc_assets_dir=str(pathlib.Path(a.init_base).parent / "assets"),
            outcomes_path=a.outcomes,
            homing_path=a.homing_onsets,
            h_goal=a.h_goal,
            discount=a.discount,
            failure_reward=a.failure_reward,
        )
        print(f"YAM transitions: {len(ds)} base frames, v_min {ds.v_min:.1f}", flush=True)
        data_iter = make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(
            project=a.wandb_project,
            entity=a.wandb_entity,
            name=a.wandb_name,
            group="fql-yam",
            config={k: str(v) for k, v in vars(a).items()},
        )
        print(f"wandb: {run.url}", flush=True)

    def save(step_i):
        # trainable experts + EMA target only; the frozen base is reproducible from --init-base
        import orbax.checkpoint as ocp

        path = (a.out / f"{step_i}").absolute()
        state = {
            "actor": params.filter(actor_filter).to_pure_dict(),
            "critic": params.filter(critic_filter).to_pure_dict(),
            "target_critic": target_critic.to_pure_dict(),
        }
        with ocp.StandardCheckpointer() as ckptr:
            ckptr.save(path, state, force=True)
        print(f"saved {path}", flush=True)

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        rng, kb, ks = jax.random.split(rng, 3)
        batch = synthetic_batch(kb) if a.synthetic else next(data_iter)
        step = step_warmup if (step_warmup is not None and s < a.critic_warmup_steps) else step_ac
        params, target_critic, opt_c, opt_a, info = step(params, target_critic, opt_c, opt_a, batch, ks)
        if s % 10 == 0 or s == a.steps - 1:
            i = {k: float(v) for k, v in info.items()}
            if run is not None:
                run.log({**i, "stage": 1 if (step is step_warmup) else 2}, step=s)
            print(
                f"step {s:5d}  l_critic {i['l_critic']:.4f}  l_actor {i['l_actor']:.4f}  "
                f"q_mean {i['q_mean']:.3f}  l_distill {i['l_distill']:.4f}  q_pi {i['q_pi']:.3f}  "
                f"({(s + 1) / (time.time() - t0):.2f} it/s)",
                flush=True,
            )
        if not a.synthetic and ((s + 1) % a.save_every == 0 or s == a.steps - 1):
            save(s + 1)
    print("FQL training done." if not a.synthetic else "FQL train-step smoke OK.", flush=True)


if __name__ == "__main__":
    main()
