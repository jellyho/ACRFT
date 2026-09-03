"""AWR policy extraction for pi0.5: advantage-weighted flow-BC fine-tune of the action expert.

Provenance:
  - Objective: Peng et al., "Advantage-Weighted Regression" (arXiv 1910.00177), Eq. 8 —
    supervised regression weighted by exp(A/beta). Weight mechanics follow the OFFICIAL code
    xbpeng/awr learning/awr_agent.py exactly: advantages are z-score normalized before
    exponentiation (awr_agent.py:403 `norm_adv = (adv - mean)/(std + eps)`), weights are
    `exp(norm_adv / temp)` (awr_agent.py:407) with temp=1.0 (awr_agent.py:43) and clipped at
    weight_clip=20 (awr_agent.py:41,409-410).
  - The regression loss for a flow policy is pi0.5's own flow-matching loss (openpi
    pi0.py:189-214), weighted per sample -- AWR is agnostic to the regression family (paper
    Sec. 4: "any supervised regression procedure").
  - A = Q(s, a_chunk) - V(s) from the frozen patch critic, precomputed by
    scripts/annotate_advantage.py; the critic and its normalization travel via input_spec.
  - Trainable set: the pi0.5 ACTION EXPERT only (llm suffix "_1" + action/time projections),
    backbone frozen -- extraction refines the head on top of the finished BC representation,
    keeping all seven extraction arms comparable (same frozen backbone, same init).
"""

# ruff: noqa: PLC0415  (heavy imports after argparse for fast --help)

import argparse
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init-ckpt",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/100000"
        ),
    )
    ap.add_argument("--advantage", type=pathlib.Path, default=R / ".scratch/extraction/advantage_fixed_tau9min")
    ap.add_argument("--temp", type=float, default=1.0, help="AWR temperature (awr_agent.py:43)")
    ap.add_argument("--weight-clip", type=float, default=20.0, help="awr_agent.py:41")
    ap.add_argument("--lr", type=float, default=5e-5, help="the BC finetune lr (house convention)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/awr_run1"))
    ap.add_argument(
        "--train-backbone",
        action="store_true",
        help="train the WHOLE model, as the BC finetune did (no freeze_filter) -- otherwise only the "
        "action expert, which keeps arms comparable but gives them a smaller budget than BC",
    )
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="jellyho_")
    ap.add_argument("--wandb-name", default="extract_awr_run1")
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.extraction import data as exdata
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    # ---- advantage -> official AWR weights (z-score, exp, clip) --------------------------------
    q = np.load(a.advantage / "q_data.npy")
    v = np.load(a.advantage / "v_data.npy")
    adv = q - v
    norm_adv = (adv - adv.mean()) / (adv.std() + 1e-5)  # awr_agent.py:403
    weights = np.minimum(np.exp(norm_adv / a.temp), a.weight_clip).astype(np.float32)  # :407,:409
    print(
        f"weights: mean {weights.mean():.3f} max {weights.max():.1f} frac@clip {(weights >= a.weight_clip).mean():.4f}"
    )

    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    ds = exdata.AnnotatedBC(dataset, {"w": weights})
    it = exdata.make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)

    # ---- pi0.5 with the BC checkpoint's params -------------------------------------------------
    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    # proven loader path (train_fql.py --init-base): base params load, anything missing keeps init
    loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    graphdef = nnx.graphdef(model)
    params = nnx.state(model)

    # trainable = action expert only (same filter family as train_fql.py flow_filter)
    expert_filter = nnx.Any(
        nnx_utils.PathRegex(".*llm.*_1.*"),
        nnx_utils.PathRegex(".*(action_(in|out)_proj|time_mlp_(in|out)|state_proj).*"),
    )
    if a.train_backbone:
        # BC trained everything (its config sets no freeze_filter), so matching its budget means
        # matching what it was allowed to move, not just steps and batch.
        expert_filter = nnx.Param
    tx = optax.adam(a.lr)
    opt = tx.init(params.filter(expert_filter))

    def loss_fn(model, rng, obs, actions, w):
        # per-sample flow-matching loss (pi0.py compute_loss returns [B, H] per-step MSE)
        per = model.compute_loss(rng, obs, actions, train=True)  # [B, H]
        per = per.mean(axis=tuple(range(1, per.ndim)))  # -> [B]
        return jnp.mean(w * per), {"unweighted": jnp.mean(per)}

    import functools

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt, rng, obs, actions, w):
        model = nnx.merge(graphdef, params)
        (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, expert_filter), has_aux=True)(
            model, rng, obs, actions, w
        )
        p = params.filter(expert_filter)
        upd, opt = tx.update(grads, opt, p)
        nnx.update(model, optax.apply_updates(p, upd))
        return nnx.state(model), opt, loss, info

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(
            project="yam-rlt",
            entity=a.wandb_entity,
            name=a.wandb_name,
            group="extraction",
            config=vars(a) | {"method": "awr"},
        )
        print(f"wandb: {run.url}", flush=True)

    def save(step_i):
        from openpi.extraction.checkpoint import save_servable

        # the WHOLE model in the BC layout (<out>/<step>/params + assets): servable as-is, no export
        # step, whichever subset was trained -- see openpi/extraction/checkpoint.py
        path = save_servable(a.out / f"{step_i}", params.to_pure_dict(), assets_from=a.init_ckpt)
        print(f"saved {path}", flush=True)

    import time

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        obs, actions, ann = next(it)
        rng, k = jax.random.split(rng)
        params, opt, loss, info = step(params, opt, k, obs, jnp.asarray(actions), jnp.asarray(ann["w"]))
        if s % 100 == 0:
            print(
                f"step {s:6d}  awr_loss {float(loss):.5f}  bc_loss {float(info['unweighted']):.5f}  ({(s + 1) / (time.time() - t0):.2f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log({"awr_loss": float(loss), "bc_loss": float(info["unweighted"])}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print("AWR extraction done.", flush=True)


if __name__ == "__main__":
    main()
