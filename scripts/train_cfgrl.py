"""CFGRL extraction for pi0.5: optimality-conditioned fine-tune + CFG sampling.

Provenance in src/openpi/models/pi0_cfgrl.py (official kvfrans/cfgrl value-based variant,
rlbase/algs_offline/iql_diffusion.py). Label O = 1{A > 0} with A from annotate_advantage.py
(iql_diffusion.py:157). Trainable: action expert + opt_embed, backbone frozen (comparability
across the seven extraction arms; same init BC checkpoint).
"""

# ruff: noqa: PLC0415

import argparse
import functools
import pathlib
import time

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
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16, help="doubled internally for the two CFG branches")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/acrft_ckpts/extraction/cfgrl_run1")
    )
    ap.add_argument(
        "--train-backbone",
        action="store_true",
        help="train the WHOLE model, as the BC finetune did (no freeze_filter) -- otherwise only the "
        "action expert, which keeps arms comparable but gives them a smaller budget than BC",
    )
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="jellyho_")
    ap.add_argument("--wandb-name", default="extract_cfgrl_run1")
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.extraction import data as exdata
    from openpi.models.pi0_cfgrl import Pi0CFGRLConfig
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    q = np.load(a.advantage / "q_data.npy")
    v = np.load(a.advantage / "v_data.npy")
    label = (q - v > 0).astype(np.float32)  # iql_diffusion.py:157 — hard indicator, threshold 0
    print(f"optimality labels: frac positive {label.mean():.3f}")

    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    ds = exdata.AnnotatedBC(dataset, {"o": label})
    it = exdata.make_loader(ds, batch_size=a.batch, num_workers=a.num_workers)

    mcfg = Pi0CFGRLConfig(pi05=True, action_horizon=cfg.model.action_horizon, action_dim=cfg.model.action_dim)
    model = mcfg.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    graphdef = nnx.graphdef(model)
    params = nnx.state(model)

    train_filter = nnx.Any(
        nnx_utils.PathRegex(".*llm.*_1.*"),
        nnx_utils.PathRegex(".*(action_(in|out)_proj|time_mlp_(in|out)|state_proj|opt_embed).*"),
    )
    if a.train_backbone:
        # BC trained everything (its config sets no freeze_filter), so matching its budget
        # means matching what it was allowed to move, not just steps and batch.
        train_filter = nnx.Param
    tx = optax.adam(a.lr)
    opt = tx.init(params.filter(train_filter))

    def loss_fn(model, rng, obs, actions, label):
        return model.compute_loss_cfgrl(rng, obs, actions, label)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt, rng, obs, actions, label):
        model = nnx.merge(graphdef, params)
        (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, train_filter), has_aux=True)(
            model, rng, obs, actions, label
        )
        p = params.filter(train_filter)
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
            config={k: str(v) for k, v in vars(a).items()} | {"method": "cfgrl"},
        )
        print(f"wandb: {run.url}", flush=True)

    def save(step_i):
        import orbax.checkpoint as ocp

        path = (a.out / f"{step_i}").absolute()
        with ocp.StandardCheckpointer() as c:
            c.save(path, {"expert": params.filter(train_filter).to_pure_dict()}, force=True)
        print(f"saved {path}", flush=True)

    rng = jax.random.key(0)
    t0 = time.time()
    for s in range(a.steps):
        obs, actions, ann = next(it)
        rng, k = jax.random.split(rng)
        params, opt, loss, info = step(params, opt, k, obs, jnp.asarray(actions), jnp.asarray(ann["o"]))
        if s % 100 == 0:
            print(
                f"step {s:6d}  loss {float(loss):.5f}  cond {float(info['cond']):.5f}  "
                f"uncond {float(info['uncond']):.5f}  ({(s + 1) / (time.time() - t0):.2f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log({k2: float(v2) for k2, v2 in info.items()} | {"loss": float(loss)}, step=s)
        if (s + 1) % a.save_every == 0 or s == a.steps - 1:
            save(s + 1)
    print("CFGRL extraction done.", flush=True)


if __name__ == "__main__":
    main()
