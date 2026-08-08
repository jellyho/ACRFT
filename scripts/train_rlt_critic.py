"""Train a QC or ARQ critic on precomputed RL tokens (see annotate_rlt.py).

Thin CLI over ``openpi.rlt_critic``: data loading lives in ``rlt_critic.data``, the update and
diagnostics in ``rlt_critic.training``, the networks in ``rlt_critic.critic``.

Target, for the prefix of length h starting at frame t:

    y_h = sum_{i<h} gamma^i r_{t+i}  +  gamma^h * [episode still running at t+h] * V(s_{t+h})

where V(s') is either the max of Q_target over the stored VLA candidates (--objective td) or a
state-value network trained by expectile regression (--objective iql: no candidate array and no
arg-max).

Usage:
    uv run scripts/train_rlt_critic.py --data $CACHE_DIR/annot/noprop --out .../run --objective iql
"""

import argparse
import json
import logging
import os as _os
import pathlib
import socket as _socket
import time

import flax.serialization as fser
import jax
import jax.numpy as jnp
import optax  # noqa: F401  (re-exported through training; kept for direct experimentation)

from openpi.rlt_critic import annotation as _annot
from openpi.rlt_critic import critic as _critic
from openpi.rlt_critic import training as _training
from openpi.rlt_critic.data import load_data

logger = logging.getLogger(__name__)

STEPS_PER_DISPATCH = 100  # update steps fused into one jit dispatch
LOG_EVERY = 1000


def _where_it_ran() -> dict:
    """Host and GPU, recorded in config/wandb so a slow or dead run can be tied to its node."""
    try:
        d = jax.devices()[0]
        gpu = getattr(d, "device_kind", None) or str(d)
    except Exception:
        gpu = "unknown"
    return {
        "slurm_node": _os.environ.get("SLURMD_NODENAME") or _socket.gethostname(),
        "slurm_job": _os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": _os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": gpu,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("data / output")
    io.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    io.add_argument("--out", type=pathlib.Path, default=None)
    io.add_argument("--max-frames", type=int, default=0, help="train on only the first N frames")
    io.add_argument("--save-every", type=int, default=50_000)
    io.add_argument("--eval-every", type=int, default=5_000, help="within-state diagnostics interval")

    obj = ap.add_argument_group("objective")
    obj.add_argument(
        "--objective",
        choices=["td", "iql", "calql"],
        default="td",
        help="td bootstraps max Q over stored candidates; iql bootstraps an expectile-trained V "
        "(no candidate array, no arg-max)",
    )
    obj.add_argument("--expectile", type=float, default=0.7, help="iql: 0.5 = mean, ->1 approaches max_a Q")
    obj.add_argument(
        "--cql-alpha", type=float, default=1.0, help="calql: weight of the conservative candidate-suppression term"
    )
    obj.add_argument("--cql-temp", type=float, default=0.1, help="calql: logsumexp temperature over the 16 candidates")
    obj.add_argument(
        "--aqc-baseline",
        action="store_true",
        help="iql: also train per-prefix baselines b_h(z) (extra value-net heads) for AQC-style "
        "commit-length selection: score = (Q_h - b_h) / gamma^h at deployment.",
    )
    obj.add_argument("--baseline-expectile", type=float, default=0.9, help="expectile for the per-prefix baselines")
    obj.add_argument(
        "--dueling",
        action="store_true",
        help="iql+arq+scalar only: Q = V + zero-mean advantage, so the Q head fits only the within-state contrast",
    )
    obj.add_argument(
        "--mc-lower-bound",
        action="store_true",
        help="floor the target at the return the data actually collected from this state",
    )
    obj.add_argument(
        "--discount",
        type=float,
        default=0.995,
        help="project default 0.995 (2026-08-07); auto-builds the matching dataset variant. "
        "Pass the annotation's own discount explicitly to opt out.",
    )

    trn = ap.add_argument_group("training")
    trn.add_argument("--steps", type=int, default=200_000)
    trn.add_argument("--batch-size", type=int, default=1024)
    trn.add_argument("--lr", type=float, default=3e-4)
    trn.add_argument("--seed", type=int, default=0)
    trn.add_argument(
        "--bootstrap",
        choices=["target", "online"],
        default="target",
        help="which params score the bootstrap (online = no target network)",
    )
    trn.add_argument("--target-tau", type=float, default=0.005)
    trn.add_argument(
        "--boot-op",
        choices=["max", "softmax", "aqcmax"],
        default="max",
        help="td/calql: bootstrap operator over the candidate pool - hard max, smooth logsumexp max, "
        "or AQC-style baseline-corrected max (trains per-prefix baseline heads alongside)",
    )
    trn.add_argument("--boot-temp", type=float, default=0.05, help="softmax bootstrap temperature")

    arch = ap.add_argument_group("architecture")
    arch.add_argument("--kind", choices=["qc", "arq"], default="arq")
    arch.add_argument("--num-atoms", type=int, default=1, help="1 = scalar Q; >1 = HL-Gauss histogram")
    arch.add_argument("--num-critics", type=int, default=2, help="full-ensemble size (min-aggregated)")
    arch.add_argument(
        "--head-ensemble",
        type=int,
        default=1,
        help="arq: >1 = ONE trunk with K MLP head banks instead of K full critics",
    )
    arch.add_argument("--macro-group-size", type=int, default=2, help="arq: steps per macro token")
    arch.add_argument("--num-layers", type=int, default=3)
    arch.add_argument("--num-heads", type=int, default=8)
    arch.add_argument("--head-dim", type=int, default=48)
    arch.add_argument("--mlp-dim", type=int, default=1024)
    arch.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512], help="qc MLP / iql V widths")
    arch.add_argument(
        "--history",
        type=int,
        default=0,
        help="arq: K past rl_tokens as extra observation positions (short-history conditioning; "
        "single frames are ambiguous under occlusion/repeated motion - Robo-ValueRL ablation)",
    )
    arch.add_argument("--history-stride", type=int, default=8, help="env steps between history frames")

    wb = ap.add_argument_group("wandb")
    wb.add_argument("--wandb-project", default=None)
    wb.add_argument("--wandb-entity", default="RSS-PFT_RLLAB")
    wb.add_argument("--wandb-group", default=None)
    wb.add_argument("--wandb-name", default=None)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args()
    if cfg.dueling and (cfg.objective != "iql" or cfg.num_atoms > 1 or cfg.kind != "arq"):
        raise ValueError("--dueling needs --objective iql, --num-atoms 1 and --kind arq")

    # The discount selects a DATASET (mc_return and the value support move with it); a mismatched
    # request builds the matching annotation in place rather than silently mixing scales.
    cfg.data = _annot.ensure_discount(cfg.data, cfg.discount)
    meta = json.loads((cfg.data / "meta.json").read_text())
    if cfg.discount is None:
        cfg.discount = meta.get("discount", 0.99)
    support = tuple(meta.get("value_support", [0.0, 1.0]))
    logger.info(f"scheme {meta.get('reward_scheme', 'sparse')!r}: discount {cfg.discount}, support {support}")

    data = load_data(cfg.data, max_frames=cfg.max_frames)
    if data.horizon % cfg.macro_group_size:
        raise ValueError(f"macro_group_size {cfg.macro_group_size} must divide horizon {data.horizon}")

    arch = (
        {
            "macro_group_size": cfg.macro_group_size,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "history": cfg.history,
            "mlp_dim": cfg.mlp_dim,
        }
        if cfg.kind == "arq"
        else {"hidden_dims": tuple(cfg.hidden_dims)}
    )
    if cfg.head_ensemble > 1:
        if cfg.kind != "arq":
            raise ValueError("--head-ensemble needs --kind arq")
        net = _critic.make_critic(
            cfg.kind,
            action_dim=data.action_dim,
            horizon=data.horizon,
            num_atoms=cfg.num_atoms,
            head_ensemble=cfg.head_ensemble,
            **arch,
        )
    else:
        net = _critic.Ensemble(
            make_critic=lambda: _critic.make_critic(
                cfg.kind, action_dim=data.action_dim, horizon=data.horizon, num_atoms=cfg.num_atoms, **arch
            ),
            num_critics=cfg.num_critics,
        )
    hl = _critic.HLGauss(v_min=support[0], v_max=support[1], num_atoms=max(cfg.num_atoms, 2))

    run = None
    if cfg.wandb_project:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            name=cfg.wandb_name or (cfg.out.name if cfg.out else f"critic_{cfg.kind}"),
            config=vars(cfg) | {"data": str(cfg.data), "frames": data.token.shape[0]} | _where_it_ran(),
        )

    rng = jax.random.key(cfg.seed)
    _obs0 = data.obs_at(jnp.arange(1), history=cfg.history, history_stride=cfg.history_stride)
    params = net.init(rng, _obs0, data.chunk[:1])
    logger.info(f"{cfg.kind.upper()} critic: {sum(x.size for x in jax.tree.leaves(params)) / 1e6:.2f}M params")

    num_prefixes = 1 if cfg.kind == "qc" else data.chunk.shape[1] // cfg.macro_group_size
    cfg.num_prefixes = num_prefixes
    if cfg.boot_op == "aqcmax":
        cfg.aqc_baseline = True
    v_out_dim = 1 + num_prefixes if cfg.aqc_baseline else 1
    v_net = (
        _critic.ValueNet(hidden_dims=tuple(cfg.hidden_dims), out_dim=v_out_dim)
        if (cfg.objective == "iql" or cfg.boot_op == "aqcmax")
        else None
    )
    v_params = v_net.init(jax.random.fold_in(rng, 1), _obs0) if v_net is not None else {}
    step_fn, tx, tx_v = _training.make_update(data, cfg, net, hl, support, v_net=v_net)
    carry = (params, params, tx.init(params), v_params, tx_v.init(v_params))
    diag = _training.make_diag(data, cfg, net, hl)

    @jax.jit
    def run_chunk(carry, rng, d):
        return jax.lax.scan(lambda c, r: step_fn(d, c, r), carry, jax.random.split(rng, STEPS_PER_DISPATCH))

    out_dir = cfg.out or (cfg.data / f"critic_{cfg.kind}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_config(*, final: bool):
        payload = vars(cfg) | {
            "data": str(cfg.data),
            "out": str(out_dir),
            "action_norm": {"mean": data.action_mean.tolist(), "std": data.action_std.tolist()},
            "proprio_norm": {"mean": data.proprio_mean.tolist(), "std": data.proprio_std.tolist()},
            "v_min": support[0],
            "v_max": support[1],
        }
        if final:
            payload |= _where_it_ran()
        if run is not None:
            payload["wandb_id"] = run.id
        (out_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))

    # Resume with the FULL carry (params + target net + both optimizers): params alone would restart
    # Adam, which is a different optimisation, not a continuation.
    start_step = 0
    state_f = out_dir / "state.msgpack"
    if state_f.exists():
        blob = fser.msgpack_restore(state_f.read_bytes())
        start_step = int(blob["step"])
        carry = jax.tree.map(jnp.asarray, tuple(blob["carry"]))
        logger.info(f"resuming from {state_f} at step {start_step}")

    t0 = time.perf_counter()
    for s in range(start_step, cfg.steps, STEPS_PER_DISPATCH):
        carry, infos = run_chunk(carry, jax.random.fold_in(rng, s), data)
        step = s + STEPS_PER_DISPATCH
        if s % LOG_EVERY == 0:
            info = jax.tree.map(lambda x: float(jnp.mean(x)), infos)
            rate = (step - start_step) / max(time.perf_counter() - t0, 1e-6)
            logger.info(
                f"step {step}/{cfg.steps}  {rate:.0f} it/s  " + "  ".join(f"{k}={v:.4f}" for k, v in info.items())
            )
            if run is not None:
                run.log({f"train/{k}": v for k, v in info.items()} | {"it_per_s": rate}, step=step)
        if cfg.eval_every and step % cfg.eval_every == 0:
            d = {k: float(v) for k, v in diag(carry[0]).items()}
            logger.info(
                f"  [diag @ {step}] range/std {d['diag_step/range_over_std']:.2f}  "
                f"rank {d['diag_step/ranking_demo_vs_cand']:.3f}"
                + (
                    f"  horizon {d.get('diag_step/avg_chosen_horizon', 0):.2f}/{data.horizon}"
                    if cfg.kind == "arq"
                    else ""
                )
            )
            if run is not None:
                run.log(d, step=step)
        if cfg.save_every and step % cfg.save_every == 0 and step < cfg.steps:
            (out_dir / f"params_{step}.msgpack").write_bytes(fser.to_bytes(carry[0]))
            if cfg.objective == "iql" or getattr(cfg, "aqc_baseline", False):
                (out_dir / f"vparams_{step}.msgpack").write_bytes(fser.to_bytes(carry[3]))
            tmp = out_dir / "state.msgpack.tmp"
            tmp.write_bytes(fser.to_bytes({"step": step, "carry": list(carry)}))
            tmp.replace(state_f)
            save_config(final=False)
            logger.info(f"  saved checkpoint params_{step}.msgpack")

    (out_dir / "params.msgpack").write_bytes(fser.to_bytes(carry[0]))
    if cfg.objective == "iql" or getattr(cfg, "aqc_baseline", False):
        (out_dir / "vparams.msgpack").write_bytes(fser.to_bytes(carry[3]))
    state_f.unlink(missing_ok=True)  # finished - a stale resume state would only confuse
    save_config(final=True)
    logger.info(f"saved to {out_dir} ({(time.perf_counter() - t0) / 60:.1f} min)")
    if run is not None:
        diag_f = out_dir / "diag.json"
        if diag_f.exists():
            run.summary.update(
                {f"diag/{k}": v for k, v in json.loads(diag_f.read_text()).items() if isinstance(v, int | float)}
            )
        run.finish()


if __name__ == "__main__":
    main()
