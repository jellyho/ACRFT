"""Is the learned value a function of distance-to-goal at all?

Every finished run reports q_demo_mean ~0.003 against mc_return_mean 0.182, between-state std ~0.002,
and spearman(Q, mc_return) ~ -0.73. That is not "the critic ignores the action" — it is "the critic is
very nearly a constant, and what little it varies runs the wrong way". Three explanations, and they
are distinguished by how Q behaves as a function of steps-to-success:

  UNDER-PROPAGATED  Q is correct near the terminal and decays to 0 too fast going back. With gamma
                    0.99 and a median episode of 521 steps, the true value is gamma^d, already 0.005
                    at d=521, so most states genuinely ARE near zero. Under-propagation looks like a
                    curve that hugs the true one near d=0 and falls below it too early.
  BUG               Q is unrelated or inverted with respect to d.
  CONVERGED-CORRECT Q tracks gamma^d over the whole range, and the low mean is just what the reward
                    scheme implies — in which case the value function is fine and only the ACTION
                    dependence is missing.

    uv run slurm/value_vs_distance.py --run $CRITIC_RUNS/pro_main/base
"""

import argparse
import json
import pathlib

import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np

import openpi.rlt_critic.critic as _critic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=pathlib.Path, help="dir with params.msgpack + config.json")
    ap.add_argument("--n-states", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    data = pathlib.Path(cfg["data"])
    meta = json.loads((data / "meta.json").read_text())
    T, D, H, A, N = (meta[k] for k in ("num_frames", "token_dim", "horizon", "action_dim", "num_samples"))
    g = float(meta.get("discount", 0.99))

    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]
    mm = lambda n, s, d: np.asarray(np.memmap(data / f"{n}.dat", d, "r", shape=s))  # noqa: E731
    done, episode = mm("done", (T,), np.int8), mm("episode_index", (T,), np.int32)
    mc = mm("mc_return", (T,), np.float32)

    dist = np.full(T, -1, np.int64)
    for e in np.unique(episode):
        w = np.flatnonzero(episode == e)
        fired = w[done[w] > 0]
        if len(fired):
            t = fired[0]
            dist[w[0] : t + 1] = np.arange(t - w[0], -1, -1)
    rng = np.random.default_rng(args.seed)
    ok = np.flatnonzero(dist >= 0)
    pick = np.sort(rng.choice(ok, size=min(args.n_states, len(ok)), replace=False))

    obs = mm("rl_token", (T, D), dt)[pick].astype(np.float32)
    if cfg.get("use_proprio"):
        pdim = meta["proprio_dim"]
        pro = mm("proprio", (T, pdim), np.float32)[pick]
        mu, sd = np.asarray(meta["proprio_mean"], np.float32), np.asarray(meta["proprio_std"], np.float32)
        obs = np.concatenate([obs, np.where(sd > 1e-6, (pro - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)], axis=1)
    chunk = mm("action_chunk", (T, H, A), dt)[pick].astype(np.float32)

    kind, atoms = cfg.get("kind", "arq"), cfg.get("num_atoms", 1)
    arch = (
        {
            "macro_group_size": cfg["macro_group_size"],
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "head_dim": cfg["head_dim"],
            "mlp_dim": cfg["mlp_dim"],
        }
        if kind == "arq"
        else {"hidden_dims": tuple(cfg["hidden_dims"])}
    )
    if cfg.get("use_proprio") and cfg.get("proprio_mode") == "token":
        arch["proprio_dim"] = meta["proprio_dim"]
    net = _critic.Ensemble(
        make_critic=lambda: _critic.make_critic(kind, action_dim=A, horizon=H, num_atoms=atoms, **arch),
        num_critics=cfg["num_critics"],
    )
    hl = _critic.HLGauss(v_min=cfg["v_min"], v_max=cfg["v_max"], num_atoms=max(atoms, 2))
    params = fser.msgpack_restore((args.run / "params.msgpack").read_bytes())

    @jax.jit
    def qf(p, o, c):
        out = net.apply(p, o, c)
        out = hl.from_logits(out) if atoms > 1 else out
        out = jnp.min(out, axis=0)
        return out[..., -1] if out.ndim == 2 else out

    q = np.asarray(qf(params, jnp.asarray(obs), jnp.asarray(chunk)))
    d, m = dist[pick], mc[pick]

    print(f"run   : {args.run}   gamma {g}")
    print(f"Q     : mean {q.mean():.6f}  std {q.std():.6f}  min {q.min():.6f}  max {q.max():.6f}")
    print(f"mc    : mean {m.mean():.6f}  max {m.max():.6f}")
    print(f"corr(Q, mc_return)      = {np.corrcoef(q, m)[0, 1]:+.4f}")
    print(f"corr(Q, distance)       = {np.corrcoef(q, d)[0, 1]:+.4f}   (true value falls with distance)")
    print(f"\n{'steps to goal':>16} {'n':>6} {'true mc (g^d)':>14} {'learned Q':>12} {'ratio':>8}")
    bins = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 10**6)]
    rows = []
    for lo, hi in bins:
        s = (d >= lo) & (d < hi)
        if not s.any():
            continue
        tm, tq = float(m[s].mean()), float(q[s].mean())
        rows.append({"lo": lo, "hi": hi, "n": int(s.sum()), "mc": tm, "q": tq})
        print(f"{f'{lo}-{hi}':>16} {int(s.sum()):>6} {tm:>14.6f} {tq:>12.6f} {tq / (tm + 1e-12):>8.3f}")

    near = (d < 16).mean()
    print(f"\nfraction of sampled states within 16 steps of the goal: {near:.3%}")
    if q.std() < 1e-3:
        print("VERDICT: Q is essentially constant — it is not a value function of the state at all.")
    elif np.corrcoef(q, d)[0, 1] > 0:
        print("VERDICT: Q INCREASES with distance to goal — inverted, this is a defect, not slow propagation.")
    else:
        print("VERDICT: Q decreases with distance as it should; check the ratio column for how far it propagated.")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "gamma": g,
                    "q_mean": float(q.mean()),
                    "q_std": float(q.std()),
                    "corr_q_mc": float(np.corrcoef(q, m)[0, 1]),
                    "corr_q_dist": float(np.corrcoef(q, d)[0, 1]),
                    "bins": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
