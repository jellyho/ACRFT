"""Measure the learned value profile ACROSS prefix lengths for trained ARQ critics.

`prefix_bias_analysis.py` shows the *targets* are asymmetric: a prefix-h head can never be trained
above gamma^h, because prefixes that cross the terminal produce no transition. This measures whether
that asymmetry actually lands in the learned Q, and whether it explains the short commitments seen in
rollout video.

Deployment (`eval_critic.make_policy_fn`) takes a joint arg-max over (candidate, prefix) and executes
`(pp+1)*macro` steps, so the quantity that decides commit length is `argmax_p Q(z, a, p)`. Reported:

  mean_q_by_prefix     the learned profile. Falling with p = a bias toward short commitments.
  argmax_hist          where the deployment arg-max actually lands, per prefix.
  frac_shortest        how often it picks the shortest prefix. 1.0 = adaptive chunking has degenerated.
  spearman_q_vs_p      monotonicity of the profile; -1 = strictly decreasing in prefix length.
  near_goal_*          the same, restricted to states close to the terminal. The coverage asymmetry
                       bites hardest there, so if the mechanism is real the tilt is stronger here.

    uv run slurm/prefix_profile.py --runs /scratch/jellyho/acrft/critic_runs/abl_main
"""

import argparse
import json
import pathlib

import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np

import openpi.rlt_critic.critic as _critic


def load_data(data: pathlib.Path, n_states: int, seed: int, use_proprio: bool = False):
    import ml_dtypes

    meta = json.loads((data / "meta.json").read_text())
    T, D, H, A, N = (meta[k] for k in ("num_frames", "token_dim", "horizon", "action_dim", "num_samples"))
    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]
    mm = lambda n, s, d: np.asarray(np.memmap(data / f"{n}.dat", d, "r", shape=s))  # noqa: E731
    done = mm("done", (T,), np.int8)
    episode = mm("episode_index", (T,), np.int32)

    # steps-to-terminal per frame: the axis the coverage asymmetry acts along.
    dist = np.full(T, 10**6, np.int64)
    for e in np.unique(episode):
        w = np.flatnonzero(episode == e)
        fired = w[done[w] > 0]
        if len(fired):
            t = fired[0]
            dist[w[0] : t + 1] = np.arange(t - w[0], -1, -1)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(np.flatnonzero(dist < 10**6), size=min(n_states, int((dist < 10**6).sum())), replace=False))
    obs = mm("rl_token", (T, D), dt)[pick].astype(np.float32)
    if use_proprio:
        # Must match the trainer exactly: same columns, same per-dim z-scoring, constant dims zeroed.
        pdim = meta["proprio_dim"]
        pro = mm("proprio", (T, pdim), np.float32)[pick]
        mu, sd = np.asarray(meta["proprio_mean"], np.float32), np.asarray(meta["proprio_std"], np.float32)
        obs = np.concatenate([obs, np.where(sd > 1e-6, (pro - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)], axis=1)
    return meta, jnp.asarray(obs), jnp.asarray(mm("base_action", (T, N, H, A), dt)[pick]), dist[pick]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=pathlib.Path, help="sweep dir holding <run>/params.msgpack")
    ap.add_argument("--n-states", type=int, default=2048)
    ap.add_argument("--near-goal", type=int, default=32, help="'near goal' = within this many steps of success")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    results = {}
    cache = {}
    for d in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        params_f, cfg_f = d / "params.msgpack", d / "config.json"
        if not (params_f.exists() and cfg_f.exists()):
            continue
        cfg = json.loads(cfg_f.read_text())
        if cfg.get("kind") != "arq":  # only ARQ has a prefix axis
            continue
        key = (cfg["data"], bool(cfg.get("use_proprio")))
        if key not in cache:
            cache[key] = load_data(pathlib.Path(cfg["data"]), args.n_states, args.seed, bool(cfg.get("use_proprio")))
        meta, tok, cand, dist = cache[key]
        H, A, N = meta["horizon"], meta["action_dim"], meta["num_samples"]
        g = cfg["macro_group_size"]

        net = _critic.Ensemble(
            make_critic=lambda: _critic.make_critic(
                "arq",
                action_dim=A,
                horizon=H,
                num_atoms=cfg["num_atoms"],
                macro_group_size=g,
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                head_dim=cfg["head_dim"],
                mlp_dim=cfg["mlp_dim"],
                **({"proprio_dim": meta["proprio_dim"]} if cfg.get("proprio_mode") == "token" and cfg.get("use_proprio") else {}),
            ),
            num_critics=cfg["num_critics"],
        )
        hl = _critic.HLGauss(v_min=cfg["v_min"], v_max=cfg["v_max"], num_atoms=max(cfg["num_atoms"], 2))
        params = fser.msgpack_restore(params_f.read_bytes())

        @jax.jit
        def q_of(p, z, c):
            out = net.apply(p, jnp.repeat(z[:, None], c.shape[1], axis=1), c)
            out = hl.from_logits(out) if cfg["num_atoms"] > 1 else out
            return jnp.min(out, axis=0)  # ensemble-min -> [S, N, P], as deployment does

        q = np.asarray(q_of(params, tok, cand))  # [S, N, P]
        P = q.shape[-1]
        prefixes = np.arange(g, H + 1, g)

        def summarize(mask):
            qq = q[mask]
            if len(qq) == 0:
                return None
            flat = qq.reshape(len(qq), -1)
            am = np.array(np.unravel_index(flat.argmax(-1), qq.shape[1:])).T  # [(cand, prefix)]
            hist = np.bincount(am[:, 1], minlength=P) / len(qq)
            mean_by_p = qq.mean(axis=(0, 1))
            rho = float(np.corrcoef(np.argsort(np.argsort(mean_by_p)), np.arange(P))[0, 1]) if P > 1 else float("nan")
            return {
                "n_states": int(len(qq)),
                "mean_q_by_prefix": [float(x) for x in mean_by_p],
                "argmax_hist": [float(x) for x in hist],
                "frac_shortest": float(hist[0]),
                "mean_exec_steps": float((am[:, 1] + 1).mean() * g),
                "spearman_q_vs_prefix": rho,
            }

        results[d.name] = {
            "data": cfg["data"],
            "discount": cfg["discount"],
            "macro_group_size": g,
            "prefixes": [int(p) for p in prefixes],
            "all": summarize(np.ones(len(q), bool)),
            "near_goal": summarize(dist <= args.near_goal),
            "far_from_goal": summarize(dist > args.near_goal),
        }
        r = results[d.name]["all"]
        print(
            f"{d.name:10s} g={cfg['discount']:<7} shortest {100 * r['frac_shortest']:5.1f}%  "
            f"mean_exec {r['mean_exec_steps']:5.2f}/{H}  rho(Q,prefix) {r['spearman_q_vs_prefix']:+.2f}  "
            f"near_goal shortest {100 * results[d.name]['near_goal']['frac_shortest']:5.1f}%"
            if results[d.name]["near_goal"]
            else ""
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
