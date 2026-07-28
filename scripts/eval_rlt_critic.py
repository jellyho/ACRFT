"""Does the critic actually rank actions, or is it a state-value function in disguise?

TD loss and Q-vs-return correlation both look fine for a critic that ignores its action argument:
most of the variance in return is explained by the state, so a `Q(z, a) = V(z)` would score well on
either while being useless here - every candidate would receive the same value and best-of-N would
select at random. Every diagnostic below is therefore computed WITHIN a state.

Necessary conditions (a critic failing these cannot support the method):

  action sensitivity   within-state spread of Q relative to between-state spread. ~0 means the
                       action argument is being ignored.
  ranking accuracy     at the same state, is Q(demonstrated chunk) > Q(a chunk from elsewhere)?
                       Chance is 0.5.
  prefix spread        does arg-max_k Q(z, a_1:k) actually vary across states? A spike at one k
                       means adaptive chunking has degenerated to a fixed k.

Then the two failure modes specific to selecting by arg-max over sampled candidates:

  bias curve           value of the arg-max as a function of how many candidates it ranges over.
                       An unbiased critic saturates; max-over-noisy-estimates keeps climbing.
  held-out generalization
                       ranking quality on candidates the bootstrap never saw (annotate_rlt.py
                       --num-heldout). A gap means the critic has fitted the stored sample rather
                       than the policy's action distribution.

Usage:
    uv run scripts/eval_rlt_critic.py --data data/rlt_critic/PrepareCoffee --params critic.msgpack
"""

import argparse
import json
import logging
import pathlib

import flax
import jax.numpy as jnp
import numpy as np

import openpi.rlt_critic.critic as _critic

logger = logging.getLogger(__name__)


def _load(path: pathlib.Path, name: str, shape, dtype):
    """annotate_rlt.py writes raw memmaps as `<name>.dat` in the dtype recorded in meta.json."""
    f = path / f"{name}.dat"
    if not f.exists():
        return None
    return np.memmap(f, dtype=dtype, mode="r", shape=shape)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="annotate_rlt.py output dir")
    ap.add_argument("--params", required=True, type=pathlib.Path, help="trained critic params (msgpack)")
    ap.add_argument("--kind", choices=["qc", "arq"], default="arq")
    ap.add_argument("--num-critics", type=int, default=2)
    ap.add_argument("--num-atoms", type=int, default=1)
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=1.0)
    ap.add_argument("--macro-group-size", type=int, default=2)
    ap.add_argument("--num-states", type=int, default=2048, help="States sampled for the diagnostics.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    T, N, H, A, D = (meta["num_frames"], meta["num_samples"], meta["horizon"], meta["action_dim"], meta["token_dim"])
    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[meta.get("dtype", "float32")]
    tok = _load(args.data, "rl_token", (T, D), dt)
    chunk = _load(args.data, "action_chunk", (T, H, A), dt)
    cand = _load(args.data, "base_action", (T, N, H, A), dt)
    nh = meta.get("num_heldout", 0)
    held = _load(args.data, "base_action_heldout", (T, nh, H, A), dt) if nh else None
    logger.info(f"{T} frames, N={N} candidates, {nh} held-out, horizon {H}, token {D}")

    net = _critic.make_critic(
        kind=args.kind,
        action_dim=A,
        horizon=H,
        macro_group_size=args.macro_group_size,
        num_atoms=args.num_atoms,
        num_critics=args.num_critics,
    )
    hl = _critic.HLGauss(v_min=args.v_min, v_max=args.v_max, num_atoms=max(args.num_atoms, 2))
    params = flax.serialization.msgpack_restore(args.params.read_bytes())

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(T, size=min(args.num_states, T), replace=False))
    z = jnp.asarray(np.asarray(tok[idx], np.float32))

    def q_of(actions):
        """[S, M, H, A] -> ensemble-min Q, reduced over prefixes to the full-chunk value."""
        m = actions.shape[1]
        out = net.apply(params, jnp.repeat(z[:, None], m, axis=1), jnp.asarray(np.asarray(actions, np.float32)))
        out = hl.from_logits(out) if args.num_atoms > 1 else out
        return jnp.min(out, axis=0)  # ensemble -> [S, M] for qc, [S, M, P] for arq

    q_cand = np.asarray(q_of(np.asarray(cand[idx])))
    q_full = q_cand[..., -1] if q_cand.ndim == 3 else q_cand  # value of committing to the whole chunk
    q_demo = np.asarray(q_of(np.asarray(chunk[idx])[:, None]))
    q_demo = q_demo[..., -1] if q_demo.ndim == 3 else q_demo
    q_demo = q_demo[:, 0]

    res = {}

    # --- necessary conditions -------------------------------------------------------------------
    within = float(np.mean(np.var(q_full, axis=1)))
    between = float(np.var(np.mean(q_full, axis=1)))
    res["action_sensitivity"] = within / (between + 1e-12)
    res["within_state_std"] = float(np.mean(np.std(q_full, axis=1)))
    res["between_state_std"] = float(np.sqrt(between))

    # Demonstrated chunk vs a chunk borrowed from a different state, scored at the SAME state.
    other = np.asarray(chunk[rng.permutation(T)[: len(idx)]])
    q_other = np.asarray(q_of(other[:, None]))
    q_other = (q_other[..., -1] if q_other.ndim == 3 else q_other)[:, 0]
    res["ranking_accuracy_demo_vs_other"] = float(np.mean(q_demo > q_other))

    # --- prefix behaviour (ARQ only) --------------------------------------------------------------
    if q_cand.ndim == 3:
        best_k = np.argmax(q_cand.max(axis=1), axis=-1)  # per state, best prefix of the best candidate
        counts = np.bincount(best_k, minlength=q_cand.shape[-1]).astype(float)
        p = counts / counts.sum()
        res["prefix_argmax_hist"] = p.tolist()
        res["prefix_argmax_entropy"] = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
        res["prefix_spread"] = float(np.mean(np.std(q_cand.max(axis=1), axis=-1)))

    # --- maximisation bias: does the arg-max keep climbing as it ranges over more candidates? ----
    curve = []
    for m in sorted({1, 2, 4, 8, min(16, N), N}):
        if m > N:
            continue
        vals = [float(np.mean(np.max(q_full[:, rng.permutation(N)[:m]], axis=1))) for _ in range(8)]
        curve.append({"m": int(m), "v_max": float(np.mean(vals))})
    res["bias_curve"] = curve
    if len(curve) >= 2:
        res["bias_growth_last_double"] = curve[-1]["v_max"] - curve[-2]["v_max"]

    # --- generalization to candidates the bootstrap never saw -------------------------------------
    if held is not None:
        q_held = np.asarray(q_of(np.asarray(held[idx])))
        q_held = q_held[..., -1] if q_held.ndim == 3 else q_held
        res["heldout_mean_q"] = float(np.mean(q_held))
        res["train_mean_q"] = float(np.mean(q_full))
        # If the critic has fitted the stored candidates, their arg-max sits above the held-out one.
        res["argmax_gap_train_minus_heldout"] = float(np.mean(np.max(q_full, 1)) - np.mean(np.max(q_held, 1)))
        res["heldout_within_state_std"] = float(np.mean(np.std(q_held, axis=1)))

    print("\n=== critic diagnostics ===")
    for k, v in res.items():
        if isinstance(v, list):
            continue
        print(f"  {k:38s} {v:+.4f}")
    if "bias_curve" in res:
        print(
            "  bias curve (v_max vs #candidates):", " ".join(f"{c['m']}:{c['v_max']:+.3f}" for c in res["bias_curve"])
        )
    if "prefix_argmax_hist" in res:
        print("  prefix arg-max histogram:", " ".join(f"{x:.2f}" for x in res["prefix_argmax_hist"]))

    print("\n=== verdict ===")
    ok = True
    if res["action_sensitivity"] < 0.01:
        print("  FAIL  action sensitivity ~ 0: the critic ignores the action, best-of-N is a no-op")
        ok = False
    if res["ranking_accuracy_demo_vs_other"] < 0.55:
        print("  FAIL  cannot tell the demonstrated chunk from an unrelated one at the same state")
        ok = False
    if "prefix_argmax_entropy" in res and res["prefix_argmax_entropy"] < 0.1:
        print("  WARN  prefix arg-max is nearly constant: adaptive chunking degenerates to fixed k")
    if ok:
        print("  necessary conditions met; proceed to rollout evaluation")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
