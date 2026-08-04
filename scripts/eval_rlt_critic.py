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
    # Architecture normally comes from the config.json that training saved beside the params; these
    # are only consulted when that file is missing.
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
    # A critic trained with --use-proprio expects token+proprio as its observation, so the same
    # concatenation and the same z-scoring have to be reproduced here or the network is fed the wrong
    # width. Whether it was is recorded in the run's config.json, read a few lines below; this is
    # resolved early because `tok` is needed before that.
    _tcfg = (
        json.loads((args.params.parent / "config.json").read_text())
        if (args.params.parent / "config.json").exists()
        else {}
    )
    if _tcfg.get("use_proprio"):
        pdim = meta["proprio_dim"]
        pro = _load(args.data, "proprio", (T, pdim), np.float32)
        mu, sd = pro.mean(0), pro.std(0)
        tok = np.concatenate([tok, np.where(sd > 1e-6, (pro - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)], axis=1)
        D = D + pdim
        logger.info(f"proprio: +{pdim} dims -> obs {D}")
    chunk = _load(args.data, "action_chunk", (T, H, A), dt)
    cand = _load(args.data, "base_action", (T, N, H, A), dt)
    mc = _load(args.data, "mc_return", (T,), np.float32)
    nh = meta.get("num_heldout", 0)
    held = _load(args.data, "base_action_heldout", (T, nh, H, A), dt) if nh else None
    logger.info(f"{T} frames, N={N} candidates, {nh} held-out, horizon {H}, token {D}")

    # The network built here has to match the checkpoint exactly, and train_rlt_critic.py already
    # writes the settings it used next to the params. Read those instead of re-declaring them on the
    # command line, where one stale flag silently builds a different network. The CLI values remain
    # as fallbacks for params saved before that file existed.
    cfg_path = args.params.parent / "config.json"
    tcfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    logger.info(f"architecture from {cfg_path}" if tcfg else f"no {cfg_path}; using command-line architecture")

    kind = tcfg.get("kind", args.kind)
    num_atoms = tcfg.get("num_atoms", args.num_atoms)
    arch = (
        {
            "macro_group_size": tcfg.get("macro_group_size", args.macro_group_size),
            "num_layers": tcfg.get("num_layers", 3),
            "num_heads": tcfg.get("num_heads", 8),
            "head_dim": tcfg.get("head_dim", 48),
            "mlp_dim": tcfg.get("mlp_dim", 1024),
        }
        if kind == "arq"
        else {"hidden_dims": tuple(tcfg.get("hidden_dims", [512, 512, 512]))}
    )
    net = _critic.Ensemble(
        make_critic=lambda: _critic.make_critic(kind, action_dim=A, horizon=H, num_atoms=num_atoms, **arch),
        num_critics=tcfg.get("num_critics", args.num_critics),
    )
    hl = _critic.HLGauss(
        v_min=tcfg.get("v_min", args.v_min), v_max=tcfg.get("v_max", args.v_max), num_atoms=max(num_atoms, 2)
    )
    params = flax.serialization.msgpack_restore(args.params.read_bytes())

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(T, size=min(args.num_states, T), replace=False))
    z = jnp.asarray(np.asarray(tok[idx], np.float32))

    def q_of(actions):
        """[S, M, H, A] -> ensemble-min Q, reduced over prefixes to the full-chunk value."""
        m = actions.shape[1]
        out = net.apply(params, jnp.repeat(z[:, None], m, axis=1), jnp.asarray(np.asarray(actions, np.float32)))
        out = hl.from_logits(out) if num_atoms > 1 else out
        return jnp.min(out, axis=0)  # ensemble -> [S, M] for qc, [S, M, P] for arq

    q_cand = np.asarray(q_of(np.asarray(cand[idx])))
    q_full = q_cand[..., -1] if q_cand.ndim == 3 else q_cand  # value of committing to the whole chunk
    q_demo = np.asarray(q_of(np.asarray(chunk[idx])[:, None]))
    q_demo = q_demo[..., -1] if q_demo.ndim == 3 else q_demo
    q_demo = q_demo[:, 0]

    res = {}

    # --- calibration against the return that was actually obtained --------------------------------
    # mc_return is the discounted return the behaviour policy collected from this very frame, so it
    # and Q(z, demonstrated chunk) are two estimates of the same state, measured the same way. Q
    # should sit ABOVE it - taking the best of N candidates is an improvement on what the data did -
    # but the size of that gap is the claim the critic is making, and a gap past the best outcome
    # anywhere in the data is the arg-max inflating rather than the policy improving. The ordering is
    # reported separately, because an inflated critic can still rank states correctly and the
    # ordering is the only part best-of-N consumes.
    if mc is not None:
        mc_s = np.asarray(mc[idx], np.float32)
        mc_all = np.asarray(mc[:], np.float32)
        res["mc_return_mean"] = float(mc_all.mean())
        res["mc_return_max"] = float(mc_all.max())
        res["q_demo_mean"] = float(q_demo.mean())
        res["q_demo_minus_mc_mean"] = float((q_demo - mc_s).mean())
        res["frac_q_demo_above_data_max"] = float((q_demo > mc_all.max()).mean())

        def _spearman(a, b):
            ra = np.argsort(np.argsort(a)).astype(np.float64)
            rb = np.argsort(np.argsort(b)).astype(np.float64)
            return float(np.corrcoef(ra, rb)[0, 1])

        res["spearman_q_demo_vs_mc"] = _spearman(q_demo, mc_s)

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

    # --- the decisive within-state test ----------------------------------------------------------
    # demo-vs-other is too easy: the two chunks come from different states, so telling them apart
    # only needs the critic to know where it is. The question best-of-N actually asks is whether it
    # can rank chunks the POLICY would emit HERE, which are all plausible and all similar.
    #
    #   demo vs candidate      chance is 0.5. The demonstration is what the data executed and what
    #                          mc_return was collected under, so it is the one chunk at this state
    #                          with a known outcome.
    #   Q vs distance to demo  within each state, rank the candidates by value and by how far they
    #                          sit from the executed chunk. A critic with real action knowledge
    #                          prefers the ones closer to what worked; the shuffled control says how
    #                          much of any such correlation is an artifact of the sample size.
    #
    # A near-zero result on BOTH, with a candidate set that is genuinely diverse, means the failure
    # is not the critic's: chunks the policy draws at one state are worth the same, and best-of-N has
    # nothing to select. That is a statement about the task and the reward, not about this network.
    def _rank(a, axis=-1):
        return np.argsort(np.argsort(a, axis=axis), axis=axis).astype(np.float64)

    res["ranking_accuracy_demo_vs_candidate"] = float(np.mean(q_demo[:, None] > q_full))
    dist = np.linalg.norm(
        np.asarray(cand[idx], np.float32).reshape(len(idx), N, -1)
        - np.asarray(chunk[idx], np.float32).reshape(len(idx), 1, -1),
        axis=-1,
    )  # [S, N]
    rq, rd = _rank(q_full), _rank(-dist)
    rq -= rq.mean(-1, keepdims=True)
    rd -= rd.mean(-1, keepdims=True)
    rho = (rq * rd).sum(-1) / np.sqrt((rq**2).sum(-1) * (rd**2).sum(-1) + 1e-12)
    res["spearman_q_vs_closeness_to_demo"] = float(rho.mean())
    perm = rng.permuted(np.tile(np.arange(N), (len(idx), 1)), axis=1)
    rqs = np.take_along_axis(rq, perm, axis=1)
    res["spearman_shuffled_control"] = float(
        (rqs * rd).sum(-1).mean() / np.sqrt(((rq**2).sum(-1) * (rd**2).sum(-1)).mean() + 1e-12)
    )
    res["within_state_q_range"] = float(np.mean(q_full.max(1) - q_full.min(1)))

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
    if abs(res["spearman_q_vs_closeness_to_demo"]) < 0.05 and res["ranking_accuracy_demo_vs_candidate"] < 0.55:
        print("  NOTE  no within-state ranking signal at all. Check the candidate spread before")
        print("        blaming the critic: if the chunks the policy draws here are worth the same,")
        print("        best-of-N has nothing to select and the reward is what needs changing.")
    if "prefix_argmax_entropy" in res and res["prefix_argmax_entropy"] < 0.1:
        print("  WARN  prefix arg-max is nearly constant: adaptive chunking degenerates to fixed k")
    if ok:
        print("  necessary conditions met; proceed to rollout evaluation")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))

    # If the critic was trained with wandb, log these diagnostics onto the same run rather than a new
    # one - the training curves and the final numbers then live together. The run id is in config.json.
    wid = tcfg.get("wandb_id")
    proj, ent = tcfg.get("wandb_project"), tcfg.get("wandb_entity")
    if wid and proj:
        import wandb

        run = wandb.init(project=proj, entity=ent, id=wid, resume="allow")
        run.summary.update({f"diag/{k}": v for k, v in res.items() if isinstance(v, int | float)})
        if "prefix_argmax_hist" in res:
            run.summary["diag/prefix_argmax_hist"] = res["prefix_argmax_hist"]
        run.finish()
        logger.info(f"logged diagnostics to wandb run {wid}")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
