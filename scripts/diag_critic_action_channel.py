"""How much of the action signal that IS in the data does the trained critic actually deliver?

Two measurements have been sitting next to each other without being comparable.

  ridge, model-free      an action chunk adds +0.034 R2 to a prediction of time-to-goal beyond the
                         state alone (episode-held-out; a LINEAR model, so a lower bound)
  the trained critic     the action explains 0.001-0.002% of its Q variance on dataset frames

The first says the data has action signal. The second says the critic barely uses one. But they are
different quantities on different scales, so the gap between them is suggestive and not evidence.
This script puts both on ONE axis: the same frames, the same target, the same state features, the
same cross-validation.

The construction. For each frame of a successful episode, take

    a_data   the demonstrator's actual next 30 actions -- the chunk whose consequence IS the
             time-to-goal being predicted
    a_bc     one draw from the base policy at that same state -- what the policy would have done

and score both with the critic. Their difference

    dQ = Q(s, a_data) - Q(s, a_bc)

is the critic's ACTION channel with the state cancelled out: whatever the critic knows about s
appears in both terms and subtracts away. So dQ is exactly "how much better does this critic think
the demonstrator's action was than a typical policy draw, here".

If the critic's action channel carries real information, dQ must predict the part of time-to-goal
that the state cannot -- an action the critic scores above the policy's own draw should be followed
by a shorter remaining time than the state alone implies.

Four regressions, all episode-held-out, all ridge at the same alpha:

    R2(S)              the state baseline
    R2(S + a_data)     the CEILING a linear model can reach from the raw action chunk
    R2(S + dQ)         what the critic's action channel delivers, in the same units
    R2(S + Q_data)     a sanity row: Q is mostly V(s), so this should be large and says little

The number that answers the question is the third against the second. If R2(S + dQ) - R2(S) is a
small fraction of R2(S + a_data) - R2(S), the critic is failing to deliver signal that a linear
model can find in the same data, and the problem is the critic rather than the dataset. If the two
are comparable, the critic is already extracting what is there and the ceiling is the dataset.

Also reported, because they cost nothing and bound the interpretation:
  - a permutation null for dQ (shuffled across frames), the same control the identifiability
    replication used, so a capacity artefact cannot masquerade as signal
  - the same numbers with dQ replaced by |dQ|, since a channel can carry magnitude without sign
"""

import argparse
import json
import logging
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "misc"))

CAMERAS = {
    "observation/image": "agentview",
    "observation/wrist_image": "wrist_left",
    "observation/image_right": "wrist_right",
}


def _obs_dict(imgs, state, task):
    out = {}
    for k, cam in CAMERAS.items():
        if cam not in imgs or imgs[cam] is None:
            return None
        out[k] = imgs[cam]
    out["observation/state"] = np.asarray(state, np.float32)
    out["prompt"] = task
    return out


def _folds(groups, folds, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {e: i % folds for i, e in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


#: Ridge strengths tried inside each training fold. A single fixed alpha is not safe here: the
#: state block is high-dimensional against a few thousand rows, and the first version of this
#: script returned R2(state) = -0.18 -- worse than predicting the mean -- purely from underfitting
#: the regulariser choice, which makes every gap measured against it meaningless.
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def r2_cv(feat, y, groups, *, folds, seed, alphas=ALPHAS):
    """Episode-held-out cross-validated R2, with alpha chosen inside each training fold.

    Whole episodes go to one fold, never split, so a model cannot see frame t-1 of a held-out
    episode while predicting frame t.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    f = _folds(groups, folds, seed)
    pred = np.empty_like(y)
    for i in range(folds):
        tr, te = f != i, f == i
        # inner split, again by episode, to pick alpha without touching the outer test fold
        inner = _folds(groups[tr], min(4, folds), seed + 17)
        best, best_score = alphas[0], -np.inf
        for al in alphas:
            ip = np.empty(int(tr.sum()), np.float32)
            ftr, ytr = feat[tr], y[tr]
            for j in range(inner.max() + 1):
                itr, ite = inner != j, inner == j
                if not itr.any() or not ite.any():
                    continue
                sc_i = StandardScaler().fit(ftr[itr])
                ip[ite] = Ridge(alpha=al).fit(sc_i.transform(ftr[itr]), ytr[itr]).predict(sc_i.transform(ftr[ite]))
            score = 1.0 - float(((ytr - ip) ** 2).sum()) / float(((ytr - ytr.mean()) ** 2).sum())
            if score > best_score:
                best, best_score = al, score
        sc = StandardScaler().fit(feat[tr])
        pred[te] = Ridge(alpha=best).fit(sc.transform(feat[tr]), y[tr]).predict(sc.transform(feat[te]))
    return 1.0 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum()), pred


def _r2(y, pred, idx):
    yy, pp = y[idx], pred[idx]
    return 1.0 - float(((yy - pp) ** 2).sum()) / float(((yy - yy.mean()) ** 2).sum())


def gap_ci(y, groups, pred_a, pred_b, *, seed, n_boot=2000):
    """95% CI on R2(a) - R2(b), bootstrapped over EPISODES.

    Resamples whole episodes and recomputes both R2 from the already-held-out predictions. Frames
    inside one episode are not independent -- consecutive frames share a scene and a demonstrator --
    so a frame-level interval would be far too narrow. Cheap because the predictions are fixed:
    the cross-validation is not redone, only the aggregation over which episodes are counted.
    """
    eps = np.unique(groups)
    where = {e: np.flatnonzero(groups == e) for e in eps}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(eps, size=len(eps), replace=True)
        idx = np.concatenate([where[e] for e in pick])
        diffs.append(_r2(y, pred_a, idx) - _r2(y, pred_b, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def collect(a):
    """Walk the dataset once and score every sampled frame. Returns plain arrays."""
    from dataset_reader import DatasetReader
    from dataset_reader import SequentialImages

    from openpi.extraction import critic_q
    import openpi.models.model as _model
    from openpi.policies import patch_critic_policy as pcp
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config
    from openpi.training import outcomes as _outcomes

    cfg = _config.get_config(a.policy_config)
    policy = _pc.create_trained_policy(cfg, pathlib.Path(a.policy_dir))
    wrapper = pcp.PatchCriticSelectPolicy(policy, a.critic, mode="bon", default_samples=1)
    critic = critic_q.load(a.critic)

    ds = pathlib.Path(a.dataset)
    reader = DatasetReader(ds.name, str(ds.parent))
    reader.load()
    verdicts = _outcomes.episode_outcomes(ds)
    if verdicts is None:
        raise SystemExit(f"{ds} carries no next.success / next.done; migrate it first")
    ok = sorted(e for e, v in verdicts.items() if v == "success")
    logging.info("%d successful episodes", len(ok))

    rng = jax.random.key(a.seed)
    pick = np.random.default_rng(a.seed)
    rows = []
    for ep in pick.permutation(ok):
        if len(rows) >= a.frames:
            break
        ep = int(ep)
        n = reader.episode_length(ep)
        acts = reader.column(ep, "action")
        states = reader.column(ep, "observation.state")
        if not n or acts is None or states is None:
            continue
        eff = min(int(n), acts.shape[0]) - a.tail
        hi = eff - a.horizon - 1
        if hi <= 0:
            continue
        want = min(a.per_episode, a.frames - len(rows))
        frames = sorted({int(x) for x in pick.integers(0, hi, size=want * 3)})[:want]
        try:
            seq = SequentialImages(str(ds), ep, cameras=list(CAMERAS.values()))
        except Exception as e:  # a camera file that will not open costs one episode, not the run
            logging.warning("episode %d: video open failed (%s)", ep, type(e).__name__)
            continue
        for fr in frames:
            obs = _obs_dict(seq.frame(fr), states[fr], "")
            if obs is None:
                continue
            raw_state = obs["observation/state"]
            feats = np.asarray(wrapper._patches_of(obs), np.float32)
            proprio = np.asarray(wrapper._critic_proprio(raw_state), np.float32)

            # the demonstrator's own continuation, in the critic's action space
            a_data = np.asarray(acts[fr : fr + a.horizon], np.float32)
            c_data = wrapper._pre.actions(a_data[None], raw_state) if wrapper._pre is not None else a_data[None]
            c_data = np.asarray(c_data, np.float32)[:, : wrapper._critic_horizon, : wrapper._critic_action_dim]

            # one draw from the base policy at the same state, mapped into the critic's space the
            # way the selection path does it -- not re-normalised by hand
            inputs = policy._input_transform(dict(obs))
            norm_state = np.asarray(inputs["state"], np.float32).reshape(-1)
            tree = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            observation = _model.Observation.from_dict(tree)
            rng, sub = jax.random.split(rng)
            _tok, draws = wrapper._extract(sub, observation, num_samples=1, num_steps=a.flow_steps)
            bc_norm = np.asarray(draws[0], np.float32)  # [1, H, model AD]
            k, c = wrapper._critic_space_affine(norm_state, raw_state)
            h, ad = k.shape
            c_bc = np.asarray(bc_norm[:, :h, :ad] * k + c, np.float32)

            chunks = np.concatenate([c_data, c_bc], 0)
            q = np.asarray(
                critic.hl.from_logits(
                    critic.net.apply(
                        {"params": critic.params},
                        jnp.asarray(feats)[None].repeat(2, 0),
                        jnp.asarray(chunks),
                        jnp.asarray(proprio)[None].repeat(2, 0),
                    )
                )[..., -1],
                np.float32,
            )  # [K, 2]
            rows.append(
                {
                    "episode": ep,
                    "frame": fr,
                    "ttg": float(eff - 1 - fr),
                    "proprio": proprio,
                    "dino": feats.mean(axis=0),  # pooled over patches, the state summary
                    "a_data": a_data.reshape(-1),
                    "q_data_mean": float(q[:, 0].mean()),
                    "q_bc_mean": float(q[:, 1].mean()),
                    "q_data_min": float(q[:, 0].min()),
                    "q_bc_min": float(q[:, 1].min()),
                }
            )
        logging.info("%d frames", len(rows))
    return rows


def analyse(rows, a):
    epi = np.array([r["episode"] for r in rows])
    ttg = np.array([r["ttg"] for r in rows], np.float32)
    dino = np.stack([r["dino"] for r in rows])
    prop = np.stack([r["proprio"] for r in rows])
    a_data = np.stack([r["a_data"] for r in rows])
    q_d = np.array([r["q_data_mean"] for r in rows], np.float32)
    q_b = np.array([r["q_bc_mean"] for r in rows], np.float32)
    dq = (q_d - q_b)[:, None]

    # The DINO block is 384-d against a few thousand rows. Compress it first, fit on the whole
    # sample (unsupervised, so it cannot leak the target) -- the alternative is a state baseline
    # that underfits, and every gap here is measured AGAINST that baseline.
    if a.dino_pcs and dino.shape[1] > a.dino_pcs:
        from sklearn.decomposition import PCA

        dino = PCA(n_components=a.dino_pcs, random_state=a.seed).fit_transform(dino)
    S = np.concatenate([dino, prop], 1)
    # The action chunk is 30x14 = 420 columns against a couple of thousand training rows, and at
    # that ratio ridge cannot estimate its contribution at all: the raw-chunk ceiling came out
    # NEGATIVE at 1020 and 2016 frames and only turned positive at 3000, i.e. it was measuring
    # sample size, not signal. Compressing it makes the ceiling estimable at this n. It is still a
    # lower bound (PCA is unsupervised and linear), and it is still far more expressive than the
    # single scalar dQ it is compared against.
    a_full = a_data
    if a.action_pcs and a_data.shape[1] > a.action_pcs:
        from sklearn.decomposition import PCA

        a_data = PCA(n_components=a.action_pcs, random_state=a.seed).fit_transform(a_data)
    kw = {"folds": a.folds, "seed": a.seed}
    base, p_base = r2_cv(S, ttg, epi, **kw)
    ceiling, p_ceil = r2_cv(np.concatenate([S, a_data], 1), ttg, epi, **kw)
    ceiling_raw, _ = r2_cv(np.concatenate([S, a_full], 1), ttg, epi, **kw)
    channel, p_chan = r2_cv(np.concatenate([S, dq], 1), ttg, epi, **kw)
    channel_abs, _ = r2_cv(np.concatenate([S, dq, np.abs(dq)], 1), ttg, epi, **kw)
    sanity, _ = r2_cv(np.concatenate([S, q_d[:, None]], 1), ttg, epi, **kw)

    # the same control the identifiability replication used: the channel attached to wrong frames
    rng = np.random.default_rng(a.seed + 1)
    null, p_null = r2_cv(np.concatenate([S, dq[rng.permutation(len(dq))]], 1), ttg, epi, **kw)

    ci = {
        "ceiling": gap_ci(ttg, epi, p_ceil, p_base, seed=a.seed),
        "critic_channel": gap_ci(ttg, epi, p_chan, p_base, seed=a.seed),
        "null": gap_ci(ttg, epi, p_null, p_base, seed=a.seed),
        "ceiling_minus_channel": gap_ci(ttg, epi, p_ceil, p_chan, seed=a.seed),
    }

    # Is the ceiling itself still rising with n? If it is, it is sample-limited and the critic's
    # share below is an OVERestimate.
    curve = {}
    for frac in (0.34, 0.67, 1.0):
        eps_all = np.unique(epi)
        take = set(eps_all[: max(2, int(round(frac * len(eps_all))))])
        m = np.array([e in take for e in epi])
        if m.sum() < 200:
            continue
        b_i, _ = r2_cv(S[m], ttg[m], epi[m], **kw)
        c_i, _ = r2_cv(np.concatenate([S, a_data], 1)[m], ttg[m], epi[m], **kw)
        curve[f"{int(m.sum())} frames"] = c_i - b_i

    gap_ceiling = ceiling - base
    gap_channel = channel - base
    gap_null = null - base
    delivered = gap_channel / gap_ceiling if abs(gap_ceiling) > 1e-9 else float("nan")
    out = {
        "n_frames": len(rows),
        "n_episodes": int(len(np.unique(epi))),
        "state_dims": int(S.shape[1]),
        "r2": {
            "state": base,
            "state_plus_action_chunk": ceiling,
            "state_plus_dQ": channel,
            "state_plus_dQ_and_absdQ": channel_abs,
            "state_plus_Q_data": sanity,
            "state_plus_dQ_permuted": null,
            "state_plus_action_chunk_raw": ceiling_raw,
        },
        "action_pcs": a.action_pcs,
        "gaps": {"ceiling": gap_ceiling, "critic_channel": gap_channel, "null": gap_null},
        "gap_ci95_episode_bootstrap": ci,
        "ceiling_vs_sample_size": curve,
        "fraction_of_ceiling_delivered": delivered,
        "dQ": {
            "mean": float(dq.mean()),
            "std": float(dq.std()),
            "frac_positive": float((dq > 0).mean()),
        },
    }
    print(f"\nframes {out['n_frames']} from {out['n_episodes']} episodes; state features {out['state_dims']}")
    if base <= 0.05:
        print(
            "\n  !! R2(state) is not meaningfully positive. The state baseline does not predict\n"
            "     time-to-goal at all here, so the gaps below are differences between two broken\n"
            "     models and mean nothing. Collect more frames or shrink the state block."
        )
    print(f"\n  R2(state)                      {base:+.4f}")
    print(f"  R2(state + action, {a.action_pcs:>3d} PCs)    {ceiling:+.4f}   gap {gap_ceiling:+.4f}   <- the ceiling")
    print(
        f"  R2(state + action, all 420)    {ceiling_raw:+.4f}   gap {ceiling_raw - base:+.4f}   (unestimable at this n)"
    )
    print(f"  R2(state + dQ)                 {channel:+.4f}   gap {gap_channel:+.4f}   <- the critic")
    print(f"  R2(state + dQ, |dQ|)           {channel_abs:+.4f}   gap {channel_abs - base:+.4f}")
    print(f"  R2(state + dQ permuted)        {null:+.4f}   gap {gap_null:+.4f}   <- null")
    print(f"  R2(state + Q(s,a_data))        {sanity:+.4f}   (sanity: Q is mostly V(s))")
    print("\n  gaps against the state baseline, 95% CI bootstrapped over episodes:")
    for k, (m, lo, hi) in ci.items():
        star = "" if lo <= 0 <= hi else "   *"
        print(f"    {k:24s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")
    if curve:
        print("\n  is the ceiling sample-limited?  gap vs frames used:")
        for k, v in curve.items():
            print(f"    {k:16s} {v:+.4f}")
    print(f"\n  the critic delivers {100 * delivered:.1f}% of the linearly available action signal")
    print(f"  dQ = Q(demonstrator) - Q(BC draw): mean {dq.mean():+.3f}, std {dq.std():.3f}, ")
    print(f"      positive on {100 * (dq > 0).mean():.1f}% of frames")
    return out


def save_rows(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        episode=np.array([r["episode"] for r in rows]),
        frame=np.array([r["frame"] for r in rows]),
        ttg=np.array([r["ttg"] for r in rows], np.float32),
        proprio=np.stack([r["proprio"] for r in rows]),
        dino=np.stack([r["dino"] for r in rows]),
        a_data=np.stack([r["a_data"] for r in rows]),
        q_data_mean=np.array([r["q_data_mean"] for r in rows], np.float32),
        q_bc_mean=np.array([r["q_bc_mean"] for r in rows], np.float32),
        q_data_min=np.array([r["q_data_min"] for r in rows], np.float32),
        q_bc_min=np.array([r["q_bc_min"] for r in rows], np.float32),
    )
    print("wrote", path)


def load_rows(path):
    z = np.load(path)
    keys = (
        "episode",
        "frame",
        "ttg",
        "proprio",
        "dino",
        "a_data",
        "q_data_mean",
        "q_bc_mean",
        "q_data_min",
        "q_bc_min",
    )
    return [{k: z[k][i] for k in keys} for i in range(len(z["ttg"]))]


def main(a):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.from_dump:
        rows = load_rows(a.from_dump)
        print(f"loaded {len(rows)} frames from {a.from_dump}")
    else:
        rows = collect(a)
        if a.dump:
            save_rows(rows, a.dump)
    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} frames collected; the CV would be noise")
    out = analyse(rows, a)
    out["critic"] = pathlib.Path(a.critic).name if a.critic else "(from dump)"
    out["dataset"] = str(a.dataset)
    out["tail_dropped"] = a.tail
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(pathlib.Path.home() / "lerobot_data/yam_lego_taxi"))
    ap.add_argument("--critic", default=None)
    ap.add_argument("--policy-dir", default=None)
    ap.add_argument("--policy-config", default="pi05_yam_lego_taxi")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--tail", type=int, default=100, help="frames dropped at each episode end (a homing stand-in)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="save the collected frames so the analysis can be redone free")
    ap.add_argument("--from-dump", default=None, help="re-analyse a saved collection; no GPU")
    ap.add_argument("--dino-pcs", type=int, default=32, help="PCA the pooled DINO block to this many dims (0 = off)")
    ap.add_argument("--action-pcs", type=int, default=32, help="PCA the action chunk to this many dims (0 = off)")
    main(ap.parse_args())
