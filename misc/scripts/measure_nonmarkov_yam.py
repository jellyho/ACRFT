"""Measure the non-Markovian content of the YAM teleop dataset via history-vs-Markov action prediction.

The DeltaL_val protocol (Lazzati/Metelli's policy-side metric; the policy twin of our critic-side
Q_reg - Q_syn): fit two CAPACITY-MATCHED predictors of the current action on frozen DINOv2 features
(the patch-critic cache) and compare held-out MSE --
    markov  : input = the current frame block repeated K times
    hist(k) : input = K frames evenly spaced over [t-k, t]
Both arms share the identical architecture and input dimensionality, so any held-out gap is
information in the past frames, not capacity. Positive gap = the dataset's actions depend on
history beyond the current observation = non-Markovian content (temporal teleop correlations,
hidden intent). Park's caveat does not apply: we never roll these predictors out -- this is a
dataset property measurement, not a policy.

Outputs .scratch/nonmarkov_yam/results.json: per-arm val MSE, Delta(k) relative gaps,
success/failure splits, and per-frame Delta maps for the val episodes (for correlation against
the k* and leak maps). Single L40S, ~1-2h including the one-off feature pooling pass.
"""

# ruff: noqa: PLC0415  (matplotlib.use must precede pyplot; probe-local imports intentional)

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path("/data1/jellyho/pc_cache/yam_s347")
PPOOL = CACHE / "features_pooled_f32.npy"
PROPRIO = [0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27]  # pos-14, critic convention


def pooled_features(n, npatch, emb):
    """Mean-pool patches once (sequential pass over the 138GB fp16 memmap), cache 1.4GB fp32."""
    if PPOOL.exists():
        return np.load(PPOOL, mmap_mode="r")
    feats = np.memmap(CACHE / "features.dat", np.float16, "r", shape=(n, npatch, emb))
    out = np.empty((n, emb), np.float32)
    step = 4096
    for i in range(0, n, step):
        out[i : i + step] = feats[i : i + step].astype(np.float32).mean(axis=1)
        if (i // step) % 32 == 0:
            print(f"pooling {i}/{n}", flush=True)
    np.save(PPOOL, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[15, 30, 60, 150])
    ap.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=None,
        help="single-lag mode (fairer attribution): each arm's input is [frame(t-n), frame(t)] for one n, "
        "Markov arm = frame(t) twice. Overrides --ks/--frames; isolates the information at exactly lag n "
        "instead of confounding window length with frame density",
    )
    ap.add_argument("--frames", type=int, default=6, help="K frames per input (both arms)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=2, help="frame subsampling for train/val pools")
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/nonmarkov_yam")
    ap.add_argument(
        "--zero-hist",
        choices=["none", "feat", "proprio", "vel"],
        default="none",
        help="zero this channel in the HISTORY frames (current frame untouched): decomposes the gap "
        "into visual-history vs proprio-history (executed-velocity, the shallow cousin of action-copy). "
        "'vel' = visual history zeroed AND history proprio slots replaced by the repeated CURRENT "
        "instantaneous velocity (p_t - p_{t-1}); vs the proprio arm this isolates what position "
        "HISTORY adds beyond one velocity (worker C's suggestion)",
    )
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax

    meta = json.loads((CACHE / "meta.json").read_text())
    n, npatch, emb = meta["N"], meta["npatch"], meta["emb"]
    feats = np.asarray(pooled_features(n, npatch, emb))
    states = np.asarray(np.memmap(CACHE / "state.dat", np.float32, "r", shape=(n, meta["sd"])))[:, PROPRIO]
    actions = np.asarray(np.memmap(CACHE / "action.dat", np.float32, "r", shape=(n, meta["ad"])))

    # target: joint-delta in z-scored units (grippers absolute), computed from the cache itself
    ref = np.array([0, 1, 2, 3, 4, 5, -1, 21, 22, 23, 24, 25, 26, -1])
    full_states = np.asarray(np.memmap(CACHE / "state.dat", np.float32, "r", shape=(n, meta["sd"])))
    tgt = actions.copy()
    for i, r_ in enumerate(ref):
        if r_ >= 0:
            tgt[:, i] -= full_states[:, r_]

    eps = sorted(meta["episodes"].items(), key=lambda kv: int(kv[0]))
    rng = np.random.default_rng(0)
    succ_eps = [int(k) for k, v in eps if v["success"]]
    fail_eps = [int(k) for k, v in eps if not v["success"]]
    val_eps = set(rng.choice(succ_eps, int(len(succ_eps) * a.val_frac), replace=False).tolist())
    val_eps |= set(rng.choice(fail_eps, max(2, int(len(fail_eps) * a.val_frac)), replace=False).tolist())
    print(f"val episodes: {len(val_eps)} ({sum(1 for e in val_eps if e in set(fail_eps))} failure)")

    if a.lags is not None:
        a.ks = a.lags
    kmax = max(a.ks)
    per_ep = {int(k): (v["offset"], v["full_len"], v["success"]) for k, v in eps}
    tr_idx, va_idx = [], []
    for e, (off, ln, _s) in per_ep.items():
        idx = np.arange(off + kmax, off + ln, a.stride)  # base frames with full history available
        (va_idx if e in val_eps else tr_idx).append(idx)
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)

    mu, sd = tgt[tr_idx].mean(0), tgt[tr_idx].std(0) + 1e-6
    tgt = (tgt - mu) / sd
    fmu, fsd = feats[tr_idx].mean(0), feats[tr_idx].std(0) + 1e-6
    smu, ssd = states[tr_idx].mean(0), states[tr_idx].std(0) + 1e-6
    _v = states[tr_idx] - states[tr_idx - 1]
    vmu, vsd = _v.mean(0), _v.std(0) + 1e-6

    def offsets(k):  # K frames evenly spaced over [t-k, t]; k=0 -> all zeros (markov arm)
        if a.lags is not None:  # single-lag mode: [t-n, t] pairs, capacity-matched [t, t] baseline
            return np.array([-k, 0], np.int64) if k else np.zeros(2, np.int64)
        if k == 0:
            return np.zeros(a.frames, np.int64)
        return np.unique(np.round(np.linspace(-k, 0, a.frames)).astype(np.int64))

    def gather(idx, offs):
        rows = idx[:, None] + offs[None, :]
        f = (feats[rows.ravel()].reshape(len(idx), len(offs), emb) - fmu) / fsd
        s = (states[rows.ravel()].reshape(len(idx), len(offs), len(PROPRIO)) - smu) / ssd
        if a.zero_hist != "none" and len(offs) > 1:
            hist = np.asarray(offs) != 0  # every non-current frame
            if a.zero_hist == "feat":
                f[:, hist, :] = 0.0
            elif a.zero_hist == "proprio":
                s[:, hist, :] = 0.0
            else:  # vel: no visual history; history proprio slots carry only the current velocity
                f[:, hist, :] = 0.0
                vel = (states[idx] - states[idx - 1] - vmu) / vsd
                s[:, hist, :] = vel[:, None, :]
        return np.concatenate([f, s], axis=-1).reshape(len(idx), -1).astype(np.float32)

    def init_params(key, din, dout):
        ks = jax.random.split(key, 4)
        dims = [din, a.width, a.width, a.width, dout]
        return [
            (jax.random.normal(ks[i], (dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i]), jnp.zeros(dims[i + 1]))
            for i in range(4)
        ]

    def fwd(p, x):
        for w, b in p[:-1]:
            x = jax.nn.relu(x @ w + b)
        w, b = p[-1]
        return x @ w + b

    @jax.jit
    def step_fn(p, opt_state, x, y, tx_idx):
        def loss_fn(p):
            return jnp.mean((fwd(p, x) - y) ** 2)

        lval, g = jax.value_and_grad(loss_fn)(p)
        upd, opt_state = tx.update(g, opt_state, p)
        return optax.apply_updates(p, upd), opt_state, lval

    results = {"config": vars(a) | {"frames": a.frames, "kmax": kmax, "cache": str(CACHE)}, "arms": {}}
    results["config"]["out"] = str(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    tx = optax.adam(a.lr)

    va_is_succ = np.concatenate(
        [
            np.full(len(np.arange(per_ep[e][0] + kmax, per_ep[e][0] + per_ep[e][1], a.stride)), per_ep[e][2])
            for e in sorted(val_eps)
        ]
    )
    va_idx_sorted = np.concatenate(
        [np.arange(per_ep[e][0] + kmax, per_ep[e][0] + per_ep[e][1], a.stride) for e in sorted(val_eps)]
    )

    for k in [0, *a.ks]:  # 0 = capacity-matched Markov arm
        offs = offsets(k)
        Xva = gather(va_idx_sorted, offs)
        Yva = tgt[va_idx_sorted]
        key = jax.random.key(k)
        p = init_params(key, Xva.shape[1], Yva.shape[1])
        opt_state = tx.init(p)
        perm = np.random.default_rng(k).permutation(len(tr_idx))
        for s_i in range(a.steps):
            bi = perm[(s_i * a.batch) % len(tr_idx) : (s_i * a.batch) % len(tr_idx) + a.batch]
            if len(bi) < a.batch:
                perm = np.random.default_rng(k + s_i).permutation(len(tr_idx))
                bi = perm[: a.batch]
            x = jnp.asarray(gather(tr_idx[bi], offs))
            p, opt_state, lval = step_fn(p, opt_state, x, jnp.asarray(tgt[tr_idx[bi]]), 0)
            if s_i % 5000 == 0:
                print(f"k={k} step {s_i} train {float(lval):.4f}", flush=True)
        pf = np.concatenate([np.asarray(fwd(p, jnp.asarray(Xva[i : i + 8192]))) for i in range(0, len(Xva), 8192)])
        err = ((pf - Yva) ** 2).mean(axis=1)
        arm = {
            "val_mse": float(err.mean()),
            "val_mse_success": float(err[va_is_succ].mean()),
            "val_mse_failure": float(err[~va_is_succ].mean()),
        }
        results["arms"][str(k)] = arm
        np.save(a.out / f"perframe_err_k{k}.npy", err.astype(np.float32))
        print(f"== k={k}: {arm}", flush=True)

    m0 = results["arms"]["0"]["val_mse"]
    for k in a.ks:
        mk = results["arms"][str(k)]["val_mse"]
        results["arms"][str(k)]["delta_rel"] = (m0 - mk) / m0
    np.save(a.out / "perframe_idx.npy", va_idx_sorted)
    np.save(a.out / "perframe_is_succ.npy", va_is_succ)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote", a.out / "results.json")


if __name__ == "__main__":
    main()
