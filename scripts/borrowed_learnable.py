"""Can a critic learn the borrowed-return target and GENERALISE across candidates, or only memorise?

critic8 will regress Q(z, candidate) toward each candidate's borrowed return. Before spending GPU on
it, this checks the one thing that could make it hollow: the borrowed return comes from a token-space
search, and the critic also reads the token, so it might fit the target by memorising which token sits
near which return rather than learning anything about the action. If so the within-state ordering
would be an artifact and would not transfer to candidates the model was not trained on.

The test splits each state's sixteen candidates into a train half and a test half, fits a small MLP
(token, candidate) -> borrowed return on the train half across all states, and measures within-state
rank correlation on the HELD-OUT half. Memorisation cannot rank held-out candidates; genuine
action-conditioned structure can. A token-only baseline (no candidate input) bounds how much of any
success is just the state.

    token+action   the real model
    token-only     the state alone - if it matches token+action, the action is being ignored

Usage:
    uv run scripts/borrowed_learnable.py --data .scratch/annot_pilot
"""

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = json.loads((args.data / "meta.json").read_text())
    T, N, H, A, D = m["num_frames"], m["num_samples"], m["horizon"], m["action_dim"], m["token_dim"]
    if not (args.data / "borrowed_return.dat").exists():
        raise SystemExit("no borrowed_return.dat; run compute_borrowed_returns.py first")
    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[m.get("dtype", "float32")]
    Tn = min(T, args.max_frames)

    def rd(name, shape, d=None):
        return np.asarray(np.memmap(args.data / f"{name}.dat", dtype=d or dt, mode="r", shape=shape)[:Tn])

    tok = rd("rl_token", (T, D)).astype(np.float32)
    cand = rd("base_action", (T, N, H, A)).astype(np.float32).reshape(Tn, N, H * A)
    bor = rd("borrowed_return", (T, N), np.float32)

    rng = np.random.default_rng(args.seed)
    # Split candidate slots into train / test halves, fixed across all states.
    perm = rng.permutation(N)
    tr_c, te_c = perm[: N // 2], perm[N // 2 :]
    # On device so the jitted step can gather with traced indices.
    tok_d, cand_d, bor_d, trc_d = jnp.asarray(tok), jnp.asarray(cand), jnp.asarray(bor), jnp.asarray(tr_c)

    def mlp_apply(params, z, a):
        x = jnp.concatenate([z, a], axis=-1) if a is not None else z
        for w, b in params[:-1]:
            x = jax.nn.gelu(x @ w + b)
        w, b = params[-1]
        return (x @ w + b)[..., 0]

    def init(use_action, key):
        dims = [D + (H * A if use_action else 0), args.width, args.width, 1]
        ps = []
        for i in range(len(dims) - 1):
            key, k = jax.random.split(key)
            ps.append((jax.random.normal(k, (dims[i], dims[i + 1])) * (2.0 / dims[i]) ** 0.5, jnp.zeros(dims[i + 1])))
        return ps

    results = {}
    for name, use_action in (("token+action", True), ("token-only", False)):
        params = init(use_action, jax.random.key(args.seed))
        opt = optax.adam(3e-4)
        st = opt.init(params)

        @jax.jit
        def step(params, st, key, _ua=use_action, _opt=opt):
            i = jax.random.randint(key, (args.batch,), 0, Tn)
            c = trc_d[jax.random.randint(jax.random.fold_in(key, 1), (args.batch,), 0, len(tr_c))]
            z, y = tok_d[i], bor_d[i, c]
            a = cand_d[i, c] if _ua else None
            lo, g = jax.value_and_grad(lambda p: jnp.mean(jnp.square(mlp_apply(p, z, a) - y)))(params)
            u, st2 = _opt.update(g, st)
            return optax.apply_updates(params, u), st2, lo

        t0 = time.perf_counter()
        for s in range(args.steps):
            params, st, lo = step(params, st, jax.random.fold_in(jax.random.key(args.seed), s))

        # Within-state ranking on the HELD-OUT candidate slots.
        idx = rng.choice(Tn, size=min(2000, Tn), replace=False)
        z = jnp.asarray(tok[idx])
        cs = []
        for c in te_c:
            a = jnp.asarray(cand[idx, c]) if use_action else None
            pred = np.asarray(mlp_apply(params, z, a))
            cs.append(pred)

        def wcorr(slots, params=params, z=z, use_action=use_action, idx=idx):
            cs2 = [np.asarray(mlp_apply(params, z, jnp.asarray(cand[idx, c]) if use_action else None)) for c in slots]
            pr = np.stack(cs2, axis=1)
            tr2 = bor[idx][:, slots]
            rr = [
                np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
                for a, b in zip(pr, tr2, strict=True)
                if np.std(a) > 1e-9 and np.std(b) > 1e-9
            ]
            return float(np.mean(rr)) if rr else float("nan")

        r_tr, r_te = wcorr(tr_c), wcorr(te_c)
        results[name] = r_te
        print(f"  {name:14s} train-slot {r_tr:+.3f}   held-out {r_te:+.3f}   ({time.perf_counter() - t0:.0f}s)")

    print("\n=== can the borrowed target be learned and generalise across candidates? ===")
    gap = results["token+action"] - results["token-only"]
    print(
        f"  token+action {results['token+action']:+.3f}   token-only {results['token-only']:+.3f}   action adds {gap:+.3f}"
    )
    if results["token+action"] > 0.2 and gap > 0.1:
        print("  -> a critic CAN rank held-out candidates from the borrowed target using the action;")
        print("     critic8 is well-motivated - the signal is learnable, not memorised.")
    elif results["token+action"] > 0.2:
        print("  -> ranking works but token-only matches it: the target is state-driven, action ignored.")
    else:
        print("  -> even a direct regressor cannot rank held-out candidates; the borrowed target does")
        print("     not carry generalisable action structure and critic8 would fit noise.")


if __name__ == "__main__":
    main()
