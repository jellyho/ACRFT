"""Why would the critic systematically prefer SHORT commitments?

At deployment `make_policy_fn` takes a joint arg-max over (candidate, prefix) and executes
`(pp+1)*macro` steps, so any systematic tilt of Q across the prefix axis decides the commit length.
Rollout videos show it almost always commits short. Three mechanisms could produce that, and two of
them are visible in the annotation alone — no critic, no GPU:

  (1) FLAT TRUTH. Under the sparse terminal scheme the ideal target is the same for every prefix:
      y_h = sum_{i<h} g^i r + g^h V(s_{t+h}) = g^(steps to success), independent of h. So the true
      prefix profile is flat, every prefix is a tie, and ANY small asymmetry fully decides the
      arg-max. This is the precondition that makes (2) and (3) decisive rather than second-order.

  (2) DISCOUNTED OVERESTIMATION. V is a max over N*P noisy estimates, so it carries an upward bias b.
      The target adds g^h * b, and g^h shrinks with h — the SHORT prefixes get more of the inflation.
      Measured here as the ratio g^h_min / g^h_max, i.e. how much more bias the shortest prefix keeps.

  (3) VALIDITY ASYMMETRY. A prefix that steps past its episode's terminal produces no transition at
      all (`valid` in train_rlt_critic.targets). Near the goal only the short prefixes survive, so the
      long-prefix heads are trained on a subset that systematically EXCLUDES the highest-value states.
      A head that never sees the top of the value range cannot predict it.

(3) is the one that would be a genuine defect rather than a known bias, so it is measured exactly as
the trainer computes it, reusing the same helper.

    uv run slurm/prefix_bias_analysis.py --data /scratch/jellyho/acrft/annot/noprop
"""

import argparse
import json
import pathlib

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--macro-group-size", type=int, default=2)
    ap.add_argument("--out", type=pathlib.Path, default=None, help="write the numbers as JSON")
    args = ap.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    T, H = meta["num_frames"], meta["horizon"]
    g = float(meta.get("discount", 0.99))
    gsz = args.macro_group_size
    prefixes = np.arange(gsz, H + 1, gsz)  # exactly train_rlt_critic's `prefixes`

    rd = lambda n, dt: np.asarray(np.memmap(args.data / f"{n}.dat", dt, "r", shape=(T,)))  # noqa: E731
    reward, done, episode, mc = rd("reward", np.float32), rd("done", np.int8), rd("episode_index", np.int32), rd(
        "mc_return", np.float32
    )

    # --- reproduce the trainer's terminal bookkeeping verbatim -------------------------------------
    done_i = done.astype(np.int64)
    done_cum = np.cumsum(done_i)
    alive = np.ones(T, np.float32)
    for e in np.unique(episode):
        w = np.flatnonzero(episode == e)
        fired = w[done_i[w] > 0]
        if len(fired):
            alive[fired[0] + 1 :][: w[-1] - fired[0]] = 0.0

    idx = np.arange(T)
    print(f"data      : {args.data}")
    print(f"frames {T}  horizon {H}  macro_group {gsz}  gamma {g}  prefixes {list(prefixes)}\n")

    # --- (2) how much more overestimation bias the short prefixes keep -----------------------------
    gam = g ** prefixes.astype(np.float64)
    print("(2) DISCOUNTED OVERESTIMATION — the target adds g^h * b, so short prefixes keep more of b")
    print(f"    g^{prefixes[0]} = {gam[0]:.4f}   g^{prefixes[-1]} = {gam[-1]:.4f}")
    print(f"    shortest prefix keeps {gam[0] / gam[-1]:.3f}x the bias of the longest")
    print("    -> with a flat true profile, this alone makes the arg-max pick the shortest prefix\n")

    # --- (3) validity and value coverage per prefix -------------------------------------------------
    print("(3) VALIDITY ASYMMETRY — which transitions each prefix head is actually trained on")
    print(f"    {'prefix':>7} {'valid':>9} {'% of alive':>11} {'mean mc':>9} {'max mc':>8} {'p99 mc':>8}")
    rows = []
    alive_n = int((alive > 0).sum())
    for h in prefixes:
        nxt = np.clip(idx + h, 0, T - 1)
        crossed = done_cum[nxt] - np.where(idx > 0, done_cum[idx - 1], 0)
        lands = (crossed == 1) & (done_i[nxt] > 0)
        boot = (crossed == 0) & (idx + h < T)
        valid = (boot | lands) & (alive > 0)
        v_mc = mc[valid]
        rows.append(
            {
                "prefix": int(h),
                "valid": int(valid.sum()),
                "frac_of_alive": float(valid.sum() / alive_n),
                "mean_mc": float(v_mc.mean()) if len(v_mc) else float("nan"),
                "max_mc": float(v_mc.max()) if len(v_mc) else float("nan"),
                "p99_mc": float(np.percentile(v_mc, 99)) if len(v_mc) else float("nan"),
            }
        )
        r = rows[-1]
        print(
            f"    {r['prefix']:>7} {r['valid']:>9} {100 * r['frac_of_alive']:>10.2f}% "
            f"{r['mean_mc']:>9.4f} {r['max_mc']:>8.4f} {r['p99_mc']:>8.4f}"
        )

    lo, hi = rows[0], rows[-1]
    d_mean = hi["mean_mc"] - lo["mean_mc"]
    print(
        f"\n    longest vs shortest: {hi['valid'] - lo['valid']:+d} transitions "
        f"({100 * (hi['frac_of_alive'] - lo['frac_of_alive']):+.2f} pts), mean mc_return {d_mean:+.4f}"
    )
    # The question is not whether the counts differ (they must) but whether the long prefixes are
    # deprived of the HIGH-value states specifically. That is what would tilt their learned values down.
    if d_mean < -0.002:
        print("    -> long prefixes are trained on systematically LOWER-return states: a real tilt toward short")
    elif d_mean > 0.002:
        print("    -> long prefixes see HIGHER-return states; validity does not explain a short preference")
    else:
        print("    -> value coverage is essentially identical; (3) is NOT the explanation, leaving (1)+(2)")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "gamma": g,
                    "prefixes": [int(p) for p in prefixes],
                    "bias_ratio_short_over_long": float(gam[0] / gam[-1]),
                    "per_prefix": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
