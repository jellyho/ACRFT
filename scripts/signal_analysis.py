"""Is there a within-state ranking signal at all, and how large would it have to be to matter?

Every critic so far reports the same thing: the spread of Q across a state's sixteen candidates,
divided by the critic's own noise, is about 3.1 - which is what sixteen draws of pure noise give. The
conclusion drawn from that is "no signal". But 3.1 is an average over states, and a signal that lives
only at a few decision points would be washed out of an average while still being real. And "no
signal" is only actionable if we know how much signal best-of-N would need. This answers both from
the stored annotation alone, with no critic and no GPU.

Part 1 - is the null the whole story, per state?
    For each state, take the sixteen candidate chunks and ask how different their OUTCOMES would be.
    The outcome of a candidate is not in the data - it was never executed - so it is approximated by
    the return of the nearest actually-executed chunk in the same episode neighbourhood: find, among
    frames close in token space, the demonstrated chunk most similar to the candidate, and read its
    mc_return. The spread of those borrowed returns across a state's candidates is a data-side
    estimate of how much the choice could matter there, with no critic in the loop.

Part 2 - the SNR best-of-N needs.
    best-of-N beats random only if the true value gap between the best and a typical candidate exceeds
    the critic's ranking noise. Given the per-state outcome spread from Part 1 (signal) and the
    critic's measured within-state noise (from a diag.json if given), the required and available SNR
    are put side by side, and the implied data multiplier - noise falls as 1/sqrt(frames) - is the
    number that says whether more data could ever close it.

Usage:
    uv run scripts/signal_analysis.py --data .scratch/annot_noprop [--diag .scratch/critic5_noprop/diag.json]
"""

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument(
        "--diag", type=pathlib.Path, default=None, help="A critic diag.json, for the noise side of the SNR."
    )
    ap.add_argument("--critic", type=pathlib.Path, default=None, help="Trained critic params, for the extraction test.")
    ap.add_argument("--num-states", type=int, default=1024)
    ap.add_argument("--knn", type=int, default=64, help="Token-space neighbours searched for a candidate's twin.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/signal_analysis.png"))
    args = ap.parse_args()

    m = json.loads((args.data / "meta.json").read_text())
    T, N, H, A, D = m["num_frames"], m["num_samples"], m["horizon"], m["action_dim"], m["token_dim"]
    import ml_dtypes

    dt = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}[m.get("dtype", "float32")]

    def rd(name, shape, d=None):
        return np.asarray(np.memmap(args.data / f"{name}.dat", dtype=d or dt, mode="r", shape=shape))

    tok = rd("rl_token", (T, D)).astype(np.float32)
    chunk = rd("action_chunk", (T, H, A)).astype(np.float32)
    cand = rd("base_action", (T, N, H, A)).astype(np.float32)
    mc = rd("mc_return", (T,), np.float32)
    ep = rd("episode_index", (T,), np.int32)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(T, size=min(args.num_states, T), replace=False))
    print(f"{T:,} frames, {len(idx)} states sampled, N={N} candidates")

    # --- Part 1: per-state outcome spread ---------------------------------------------------------
    # Normalise the token so cosine distance is a dot product; do the neighbour search in a random
    # low-dim projection to keep it cheap without changing which frames are close.
    proj = rng.standard_normal((D, 128)).astype(np.float32) / np.sqrt(128)
    tok_p = tok @ proj
    tok_p /= np.linalg.norm(tok_p, axis=1, keepdims=True) + 1e-9
    chunk_flat = chunk.reshape(T, -1)

    frame_idx = rd("frame_index", (T,), np.int32)

    def neighbourhood(i):
        """Executed chunks near state i in token space, SAME episode, but excluding i and the frames
        right around it - a candidate must borrow from a genuinely different transition, or it just
        reads back the current state's own return and the outcome framing degenerates to V(s)."""
        sims = tok_p @ tok_p[i]
        nbr = np.argpartition(-sims, args.knn + 8)[: args.knn + 8]
        keep = (ep[nbr] == ep[i]) & (np.abs(frame_idx[nbr].astype(np.int64) - int(frame_idx[i])) > 8)
        return nbr[keep]

    per_state_spread, demo_borrowed, used = [], [], []
    for i in idx:
        nbr = neighbourhood(i)
        if len(nbr) < 4:
            continue
        nbr_chunks, nbr_mc = chunk_flat[nbr], mc[nbr]
        cc = cand[i].reshape(N, -1)  # [N, H*A]
        d = np.linalg.norm(cc[:, None, :] - nbr_chunks[None, :, :], axis=-1)  # [N, K]
        borrowed = nbr_mc[np.argmin(d, axis=1)]  # [N]
        per_state_spread.append(borrowed.max() - borrowed.min())
        dd = np.linalg.norm(chunk_flat[i][None] - nbr_chunks, axis=-1)
        demo_borrowed.append(float(nbr_mc[np.argmin(dd)]))
        used.append(i)
    spread = np.array(per_state_spread)

    # HONESTY GATE: the demonstrated chunk WAS executed, so its true return is mc[i]. If borrowing
    # from the neighbourhood recovers that, the borrowed returns for the (never-executed) candidates
    # are trustworthy too; if it does not, Parts 1 and 2 are measuring matching noise, not signal.
    db = np.array(demo_borrowed)
    true = mc[np.array(used)]  # the demo's OWN known return, at the same states, same loop order
    r = float(np.corrcoef(db, true)[0, 1]) if len(db) > 2 else float("nan")
    mae = float(np.mean(np.abs(db - true)))
    print("\n=== HONESTY GATE: does borrowing recover the demo's OWN known return? ===")
    print(f"  corr(borrowed, true mc_return) = {r:+.3f}   MAE {mae:.4f}   (return std {true.std():.4f})")
    if r < 0.5:
        print("  -> borrowing is unreliable; treat Parts 1-2 as an upper bound on signal, not a measurement")
    else:
        print("  -> borrowing tracks the true return; the candidate estimates below are trustworthy")

    print("\n=== Part 1: per-state outcome spread (borrowed returns) ===")
    print(f"  states with a usable neighbourhood: {len(spread)}")
    print(
        f"  spread of borrowed returns across candidates: median {np.median(spread):.4f}  p90 {np.percentile(spread, 90):.4f}"
    )
    print(f"  fraction of states with spread > 0.05 (a 5-step timing difference): {(spread > 0.05).mean():.1%}")
    print(f"  fraction essentially flat (< 0.01): {(spread < 0.01).mean():.1%}")

    # --- Part 2: required vs available SNR --------------------------------------------------------
    # Signal: how much the best candidate beats a typical one, in return units - the mean over states
    # of (max - median) of the borrowed returns. Noise: the critic's within-state std, if a diag is
    # given. best-of-N helps when signal/noise clears ~1; the 1/sqrt(n) law then gives the data factor.
    gaps = []
    for i in idx:
        nbr = neighbourhood(i)
        if len(nbr) < 4:
            continue
        cc = cand[i].reshape(N, -1)
        d = np.linalg.norm(cc[:, None, :] - chunk_flat[nbr][None], axis=-1)
        b = mc[nbr][np.argmin(d, axis=1)]
        gaps.append(float(np.max(b) - np.median(b)))
    signal = float(np.mean(gaps))
    print("\n=== Part 2: SNR ===")
    print(f"  signal (best - median candidate return, data estimate): {signal:.4f}")
    if args.diag and args.diag.exists():
        dg = json.loads(args.diag.read_text())
        noise = dg.get("within_state_std", np.nan)
        print(f"  critic within-state noise (from {args.diag.name}): {noise:.4f}")
        snr = signal / (noise + 1e-9)
        print(f"  available SNR = signal / noise = {snr:.2f}   (best-of-N needs roughly >= 1)")
        if snr < 1:
            factor = (1.0 / snr) ** 2
            print(
                f"  noise ~ 1/sqrt(frames), so closing it needs about {factor:.0f}x the data ({factor * T / 1e6:.1f}M frames)"
            )
            print("  -> if that factor is enormous, the reward is the lever, not the data")
        else:
            print("  -> the signal already clears the noise; the critic should be able to rank")

    # --- Part 3: does the critic EXTRACT the signal Parts 1-2 show is present? --------------------
    if args.critic and args.critic.exists():
        import jax

        import openpi.rlt_critic.critic as _critic

        score = jax.jit(_critic.load_trained(args.critic, action_dim=A, horizon=H)[0])
        borrowed_all, q_all, spread_all = [], [], []
        for i in idx:
            nbr = neighbourhood(i)
            if len(nbr) < 4:
                continue
            cc = cand[i].reshape(N, -1)
            d = np.linalg.norm(cc[:, None, :] - chunk_flat[nbr][None], axis=-1)
            b = mc[nbr][np.argmin(d, axis=1)]  # borrowed return per candidate [N]
            q = np.min(np.asarray(score(np.repeat(tok[i][None, None], N, axis=1), cand[i][None])), axis=0)[0]
            q = q[:, -1] if q.ndim == 2 else q  # full-chunk value [N]
            borrowed_all.append(b)
            q_all.append(q)
            spread_all.append(float(b.max() - b.min()))

        def within_corr(qs, bs):
            cs = [
                np.corrcoef(np.argsort(np.argsort(q)), np.argsort(np.argsort(b)))[0, 1]
                for q, b in zip(qs, bs, strict=True)
                if np.std(q) > 1e-9 and np.std(b) > 1e-9
            ]
            return float(np.mean(cs)) if cs else float("nan")

        sp = np.array(spread_all)
        hi = sp > np.percentile(sp, 75)  # decision points: the top-quartile signal states
        rho_all = within_corr(q_all, borrowed_all)
        rho_hi = within_corr(
            [q for q, h in zip(q_all, hi, strict=True) if h], [b for b, h in zip(borrowed_all, hi, strict=True) if h]
        )
        print("\n=== Part 3: does the critic extract the signal? ===")
        print(f"  within-state Spearman(Q, borrowed return), all states:          {rho_all:+.3f}")
        print(f"  restricted to top-quartile signal states (decision points):     {rho_hi:+.3f}")
        if rho_hi < 0.15:
            print("  -> the critic does NOT track the outcome even where candidates clearly differ;")
            print("     the signal is present (Parts 1-2) but the critic fails to extract it.")
        else:
            print("  -> the critic tracks the outcome at decision points; extraction works.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    ax[0].hist(spread, bins=50, color="tab:blue", alpha=0.8)
    ax[0].axvline(0.05, color="k", ls="--", lw=1)
    ax[0].set_title(
        f"per-state outcome spread\nmedian {np.median(spread):.3f}, {(spread > 0.05).mean():.0%} above a 5-step gap",
        fontsize=9,
    )
    ax[0].set_xlabel("max - min borrowed return across candidates")
    g = np.array(gaps)
    ax[1].hist(g, bins=50, color="tab:green", alpha=0.8)
    ax[1].set_title(f"best-minus-median gap\nmean {signal:.3f} (the signal best-of-N could exploit)", fontsize=9)
    ax[1].set_xlabel("max - median borrowed return")
    for a in ax:
        a.grid(visible=True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
