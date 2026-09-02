"""Why a critic that looks fine offline does not help on the robot — two failures, two figures.

The measurement is on the critic's OWN training set, so "off-support" means what it meant during
fitting, and every number is relative to THE DEMONSTRATOR'S OWN next 30 actions at that frame,
taken from successful episodes only: an action known to have reached the goal, scored by a
cost-to-goal critic that should not be able to beat it by much.

    fig 1  why best-of-N does not help    the prize is smaller than the selection bias
    fig 2  why steering breaks            the gradient points where the policy never goes
    fig 3  nine critics                   is it this checkpoint, or the method

Q is cost-to-goal: +1 means "one control step (33 ms at 30 fps) closer to the goal, says the critic".
"""

import argparse
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))

RHO = 0.5


def load(probe, name=None):
    p = pathlib.Path(probe)
    d = json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_text())
    name = name or d["critics"][0]
    eps = [r["episode"] for r in d["rows"] if name in r["critics"]]
    rows = [r["critics"][name] for r in d["rows"] if name in r["critics"]]
    return d, name, eps, rows


def ci(x, groups):
    """95% t half-width, clustered by EPISODE — two frames from one trajectory are not two draws."""
    from scipy import stats

    x = np.asarray(x, np.float64)
    keys = sorted(set(groups))
    per = np.stack([x[[i for i, g in enumerate(groups) if g == k]].mean(axis=0) for k in keys])
    n = len(keys)
    if n < 2:
        return per.mean(axis=0), np.zeros(per.shape[1:])
    return per.mean(axis=0), per.std(axis=0, ddof=1) / np.sqrt(n) * stats.t.ppf(0.975, n - 1)


def bon_numbers(rows, eps, rng_seed=0):
    """The prize, the noise, and the bias -- the three numbers figure 1 is about.

    prize   how much a PERFECT oracle could win by picking the best of N. Bounded by how much the
            candidates actually differ in value, which is the signal spread with the ranking noise
            taken out.
    noise   how much the two ensemble heads disagree about the candidates AFTER removing their
            per-frame common offset. That offset is large (~8 cost-to-goal steps) and completely
            irrelevant to ranking: it shifts all 16 candidates together.
    bias    what the critic claims its own pick is worth, over the demonstrator -- the winner's
            curse, measured.
    """
    q = np.asarray([r["q_bc"] for r in rows], np.float64)  # [F, K, N]
    qc = q - q.mean(axis=2, keepdims=True)  # centre each head, each frame
    est = qc.mean(axis=1)  # [F, N] the score bon ranks on
    noise = ((qc[:, 0] - qc[:, 1]) / 2).std(axis=1)  # [F]
    est_sd = est.std(axis=1)
    signal = np.sqrt(np.maximum(est_sd**2 - noise**2, 0.0))  # de-noised spread

    rng = np.random.default_rng(rng_seed)
    ns = [n for n in (1, 2, 4, 8, 16) if n <= q.shape[2]]
    # An oracle ranks on the SIGNAL alone: sample N true values, take the best. The critic ranks on
    # signal+noise, so it sometimes picks a worse candidate AND overstates what it picked.
    prize, bias = [], []
    qmin = q.min(axis=1)  # bon selects on the min-ensemble score
    data_min = np.asarray([r["q_data"] for r in rows], np.float64)[..., 0].min(axis=1)
    for n in ns:
        idx = rng.integers(0, q.shape[2], size=(len(rows), 400, n))
        prize.append(
            np.take_along_axis(est[:, None, :], idx, 2).max(axis=2).mean(axis=1) * (signal / np.maximum(est_sd, 1e-9))
        )
        bias.append(np.take_along_axis(qmin[:, None, :], idx, 2).max(axis=2).mean(axis=1) - data_min)
    return {
        "ns": ns,
        "prize": np.stack(prize, 1),
        "bias": np.stack(bias, 1),
        "signal": signal,
        "noise": noise,
        "est": est,
        "qc": qc,
        "eps": eps,
    }


def fig_bon(rows, eps, name, out):
    import matplotlib.pyplot as plt
    from plot_style import GRAY
    from plot_style import PALETTE
    from plot_style import apply

    apply()
    B = bon_numbers(rows, eps)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))

    # --- 1. how different ARE the candidates -----------------------------------------------
    ax = axes[0]
    pooled = B["est"].reshape(-1)
    ax.hist(pooled, bins=40, color=PALETTE[0], alpha=0.85)
    s_m, s_h = ci(B["signal"][:, None], eps)
    n_m, n_h = ci(B["noise"][:, None], eps)
    ax.axvline(0, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("candidate's score − the frame's own mean  (cost-to-goal steps)")
    ax.set_ylabel("candidates")
    ax.set_title("how much the 16 draws differ")
    ax.text(
        0.03,
        0.95,
        f"real spread   {s_m[0]:.2f} ± {s_h[0]:.2f}\nranking noise {n_m[0]:.2f} ± {n_h[0]:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    # --- 2. the prize against the bias -------------------------------------------------------
    ax = axes[1]
    pm, ph = ci(B["prize"], eps)
    bm, bh = ci(B["bias"], eps)
    ax.errorbar(
        B["ns"],
        pm,
        yerr=ph,
        color=PALETTE[2],
        lw=2,
        marker="o",
        ms=4,
        capsize=3,
        label="best a PERFECT picker could win",
    )
    ax.errorbar(
        B["ns"],
        bm,
        yerr=bh,
        color=PALETTE[3],
        lw=2,
        marker="s",
        ms=4,
        capsize=3,
        label="what the critic CLAIMS its pick won",
    )
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (candidates drawn)")
    ax.set_ylabel("cost-to-goal steps vs the demonstrator")
    ax.set_title("the prize is smaller than the claim")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # --- 3. do the two heads even agree ------------------------------------------------------
    ax = axes[2]
    qc = B["qc"]
    ax.scatter(qc[:, 0].reshape(-1), qc[:, 1].reshape(-1), s=6, alpha=0.25, lw=0, color=PALETTE[4])
    lim = float(np.abs(qc).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, ls="--", lw=1)
    agree = np.mean([qc[f, 0].argmax() == qc[f, 1].argmax() for f in range(len(qc))])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("head 1's score for a candidate")
    ax.set_ylabel("head 2's score")
    ax.set_title("the two heads, on the same candidates")
    ax.text(
        0.03,
        0.95,
        f"they pick the same best\ncandidate {100 * agree:.0f}% of frames\n(chance = {100 / qc.shape[2]:.0f}%)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)
    return {
        "prize": [pm.tolist(), ph.tolist()],
        "bias": [bm.tolist(), bh.tolist()],
        "ns": B["ns"],
        "signal": [float(s_m[0]), float(s_h[0])],
        "noise": [float(n_m[0]), float(n_h[0])],
        "argmax_agree": float(agree),
        "critic": name,
    }


def fig_steer(rows, eps, name, out):
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    from plot_style import GRAY
    from plot_style import PALETTE
    from plot_style import apply

    apply()
    ts = np.asarray(rows[0]["abs_ts"], np.float64)
    ray = np.asarray([r["q_absray"] for r in rows], np.float64)  # [F, K, T]
    data = np.asarray([r["q_data"] for r in rows], np.float64)[..., 0]  # [F, K]
    d_mean = ray.mean(axis=1) - data.mean(axis=1)[:, None]
    std = ray.std(axis=1)
    span = np.asarray([r["grad_in_draw_span"] for r in rows], np.float64)
    chance = float(np.mean([r["grad_in_draw_span_chance"] for r in rows]))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    # --- 1. where the policy lives, and how flat Q is there ---------------------------------
    ax = axes[0]
    n = int(round(len(rows[0]["q_far"][0]) ** 0.5))
    far = (
        (np.asarray([r["q_far"] for r in rows], np.float64).mean(axis=1) - data.mean(axis=1)[:, None])
        .mean(axis=0)
        .reshape(n, n)
        .T
    )
    g = np.linspace(-1, 1, n)
    lim = float(np.abs(far).max())
    im = ax.pcolormesh(g, g, far, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    sig = np.asarray([r["pc_sigma"] for r in rows], np.float64).mean(axis=0)
    ax.add_patch(Ellipse((0, 0), 4 * sig[0], 4 * sig[1], fill=False, color="k", lw=1.6))
    ax.text(0, 2.05 * sig[1], "the policy's own draws (±2σ)", ha="center", fontsize=8.5)
    ax.set_xlabel("PC1 of the draws  (normalized action units)")
    ax.set_ylabel("PC2")
    ax.set_title(f"where the policy varies, Q barely moves\n(range {far.min():+.1f} to {far.max():+.1f})", fontsize=11)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax).set_label("Q − Q(demonstrator)", fontsize=9)

    # --- 2. along the gradient, it moves a lot ----------------------------------------------
    ax = axes[1]
    m, h = ci(d_mean, eps)
    ax.plot(ts, m, color=PALETTE[3], lw=2.2)
    ax.fill_between(ts, m - h, m + h, color=PALETTE[3], alpha=0.2, lw=0)
    ax.axhline(0, color=GRAY, ls="--", lw=1.2)
    ax.text(ts[-1], 0.4, "the demonstrator, who reached the goal", ha="right", fontsize=8.5, color=GRAY)
    ax.axvline(1.0, color=GRAY, ls=":", lw=1.2)
    ax.text(1.03, m.max() * 0.35, " box edge", fontsize=8.5, color=GRAY)
    ax.set_xlabel("distance pushed along $\\nabla_a Q$  (normalized action units)")
    ax.set_ylabel("Q − Q(demonstrator)   (cost-to-goal steps)")
    ax.set_title(
        f"along the gradient it climbs\n(+{m[-1]:.0f} steps ≈ {m[-1] / 30:.1f} s of 'progress' the data never saw)",
        fontsize=11,
    )

    # --- 3. how much pessimism it would take -------------------------------------------------
    ax = axes[2]
    # A RATIO OF AGGREGATES, not the average of per-frame ratios. Some frames have a near-zero
    # ensemble gap and d/std explodes there -- averaging those gave spikes of 800 and a band wider
    # than the panel. Numerator and denominator are both well estimated in aggregate; the band is
    # a bootstrap over episodes of that ratio.
    keys = sorted(set(eps))
    grp = {k: [i for i, g in enumerate(eps) if g == k] for k in keys}
    per_d = np.stack([d_mean[grp[k]].mean(axis=0) for k in keys])
    per_s = np.stack([std[grp[k]].mean(axis=0) for k in keys])
    km = per_d.mean(axis=0) / np.maximum(per_s.mean(axis=0), 1e-9)
    rng = np.random.default_rng(0)
    boot = np.stack(
        [
            per_d[bi].mean(axis=0) / np.maximum(per_s[bi].mean(axis=0), 1e-9)
            for bi in rng.integers(0, len(keys), size=(400, len(keys)))
        ]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.plot(ts[1:], km[1:], color=PALETTE[4], lw=2.2)
    ax.fill_between(ts[1:], lo[1:], hi[1:], color=PALETTE[4], alpha=0.2, lw=0)
    ax.set_ylim(0, max(float(hi[1:].max()) * 1.15, 1.4))
    for k, lab in ((RHO, f"ρ={RHO}  what QPILOTS uses"), (1.0, "ρ=1  the strongest available (min, K=2)")):
        ax.axhline(k, color=GRAY, ls="--", lw=1.2)
        ax.text(ts[-1], k + 0.05, lab, ha="right", fontsize=8.5, color=GRAY)
    ax.set_xlabel("distance pushed along $\\nabla_a Q$")
    ax.set_ylabel("ρ needed, in units of ensemble disagreement")
    ax.set_title("pessimism cannot reach it", fontsize=11)
    ax.text(
        0.03,
        0.95,
        f"{100 * (1 - span.mean()):.0f}% of $\\nabla_a Q$ points OUT of\nthe subspace the policy explores\n(chance would be {100 * (1 - chance):.0f}%)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)
    i_box = int(np.argmin(abs(ts - 1.0)))
    return {
        "critic": name,
        "abs_ts": ts.tolist(),
        "dq": [m.tolist(), h.tolist()],
        "k_needed_box": [float(km[i_box]), float(lo[i_box]), float(hi[i_box])],
        "k_needed_end": [float(km[-1]), float(lo[-1]), float(hi[-1])],
        "grad_out_of_span": float(1 - span.mean()),
        "chance_out": float(1 - chance),
        "flat_range": [float(far.min()), float(far.max())],
    }


def main(a):
    d, name, eps, rows = load(a.probe, a.critic_name)
    print(f"critic: {name}  ({len(rows)} frames, {len(set(eps))} episodes)")
    out = {"frames": len(rows), "episodes": len(set(eps))}
    out["bon"] = fig_bon(rows, eps, name, a.out_bon)
    out["steer"] = fig_steer(rows, eps, name, a.out_steer)
    pathlib.Path(a.summary).write_text(json.dumps(out, indent=1))
    print("wrote", a.summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="slurm/probes/q_landscape.json.gz")
    ap.add_argument("--critic-name", default="patch_critic_yam_s347_fixed_tau9_min_200k")
    ap.add_argument("--out-bon", default="hub_figs/ood_1_bon.png")
    ap.add_argument("--out-steer", default="hub_figs/ood_2_steering.png")
    ap.add_argument("--summary", default="ood_summary.json")
    main(ap.parse_args())
