"""Figures for the Q-landscape probe. Every number is recomputed from the probe's JSON.

Four panels, answering four separate questions:

  1  Q along the exploit direction, against the demonstrator's own action
     Does Q rise where the demonstrations do not go? An honest cost-to-goal critic should not
     rank an action the data never took above the one that reached the goal.

  2  Ensemble disagreement along the same axis
     Pessimism can only defend against error it can SEE. If std stays flat while the mean rises,
     no rho and no min-over-K stops best-of-N from selecting the optimism.

  3  The pessimistic values themselves
     min and mean - rho*std plotted next to the mean, so the question "does pessimism cancel it"
     is answered by looking rather than by arguing about rho.

  4  Best-of-N selection bias
     The value of the SELECTED candidate as N grows, minus the demonstrator's. This is the
     quantity a bon run reports as its own estimate, and it is the one that drifts.
"""

import argparse
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))

RHO = 0.5  # QPILOTS Eq. 12 default, and what serving.py's ArmSpec uses


def _stack(rows, key):
    """[frames, K, N] for a per-member field."""
    return np.asarray([r[key] for r in rows], np.float32)


def _ci(x, groups):
    """Mean and 95% t half-width, CLUSTERED BY EPISODE.

    40 frames come from 20 episodes, two each, and two chunks from the same smooth trajectory are
    not independent draws -- a frame-level CI treats them as if they were and reports a band that
    is too narrow. The repo's convention is run-level, and here the run is the episode: average
    within an episode first, then take the t-CI over the 20 episode means.
    """
    from scipy import stats

    x = np.asarray(x, np.float64)
    keys = sorted(set(groups))
    per = np.stack([x[[i for i, g in enumerate(groups) if g == k]].mean(axis=0) for k in keys])
    n = len(keys)
    m = per.mean(axis=0)
    if n < 2:
        return m, np.zeros_like(m)
    return m, per.std(axis=0, ddof=1) / np.sqrt(n) * stats.t.ppf(0.975, n - 1)


def main(a):
    import matplotlib.pyplot as plt
    from plot_style import GRAY
    from plot_style import PALETTE
    from plot_style import apply

    apply()
    # .gz because the raw probe is 7.5 MB and the figures must regenerate from it, not from a
    # transcribed summary -- a figure that can drift from its data is the thing this repo bans.
    pp = pathlib.Path(a.probe)
    d = json.loads(gzip.decompress(pp.read_bytes()) if pp.suffix == ".gz" else pp.read_text())
    # The probe scores every critic on the same frames, so a row is {critic name -> measurements}.
    # This figure is the anatomy of ONE of them; the nine-critic comparison is its own script.
    name = a.critic_name or d["critics"][0]
    pairs = [(r["episode"], r["critics"][name]) for r in d["rows"] if name in r["critics"]]
    if not pairs:
        raise SystemExit(f"{name!r} not in this probe: {d['critics']}")
    eps = [e for e, _ in pairs]
    rows = [x for _, x in pairs]
    print(f"critic: {name}  ({len(rows)} frames from {len(set(eps))} episodes)")
    ts = np.asarray(rows[0]["abs_ts"], np.float32)

    absray = _stack(rows, "q_absray")  # [F, K, T]
    data = _stack(rows, "q_data")[..., 0]  # [F, K]
    bc = _stack(rows, "q_bc")  # [F, K, N]

    # Everything relative to the demonstrator's own continuation: it is the only action here known
    # to have reached the goal, so it is the anchor a critic should not beat off-support.
    # Each estimator is compared against ITSELF applied to the demonstrator. Scoring `min` against
    # a `mean` anchor makes min start below zero at t=0 purely from the mismatch -- an artifact
    # that reads as "the critic already beats the demonstrator", which it does not.
    def rel(f):
        return f(absray, axis=1) - f(data, axis=1)[:, None]

    d_mean = rel(np.mean)
    d_min = rel(np.min)
    d_pess = (absray.mean(axis=1) - RHO * absray.std(axis=1)) - (data.mean(axis=1) - RHO * data.std(axis=1))[:, None]
    std = absray.std(axis=1)  # [F, T]

    # With K=2, min IS mean - 1*std exactly. So mean / mean-0.5s / min is not three estimators but
    # one family at k = 0, 0.5, 1 -- the honest statement is which k would restore parity, not
    # "even the min fails".
    #
    # A RATIO OF AGGREGATES, per episode, not the average of per-frame d/std. This used to be the
    # per-frame ratio, which explodes on frames where the two heads nearly agree: on the pinned
    # checkpoint it read 29.1 at the box edge and 176.5 at t=3 against the 2.50 and 4.64 that
    # plot_ood.py computes on the same probe -- and 2.50 / 4.64 are what the report publishes.
    # plot_ood.py:250 already carried the fix and the reason; this script did not.
    def _k_needed(eps_):
        keys = sorted(set(eps_))
        grp = {k: [i for i, g in enumerate(eps_) if g == k] for k in keys}
        per_d = np.stack([d_mean[grp[k]].mean(axis=0) for k in keys])
        per_s = np.stack([std[grp[k]].mean(axis=0) for k in keys])
        return per_d, per_s

    per_d, per_s = _k_needed(eps)
    k_needed = per_d.mean(axis=0) / np.maximum(per_s.mean(axis=0), 1e-9)  # [T]

    fig, axes = plt.subplots(1, 5, figsize=(21, 3.6))

    ax = axes[0]
    m, h = _ci(d_mean, eps)
    ax.plot(ts, m, color=PALETTE[0], lw=2)
    ax.fill_between(ts, m - h, m + h, color=PALETTE[0], alpha=0.2, lw=0)
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.axvline(1.0, color=PALETTE[3], ls=":", lw=1.5)
    ax.set_xlabel("distance along $\\nabla_a Q$  (normalized action units)")
    ax.set_ylabel("$Q - Q(\\mathrm{demonstrator})$")
    ax.set_title("Q off-support")
    ax.text(1.02, ax.get_ylim()[0], " box edge", color=PALETTE[3], fontsize=9, va="bottom")

    ax = axes[1]
    m, h = _ci(std, eps)
    ax.plot(ts, m, color=PALETTE[2], lw=2)
    ax.fill_between(ts, m - h, m + h, color=PALETTE[2], alpha=0.2, lw=0)
    ax.axvline(1.0, color=PALETTE[3], ls=":", lw=1.5)
    ax.set_xlabel("distance along $\\nabla_a Q$")
    ax.set_ylabel("ensemble std  (K=2, so this is $|Q_1-Q_2|/2$)")
    ax.set_title("what pessimism can see")

    ax = axes[2]
    for arr, lab, col in (
        (d_mean, "mean", PALETTE[0]),
        (d_pess, f"mean $-$ {RHO}$\\sigma$", PALETTE[4]),
        (d_min, "min", PALETTE[1]),
    ):
        m, h = _ci(arr, eps)
        ax.plot(ts, m, color=col, lw=2, label=lab)
        ax.fill_between(ts, m - h, m + h, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.axvline(1.0, color=PALETTE[3], ls=":", lw=1.5)
    ax.set_xlabel("distance along $\\nabla_a Q$")
    ax.set_ylabel("$Q - Q(\\mathrm{demonstrator})$")
    ax.set_title("does pessimism cancel it  (k = 0, 0.5, 1)")
    ax.legend(frameon=False, fontsize=9)

    # best-of-N: the value of the selected candidate, as a function of N. Subsets drawn from the
    # frame's own BC draws, so this is the bias bon actually incurs at this state.
    ax = axes[3]
    qmin_bc = bc.min(axis=1)  # [F, N] -- bon selects on the min-ensemble Q
    rng = np.random.default_rng(0)
    ns = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= qmin_bc.shape[1]]
    sel = np.stack(
        [
            np.stack([qmin_bc[f][rng.integers(0, qmin_bc.shape[1], size=(200, n))].max(axis=1).mean() for n in ns])
            for f in range(qmin_bc.shape[0])
        ]
    )
    m, h = _ci(sel - data.min(axis=1)[:, None], eps)
    ax.errorbar(ns, m, yerr=h, color=PALETTE[3], lw=2, marker="o", ms=4, capsize=3)
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (candidates)")
    ax.set_ylabel("$Q(\\mathrm{selected}) - Q(\\mathrm{demonstrator})$")
    ax.set_title("best-of-N selection bias")

    # The landscape itself: Q on the plane spanned by the exploit direction and one orthogonal to
    # it, averaged over frames after each is centred on its own demonstrator value. Colour is the
    # same quantity as panel 1, so the two are read together -- the ridge running along e1 IS the
    # rise in panel 1, and its flatness along e2 says the optimism is directional, not a general
    # inflation of everything off the data.
    ax = axes[4]
    gts = np.asarray(rows[0]["grid_ts"], np.float32)
    n = len(gts)
    grid = _stack(rows, "q_grid").mean(axis=1)  # [F, n*n] ensemble mean
    z = (grid - data.mean(axis=1)[:, None]).mean(axis=0).reshape(n, n).T  # rows vary e1(a), cols e2(b) -> transpose
    lim = float(np.abs(z).max())
    im = ax.pcolormesh(gts, gts, z, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    ax.contour(gts, gts, z, levels=8, colors="k", linewidths=0.4, alpha=0.4)
    ax.plot(0, 0, "o", color="k", ms=5)
    # the BC cloud, to scale: best-of-N cannot leave it, steering is not bounded by it
    # The BC cloud drawn to scale, using its spread ALONG THIS PLANE -- r["sigma"] is the mean
    # per-coordinate std (~0.009) and is NOT the cloud's radius in a projected direction, where
    # variance concentrates (top PC holds ~44%). Using it would draw the cloud 20x too small.
    sig = (
        float(np.mean([r["pc_sigma"][0] for r in rows]))
        if "pc_sigma" in rows[0]
        else float(np.mean([r["sigma"] for r in rows]))
    )
    ax.add_patch(plt.Circle((0, 0), 2 * sig, fill=False, color="k", lw=1.2))
    ax.set_xlabel("along $\\nabla_a Q$   (normalized action units)")
    ax.set_ylabel("orthogonal")
    ax.set_title("the landscape")
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("$Q - Q(\\mathrm{demonstrator})$", fontsize=9)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)

    # The numbers the figure is made of, so a report never transcribes them by hand.
    j = {"frames": len(rows), "rho": RHO, "abs_ts": ts.tolist()}
    for name, arr in (("dq_mean", d_mean), ("dq_min", d_min), ("dq_pess", d_pess), ("std", std)):
        m, h = _ci(arr, eps)
        j[name] = {"mean": m.tolist(), "ci95": h.tolist()}
    m, h = _ci(sel - data.min(axis=1)[:, None], eps)
    j["bon"] = {"n": ns, "mean": m.tolist(), "ci95": h.tolist()}
    j["critic"] = name
    j["episodes"] = len(set(eps))
    j["ci_unit"] = "episode (frames averaged within an episode first)"
    j["ensemble_K"] = int(absray.shape[1])
    # PAIRED growth of the disagreement: per frame, std(t_end) - std(t=0). The level CIs at the two
    # ends overlap, which says nothing about the growth -- between-frame variance dominates them
    # and cancels in the difference.
    gm, gh = _ci((std[:, -1] - std[:, 0])[:, None], eps)
    j["std_growth_paired"] = {
        "mean": float(gm[0]),
        "ci95": float(gh[0]),
        "start": float(_ci(std, eps)[0][0]),
        "end": float(_ci(std, eps)[0][-1]),
    }
    # The k that would restore parity with the demonstrator, per frame, at the box edge and the end.
    i_box = int(np.argmin(abs(ts - 1.0)))
    for tag, idx in (("at_box", i_box), ("at_end", -1)):
        rng_b = np.random.default_rng(0)
        boot = np.stack(
            [
                per_d[b].mean(axis=0) / np.maximum(per_s[b].mean(axis=0), 1e-9)
                for b in rng_b.integers(0, per_d.shape[0], size=(400, per_d.shape[0]))
            ]
        )
        lo_b, hi_b = np.percentile(boot[:, idx], [2.5, 97.5])
        j[f"k_needed_{tag}"] = {"mean": float(k_needed[idx]), "ci95_lo": float(lo_b), "ci95_hi": float(hi_b)}
    # How much of the chunk actually left the box at each distance -- "box edge" on the x axis is a
    # displacement LENGTH, not a statement that any coordinate left [-1, 1].
    j["outbox_absray"] = float(np.mean([r["outbox_absray"] for r in rows]))
    j["outbox_bc"] = float(np.mean([r["outbox_bc"] for r in rows]))
    pathlib.Path(a.summary).write_text(json.dumps(j, indent=1))
    print("wrote", a.summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="slurm/probes/q_landscape.json.gz")
    ap.add_argument("--critic-name", default=None, help="which critic to anatomise (default: the first)")
    ap.add_argument("--out", default="hub_figs/q_landscape.png")
    ap.add_argument("--summary", default="q_landscape_summary.json")
    main(ap.parse_args())
