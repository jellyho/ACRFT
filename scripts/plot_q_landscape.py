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


def _ci(x, axis=0):
    """mean and 95% t half-width over frames -- run-level, the house convention."""
    from scipy import stats

    m = x.mean(axis=axis)
    n = x.shape[axis]
    if n < 2:
        return m, np.zeros_like(m)
    se = x.std(axis=axis, ddof=1) / np.sqrt(n)
    return m, se * stats.t.ppf(0.975, n - 1)


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
    rows = [r["critics"][name] for r in d["rows"] if name in r["critics"]]
    if not rows:
        raise SystemExit(f"{name!r} not in this probe: {d['critics']}")
    print(f"critic: {name}  ({len(rows)} frames)")
    ts = np.asarray(rows[0]["abs_ts"], np.float32)

    absray = _stack(rows, "q_absray")  # [F, K, T]
    data = _stack(rows, "q_data")[..., 0]  # [F, K]
    bc = _stack(rows, "q_bc")  # [F, K, N]

    # Everything relative to the demonstrator's own continuation: it is the only action here known
    # to have reached the goal, so it is the anchor a critic should not beat off-support.
    anchor = data.mean(axis=1)[:, None]  # [F, 1] ensemble-mean Q of the data action
    d_mean = absray.mean(axis=1) - anchor  # [F, T]
    d_min = absray.min(axis=1) - anchor
    d_pess = (absray.mean(axis=1) - RHO * absray.std(axis=1)) - anchor
    std = absray.std(axis=1)  # [F, T]

    fig, axes = plt.subplots(1, 5, figsize=(21, 3.6))

    ax = axes[0]
    m, h = _ci(d_mean)
    ax.plot(ts, m, color=PALETTE[0], lw=2)
    ax.fill_between(ts, m - h, m + h, color=PALETTE[0], alpha=0.2, lw=0)
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.axvline(1.0, color=PALETTE[3], ls=":", lw=1.5)
    ax.set_xlabel("distance along $\\nabla_a Q$  (normalized action units)")
    ax.set_ylabel("$Q - Q(\\mathrm{demonstrator})$")
    ax.set_title("Q off-support")
    ax.text(1.02, ax.get_ylim()[0], " box edge", color=PALETTE[3], fontsize=9, va="bottom")

    ax = axes[1]
    m, h = _ci(std)
    ax.plot(ts, m, color=PALETTE[2], lw=2)
    ax.fill_between(ts, m - h, m + h, color=PALETTE[2], alpha=0.2, lw=0)
    ax.axvline(1.0, color=PALETTE[3], ls=":", lw=1.5)
    ax.set_xlabel("distance along $\\nabla_a Q$")
    ax.set_ylabel("ensemble std")
    ax.set_title("what pessimism can see")

    ax = axes[2]
    for arr, lab, col in (
        (d_mean, "mean", PALETTE[0]),
        (d_pess, f"mean $-$ {RHO}$\\sigma$", PALETTE[4]),
        (d_min, "min", PALETTE[1]),
    ):
        m, h = _ci(arr)
        ax.plot(ts, m, color=col, lw=2, label=lab)
        ax.fill_between(ts, m - h, m + h, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.axvline(1.0, color=PALETTE[3], ls=":", lw=1.5)
    ax.set_xlabel("distance along $\\nabla_a Q$")
    ax.set_ylabel("$Q - Q(\\mathrm{demonstrator})$")
    ax.set_title("does pessimism cancel it")
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
    m, h = _ci(sel - data.min(axis=1)[:, None])
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
    z = (grid - anchor).mean(axis=0).reshape(n, n).T  # rows vary e1(a), cols e2(b) -> transpose
    lim = float(np.abs(z).max())
    im = ax.pcolormesh(gts, gts, z, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    ax.contour(gts, gts, z, levels=8, colors="k", linewidths=0.4, alpha=0.4)
    ax.plot(0, 0, "o", color="k", ms=5)
    # the BC cloud, to scale: best-of-N cannot leave it, steering is not bounded by it
    sig = float(np.mean([r["sigma"] for r in rows]))
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
        m, h = _ci(arr)
        j[name] = {"mean": m.tolist(), "ci95": h.tolist()}
    m, h = _ci(sel - data.min(axis=1)[:, None])
    j["bon"] = {"n": ns, "mean": m.tolist(), "ci95": h.tolist()}
    pathlib.Path(a.summary).write_text(json.dumps(j, indent=1))
    print("wrote", a.summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="slurm/probes/q_landscape.json.gz")
    ap.add_argument("--critic-name", default=None, help="which critic to anatomise (default: the first)")
    ap.add_argument("--out", default="hub_figs/q_landscape.png")
    ap.add_argument("--summary", default="q_landscape_summary.json")
    main(ap.parse_args())
