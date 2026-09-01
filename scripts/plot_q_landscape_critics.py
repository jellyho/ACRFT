"""Is off-support overestimation a property of ONE checkpoint, or of the method?

The single-critic probe showed Q rising +33 (cost-to-goal steps) as an action moves 3 box-widths
along grad_a Q, with the ensemble's own disagreement growing only 1.7x -- so pessimism sees it and
cannot cancel it. That is one checkpoint. This runs the same frames, the same BC draws and the
same measurement through every critic trained on this dataset, which differ along four axes:

    expectile   0.70 / 0.90    a higher IQL expectile is a more optimistic target
    macro       30 / 5         commitment granularity
    mc_floor    on / off       whether the Monte-Carlo return floors the bootstrap
    aug         on / noaug     image augmentation -- the axis most directly about generalising
                               off the training distribution

If the rise is common to all nine, it is the method. If it tracks one axis, that axis is the knob.
Nothing here is a claim about which critic is BEST at the task: this measures one failure mode,
and a critic can be well-calibrated off-support and still rank on-support actions poorly.
"""

import argparse
import gzip
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))

RHO = 0.5


def axes_of(name: str) -> dict:
    """The four training knobs, read off the checkpoint name -- they are not all in config.json."""
    return {
        "expectile": 0.90 if "tau9" in name else 0.70,
        "macro": 5 if "_g5" in name else 30,
        "floor": "nofloor" not in name,
        "aug": "noaug" not in name,
    }


def _ci(x, axis=0):
    from scipy import stats

    m = x.mean(axis=axis)
    n = x.shape[axis]
    if n < 2:
        return m, np.zeros_like(m)
    return m, x.std(axis=axis, ddof=1) / np.sqrt(n) * stats.t.ppf(0.975, n - 1)


def curves(rows, name):
    """Per-frame Q(t) - Q(demonstrator), for one critic. [F, T]"""
    got = [r["critics"][name] for r in rows if name in r["critics"]]
    ray = np.asarray([g["q_absray"] for g in got], np.float32)  # [F, K, T]
    data = np.asarray([g["q_data"] for g in got], np.float32)[..., 0]  # [F, K]
    anchor = data.mean(axis=1)[:, None]
    return {
        "mean": ray.mean(axis=1) - anchor,
        "min": ray.min(axis=1) - anchor,
        "pess": (ray.mean(axis=1) - RHO * ray.std(axis=1)) - anchor,
        "std": ray.std(axis=1),
        "bc": np.asarray([g["q_bc"] for g in got], np.float32),
        "data": data,
    }


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
    rows, names = d["rows"], d["critics"]
    ts = np.asarray(rows[0]["critics"][names[0]]["abs_ts"], np.float32)
    C = {n: curves(rows, n) for n in names}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))

    # 1-2: every critic's rise, coloured by the two axes most likely to explain it
    for ax, key, title in ((axes[0], "expectile", "by IQL expectile"), (axes[1], "aug", "by augmentation")):
        vals = sorted({axes_of(n)[key] for n in names})
        for i, v in enumerate(vals):
            sub = [n for n in names if axes_of(n)[key] == v]
            stack = np.concatenate([C[n]["mean"] for n in sub])
            m, h = _ci(stack)
            lab = f"{key}={v}  (n={len(sub)})"
            ax.plot(ts, m, color=PALETTE[i], lw=2, label=lab)
            ax.fill_between(ts, m - h, m + h, color=PALETTE[i], alpha=0.2, lw=0)
        ax.axhline(0, color=GRAY, ls="--", lw=1)
        ax.axvline(1.0, color=GRAY, ls=":", lw=1)
        ax.set_xlabel("distance along $\\nabla_a Q$")
        ax.set_ylabel("$Q - Q(\\mathrm{demonstrator})$")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)

    # 3: the whole set, one line each -- is anyone flat?
    ax = axes[2]
    for i, n in enumerate(sorted(names)):
        m, _ = _ci(C[n]["mean"])
        ax.plot(ts, m, color=PALETTE[i % len(PALETTE)], lw=1.6, label=re.sub(r"^patch_critic_yam_s347_|_200k$", "", n))
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.axvline(1.0, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("distance along $\\nabla_a Q$")
    ax.set_ylabel("$Q - Q(\\mathrm{demonstrator})$")
    ax.set_title("every critic")
    ax.legend(frameon=False, fontsize=7, ncol=1)

    # 4: how much of the rise pessimism removes, per critic. The bar is what min() is worth --
    # the quantity bon relies on and steering's rho approximates.
    ax = axes[3]
    order = sorted(names, key=lambda n: _ci(C[n]["mean"])[0][-1])
    xs = np.arange(len(order))
    mm = np.array([_ci(C[n]["mean"])[0][-1] for n in order])
    mh = np.array([_ci(C[n]["mean"])[1][-1] for n in order])
    pm = np.array([_ci(C[n]["min"])[0][-1] for n in order])
    ph = np.array([_ci(C[n]["min"])[1][-1] for n in order])
    ax.barh(xs - 0.2, mm, height=0.38, xerr=mh, color=PALETTE[0], label="mean", error_kw={"lw": 1})
    ax.barh(xs + 0.2, pm, height=0.38, xerr=ph, color=PALETTE[1], label="min", error_kw={"lw": 1})
    ax.set_yticks(xs)
    ax.set_yticklabels([re.sub(r"^patch_critic_yam_s347_|_200k$", "", n) for n in order], fontsize=7)
    ax.set_xlabel(f"$Q - Q(\\mathrm{{demonstrator}})$ at t={ts[-1]:.0f}")
    ax.set_title("what pessimism removes")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(visible=True, axis="x")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)

    summary = {"frames": len(rows), "rho": RHO, "abs_ts": ts.tolist(), "critics": {}}
    for n in names:
        c = C[n]
        m, h = _ci(c["mean"])
        mn, hn = _ci(c["min"])
        sd, _ = _ci(c["std"])
        summary["critics"][n] = {
            **axes_of(n),
            "dq_mean_end": [float(m[-1]), float(h[-1])],
            "dq_min_end": [float(mn[-1]), float(hn[-1])],
            "dq_mean_at_box": [float(m[np.argmin(abs(ts - 1.0))]), float(h[np.argmin(abs(ts - 1.0))])],
            "std_start_end": [float(sd[0]), float(sd[-1])],
            "dq_mean_curve": m.tolist(),
        }
    pathlib.Path(a.summary).write_text(json.dumps(summary, indent=1))
    print("wrote", a.summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="slurm/probes/q_landscape.json.gz")
    ap.add_argument("--out", default="hub_figs/q_landscape_critics.png")
    ap.add_argument("--summary", default="q_landscape_critics_summary.json")
    main(ap.parse_args())
