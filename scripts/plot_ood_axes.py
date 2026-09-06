"""What changes the two failures, and what does not.

Three candidate explanations were on the table for why an IQL critic does not help this policy.
Each is a knob we can turn, and turning them separates the failures cleanly:

  base strength   100k / 150k / 200k BC checkpoints. A weaker base has more spread among its
                  draws, so selection has more to work with -- V-GPS's advantage, quantified.
  N               8 / 16 / 50 candidates. DEAS's real-robot run and V-GPS both use ~50 and we
                  used 8, so this is the obvious "you under-powered it" objection.
  where the state came from   demonstrations vs the states a deployed policy actually reaches.

The answer is that the two failures live on different axes: selection improves with a weaker base
and does not care about N, while the gradient's overestimation is FLAT in both -- it is a property
of the critic, not of the policy or the search budget.
"""

import argparse
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))
NAME = "patch_critic_yam_s347_fixed_tau9_min_200k"


def load(p):
    p = pathlib.Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_text())


def measure(path):
    """The five numbers each panel needs, from one probe file."""
    from scipy import stats

    d = load(path)
    rows, names = d["rows"], d["critics"]
    eps = [r["episode"] for r in rows]
    R = [r["critics"][NAME] for r in rows]
    keys = sorted(set(eps))

    def ci(x):
        x = np.asarray(x, np.float64)
        per = np.stack([x[[i for i, g in enumerate(eps) if g == k]].mean(0) for k in keys])
        if per.ndim == 1:
            per = per[:, None]
        return per.mean(0), per.std(0, ddof=1) / np.sqrt(len(keys)) * stats.t.ppf(0.975, len(keys) - 1)

    q = np.asarray([r["q_bc"] for r in R], np.float64)
    qc = q - q.mean(2, keepdims=True)  # per-head, per-frame centring: the offset cannot affect a ranking
    noise = ((qc[:, 0] - qc[:, 1]) / 2).std(1)
    signal = np.sqrt(np.maximum(qc.mean(1).std(1) ** 2 - noise**2, 0))
    g = np.asarray([r["q_absray"] for r in R], np.float64).mean(1)
    g = g - g[:, :1]
    gm, gh = ci(g)

    # bon: pick with one critic, score with the other eight. The bias is what the picker adds.
    Q = np.stack([np.stack([np.asarray(r["critics"][n]["q_bc"]).mean(0) for r in rows]) for n in names])
    Q = Q - Q.mean(2, keepdims=True)
    C, F, N = Q.shape
    rng = np.random.default_rng(0)
    # Select over the WHOLE pool. This used to be min(16, N), which capped the subset at 16 in every
    # arm -- so the 50-draw probe was still a best-of-16 and the panel's "more candidates changes
    # neither" was true by construction. It is not: sweeping k on the same frozen 50-draw probe is
    # monotone (0.24, 0.43, 0.59, 0.71, 0.82, 0.88 at k = 2, 4, 8, 16, 32, 50).
    idx = rng.integers(0, N, size=(F, 300, N))
    own, real = [], []
    for a in range(C):
        s = np.take_along_axis(Q[a][:, None, :], idx, 2)
        pk = s.argmax(2, keepdims=True)
        own.append(np.take_along_axis(s, pk, 2)[..., 0].mean(1))
        real.append(
            np.mean(
                [
                    np.take_along_axis(np.take_along_axis(Q[b][:, None, :], idx, 2), pk, 2)[..., 0].mean(1)
                    for b in range(C)
                    if b != a
                ],
                0,
            )
        )
    rm, rh = ci(np.mean(real, 0))
    om, _ = ci(np.mean(own, 0))
    return {
        "n_draws": N,
        "grad_end": (float(gm[-1]), float(gh[-1])),
        "bon_real": (float(rm[0]), float(rh[0])),
        "bon_bias": float(om[0] - rm[0]),
        "signal": float(ci(signal)[0][0]),
        "noise": float(ci(noise)[0][0]),
        "action_frac": float(
            np.stack([np.asarray(r["q_bc"]).mean(0) for r in R]).std(1).mean() ** 2
            / (
                np.stack([np.asarray(r["q_bc"]).mean(0) for r in R]).mean(1).std() ** 2
                + np.stack([np.asarray(r["q_bc"]).mean(0) for r in R]).std(1).mean() ** 2
            )
        ),
    }


def _twin(ax, xs, base, grad, xlabel, title, xticks=None):
    """Selection gain on the left axis, gradient overestimation on the right — the whole point is
    that one moves and the other does not, so they share an x and nothing else."""
    from plot_style import GRAY
    from plot_style import PALETTE

    bm = [b[0] for b in base]
    bh = [b[1] for b in base]
    ax.errorbar(xs, bm, yerr=bh, color=PALETTE[2], lw=2, marker="o", ms=5, capsize=3, label="best-of-N real gain")
    ax.set_ylabel("cost-to-goal steps won by selecting", color=PALETTE[2])
    ax.tick_params(axis="y", labelcolor=PALETTE[2])
    ax.set_ylim(0, max(bm) * 1.6)
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax2 = ax.twinx()
    gm = [g[0] for g in grad]
    gh = [g[1] for g in grad]
    ax2.errorbar(xs, gm, yerr=gh, color=PALETTE[3], lw=2, marker="s", ms=5, capsize=3, label="gradient overestimation")
    ax2.set_ylabel("Q − Q(demonstrator) along $\\nabla_a Q$", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])
    ax2.set_ylim(0, max(gm) * 1.6)
    ax2.grid(visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    if xticks:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticks)
    return ax2


def main(a):
    import matplotlib.pyplot as plt
    from plot_style import GRAY
    from plot_style import PALETTE
    from plot_style import apply

    apply()
    root = pathlib.Path(a.root)
    base = {
        s: measure(root / f"probe_base_{s}.json") for s in (100000, 150000) if (root / f"probe_base_{s}.json").exists()
    }
    base[200000] = measure(a.probe)
    n50 = measure(root / "probe_n50.json") if (root / "probe_n50.json").exists() else None
    rollouts = {p.stem.replace("probe_rollout_", ""): measure(p) for p in sorted(root.glob("probe_rollout_*.json"))}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # 1 --- base strength
    ks = sorted(base)
    ax2 = _twin(
        axes[0],
        [k / 1000 for k in ks],
        [base[k]["bon_real"] for k in ks],
        [base[k]["grad_end"] for k in ks],
        "BC checkpoint (thousand steps)",
        "a weaker base helps selection, not the gradient",
        xticks=[f"{k // 1000}k" for k in ks],
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper center")
    ax2.legend(frameon=False, fontsize=8, loc="lower center")

    # 2 --- N
    if n50:
        pts = [(base[200000]["n_draws"], base[200000]), (n50["n_draws"], n50)]
        xs = [p[0] for p in pts]
        _twin(
            axes[1],
            xs,
            [p[1]["bon_real"] for p in pts],
            [p[1]["grad_end"] for p in pts],
            "N (candidates drawn)",
            "more candidates changes neither",
            xticks=[str(x) for x in xs],
        )
        axes[1].annotate(
            "what DEAS's real robot\nand V-GPS use",
            (xs[-1], base[200000]["bon_real"][0] * 0.35),
            ha="right",
            fontsize=8,
            color=GRAY,
        )

    # 3 --- where the state came from
    order = ["(demos)"] + [k for k in ("implicit", "bon8_v2", "qpilots_0_05", "qpilots_0_1") if k in rollouts]
    vals = {"(demos)": base[200000], **rollouts}
    ax = axes[2]
    xs = np.arange(len(order))
    sn = [vals[k]["signal"] / max(vals[k]["noise"], 1e-9) for k in order]
    ax.bar(xs, sn, color=[PALETTE[0]] + [PALETTE[1]] * (len(order) - 1))
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.2)
    ax.text(len(order) - 0.5, 1.03, "signal = noise", ha="right", fontsize=8.5, color=GRAY)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [k.replace("_v2", "").replace("qpilots_0_", "qpilots α=0.") for k in order],
        rotation=20,
        ha="right",
        fontsize=8.5,
    )
    ax.set_ylabel("ranking signal ÷ ranking noise")
    ax.set_title("the states a deployed policy reaches", fontsize=11)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)
    summary = {"base": {str(k): v for k, v in base.items()}, "n50": n50, "rollouts": rollouts}
    pathlib.Path(a.summary).write_text(json.dumps(summary, indent=1))
    print("wrote", a.summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="slurm/probes/q_landscape.json.gz")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="hub_figs/ood_3_axes.png")
    ap.add_argument("--summary", default="ood_axes_summary.json")
    main(ap.parse_args())
