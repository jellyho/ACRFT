"""Figures from the rollout statistics.

Three pictures, each answering a question the table can only hint at:

1. **Commitment distribution** -- what the critic actually chose, not its average. A mean of 14
   steps is either every replan committing 14 or a split between 5 and 30, and only the shape tells
   which. This is the figure that shows a fixed critic has no choice to make.
2. **Commitment length** -- the mean per arm with a 95% t-CI over episodes, so arms are comparable.
3. **The splice** -- the largest joint step across a replan boundary against the same statistic
   inside a chunk. The boundary bar alone says nothing; the pair is the measurement.

Style follows the house convention (slurm/plot_style.py): white, y-grid only, seaborn deep, plain
titles, CI as error bars.
"""

import argparse
import json
import pathlib
import sys

import numpy as np


def _style():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))
    import plot_style

    plot_style.apply()
    return plot_style


def _short(repo_id: str) -> str:
    """Drop the prefix every arm shares -- it costs tick space and distinguishes nothing."""
    name = repo_id.rstrip("/").split("/")[-1]
    for prefix in ("yam_s300_h30_", "yam_s300_rel_200k_", "yam_s300_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def commitment_distribution(results: list, out: pathlib.Path, title: str = "") -> pathlib.Path:
    """Share of replans at each commitment length, grouped bars.

    Bars rather than lines: the commitment is discrete (multiples of the macro-group size), and
    overlaid lines hid four of six arms behind each other -- with no way to tell an arm that
    coincides from one that is missing.
    """
    import matplotlib.pyplot as plt

    ps = _style()
    hists = []
    for r in results:
        # int keys through JSON come back as strings; normalize rather than index with either.
        h = {int(k): int(v) for k, v in (r["aggregate"].get("chunk_hist") or {}).items()}
        total = sum(h.values()) or 1
        hists.append({k: 100 * v / total for k, v in h.items()})

    # Lengths worth their own bar, plus one bucket for the long tail of odd values (a chunk cut
    # short by the end of an episode or an intervention) so the shares still sum to 100.
    major = sorted({k for h in hists for k, share in h.items() if share >= 2.0})
    labels = [str(k) for k in major] + ["other"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    n = len(results)
    width = 0.8 / n
    x = np.arange(len(labels))
    for i, (r, h) in enumerate(zip(results, hists, strict=False)):
        vals = [h.get(k, 0.0) for k in major]
        vals.append(sum(share for k, share in h.items() if k not in major))
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, color=ps.PALETTE[i % len(ps.PALETTE)],
               label=_short(r["repo_id"]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("steps committed per replan")
    ax.set_ylabel("share of replans (%)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.savefig(out)
    plt.close(fig)
    return out


def _bar_with_ci(ax, results, key, color_idx=0, label=None, offset=0.0, width=0.8):
    xs, means, errs, names = [], [], [], []
    for i, r in enumerate(results):
        m = r["aggregate"].get(key)
        if not m or m.get("mean") is None:
            continue
        xs.append(i + offset)
        means.append(m["mean"])
        # A single episode has no interval; drawing 0 would read as agreement across runs.
        errs.append(m.get("ci") or 0.0)
        names.append(_short(r["repo_id"]))
    ps = _style()
    ax.bar(xs, means, width, yerr=errs, capsize=3, color=ps.PALETTE[color_idx % len(ps.PALETTE)], label=label,
           error_kw={"ecolor": ps.GRAY, "lw": 1.1})
    return xs, names


def commitment_length(results: list, out: pathlib.Path, title: str = "") -> pathlib.Path:
    """Mean steps committed per replan, 95% t-CI over episodes."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs, names = _bar_with_ci(ax, results, "chunk_mean")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([_short(r["repo_id"]) for r in results], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("steps committed per replan")
    if title:
        ax.set_title(title)
    fig.savefig(out)
    plt.close(fig)
    return out


def splice(results: list, out: pathlib.Path, title: str = "") -> pathlib.Path:
    """Joint step at a replan boundary vs inside a chunk (95th pct), 95% t-CI over episodes."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bar_with_ci(ax, results, "boundary_jump_p95", color_idx=3, label="across a replan boundary", offset=-0.2, width=0.4)
    _bar_with_ci(ax, results, "within_jump_p95", color_idx=7, label="inside a chunk", offset=0.2, width=0.4)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([_short(r["repo_id"]) for r in results], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("max joint step, 95th pct (rad)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return out


def macro_choice(results: list, out: pathlib.Path, title: str = "") -> pathlib.Path:
    """How often each macro group was the critic's commitment, per episode.

    This is the critic's DECISION, not the realized chunk -- a reply cut short by the end of an
    episode is not a shorter commitment. One point per episode with the arm's mean as a bar behind
    them, because a mean k* of 2.9 hides episodes running from 1.8 to 3.6.
    """
    import matplotlib.pyplot as plt

    ps = _style()
    arms = [r for r in results if (r["aggregate"].get("kstar_hist") or {})]
    if not arms:
        raise SystemExit("no adaptive run here -- k* needs critic_macro / critic_best_prefix")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, r in enumerate(arms):
        hist = {int(k): int(v) for k, v in r["aggregate"]["kstar_hist"].items()}
        total = sum(hist.values()) or 1
        ks = sorted(hist)
        color = ps.PALETTE[i % len(ps.PALETTE)]
        width = 0.8 / len(arms)
        offset = (i - (len(arms) - 1) / 2) * width
        ax.bar([k + offset for k in ks], [100 * hist[k] / total for k in ks], width, color=color,
               label=f"{_short(r['repo_id'])}  (k* {r['aggregate'].get('kstar_mean', {}).get('mean', float('nan')):.2f})")
        # Per-episode shares as points, so an arm's spread across runs is visible behind its mean.
        for ep in r["per_episode"]:
            h = {int(k): int(v) for k, v in (ep.get("kstar_hist") or {}).items()}
            t = sum(h.values()) or 1
            ax.plot([k + offset for k in ks], [100 * h.get(k, 0) / t for k in ks], ".", ms=3,
                    color=ps.GRAY, alpha=0.55, zorder=3)
    macro = arms[0]["aggregate"].get("macro")
    ax.set_xlabel(f"macro group committed (k*{f' x {macro} steps' if macro else ''})")
    ax.set_ylabel("share of replans (%)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(results: list, out_dir: pathlib.Path, title: str = "") -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = [
        commitment_distribution(results, out_dir / "commitment_distribution.png", title),
        commitment_length(results, out_dir / "commitment_length.png", title),
        splice(results, out_dir / "splice.png", title),
    ]
    if any(r["aggregate"].get("kstar_hist") for r in results):
        figs.append(macro_choice(results, out_dir / "macro_choice.png", title))
    return figs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", default=None, help="a --json file from `yam-misc stats` (else compute now)")
    p.add_argument("--repo-id", nargs="*", default=[])
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--out-dir", default="~/rollout_figs")
    p.add_argument("--title", default="", help="keep it short; the detail belongs in the prose")
    args = p.parse_args()

    if args.stats:
        results = json.loads(pathlib.Path(args.stats).expanduser().read_text())
    elif args.repo_id:
        from misc.rollout_stats import dataset_stats

        results = [dataset_stats(n, args.root) for n in args.repo_id]
    else:
        raise SystemExit("pass --stats <file> or --repo-id <names>")

    for f in make_all(results, pathlib.Path(args.out_dir).expanduser(), args.title):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
