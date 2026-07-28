"""Plot latent-probe success against the proprio-only baseline.

The probe is a flow-matching action head trained on the STOP-GRADIENT RL token (plus proprio, which
the downstream critic also receives). Its rollout success rate answers "how much of the policy is
recoverable from the frozen latent alone" - but only once you know how much is recoverable without
the latent at all. That is what the proprio-only baseline (train_proprio_baseline.py) provides, and
it is drawn here as the floor every probe curve should be read against.

Evals use `rlt_probe_eval_trials` rollouts, so the numbers are coarse; 95% Wilson intervals are shown
rather than left implicit, because at 20 trials a 10-point gap is one or two episodes.

Usage:
    uv run examples/robocasa/plot_probe_success.py --out probe_success.png
"""

import argparse
import pathlib
import re

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

_EVAL = re.compile(r"probe eval @ (\d+): vla=(\d+)% probe=(\d+)%")
_BASE = re.compile(r"PROPRIO-ONLY @ (\d+): success (\d+)%")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval - behaves at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def parse(path: pathlib.Path, pattern: re.Pattern, n_groups: int):
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    return [tuple(int(g) for g in m.groups()) for m in pattern.finditer(text)][:None] if n_groups else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="rlt5", help="Sweep tag: reads train_<suffix>_<variant>.slurm.log.")
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["recon", "reconprog", "noprop_reconprog", "pardec_reconprog"],
        help="Variant tags, in legend order.",
    )
    ap.add_argument("--baseline-log", type=pathlib.Path, default=pathlib.Path(".scratch/proprio.log"))
    ap.add_argument("--trials", type=int, default=20, help="Rollouts per eval (for the confidence interval).")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("probe_success.png"))
    args = ap.parse_args()

    runs = {}
    for v in args.variants:
        pts = parse(pathlib.Path(f"train_{args.suffix}_{v}.slurm.log"), _EVAL, 3)
        if pts:
            runs[v] = np.array(pts, dtype=float)  # [n, (step, vla%, probe%)]
    base = np.array(parse(args.baseline_log, _BASE, 2), dtype=float)  # [n, (step, success%)]
    if not runs:
        raise SystemExit(f"no eval lines found for suffix {args.suffix!r}")

    cmap = plt.get_cmap("tab10")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=140, gridspec_kw={"width_ratios": [1.35, 1]})

    # --- left: probe success over training, with the proprio-only floor -------------------------
    for j, (v, a) in enumerate(runs.items()):
        step, probe = a[:, 0], a[:, 2] / 100
        lo, hi = zip(*[wilson(int(round(p * args.trials)), args.trials) for p in probe], strict=True)
        axL.fill_between(step, lo, hi, color=cmap(j), alpha=0.12, linewidth=0)
        axL.plot(step, probe, "-o", color=cmap(j), lw=2, ms=5, label=v)
    if len(base):
        axL.plot(base[:, 0], base[:, 1] / 100, "--s", color="0.35", lw=2, ms=5, label="proprio-only (no token)")
        axL.fill_between(
            [0, max(a[:, 0].max() for a in runs.values())],
            0,
            max(base[:, 1].max() / 100, 0.005),
            color="0.6",
            alpha=0.25,
            linewidth=0,
        )
    axL.set_xlabel("training step")
    axL.set_ylabel("rollout success rate")
    axL.set_title(
        f"Latent probe: policy recovered from the frozen RL token\n"
        f"(shaded = 95% Wilson interval, {args.trials} trials/eval)",
        fontsize=10,
    )
    axL.set_ylim(-0.02, 1.0)
    axL.grid(visible=True, lw=0.5, alpha=0.4)
    axL.legend(fontsize=9, loc="upper left")

    # --- right: latest step, probe vs the VLA it is distilled from -------------------------------
    names, probes, vlas = [], [], []
    for v, a in runs.items():
        names.append(v)
        probes.append(a[-1, 2] / 100)
        vlas.append(a[-1, 1] / 100)
    y = np.arange(len(names))
    axR.barh(y + 0.19, vlas, height=0.34, color="0.75", label="full VLA")
    err = np.array(
        [
            [p - wilson(int(round(p * args.trials)), args.trials)[0] for p in probes],
            [wilson(int(round(p * args.trials)), args.trials)[1] - p for p in probes],
        ]
    )
    axR.barh(
        y - 0.19,
        probes,
        height=0.34,
        color=[cmap(j) for j in range(len(names))],
        xerr=err,
        error_kw={"ecolor": "0.3", "lw": 1.2, "capsize": 3},
        label="latent probe",
    )
    if len(base):
        axR.axvline(base[-1, 1] / 100, color="0.35", ls="--", lw=2, label="proprio-only")
    axR.set_yticks(y, names, fontsize=9)
    axR.invert_yaxis()
    axR.set_xlabel("rollout success rate")
    step_txt = ", ".join(f"{v}@{int(a[-1, 0]):,}" for v, a in runs.items())
    axR.set_title(f"Latest eval\n{step_txt}", fontsize=9)
    axR.set_xlim(0, 1.0)
    axR.grid(visible=True, axis="x", lw=0.5, alpha=0.4)
    axR.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
