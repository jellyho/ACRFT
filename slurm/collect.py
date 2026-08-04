"""Summarise a finished critic sweep into one table (and a CSV).

Pulls two things per variant: the offline diagnostics eval_rlt_critic.py wrote to `diag.json`, and
the last critic-vs-VLA rollout line from the job's log. The columns are the ones that decide whether
a variant is worth anything — every diagnostic here is computed WITHIN a state, because a critic that
has collapsed to Q(z, a) = V(z) scores well on TD loss and on Q-vs-return while ranking candidates at
chance.

    uv run slurm/collect.py critic_abl
    uv run slurm/collect.py critic_abl --csv /tmp/abl.csv
"""

import argparse
import csv
import json
import os
import pathlib
import re
import sys

# (diag.json key, header, format). action_sensitivity and the two ranking accuracies are the
# necessary conditions; the rest say how a failure looks.
COLS = [
    ("action_sensitivity", "act_sens", "{:.4f}"),
    ("ranking_accuracy_demo_vs_candidate", "rank_cand", "{:.3f}"),
    ("ranking_accuracy_demo_vs_other", "rank_other", "{:.3f}"),
    ("spearman_q_vs_closeness_to_demo", "rho_close", "{:+.3f}"),
    ("within_state_q_range", "ws_range", "{:.3f}"),
    ("prefix_argmax_entropy", "pfx_H", "{:.3f}"),
    ("bias_growth_last_double", "bias_grow", "{:+.3f}"),
    ("argmax_gap_train_minus_heldout", "held_gap", "{:+.3f}"),
    ("q_demo_minus_mc_mean", "q-mc", "{:+.3f}"),
]
ROLLOUT_RE = re.compile(r"\[rollout @ (\d+)\]\s+critic\s+([\d.]+)%\s+vla\s+([\d.]+)%")
RUN_RE = re.compile(r"^sweep/run\s*:\s*(\S+)/(\S+)\s*$", re.M)


def rollout_by_run(log_dir: pathlib.Path, sweep: str) -> dict[str, tuple[str, str, str]]:
    """Map run name -> (step, critic%, vla%) from the last rollout line in each matching job log."""
    out: dict[str, tuple[str, str, str]] = {}
    for log in sorted(log_dir.glob(f"{sweep}_*.out")):
        text = log.read_text(errors="replace")
        m = RUN_RE.search(text)
        if not m or m.group(1) != sweep:
            continue
        hits = ROLLOUT_RE.findall(text)
        if hits:
            out[m.group(2)] = hits[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="sweep name (the SWEEP= passed to sweep.sh)")
    ap.add_argument("--runs", type=pathlib.Path, default=None, help="default: $CRITIC_RUNS/<sweep>")
    ap.add_argument("--logs", type=pathlib.Path, default=None, help="default: $SLURM_LOGS")
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    args = ap.parse_args()

    cache = os.environ.get("CACHE_DIR", "/lustre/jellyho/acrft")
    runs = args.runs or pathlib.Path(os.environ.get("CRITIC_RUNS", f"{cache}/critic_runs")) / args.sweep
    logs = args.logs or pathlib.Path(os.environ.get("SLURM_LOGS", f"{cache}/logs"))
    if not runs.is_dir():
        sys.exit(f"no sweep dir at {runs}")

    roll = rollout_by_run(logs, args.sweep) if logs.is_dir() else {}
    headers = ["run", "critic%", "vla%", *(h for _, h, _ in COLS)]
    rows = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        diag = d / "diag.json"
        if not diag.exists():
            rows.append([d.name, *["-"] * (len(headers) - 1)])
            continue
        res = json.loads(diag.read_text())
        r = roll.get(d.name)
        row = [d.name, f"{float(r[1]):.0f}" if r else "-", f"{float(r[2]):.0f}" if r else "-"]
        row += [fmt.format(res[k]) if k in res else "-" for k, _, fmt in COLS]
        rows.append(row)

    if not rows:
        sys.exit(f"{runs} has no run directories")
    width = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, width, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, width, strict=True)))
    missing = sum(r[3] == "-" for r in rows)
    if missing:
        print(f"\n{missing} of {len(rows)} runs have no diag.json yet (still queued/running, or failed)")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
