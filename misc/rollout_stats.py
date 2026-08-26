"""What a deploy run actually did, in numbers.

The renderer shows one episode; this summarizes whole runs and puts several side by side, which is
how a critic arm gets compared to another (fixed vs g5 vs tau9) without watching forty videos.

Every number is recomputed from the recording, never carried over from a log. Where a value varies
per episode, the aggregate is an episode-level mean with a 95% t-CI -- the same convention the
reports use, because a mean over pooled FRAMES silently weights long episodes more and its spread
describes frames rather than runs.

Usage:
    misc/yam-misc stats --root ~/lerobot_data --repo-id run_a run_b
    misc/yam-misc stats --root ~/lerobot_data --all --json out.json
"""

import argparse
import json
import math
import pathlib

import numpy as np

#: Two-sided 95% critical values by n (df = n-1), matching slurm/make_master_report.py.
TCRIT = {2: 12.7, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 10: 2.26, 15: 2.14, 16: 2.13}


def t_crit(n: int) -> float:
    """95% two-sided t for n samples. Interpolation is not worth it; the table is conservative."""
    if n in TCRIT:
        return TCRIT[n]
    return 2.1 if n < 30 else 1.96


def mean_ci(values) -> dict:
    """Episode-level mean with a 95% t-CI. `ci` is None with fewer than 2 episodes -- one run has
    no spread, and printing 0 would read as agreement rather than absence."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "ci": None}
    if v.size == 1:
        return {"n": 1, "mean": float(v[0]), "ci": None}
    se = float(v.std(ddof=1) / math.sqrt(v.size))
    return {"n": int(v.size), "mean": float(v.mean()), "ci": t_crit(v.size) * se}


def chunk_starts(chunk_index: np.ndarray) -> list:
    """Frame indices where a new reply began, from `policy.chunk_index`.

    Same definition the renderer draws with: a boundary is where the recorded chunk id CHANGES.
    A column that never changes carries no information (a run recorded while provenance was written
    as a constant), and is reported as no boundaries rather than one episode-long chunk.
    """
    if chunk_index is None or chunk_index.size == 0:
        return []
    flat = np.asarray(chunk_index, dtype=float).reshape(len(chunk_index), -1)[:, 0]
    if np.all(flat == flat[0]):
        return []
    return [0, *(np.nonzero(np.diff(flat) != 0)[0] + 1).tolist()]


def boundary_jumps(actions: np.ndarray, starts: list) -> tuple:
    """(jump at each replan boundary, jump at each within-chunk step), in the action's own units.

    A chunk boundary splices two independently sampled trajectories together, so the joint command
    can step discontinuously there in a way it never does inside a chunk. Comparing the two
    distributions is the measurement: a boundary jump that looks like an ordinary step means the
    splice is invisible to the arm, and one that does not is what prefix guidance (RTC) removes.
    """
    if actions is None or len(actions) < 2:
        return np.empty(0), np.empty(0)
    step = np.abs(np.diff(np.asarray(actions, dtype=float), axis=0)).max(axis=1)  # [T-1]
    at_boundary = np.zeros(step.shape[0], dtype=bool)
    for s in starts:
        if 0 < s <= step.shape[0]:
            at_boundary[s - 1] = True  # the step INTO the first frame of a new reply
    return step[at_boundary], step[~at_boundary]


def episode_stats(reader, episode: int) -> dict:
    """Everything measurable about one episode. Missing columns give None, never a guess."""
    n = reader.episode_length(episode)
    fps = max(int(reader.fps or 30), 1)
    out = {"episode": episode, "frames": n, "seconds": n / fps}

    starts = chunk_starts(reader.column(episode, "policy.chunk_index"))
    if starts:
        lengths = np.diff([*starts, n])
        out |= {
            "replans": len(starts),
            "chunk_mean": float(lengths.mean()),
            "chunk_median": float(np.median(lengths)),
            "chunk_min": int(lengths.min()),
            "chunk_max": int(lengths.max()),
            "replans_per_s": len(starts) / max(out["seconds"], 1e-9),
            "chunk_hist": {int(k): int(v) for k, v in zip(*np.unique(lengths, return_counts=True), strict=False)},
        }

    for key, name in (("policy.infer_ms", "infer_ms"), ("policy.delay_ticks", "delay_ticks")):
        col = reader.column(episode, key)
        if col is not None and col.size:
            flat = col.reshape(col.shape[0], -1)[:, 0]
            flat = flat[np.isfinite(flat) & (flat > 0)]
            if flat.size:
                out[f"{name}_p50"] = float(np.median(flat))
                out[f"{name}_p95"] = float(np.percentile(flat, 95))

    scores = reader.column(episode, "critic_scores")
    if scores is not None and scores.ndim == 2 and scores.shape[1] > 1 and starts:
        per_replan = scores[starts]  # the decision is made once per reply; frames repeat it
        out["candidates"] = int(scores.shape[1])
        out["critic_spread"] = float(np.mean(per_replan.max(axis=1) - per_replan.min(axis=1)))
        out["critic_advantage"] = float(np.mean(np.sort(per_replan, axis=1)[:, -1] - np.sort(per_replan, axis=1)[:, -2]))
    choice = reader.column(episode, "critic_choice")
    if choice is not None and choice.size and starts:
        picked = choice.reshape(choice.shape[0], -1)[starts, 0].astype(int)
        out["choice_hist"] = {int(k): int(v) for k, v in zip(*np.unique(picked, return_counts=True), strict=False)}
        out["chose_first_frac"] = float(np.mean(picked == 0))

    actions = reader.column(episode, "action")
    if actions is not None and actions.ndim == 2:
        at_b, within = boundary_jumps(actions, starts)
        if at_b.size:
            out["boundary_jump_p50"] = float(np.median(at_b))
            out["boundary_jump_p95"] = float(np.percentile(at_b, 95))
            out["boundary_jump_over_0p3"] = float(np.mean(at_b > 0.3))
        if within.size:
            out["within_jump_p50"] = float(np.median(within))
            out["within_jump_p95"] = float(np.percentile(within, 95))
    return out


def dataset_stats(repo_id: str, root: str, episodes: "list | None" = None) -> dict:
    """Per-episode rows plus episode-level aggregates with 95% t-CIs."""
    from misc.dataset_reader import DatasetReader

    reader = DatasetReader(repo_id, root)
    reader.load()
    eps = list(range(reader.num_episodes)) if episodes is None else episodes
    rows = [episode_stats(reader, e) for e in eps]

    agg = {}
    for key in (
        "frames",
        "seconds",
        "replans",
        "chunk_mean",
        "chunk_median",
        "replans_per_s",
        "infer_ms_p50",
        "infer_ms_p95",
        "delay_ticks_p50",
        "critic_spread",
        "critic_advantage",
        "chose_first_frac",
        "boundary_jump_p50",
        "boundary_jump_p95",
        "boundary_jump_over_0p3",
        "within_jump_p50",
        # The boundary number means nothing alone -- a 0.4 rad step at a splice is only a splice
        # artefact if ordinary steps are much smaller. Aggregating one and not the other left the
        # comparison column silently empty.
        "within_jump_p95",
    ):
        got = [r.get(key) for r in rows if r.get(key) is not None]
        if got:
            agg[key] = mean_ci(got)

    hist: dict = {}
    for r in rows:
        for k, v in (r.get("chunk_hist") or {}).items():
            hist[int(k)] = hist.get(int(k), 0) + int(v)
    if hist:
        agg["chunk_hist"] = dict(sorted(hist.items()))
    lo = [r["chunk_min"] for r in rows if "chunk_min" in r]
    hi = [r["chunk_max"] for r in rows if "chunk_max" in r]
    if lo:
        agg["chunk_min"], agg["chunk_max"] = min(lo), max(hi)
    return {"repo_id": repo_id, "episodes": len(rows), "per_episode": rows, "aggregate": agg}


def _fmt(m: "dict | None", digits: int = 1) -> str:
    if not m or m.get("mean") is None:
        return "—"
    if m.get("ci") is None:
        return f"{m['mean']:.{digits}f}"
    return f"{m['mean']:.{digits}f} ± {m['ci']:.{digits}f}"


def format_table(results: list) -> str:
    """One row per dataset. Only columns that at least one dataset has are printed, so a table of
    teleop recordings does not carry empty critic columns."""
    cols = [
        ("dataset", lambda r: r["repo_id"].rstrip("/").split("/")[-1], 34),
        ("eps", lambda r: str(r["episodes"]), 4),
        ("sec", lambda r: _fmt(r["aggregate"].get("seconds"), 0), 12),
        ("replans", lambda r: _fmt(r["aggregate"].get("replans"), 0), 13),
        ("chunk len", lambda r: _fmt(r["aggregate"].get("chunk_mean"), 1), 14),
        ("range", lambda r: f"{r['aggregate'].get('chunk_min', '—')}–{r['aggregate'].get('chunk_max', '—')}", 8),
        ("replan/s", lambda r: _fmt(r["aggregate"].get("replans_per_s"), 2), 12),
        ("infer p50", lambda r: _fmt(r["aggregate"].get("infer_ms_p50"), 0), 11),
        ("spread", lambda r: _fmt(r["aggregate"].get("critic_spread"), 2), 13),
        ("adv", lambda r: _fmt(r["aggregate"].get("critic_advantage"), 2), 12),
        ("pick#0", lambda r: _fmt(r["aggregate"].get("chose_first_frac"), 2), 11),
        ("jump@bnd", lambda r: _fmt(r["aggregate"].get("boundary_jump_p95"), 3), 13),
        ("jump@in", lambda r: _fmt(r["aggregate"].get("within_jump_p95"), 3), 13),
    ]
    keep = [c for c in cols if any(c[1](r) != "—" for r in results)]
    lines = ["  ".join(h.ljust(w) for h, _, w in keep), "  ".join("-" * w for _, _, w in keep)]
    for r in results:
        lines.append("  ".join(f(r).ljust(w) for _, f, w in keep))
    lines.append("")
    lines.append("mean ± 95% t-CI over EPISODES (— - not recorded, or one episode so no spread).")
    lines.append("spread = best-worst candidate value per replan; adv = best minus runner-up.")
    lines.append("jump@bnd / jump@in = 95th pct of max joint step across a replan boundary vs inside a chunk.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", nargs="*", default=[], help="one or more datasets, compared side by side")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--all", action="store_true", help="every dataset under --root")
    p.add_argument("--episodes", default="all", help='"all", "3", "0-9", "0,3,5-7"')
    p.add_argument("--json", dest="json_out", default=None, help="also write the full per-episode numbers here")
    args = p.parse_args()

    from misc.dataset_reader import DatasetReader, list_datasets
    from misc.render_bulk import parse_episodes

    names = list(args.repo_id)
    if args.all:
        names = list_datasets(args.root)
    if not names:
        raise SystemExit("nothing to summarize -- pass --repo-id or --all")

    results = []
    for name in names:
        try:
            eps = None
            if args.episodes != "all":
                r = DatasetReader(name, args.root)
                r.load()
                eps = parse_episodes(args.episodes, r.num_episodes)
            results.append(dataset_stats(name, args.root, eps))
        except (Exception, SystemExit) as e:  # noqa: BLE001 - one unreadable dataset must not end the sweep
            print(f"{name}: skipped -- {e}")
    if not results:
        raise SystemExit("no dataset could be read")

    print(format_table(results))
    if args.json_out:
        out = pathlib.Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
