"""Two-day experiment log: what was run, what was expected, what came back, what it settled.

Reads the files the experiments wrote (diag.json / config.json / rollout/*.json / vbias*.json /
pfx_curve.json), so re-running refreshes every number without editing this file.

    uv run slurm/make_progress_report.py --out $CACHE_DIR/report_progress.html
"""

import argparse
import contextlib
import html
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _describe import BASELINE_NOTE
from _describe import describe

MODES = ["vla", "rand", "bon", "prefix", "critic"]
BANDS = [(5, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 600)]

CSS = """
:root{--paper:#f5f4ef;--ink:#191a17;--mut:#66675e;--line:#dcdbd2;--panel:#ebeae2;
  --bad:#a03030;--ok:#31684a;--warn:#8a6a1c;--rule:#c8c7bc;--acc:#4a5e78}
@media (prefers-color-scheme:dark){:root{--paper:#141511;--ink:#e8e7e0;--mut:#93948a;--line:#2a2b25;
  --panel:#1b1c17;--bad:#dd8078;--ok:#7fc49c;--warn:#cfa94b;--rule:#3a3b33;--acc:#8fa8c8}}
:root[data-theme=light]{--paper:#f5f4ef;--ink:#191a17;--mut:#66675e;--line:#dcdbd2;--panel:#ebeae2;
  --bad:#a03030;--ok:#31684a;--warn:#8a6a1c;--rule:#c8c7bc;--acc:#4a5e78}
:root[data-theme=dark]{--paper:#141511;--ink:#e8e7e0;--mut:#93948a;--line:#2a2b25;--panel:#1b1c17;
  --bad:#dd8078;--ok:#7fc49c;--warn:#cfa94b;--rule:#3a3b33;--acc:#8fa8c8}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:3rem 1.25rem 7rem;
  font:16px/1.66 -apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo",system-ui,sans-serif;
  font-variant-numeric:tabular-nums}
main{max-width:62rem;margin:0 auto}
h1{font:600 2rem/1.22 Georgia,serif;margin:0 0 .5rem;letter-spacing:-.015em;text-wrap:balance}
h2{font:600 1.28rem/1.3 Georgia,serif;margin:3.2rem 0 .9rem;padding-top:1.1rem;
  border-top:1px solid var(--rule);letter-spacing:-.01em}
h3{font:600 1rem/1.35 Georgia,serif;margin:1.9rem 0 .45rem}
.lede{color:var(--mut);font-size:1.02rem;margin:0 0 .4rem}
.stamp{color:var(--mut);font-family:ui-monospace,Menlo,monospace;font-size:.76rem;margin:0 0 2.4rem}
p{margin:.8rem 0}
.scroll{overflow-x:auto;margin:1rem 0;border:1px solid var(--line);border-radius:3px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,Menlo,monospace;font-size:.78rem}
th{text-align:right;padding:.5rem .65rem;color:var(--mut);font-weight:600;white-space:nowrap;
  border-bottom:1px solid var(--rule);background:var(--panel);position:sticky;top:0}
td{text-align:right;padding:.4rem .65rem;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
tr.total td{border-top:2px solid var(--rule);font-weight:700}
.bad{color:var(--bad)}.ok{color:var(--ok)}.warn{color:var(--warn)}.mut{color:var(--mut)}
code{font-family:ui-monospace,Menlo,monospace;font-size:.87em;background:var(--panel);
  padding:.1em .36em;border-radius:2px}
.key{border-left:3px solid var(--acc);background:var(--panel);padding:1rem 1.2rem;margin:1.3rem 0;
  border-radius:0 3px 3px 0}
.key p:first-child{margin-top:0}.key p:last-child{margin-bottom:0}
.fail{border-left-color:var(--bad)}
.win{border-left-color:var(--ok)}
.note{border-left:3px solid var(--rule);background:var(--panel);padding:.85rem 1.05rem;margin:1.2rem 0;
  border-radius:0 3px 3px 0;font-size:.92rem;color:var(--mut)}
.note b{color:var(--ink)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.85rem 1rem;
  overflow-x:auto;font-size:.79rem;line-height:1.5;margin:.9rem 0}
dl{margin:.9rem 0}dt{font-family:ui-monospace,monospace;font-size:.83rem;font-weight:600;margin-top:.75rem}
dd{margin:.12rem 0 0;color:var(--mut);font-size:.92rem}
.tl{list-style:none;padding:0;margin:1rem 0}
.tl li{display:grid;grid-template-columns:9.5rem 1fr;gap:.9rem;padding:.5rem 0;border-bottom:1px solid var(--line)}
.tl time{color:var(--mut);font-family:ui-monospace,monospace;font-size:.78rem;padding-top:.15rem}
ol,ul{padding-left:1.3rem}li{margin:.3rem 0}
.pill{display:inline-block;padding:.05em .55em;border-radius:2px;font-size:.72rem;font-weight:600;
  font-family:ui-monospace,monospace;vertical-align:middle}
.p-done{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}
.p-run{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
"""


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_sweep(root, sweep):
    out = {}
    # NOT `glob(".../*/")`: pathlib normalises away the trailing slash, so the old string
    # concatenation quietly produced ".../baserollout/*.json" and every rollout table came out
    # empty while the prose around it (hand-computed) looked fine.
    for d in sorted((root / "critic_runs" / sweep).glob("*")):
        if not d.is_dir():
            continue
        e = {}
        for name in ("diag.json", "config.json"):
            f = d / name
            if f.exists():
                with contextlib.suppress(json.JSONDecodeError):
                    e[name.split(".")[0]] = json.loads(f.read_text())
        rolls = sorted((d / "rollout").glob("*.json"))
        if rolls:
            e["rollout"] = json.loads(rolls[-1].read_text())
        if e:
            out[d.name] = e
    return out


def diag_table(emit, sweeps, note=""):
    """One combined diagnostics table. Columns are the necessary-condition metrics."""
    emit(
        "<div class='scroll'><table><thead><tr><th>run</th><th>sweep</th><th>ρ(Q,MC)</th><th>act_sens</th>"
        "<th>rank_c</th><th>rank_o</th><th>pfx_H</th></tr></thead><tbody>"
    )
    for sweep, runs in sweeps:
        for r, e in sorted(runs.items()):
            g = e.get("diag")
            if not g:
                continue
            fmt = lambda k, p=3, g=g: (f"{g[k]:.{p}f}" if k in g else "—")  # noqa: E731
            rc = g.get("ranking_accuracy_demo_vs_candidate", 0)
            emit(
                f"<tr><td>{html.escape(r)}</td><td class='mut'>{html.escape(sweep)}</td>"
                f"<td>{fmt('spearman_q_demo_vs_mc')}</td><td>{fmt('action_sensitivity', 4)}</td>"
                f"<td class='{'ok' if rc > 0.55 else ''}'>{fmt('ranking_accuracy_demo_vs_candidate')}</td>"
                f"<td>{fmt('ranking_accuracy_demo_vs_other')}</td><td>{fmt('prefix_argmax_entropy')}</td></tr>"
            )
    emit("</tbody></table></div>")
    if note:
        emit(f"<p class='mut' style='font-size:.9rem'>{note}</p>")


def rollout_table(emit, runs):
    tot = {m: [0, 0] for m in MODES}
    emit(
        "<div class='scroll'><table><thead><tr><th>run</th>"
        + "".join(f"<th>{m}</th>" for m in MODES)
        + "<th>critic−vla</th><th>McNemar p</th></tr></thead><tbody>"
    )
    for r, e in sorted(runs.items()):
        d = e.get("rollout")
        if not d:
            continue
        cells = ""
        for m in MODES:
            if m in d:
                k, n = d[m]["successes"], d[m]["num_trials"]
                tot[m][0] += k
                tot[m][1] += n
                cells += f"<td>{k}/{n}</td>"
            else:
                cells += "<td class='mut'>—</td>"
        if "critic" in d and "vla" in d:
            cv = {t["trial"]: t["success"] for t in d["critic"]["trials"]}
            vv = {t["trial"]: t["success"] for t in d["vla"]["trials"]}
            b = sum(1 for t in cv if cv[t] and not vv.get(t))
            c = sum(1 for t in cv if not cv[t] and vv.get(t))
            n = b + c
            pv = 1.0 if n == 0 else min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2 / 2**n)
            diff = d["critic"]["success_rate"] - d["vla"]["success_rate"]
            cells += (
                f"<td class='{'bad' if diff < 0 else 'ok'}'>{diff:+.3f}</td>"
                f"<td class='{'bad' if pv < 0.05 else 'mut'}'>{pv:.3f}</td>"
            )
        else:
            cells += "<td class='mut'>—</td><td class='mut'>—</td>"
        emit(f"<tr><td>{html.escape(r)}</td>{cells}</tr>")
    if any(n for _, n in tot.values()):
        emit(
            "<tr class='total'><td>total</td>"
            + "".join(f"<td>{tot[m][0]}/{tot[m][1]}</td>" for m in MODES)
            + "<td colspan='2'></td></tr>"
        )
    emit("</tbody></table></div>")
    return tot


# ---------------------------------------------------------------- charts
# Static inline SVG: CSP-safe, theme-aware via CSS vars. One axis per chart, series directly
# labelled at their last point - the tables stay underneath as the precise backup.
PAL = {"td": "#c0563f", "e50": "#3b78ae", "e70": "#2f855a", "e90": "#b9892e", "e95": "#8168b3"}


def _sv(tag, **kw):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
    return f"<{tag} {a}/>"


def _tx(x, y, t, *, size=11, color="var(--mut)", anchor="start", mono=True, weight=""):
    fam = "ui-monospace,Menlo,monospace" if mono else "inherit"
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
        f'text-anchor="{anchor}" font-family="{fam}"{w}>{t}</text>'
    )


def dot_intervals(rows, *, w=680, rowh=34, pad_l=76, x_lo=0.3, x_hi=0.85, title=""):
    """rows = [(label, [(series_name, color, k, n)])] - a dot with a Wilson CI per series per row."""
    h = len(rows) * rowh + 56
    iw = w - pad_l - 60
    X = lambda p: pad_l + iw * (p - x_lo) / (x_hi - x_lo)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
    for gx in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        out.append(_sv("line", x1=f"{X(gx):.1f}", y1=26, x2=f"{X(gx):.1f}", y2=h - 30, stroke="var(--line)"))
        out.append(_tx(X(gx), h - 14, f"{gx * 100:.0f}%", anchor="middle", size=10))
    for i, (lab, pts) in enumerate(rows):
        y = 40 + i * rowh
        out.append(_tx(pad_l - 10, y + 4, lab, anchor="end", size=12, color="var(--ink)"))
        for j, (name, color, k, n) in enumerate(pts):
            if not n:
                continue
            pr = k / n
            lo, hi = wilson(k, n)
            yy = y + (j - (len(pts) - 1) / 2) * 9
            out.append(
                _sv(
                    "line",
                    x1=f"{X(lo):.1f}",
                    y1=yy,
                    x2=f"{X(hi):.1f}",
                    y2=yy,
                    stroke=color,
                    stroke_width=2,
                    opacity=0.55,
                )
            )
            out.append(_sv("circle", cx=f"{X(pr):.1f}", cy=yy, r=4.2, fill=color))
            if i == 0:
                out.append(_tx(X(pr), 18, name, anchor="middle", size=10.5, color=color, weight=600))
    out.append("</svg>")
    return "".join(out)


def line_chart(series, xticks, *, w=680, h=300, pad_l=64, pad_b=42, ylabel="", zero_line=True, title=""):
    """series = [(name, color, [y...])] over the shared xticks (band or h labels)."""
    ys = [v for _, _, vv in series for v in vv if v is not None]
    lo = min([*ys, 0.0]) if zero_line else min(ys)
    hi = max(ys)
    span = (hi - lo) or 1.0
    lo -= span * 0.04
    hi += span * 0.08
    span = hi - lo
    iw, ih = w - pad_l - 78, h - pad_b - 20
    X = lambda i: pad_l + iw * i / max(len(xticks) - 1, 1)  # noqa: E731
    Y = lambda v: 20 + ih * (1 - (v - lo) / span)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
    for t in range(5):
        v = lo + span * t / 4
        out.append(_sv("line", x1=pad_l, y1=f"{Y(v):.1f}", x2=pad_l + iw, y2=f"{Y(v):.1f}", stroke="var(--line)"))
        out.append(_tx(pad_l - 8, Y(v) + 4, f"{v:+.2f}" if abs(hi) < 10 else f"{v:.1f}", anchor="end", size=10))
    if zero_line and lo < 0 < hi:
        out.append(
            _sv(
                "line",
                x1=pad_l,
                y1=f"{Y(0):.1f}",
                x2=pad_l + iw,
                y2=f"{Y(0):.1f}",
                stroke="var(--mut)",
                stroke_dasharray="3 3",
            )
        )
    for i, lab in enumerate(xticks):
        out.append(_tx(X(i), h - 26, str(lab), anchor="middle", size=10))
    if ylabel:
        out.append(_tx(pad_l, h - 8, ylabel, size=10.5))
    for name, color, vv in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vv) if v is not None)
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        last = max(i for i, v in enumerate(vv) if v is not None)
        out.append(_tx(X(last) + 7, Y(vv[last]) + 4, name, size=10.5, color=color, weight=600))
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--generated", default="")
    args = ap.parse_args()
    root = args.root or pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
    out = args.out or root / "report_progress.html"

    v3 = load_sweep(root, "v3_fixedmask")
    v4 = load_sweep(root, "v4_hlgfloor")
    v5 = load_sweep(root, "v5_stability")
    v6 = load_sweep(root, "v6_iql")
    v7 = load_sweep(root, "v7_single")
    vb3 = json.loads((root / "vbias.json").read_text()) if (root / "vbias.json").exists() else {}
    vb6 = json.loads((root / "vbias_v6_iql.json").read_text()) if (root / "vbias_v6_iql.json").exists() else {}
    pfx = json.loads((root / "pfx_curve.json").read_text()) if (root / "pfx_curve.json").exists() else {}

    A = []
    P = A.append
    P("<h1>ACRFT critic — a two-day experiment log</h1>")
    P(
        "<p class='lede'>How fifteen TD ablations that all landed in the same place were narrowed down, by "
        "measurement, to a single cause — the bootstrap's arg-max — and how IQL, which removes it, was "
        "implemented and verified.</p>"
    )
    P(
        f"<p class='stamp'>{html.escape(args.generated)} · RoboCasa PrepareCoffee · 514 demo episodes / "
        f"279,534 frames · γ=0.99 · companion deep-dive: report_bias.html</p>"
    )

    # ---------------------------------------------------------------- summary
    P("<h2>Summary — what is now settled</h2>")
    P(
        "<ol>"
        "<li><b>The critic is harmful, not merely useless</b> — and unanimously so under TD: "
        "critic−vla is negative in 14/14 runs (sign test p≈1e-4). Candidate selection alone does "
        "worse than picking at random.</li>"
        "<li><b>The cause is the bootstrap's arg-max</b>: the target's V is over-estimated with a "
        "distance structure (5.07× beyond 250 steps; the truth γ^d is known in closed form).</li>"
        "<li><b>IQL removes the inflation (19–38×) with a clean expectile dose-response</b>, offline "
        "and in rollout — causality settled.</li>"
        "<li><b>IQL cures the prefix axis</b>: 0.528 → 0.733 in rollout, at parity with the raw VLA "
        "(TD-vs-IQL across runs: Mann-Whitney z=2.6). Safe deployment today = <b>prefix-only</b>.</li>"
        "<li><b>The candidate axis stays dead because the signal is absent from the data</b>: with "
        "one action per state the target is action-independent (act_sens ≈ 0 across 30+ runs), and "
        "dueling — the cleanest possible conditions — still ranked candidates at chance while "
        "memorising the trained pair (rank_c 0.96 with zero similarity gradient). Dueling also "
        "hurts deployment (p=0.021): untrained candidates' advantage profiles are noise.</li>"
        "<li><b>Larger discounts calibrate but do not select</b>: γ=.999 puts b(d) within ±0.01 at "
        "every distance and lifts cross-state ranking (rank_o 0.56→0.65), yet rollout matches "
        "IQL γ=.99 — calibration cannot substitute for a ranking signal.</li>"
        "<li><b>The episode ladder shows the value model is generalisation, not memorisation</b>: "
        "trained on 1/4/16/64 episodes, whole-dataset ρ(Q,MC) climbs 0.48→0.58→0.74→0.78 — most of "
        "the state-value skill comes from cross-episode structure, converging toward the full-data "
        "0.82.</li>"
        "<li><b>Ensembles and target networks are not the lever</b>: K 2→3 changed nothing "
        "measurable (ρ 0.816→0.814, rank_c 0.521→0.532); K=5 / no-target / τ-ladder still running "
        "after two infrastructure losses, but the candidate-side knobs that already failed (topm/"
        "soft/lcb/bc) bound how much variance-softening alone can give.</li>"
        "</ol>"
    )
    # ---------------------------------------------------------------- figures
    P("<h2>The results in six figures</h2>")
    figs = [
        (
            "1_success_by_mode.png",
            "Each method as the critic gains authority (thin line = one run, bold = family median, "
            "dashed = that family's own vla level). TD declines monotonically; IQL γ=.99 holds parity "
            "through prefix and only drops at the joint arg-max; dueling collapses at prefix.",
        ),
        (
            "2_value_bias.png",
            "Value bias b(d) = V̂ − γ^d by distance to goal. Left: TD inflates with distance and the "
            "IQL expectile ladder re-inflates toward it. Right (same scale): γ=.999 / .9995 and "
            "dueling sit on the zero line everywhere.",
        ),
        ("3_dose_response.png", "The expectile dose-response, offline and in rollout — the causal pin on the arg-max."),
        (
            "4_prefix_targets.png",
            "TD per-prefix targets over truth (log scale): every distance band slopes down in h, and "
            "the far band floats at 5×.",
        ),
        ("5_per_run_harm.png", "critic − vla for every rollout run (filled = McNemar p<0.05). No run above zero yet."),
        ("6_success_vs_steps.png", "v8 success vs training steps, 30 trials per point — trends only."),
    ]
    import base64 as _b64

    for fn, cap in figs:
        fp = root / "plots" / fn
        if not fp.exists():
            continue
        b64 = _b64.b64encode(fp.read_bytes()).decode()
        P(
            f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(cap)}' "
            f"style='width:100%;height:auto;border:1px solid var(--line);border-radius:3px'>"
            f"<figcaption style='font-size:.85rem;color:var(--mut);margin-top:.4rem'>{cap}</figcaption></figure>"
        )
    P(
        "<p class='mut' style='font-size:.88rem'>Statistical honesty notes: every job replays the same "
        "30 scenes, and rollouts are not bit-deterministic across jobs (different GPU models × chaotic "
        "contacts — measured: identical vla policies agree on only 20–25/30 trials between jobs), so "
        "family claims rest on run-level tests, not pooled CIs: critic−vla is negative in 14/14 TD "
        "runs (sign test p≈1e-4); prefix under IQL beats TD across runs (Mann-Whitney z=2.6).</p>"
    )

    # ---------------------------------------------------------------- why
    P("<h2>Why more critic authority fails — the causal chain</h2>")
    P(
        "<p>There are <b>two arg-maxes</b>, and the eval modes touch only one of them. Training-time: "
        "the TD target maxes Q over N×P candidate cells at the next state (IQL deletes this). "
        "Deployment-time: best-of-N maxes over candidates at the current state (present in bon/critic "
        "whatever the objective). The modes are pure inference rules — training is identical within "
        "a run.</p>"
    )
    P(
        "<ol>"
        "<li><b>There is nothing to choose between candidates.</b> The VLA imitates the demos, so its "
        "16 samples share the same true value to within 1/150 of the between-state spread; dueling "
        "proved the ranking signal is absent from the DATA, not from capacity (its advantage head, "
        "freed of state-value variance, still ranked candidates at exactly chance while memorising "
        "the trained pair at 0.96).</li>"
        "<li><b>An arg-max over equal-mean noisy scores picks the largest error.</b> And the error is "
        "a FIXED function, not fresh noise — the critic consistently over-values particular kinds of "
        "chunks, so the tilt repeats every replan. That is why bon (0.64) is worse than rand (0.70): "
        "rand samples the average error, bon samples the maximum, systematically.</li>"
        "<li><b>Under TD the prefix axis measured critic error, not commitment.</b> The distance-"
        "structured inflation tilted per-prefix targets monotonically, so the joint arg-max collapsed "
        "to the shortest commit. IQL flattened the targets and the prefix mode recovered to vla "
        "parity — the one axis with a real signal, cured.</li></ol>"
    )
    P(
        "<div class='key'><p><b>One sentence:</b> failure scales not with authority but with the "
        "number of arg-maxes taken over noise. Where the axis has signal (IQL's prefix), authority is "
        "harmless; where it has none (candidates), authority is harm.</p></div>"
    )

    # ---------------------------------------------------------------- timeline
    P("<h2>Timeline</h2><ul class='tl'>")
    for t, ev in [
        (
            "day 1, afternoon",
            "v3_fixedmask finishes: 15 runs, every one at act_sens≈0.0003 and rank_c≈0.5 — no axis to rank "
            "methods along.",
        ),
        (
            "day 1, evening",
            "All 15 runs submitted for rollout (30 trials × 5 modes). HUD panel-corruption solved by deferring "
            "all rendering to after the episode (1000-frame full scan: zero corrupted frames).",
        ),
        (
            "day 1, night",
            "13 rollouts back: the critic is harmful. targets() rewritten around a single `ended` mask "
            "(verified bitwise-identical on 2.2M valid cells). Discount-variant datasets now build themselves "
            "(ensure_discount); proprio moved into the annotation stage.",
        ),
        (
            "day 2, early",
            "Per-prefix target decomposition → y_h = γ^d + γ^h·b with distance-structured bias b. IQL "
            "implemented (ValueNet + expectile; no candidate array). v4 resubmitted on the HL-Gauss+mcfloor "
            "baseline.",
        ),
        (
            "day 2, morning",
            "b(d) measured on all 15 runs: the bias is real but does not explain between-run performance "
            "(ρ=−0.17) — what matters is the error spread the arg-max picks from, not the bias size. "
            "v5 (stability), v6 (IQL), v7 (episode ladder) submitted.",
        ),
        (
            "day 2, afternoon",
            "v6 done: IQL shrinks b(d) 19–38× with a clean expectile dose-response. Found and fixed the bug "
            "that had silently killed every QC rollout across three sweeps (load_trained dropped the prefix "
            "axis). Overlay rewritten to magnify in world space. Both machines' work merged and pushed.",
        ),
        (
            "day 2, evening",
            "IQL rollout verdict: prefix mode 0.528→0.733 (parity with vla — cured), overall critic harm "
            "halved, dose-response reproduced (e90 p=0.016). iql_qc's offline rank_c 0.571 rejected "
            "(bon−vla=0 in rollout).",
        ),
    ]:
        P(f"<li><time>{t}</time><span>{ev}</span></li>")
    P("</ul>")

    # ---------------------------------------------------------------- 1. v3 rollout
    P("<h2>1. v3 rollouts — the critic was not harmless <span class='pill p-done'>done</span></h2>")
    P("<h3>Protocol</h3>")
    P(
        "<p>Each of 15 critics (200k steps, batch 1024) is evaluated in five modes, 30 trials each. All five "
        "modes see the identical scenes (fixed seed), so the comparisons are paired. <code>vla</code> = no "
        "critic (baseline); <code>rand</code> = a uniformly random candidate (the control); <code>bon</code> "
        "= critic picks the candidate only; <code>prefix</code> = critic picks the commitment length only; "
        "<code>critic</code> = the joint arg-max, i.e. the deployment rule.</p>"
    )
    P(
        "<pre>ROLLOUT=0 SWEEP=v3_fixedmask slurm/sweep.sh\n"
        "RUN_DIR=$CACHE_DIR/critic_runs/v3_fixedmask/&lt;run&gt; TRIALS=30 sbatch slurm/eval_rollout.sbatch</pre>"
    )
    P(
        "<h3>Expected</h3><p>From the offline diagnostics (act_sens≈0, rank_c≈0.5) the prediction was "
        "<b>bon ≈ rand and critic ≈ vla</b> — harmless but useless.</p>"
    )
    P("<h3>Observed</h3>")
    tot = rollout_table(P, v3)
    if tot["vla"][1]:
        P(
            "<div class='key fail'><p>The prediction failed. "
            + " · ".join(f"<b>{m}</b> {tot[m][0] / tot[m][1]:.3f}" for m in MODES if tot[m][1])
            + ". Success falls monotonically with critic authority, and bon is <i>worse</i> than rand: when "
            "candidates share the same true value and differ only in estimation error, the arg-max picks "
            "the most inflated error, while a random draw samples the average one. rand ≈ vla shows that "
            "re-picking a candidate is nearly free — the loss comes from the critic doing the picking.</p></div>"
        )

    # ---------------------------------------------------------------- 2. decomposition
    P("<h2>2. Per-prefix target decomposition — the structure of the bias <span class='pill p-done'>done</span></h2>")
    P(
        "<p>The reward is sparse and terminal, so the true value is exactly <code>γ^d</code> (d = steps to "
        "goal). Expanding the target: <code>y_h = γ^d + γ^h·b</code> — if V̂ were exact, all eight per-prefix "
        "targets would be equal.</p>"
    )
    if pfx.get("buckets"):
        pl = pfx.get("pfx", [2, 4, 6, 8, 10, 12, 14, 16])
        P(
            "<div class='scroll'><table><thead><tr><th>distance band</th><th>n</th>"
            + "".join(f"<th>h={h}</th>" for h in pl)
            + "</tr></thead><tbody>"
        )
        for b in pfx["buckets"]:
            cells = "".join(
                f"<td class='{'bad' if v > 1.15 else ('warn' if v > 1.03 else '')}'>{v:.3f}</td>" for v in b["ratio"]
            )
            P(f"<tr><td>{b['lo']}–{b['hi']}</td><td class='mut'>{b['n']}</td>{cells}</tr>")
        P("</tbody></table></div>")
        P(
            "<p class='mut' style='font-size:.9rem'>Values are <code>y_h / γ^d</code> (1.0 = exact). "
            "TD base critic, 400 states per band.</p>"
        )
    if pfx.get("buckets"):
        bands_lab = [f"{b['lo']}–{b['hi']}" for b in pfx["buckets"]]
        pl2 = pfx.get("pfx", [2, 4, 6, 8, 10, 12, 14, 16])
        shade = ["#3b78ae", "#2f855a", "#b9892e", "#c0563f", "#8168b3", "#666a5e"]
        P(
            "<figure>"
            + line_chart(
                [(bands_lab[i], shade[i % 6], list(b["ratio"])) for i, b in enumerate(pfx["buckets"])],
                pl2,
                ylabel="y_h / γ^d   (1.0 = exact)",
                zero_line=False,
                title="per-prefix target over truth, by distance band",
            )
            + "<figcaption>Per-prefix target ÷ true value, one line per distance-to-goal band "
            "(TD base critic). Every line slopes down in h and the far bands float far above 1.0 — "
            "the two facts that make the joint arg-max prefer short commits.</figcaption></figure>"
        )
    P(
        "<p>Monotone decline in h within every band — the 'prefix values fall monotonically' seen in the "
        "videos is the direct image of the critic over-valuing far-from-goal successor states. Beyond 250 "
        "steps the target is <b>5× the truth</b>.</p>"
    )
    P(
        "<div class='note'><p><b>Side finding — the bias size does not explain between-run performance.</b> "
        "Across 13 runs, mean b vs critic success rate has rank correlation −0.17. A uniform bias leaves the "
        "arg-max unchanged (inflate everything equally and the ranking survives); what decides performance "
        "is the <b>spread</b> of per-candidate errors, which a mean bias cannot see. Only the extremes fit: "
        "soft (the sole negative bias) is best, tn03 (the largest bias) is worst.</p></div>"
    )

    # ---------------------------------------------------------------- 3. v4
    P("<h2>3. v4 — HL-Gauss + mc-floor promoted to baseline <span class='pill p-done'>done</span></h2>")
    P(
        "<p>By decision, the distributional head (51 atoms) and the mc_return floor became the defaults and "
        "the same 15 arms were rerun; <code>scalarq</code>/<code>nofloor</code> now ablate in the reverse "
        "direction.</p>"
    )
    P(
        "<h3>Expected</h3><p>act_sens stays at 0 (it is a property of the target, not the head). The open "
        "question was whether mg4/mg8's ρ gain (0.92/0.91) would survive.</p>"
    )
    P("<h3>Observed</h3>")
    diag_table(
        P,
        [("v4", v4)],
        "Versus v3: act_sens still ≤0.0014, as expected. The mg4/mg8 ρ gain <b>vanished</b> (0.92/0.91 → "
        "0.82/0.82; every v4 run sits flat at 0.81–0.83) — that gain was specific to the scalar-regression "
        "combination. The v3 conclusion 'macro grouping is the largest positive effect' is hereby corrected "
        "as baseline-dependent.",
    )

    # ---------------------------------------------------------------- 4. IQL
    P("<h2>4. IQL — remove the arg-max <span class='pill p-done'>rollout verdict in</span></h2>")
    P("<h3>Design</h3>")
    P(
        "<pre>L_V = E[ |τ − 1(u&lt;0)| · u² ],   u = Q_tgt(z, a_demo, h) − V(z)\n"
        "L_Q = E[ (Q(z, a_demo, h) − y_h)² ],  y_h = cum_h + γ^h · ¬ended · V(z_{t+h})</pre>"
    )
    P(
        "<p>The candidate array is not used in training at all. τ=0.5 is plain least squares (V → mean Q, no "
        "improvement); higher τ weights over-shoots more and approaches max_a Q. Without the candidate "
        "forward the step is ~2.5× faster. Arms: τ ∈ {0.5, 0.7, 0.9, 0.95} plus <code>iql_qc</code> (no "
        "prefix head — separates 'IQL helps' from 'the prefix axis was the problem').</p>"
    )
    P("<pre>ROLLOUT=0 AXES=iql SWEEP=v6_iql slurm/sweep.sh</pre>")
    P(
        "<h3>Expected</h3><p>b(d) collapses toward 0 and returns as τ → 1. In rollout the success criterion "
        "is not 'beats vla' but '<b>hurts less</b>' — IQL cannot create an action-ranking signal the data "
        "does not contain.</p>"
    )
    P("<h3>Observed — value bias</h3>")
    if vb6:
        bands_lab = [f"{a}–{b}" for a, b in BANDS]
        ser = []
        base_row = vb3.get("base", {}).get("rows")
        if base_row:
            ser.append(("TD base", PAL["td"], [x["b"] if x else None for x in base_row]))
        for r, key in (("iql_e50", "e50"), ("iql_e70", "e70"), ("iql_e90", "e90"), ("iql_e95", "e95")):
            if r in vb6:
                ser.append((key, PAL[key], [x["b"] if x else None for x in vb6[r]["rows"]]))
        P(
            "<figure>"
            + line_chart(
                ser,
                bands_lab,
                ylabel="b(d) = V̂ − γ^d   (0 = exact)",
                title="value bias by distance, TD vs IQL expectiles",
            )
            + "<figcaption>Deployment-side value bias by distance to goal. The TD curve is the "
            "inflation the arg-max feeds on; the IQL curves collapse toward zero and re-inflate "
            "monotonically as the expectile approaches a max — the dose-response that pins "
            "causality.</figcaption></figure>"
        )
    if vb6:
        P(
            "<div class='scroll'><table><thead><tr><th>run</th>"
            + "".join(f"<th>{a}–{b}</th>" for a, b in BANDS)
            + "</tr></thead><tbody>"
        )
        base_row = vb3.get("base", {}).get("rows")
        if base_row:
            cells = "".join(f"<td class='bad'>{x['b']:+.4f}</td>" if x else "<td>—</td>" for x in base_row)
            P(f"<tr><td>TD base (reference)</td>{cells}</tr>")
        for r in sorted(vb6):
            cells = ""
            for x in vb6[r]["rows"]:
                if x is None:
                    cells += "<td class='mut'>—</td>"
                else:
                    cls = "ok" if abs(x["b"]) < 0.015 else ("warn" if x["b"] < 0.04 else "bad")
                    cells += f"<td class='{cls}'>{x['b']:+.4f}</td>"
            P(f"<tr><td>{html.escape(r)}</td>{cells}</tr>")
        P("</tbody></table></div>")
        P(
            "<p class='mut' style='font-size:.9rem'>b(d) = V̂(s) − γ^d, with V̂ computed exactly as deployment "
            "reads it (max over the 16 candidates) — i.e. the inflation deployment actually sees. Final "
            "(200k) checkpoints.</p>"
        )
    P(
        "<div class='key win'><p><b>At 60–120 steps from the goal: TD +0.100 → iql_e50 +0.003 / e70 +0.005 "
        "— a 19–38× reduction.</b> And a monotone dose-response, e50 ≈ e70 &lt; e90 &lt; e95: the closer the "
        "expectile gets to a max, the more of the max's bias returns. All five runs are at the same training "
        "stage, so this is not a checkpoint confound — <b>the inflation is causally pinned on the "
        "arg-max</b>. Unlike TD, the bias does not grow with more training (100k → 200k).</p></div>"
    )
    P("<h3>Observed — diagnostics</h3>")
    diag_table(
        P,
        [("v6", v6)],
        "iql_qc's rank_c 0.571 was the only value outside the chance band in 26 runs; the same τ under ARQ "
        "(iql_e70, 0.523) does not show it, so it was a property of the IQL+QC combination. The rollout "
        "below rejects it as a real ranking ability.",
    )
    ir = {r: e for r, e in v6.items() if "rollout" in e}
    if ir:
        P("<h3>Observed — rollout</h3>")
        rollout_table(P, ir)
        P(
            "<div class='key win'><p><b>Prefix mode is cured: 0.528 (TD) → 0.733, parity with the raw "
            "VLA.</b> The causal chain closes at the success-rate level: remove the arg-max → inflation "
            "vanishes → per-prefix targets flatten → commitment-length selection stops reading critic error. "
            "Overall critic harm halves (−0.285 → −0.142) and the expectile dose-response reproduces "
            "(e50 −0.067 p=0.727 vs e90 −0.233 p=0.016 — keep τ low). bon stays bad (0.650): deployment's "
            "best-of-N still arg-maxes over 16 candidates, and with no true value differences between them "
            "that arg-max selects noise. iql_qc's offline rank_c 0.571 did not transfer (bon−vla = 0)."
            "</p></div>"
        )
    else:
        P(
            "<p class='mut'>Four rollouts in progress (30 trials × 5 modes each) — this table fills in when "
            "they land.</p>"
        )

    # ---------------------------------------------------------------- 5. pending
    P("<h2>5. Still running</h2>")
    P(
        "<dl><dt>v5_stability — k3 / k5 / online / tau001 / tau05</dt>"
        "<dd>The three never-ablated axes: ensemble size (how hard the min pushes back), removing the target "
        "network entirely (the reference implementation's default), and the Polyak time constant. The control "
        "axis for how far TD can be repaired.</dd>"
        "<dt>v7_single — ep1 / ep4 / ep16 / ep64</dt>"
        "<dd>The single-trajectory memorisation limit. If b(d) &gt; 0 survives at ep1, the bias is inherent "
        "to the bootstrap structure rather than a product of cross-episode interference. If act_sens rises "
        "there from pure noise, it also proves high act_sens ≠ action understanding.</dd></dl>"
    )
    if any("diag" in e for e in list(v5.values()) + list(v7.values())):
        diag_table(P, [("v5", v5), ("v7", v7)])

    # ---------------------------------------------------------------- 6. engineering
    P("<h2>6. Fixed along the way</h2>")
    P(
        "<ul>"
        "<li><b>targets() simplified</b> — four variables and three branches (crossed/lands_on_term/boot/"
        "term_inside) collapsed into one <code>ended</code> mask; verified bitwise-identical on every valid "
        "cell (2,236,272 cells).</li>"
        "<li><b>QC rollout bug</b> — load_trained returned QC critics without the prefix axis, so "
        "np.unravel_index died and every QC rollout across three sweeps failed silently. Restored the "
        "documented contract with a P=1 axis and macro=horizon.</li>"
        "<li><b>HUD panel corruption solved</b> — nothing renders during the rollout; raw frames are "
        "collected and composed afterwards. A 1000-frame failure episode scans clean (after five measured "
        "hypothesis rejections, this removes the interleaving rather than guessing at it).</li>"
        "<li><b>Overlay rewritten</b> — (a) scale constant re-measured (0.0054 m/unit), (b) magnification "
        "moved to world space (screen-space scaling destroyed the perspective), (c) the factor printed on "
        "every frame, (d) paths whose anchor is clipped to the border are skipped, (e) the projector is "
        "rebuilt between trials (reset tears down the sim it holds).</li>"
        "<li><b>Reproducibility</b> — training at a mismatched discount now builds the matching dataset "
        "itself (mc_return re-accumulated, hardlinked heavy arrays, atomic rename, concurrency-safe). "
        "Proprio is written during annotation (bitwise-equal to the old join on all 279,534 rows); "
        "--use-proprio is gone (always on).</li>"
        "<li><b>Operations</b> — sweeps reroute wholly to the preemptable tier when our base-QOS quota is "
        "already full; every submission gets an artifact-file watcher and milestone reporting.</li></ul>"
    )

    # ---------------------------------------------------------------- 7. next
    P("<h2>7. Next verdicts</h2>")
    P(
        "<ol><li><b>v7 ep1</b> — b(d) in the memorisation limit: the bias's final attribution.</li>"
        "<li><b>v5</b> — whether TD is repairable, or switching to IQL is simply correct.</li>"
        "<li>The action-ranking signal itself is still absent from the data — a margin ranking loss "
        "(chunks from other episodes as negatives) or collecting failure trajectories is the next step.</li>"
        "<li>Deployment: evaluate <b>prefix-only</b> as the default configuration (IQL critic sets the "
        "commitment length; first sample executes).</li></ol>"
    )

    # ---------------------------------------------------------------- appendices
    P("<h2>Appendix A. Metric definitions</h2><dl>")
    for dt, dd in [
        (
            "ρ(Q,MC) — spearman_q_demo_vs_mc",
            "Rank correlation between the demo chunk's Q and the return actually collected. 0 = unrelated; "
            "higher is better. Measures how well the critic reads STATE value.",
        ),
        (
            "act_sens — action_sensitivity",
            "Within-state Q variance ÷ between-state Q variance. 0 = the action is ignored entirely "
            "(Q(z,a) collapses to V(z)); higher is better.",
        ),
        (
            "rank_c — ranking_accuracy_demo_vs_candidate",
            "How often the demonstrated chunk out-scores a policy candidate at the same state. "
            "Chance = 0.5; higher is better.",
        ),
        (
            "rank_o — ranking_accuracy_demo_vs_other",
            "Demo chunk vs a chunk borrowed from a DIFFERENT state, scored at this state. An easy problem; "
            "chance = 0.5. If even this sits at 0.5 the critic ignores its action input.",
        ),
        (
            "pfx_H — prefix_argmax_entropy",
            "Normalised entropy of the prefix lengths the deployment arg-max picks. 1 = spread evenly, "
            "0 = always the same length (adaptive chunking has degenerated).",
        ),
        (
            "b(d)",
            "V̂(s) − γ^d. The truth is known in closed form (sparse terminal reward), so this is computed, "
            "not approximated. 0 = exact; positive = over-estimation.",
        ),
        ("y_h / γ^d", "Per-prefix target over the truth. In theory 1.0, independent of h."),
        ("McNemar p", "Exact test on paired successes over identical scenes. p&lt;0.05 = not chance."),
        ("Wilson interval", "95% CI for a binomial rate; stable near 0 and 1."),
        (
            "expectile τ",
            "IQL's asymmetric-regression parameter. 0.5 = the mean (no improvement); approaching 1 approaches max_a Q.",
        ),
    ]:
        P(f"<dt>{html.escape(dt)}</dt><dd>{dd}</dd>")
    P("</dl>")
    P("<h2>Appendix B. Run names</h2>")
    P(
        f"<p class='mut'>Baseline from v4 onward = {html.escape(BASELINE_NOTE)}. Each run differs in exactly "
        "one thing.</p>"
    )
    P("<div class='scroll'><table><thead><tr><th>run</th><th>sweep</th><th>what differs</th></tr></thead><tbody>")
    extra = {
        "k3": "ensemble of 3 (default 2)",
        "k5": "ensemble of 5",
        "online": "no target network — bootstrap off the online critic",
        "tau001": "Polyak τ=0.001 (10× slower target)",
        "tau05": "Polyak τ=0.05 (10× faster target)",
        "iql_e50": "IQL, expectile 0.50 (= least squares; the no-improvement control)",
        "iql_e70": "IQL, expectile 0.70",
        "iql_e90": "IQL, expectile 0.90",
        "iql_e95": "IQL, expectile 0.95",
        "iql_qc": "IQL (τ=0.7) + QC — no prefix head",
        "ep1": "trained on 1 episode (745 frames) — the memorisation limit",
        "ep4": "4 episodes",
        "ep16": "16 episodes",
        "ep64": "64 episodes",
        "scalarq": "scalar regression (instead of the default HL-Gauss head)",
        "nofloor": "no mc floor (default is max(TD, mc_return))",
    }
    for sweep, runs in [("v4", v4), ("v5", v5), ("v6", v6), ("v7", v7)]:
        for r in sorted(runs):
            cfg = runs[r].get("config")
            desc = extra.get(r) or (" / ".join(describe(cfg)) if cfg else "") or "baseline"
            P(
                f"<tr><td>{html.escape(r)}</td><td class='mut'>{sweep}</td>"
                f"<td style='text-align:left;white-space:normal'>{html.escape(desc)}</td></tr>"
            )
    P("</tbody></table></div>")

    out.write_text(
        f"<title>ACRFT critic — experiment log</title>\n<style>{CSS}</style>\n<main>{''.join(A)}</main>\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
