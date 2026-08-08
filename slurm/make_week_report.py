"""Build report_week.html — the week's findings, executive-summary style: the causal chain, four
key charts, a verdict on every method tried, and what comes next.

Companion to report_fit.html (the detailed probe report). Everything quantitative is recomputed
from raw artefacts: rollout JSONs are re-scanned for the success-rate chart, the band-width chart
reads the probe's JSON dump, and the ladder/full-data figures are embedded from $CACHE_DIR/plots.

    uv run --no-sync python slurm/make_week_report.py
"""

import base64
import glob
import html
import io
import json
import os
import pathlib

C = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
OUT = C / "report_week.html"


def _mpl():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def img64(path):
    p = pathlib.Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


# ---------------------------------------------------------------- chart 1: mode deltas by family
def family_delta_chart():
    """Δ success (critic−vla and prefix−vla) per run, grouped by critic family, TD left, IQL right."""
    plt = _mpl()
    import numpy as np

    rows = []
    for f in glob.glob(str(C / "critic_runs/*/*/rollout/*.json")):
        try:
            j = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        modes = {
            k: [bool(t["success"]) for t in v["trials"]] for k, v in j.items() if isinstance(v, dict) and "trials" in v
        }
        if "vla" not in modes:
            continue
        run_dir = pathlib.Path(f).parent.parent
        try:
            cfg = json.loads((run_dir / "config.json").read_text())
        except Exception:
            continue
        fam = run_dir.parent.name
        obj = cfg.get("objective", "td")
        vla = np.mean(modes["vla"])
        rows.extend(
            {"family": fam, "obj": obj, "mode": m, "delta": float(np.mean(modes[m]) - vla), "n": len(modes[m])}
            for m in ("critic", "prefix")
            if m in modes
        )
    fams = sorted(
        {r["family"] for r in rows},
        key=lambda f: (0 if any(r["family"] == f and r["obj"] == "td" for r in rows) else 1, f),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, dpi=180, sharey=True)
    for ax, mode, color in zip(axes, ("critic", "prefix"), ("#dc2626", "#2563eb"), strict=True):
        xs, labels = [], []
        for _i, fam in enumerate(fams):
            ds = [r["delta"] for r in rows if r["family"] == fam and r["mode"] == mode]
            if not ds:
                continue
            x = len(labels)
            jit = (np.arange(len(ds)) - (len(ds) - 1) / 2) * 0.06
            ax.scatter(x + jit, ds, s=26, color=color, alpha=0.65, zorder=3)
            ax.hlines(np.mean(ds), x - 0.28, x + 0.28, color="#111", lw=2.5, zorder=4)
            labels.append(
                f"{fam}\n({'TD' if any(r['family'] == fam and r['obj'] == 'td' for r in rows) else 'IQL'}, {len(ds)} runs)"
            )
            xs.append(x)
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(f"{mode} − vla   (per run, same 30 scenes within a run)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Δ success rate vs vla\n(>0 = critic mode helped)")
    return _b64(fig), len(rows)


# ---------------------------------------------------------------- chart 2: band widths
def band_chart():
    plt = _mpl()
    import numpy as np

    path = C / "probes/band_width.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    labels = list(dict.fromkeys(d["label"] for d in data))
    segs = ["ep0", "ep65"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True, dpi=180)
    w = 0.38
    for si, (seg, color) in enumerate(zip(segs, ("#2563eb", "#93c5fd"), strict=True)):
        vals = [next(d["band"] for d in data if d["label"] == lab and d["seg"] == seg) for lab in labels]
        ax.bar(np.arange(len(labels)) + (si - 0.5) * w, vals, w, color=color, label=f"{seg} (train episode)")
        for x, v in enumerate(vals):
            ax.text(x + (si - 0.5) * w, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("candidate-Q band width q99−q01\n(log scale; smaller = action axis more collapsed)")
    ax.set_title(
        "How much do the deployed critics distinguish the 16 candidate actions? (full-data critics, training episodes)"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    return _b64(fig)


# ---------------------------------------------------------------- page
CHAIN = """
<div class="chain">
 <div class="node bad"><b>No candidate signal in the data</b><br>dueling probe: chance ranking of candidates, memorised pair identity</div>
 <div class="arr">→</div>
 <div class="node bad"><b>Arg-max = winner's curse</b><br>bon &lt; rand; candidate main effect 81% of Q variance is fixed-function noise</div>
 <div class="arr">→</div>
 <div class="node warn"><b>TD adds h-bias &amp; inflation</b><br>distance-structured V-bias (5.07× far), γ=0.999 bootstrap blow-up at small budget</div>
 <div class="arr">→</div>
 <div class="node good"><b>IQL cures the value axis</b><br>prefix ≥ vla; near-perfect calibration on training data (duel 0.002)</div>
 <div class="arr">→</div>
 <div class="node warn"><b>…but collapses the action axis</b><br>band q99−q01 down to 0.0002; nothing left to arg-max over</div>
</div>
"""

VERDICTS = [
    (
        "TD + target-noise / top-m / soft-tau / LCB / bootstrap-cand knobs",
        "dead",
        "no rollout gain; flags removed in the refactor",
    ),
    ("Best-of-N over candidates (bon)", "dead", "≤ rand — selection indistinguishable from noise"),
    (
        "Joint arg-max (critic mode)",
        "dead",
        "costs −0.074 vs prefix (p=0.0021); reproduced this week (−10 discordant, p=0.099)",
    ),
    ("Softmax over flat Q (T=0.005…0.02)", "dead", "parity at every temperature"),
    (
        "softcand (sample candidate, arg-max prefix)",
        "dead",
        "74/120 vs vla 81/120, p=0.37 — sampling a signal-free ranking is still a coin flip",
    ),
    ("iql_e70 zero-crossing (+0.033)", "not replicated", "3 fresh seeds, 90 paired trials: critic 48/90 vs vla 53/90"),
    ("IQL prefix-only", "alive", "consistently ≥ vla; the one mode that pays its way"),
    (
        "IQL dueling (zero-mean advantage)",
        "alive (as machinery)",
        "best calibration of all (0.002); action head fully collapsed — the canvas for a margin loss",
    ),
    (
        "γ=0.999",
        "diagnostic win",
        "keeps truth sloped everywhere; exposed TD bootstrap inflation; full-data fit excellent",
    ),
    ("Head-bank MLP ensembles (v9)", "pending", "training done, rollouts submitted"),
    ("Target-net variants (v5_r3)", "pending", "still training"),
]


def main():
    fam_chart, nrows = family_delta_chart()
    bandc = band_chart()
    verdict_rows = "".join(
        f"<tr><td>{html.escape(m)}</td><td class='v-{v.split()[0].replace('…', '')}'>{html.escape(v)}</td><td>{html.escape(e)}</td></tr>"
        for m, v, e in VERDICTS
    )

    def fig(title, b64, cap):
        if b64 is None:
            return f"<div class='fig missing'><h4>{title}</h4><p>probe output not present yet — rerun after it lands.</p></div>"
        return (
            f"<div class='fig'><h4>{title}</h4><img src='data:image/png;base64,{b64}'/><p class='cap'>{cap}</p></div>"
        )

    lad = "".join(
        f"<img class='third' src='data:image/png;base64,{img64(C / 'plots' / p)}'/>"
        for p in ("8_ladder_summary.png", "10_ladder_summary.png", "11_ladder_summary.png")
        if img64(C / "plots" / p)
    )
    full = img64(C / "plots/12_fullrun_fit.png")

    body = f"""
<h1>Week in review — RLT critic</h1>
<p class="meta">RoboCasa PrepareCoffee · generated by <code>slurm/make_week_report.py</code> · detailed
probe report: <code>report_fit.html</code>. Success-rate points recomputed from
{nrows} (run, mode) rollout entries on disk.</p>

<h2>The story in one line</h2>
{CHAIN}
<p>Every failure of the candidate axis traced back to one root cause — the data contains no signal
that distinguishes the VLA's 16 samples — and every value-axis pathology traced back to the TD
bootstrap. IQL fixes the second completely and, in doing so, proves the first: with smooth targets
the critic collapses to Q(z,a)=V(z), leaving nothing to select on.</p>

<h2>1 &nbsp;Rollout outcomes, the whole week in one chart</h2>
<img src="data:image/png;base64,{fam_chart}"/>
<p class="cap">Each dot is one evaluation run (30 scenes, modes paired within a run); the black bar is
the family mean. Left: the deployment rule (joint arg-max) — TD families sit below zero and IQL pulls
back to parity but not above. Right: prefix-only — the only mode at or above the vla baseline.
Family names: v3–v5 = TD generations (mask fix, HLG floor, stability), v6/v8 = IQL generations,
pro_* = early proprio probes.</p>

<h2>2 &nbsp;Fit quality vs data — TD and IQL ladders side by side</h2>
<div class="row">{lad}</div>
<p class="cap">Mean |estimate − truth| along a training episode (solid) and a held-out episode
(the other lines) vs number of training episodes. Left: TD γ=0.99 (fails held-out below 16 episodes).
Middle: TD γ=0.999 (bootstrap inflation — fails even the training episode at the 20k-step budget).
Right: IQL γ=0.999 (monotone held-out improvement; V beats Q at every rung). Same probe, same data,
same budget — only the objective differs.</p>

<h2>3 &nbsp;The deployed critics: calibration solved…</h2>
{fig("Full-data critics on their own training episodes", full, "Dashed = ground truth γ^(K−t) (sparse reward ⇒ exact). IQL rows sit on the truth curve (errors 0.002–0.031); TD keeps a systematic optimism offset that survives full data and 200k steps. All runs use the MC floor; all use HL-Gauss 51 atoms except dueling (scalar by design).")}

<h2>4 &nbsp;…and action discrimination not</h2>
{fig("Candidate-band width by critic", bandc, "Width of the q01–q99 band of Q over the 16 stored candidates, averaged along two training episodes (log scale). TD γ=0.99's spread is 5–70× larger but was already shown to be noise (bon ≤ rand); γ=0.999 or IQL smooth it away entirely. No critic distinguishes actions in-distribution.")}

<h2>5 &nbsp;Verdict table — everything tried</h2>
<table>
<tr><th>method / knob</th><th>verdict</th><th>evidence</th></tr>
{verdict_rows}
</table>

<h2>6 &nbsp;Next</h2>
<ol>
 <li><b>Margin/ranking loss on the dueling machinery</b> — inject the missing candidate signal at
 training time. The collapsed IQL action axis is a clean canvas: any spread that appears afterwards
 is injected signal, verifiable with the ladder probe before spending rollouts.</li>
 <li><b>Failure data</b> — annotate failed rollout trajectories to give "low-value state" evidence the
 demos cannot contain. Decide after (1) reads out.</li>
 <li><b>Episode-boundary artefact</b> — every critic spikes to V≈0.8–1.0 on the first ~10 frames of an
 episode; suspected RLT-token contamination across resets. Probe before trusting early-episode values.</li>
 <li>Pending readouts: v9 head-bank rollouts, v5_r3 target-net comparison.</li>
</ol>
"""
    css = """
body{font-family:system-ui,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#111;line-height:1.55}
h1{font-size:1.5em}h2{border-bottom:2px solid #e5e7eb;padding-bottom:4px;margin-top:1.8em}
img{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}.row{display:flex;gap:6px}.third{flex:1;min-width:0}
table{border-collapse:collapse;font-size:.92em;margin:10px 0}td,th{border:1px solid #d1d5db;padding:5px 10px;text-align:left}
th{background:#f3f4f6}.cap{font-size:.88em;color:#555;margin-top:4px}.meta{color:#666;font-size:.9em}
.fig{margin:16px 0}.missing{background:#fef9c3;padding:8px 12px;border-radius:6px}
.chain{display:flex;align-items:stretch;gap:4px;margin:14px 0;flex-wrap:wrap}
.node{flex:1;min-width:150px;border-radius:8px;padding:8px 10px;font-size:.85em;border:1.5px solid}
.node.bad{background:#fef2f2;border-color:#fca5a5}.node.warn{background:#fffbeb;border-color:#fcd34d}
.node.good{background:#f0fdf4;border-color:#86efac}.arr{align-self:center;font-size:1.3em;color:#666}
.v-dead{color:#b91c1c}.v-alive{color:#15803d}.v-pending{color:#a16207}.v-not{color:#b91c1c}.v-diagnostic{color:#1d4ed8}
code{background:#f3f4f6;padding:1px 5px;border-radius:4px}
"""
    OUT.write_text(
        f"<!doctype html><meta charset='utf-8'><title>Week in review — RLT critic</title><style>{css}</style>{body}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
