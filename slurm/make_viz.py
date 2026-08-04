"""Put every number this investigation produced on one page, so the judgement is the reader's.

Chat and the report both arrive at conclusions. This does not: it draws what was measured, with the
null value marked on every axis that has one, and leaves the reading open. Where a metric is
ambiguous — and two of the central ones are — the chart says so next to the chart rather than in a
footnote somewhere else.

Self-contained: inline SVG, no plotting library, no network. Regenerate whenever new runs land.

    uv run slurm/make_viz.py --out /scratch/jellyho/acrft/viz.html
"""

import argparse
import collections
import html
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import base64
import hashlib
import re

from _describe import BASELINE_NOTE
from _describe import describe

# Validated with the dataviz palette checker (light: ALL PASS with a contrast WARN answered by
# direct labels on every mark; dark: ALL PASS).
PAL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"}
PAL_D = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70"}

CSS = """
:root{--surface:#fcfcfb;--panel:#f3f2ee;--ink:#0b0b0b;--ink2:#52514e;--ink3:#86847c;--line:#dedcd5;
      --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--grid:#e8e6df;--null:#a8a49a}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--panel:#232322;--ink:#fff;--ink2:#c3c2b7;
      --ink3:#8e8d84;--line:#343430;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--grid:#2c2c29;--null:#6d6b64}}
:root[data-theme=dark]{--surface:#1a1a19;--panel:#232322;--ink:#fff;--ink2:#c3c2b7;--ink3:#8e8d84;
      --line:#343430;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--grid:#2c2c29;--null:#6d6b64}
:root[data-theme=light]{--surface:#fcfcfb;--panel:#f3f2ee;--ink:#0b0b0b;--ink2:#52514e;--ink3:#86847c;
      --line:#dedcd5;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--grid:#e8e6df;--null:#a8a49a}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;padding:2.5rem 1.25rem 6rem;
     font:15px/1.65 Pretendard,"Apple SD Gothic Neo",system-ui,-apple-system,sans-serif}
main{max-width:64rem;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .3rem;letter-spacing:-.02em}
.sub{color:var(--ink3);font-family:ui-monospace,monospace;font-size:.8rem;margin:0 0 2.5rem}
h2{font-size:1.15rem;margin:3rem 0 .2rem;letter-spacing:-.01em}
.q{color:var(--ink2);font-size:.92rem;margin:.2rem 0 1rem}
figure{margin:0 0 .5rem;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:1rem}
figcaption{color:var(--ink3);font-size:.8rem;margin-top:.6rem;line-height:1.5}
svg{display:block;width:100%;height:auto;overflow:visible}
.tick{fill:var(--ink3);font:10px ui-monospace,monospace}
.lbl{fill:var(--ink);font:11px ui-monospace,monospace}
.lbl2{fill:var(--ink2);font:10px ui-monospace,monospace}
.nullline{stroke:var(--null);stroke-width:1;stroke-dasharray:3 3}
.gridline{stroke:var(--grid);stroke-width:1}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--s2);
      border-radius:0 5px 5px 0;padding:.85rem 1.1rem;margin:1rem 0;font-size:.9rem;color:var(--ink2)}
.note b{color:var(--ink)}
.legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.8rem;color:var(--ink2);margin:.2rem 0 .7rem}
.legend i{display:inline-block;width:.75rem;height:.75rem;border-radius:2px;margin-right:.35rem;
          vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink3);font:600 .72rem ui-monospace,monospace;text-transform:uppercase;letter-spacing:.05em}
td.num{text-align:right;font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;margin:.5rem 0}
.rd{color:var(--ink2);font-size:.83rem;margin:0 0 .5rem;line-height:1.5}
.when{float:right;font:400 .72rem ui-monospace,monospace;color:var(--ink3);background:var(--grid);padding:.1em .5em;border-radius:3px;margin-left:.6rem}
.vidgrid{display:grid;gap:.8rem;grid-template-columns:repeat(auto-fill,minmax(16rem,1fr));margin:.6rem 0}
.vcard{border:1px solid var(--line);border-radius:5px;overflow:hidden;background:var(--panel)}
.vcard video{width:100%;display:block;background:#000;aspect-ratio:1/1}
.vcap{padding:.4rem .6rem;font:.72rem ui-monospace,monospace;display:flex;gap:.45rem;align-items:center}
.vmode{color:var(--s1);font-weight:600}.vt{color:var(--ink3)}
.vb{margin-left:auto;padding:.05em .45em;border-radius:2px;font-size:.66rem;font-weight:600}
.vok{background:color-mix(in srgb,var(--s3) 20%,transparent);color:var(--s3)}
.vno{background:color-mix(in srgb,var(--s2) 20%,transparent);color:var(--s2)}
code{font-family:ui-monospace,monospace;font-size:.86em;background:var(--grid);padding:.08em .34em;border-radius:3px}
g.mark:hover rect,g.mark:hover circle{filter:brightness(1.15)}
g.mark title{pointer-events:none}
"""


def wilson(k, n, z=1.96):
    """Success-rate interval that stays inside [0,1] at the small n these rollouts have."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def esc(s):
    return html.escape(str(s))


def when(*paths):
    """When a result was produced, from the artefact's own mtime.

    A page that mixes numbers from different hours of a moving investigation is a trap: two charts
    can disagree simply because one predates a fix. Stamping each from the file it was read out of
    keeps that visible instead of implicit.
    """
    ts = [p.stat().st_mtime for p in paths if p and p.exists()]
    if not ts:
        return ""
    import datetime as _dt

    lo, hi = min(ts), max(ts)
    f = lambda t: _dt.datetime.fromtimestamp(t, tz=_dt.UTC).astimezone().strftime("%m-%d %H:%M")  # noqa: E731
    return f(hi) if hi - lo < 120 else f"{f(lo)} – {f(hi)}"


def stamp(*paths):
    w = when(*paths)
    return f"<span class='when'>{esc(w)}</span>" if w else ""


def bars_with_ci(rows, *, width=620, rowh=30, pad_l=92, title_null=None, null_label=""):
    """Horizontal bars + 95% CI whiskers. rows = [(label, k, n, tooltip)]. One hue: this is magnitude,
    not identity, so a categorical palette would imply a distinction that is not there."""
    h = rowh * len(rows) + 34
    x0, x1 = pad_l, width - 46
    out = [f'<svg viewBox="0 0 {width} {h}" role="img">']
    for gx in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + gx * (x1 - x0)
        out.append(f'<line class="gridline" x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{h - 20}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - 6}" text-anchor="middle">{int(gx * 100)}%</text>')
    if title_null is not None:
        xn = x0 + title_null * (x1 - x0)
        out.append(f'<line class="nullline" x1="{xn:.1f}" y1="10" x2="{xn:.1f}" y2="{h - 20}"/>')
        out.append(f'<text class="lbl2" x="{xn:.1f}" y="8" text-anchor="middle">{esc(null_label)}</text>')
    for i, (lab, k, n, tip) in enumerate(rows):
        y = 20 + i * rowh
        p, lo, hi = wilson(k, n)
        bw = max(2.0, p * (x1 - x0))
        out.append(f'<g class="mark"><title>{esc(tip)}</title>')
        out.append(f'<text class="lbl" x="{x0 - 8}" y="{y + 11}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{x0}" y="{y + 2}" width="{bw:.1f}" height="14" rx="4" fill="var(--s1)"/>')
        xa, xb = x0 + lo * (x1 - x0), x0 + hi * (x1 - x0)
        out.append(
            f'<line x1="{xa:.1f}" y1="{y + 9}" x2="{xb:.1f}" y2="{y + 9}" stroke="var(--ink3)" stroke-width="1.5"/>'
        )
        out.extend(
            f'<line x1="{xe:.1f}" y1="{y + 4}" x2="{xe:.1f}" y2="{y + 14}" stroke="var(--ink3)" stroke-width="1.5"/>'
            for xe in (xa, xb)
        )
        out.append(f'<text class="lbl" x="{x1 + 6}" y="{y + 13}">{k}/{n}</text></g>')
    out.append("</svg>")
    return "".join(out)


def diverging(rows, *, width=620, rowh=28, pad_l=92):
    """rows = [(label, left_count, right_count, tip)] — losses left, wins right, centred."""
    h = rowh * len(rows) + 30
    mid = pad_l + (width - pad_l - 40) / 2
    span = (width - pad_l - 40) / 2
    mx = max(1, *(max(a, b) for _, a, b, _ in rows))
    out = [f'<svg viewBox="0 0 {width} {h}" role="img">']
    out.append(f'<line class="nullline" x1="{mid}" y1="8" x2="{mid}" y2="{h - 18}"/>')
    for i, (lab, a, b, tip) in enumerate(rows):
        y = 16 + i * rowh
        wa, wb = a / mx * span * 0.92, b / mx * span * 0.92
        out.append(f'<g class="mark"><title>{esc(tip)}</title>')
        out.append(f'<text class="lbl" x="{pad_l - 8}" y="{y + 11}" text-anchor="end">{esc(lab)}</text>')
        if a:
            out.append(f'<rect x="{mid - wa:.1f}" y="{y + 2}" width="{wa:.1f}" height="13" rx="4" fill="var(--s2)"/>')
            out.append(f'<text class="lbl2" x="{mid - wa - 5:.1f}" y="{y + 12}" text-anchor="end">{a}</text>')
        if b:
            out.append(f'<rect x="{mid + 2:.1f}" y="{y + 2}" width="{wb:.1f}" height="13" rx="4" fill="var(--s3)"/>')
            out.append(f'<text class="lbl2" x="{mid + wb + 7:.1f}" y="{y + 12}">{b}</text>')
        out.append("</g>")
    out.append(f'<text class="lbl2" x="{mid - 6}" y="{h - 4}" text-anchor="end">← vla만 성공</text>')
    out.append(f'<text class="lbl2" x="{mid + 6}" y="{h - 4}">이 모드만 성공 →</text>')
    out.append("</svg>")
    return "".join(out)


def lines(series, xs, *, width=620, height=210, pad_l=52, pad_b=34, ylab="", xlab="", logy=False, xticks=None):
    """series = [(name, colorvar, [y...], emphasis_bool)]."""
    x0, x1, y0, y1 = pad_l, width - 14, 12, height - pad_b
    allv = [v for _, _, ys, _ in series for v in ys if v is not None and (not logy or v > 0)]
    if not allv:
        return ""
    lo, hi = min(allv), max(allv)
    if logy:
        lo, hi = math.log10(max(lo, 1e-6)), math.log10(hi)
    if hi - lo < 1e-9:
        hi = lo + 1
    fx = lambda i: x0 + (i / max(1, len(xs) - 1)) * (x1 - x0)  # noqa: E731

    def fy(v):
        t = math.log10(max(v, 1e-6)) if logy else v
        return y1 - (t - lo) / (hi - lo) * (y1 - y0)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for g in range(5):
        y = y0 + g * (y1 - y0) / 4
        val = hi - g * (hi - lo) / 4
        shown = f"{10 ** val:.3g}" if logy else f"{val:.3g}"
        out.append(f'<line class="gridline" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 6}" y="{y + 3:.1f}" text-anchor="end">{shown}</text>')
    for i, xv in enumerate(xs):
        if xticks is None or xv in xticks:
            out.append(
                f'<text class="tick" x="{fx(i):.1f}" y="{height - pad_b + 14}" text-anchor="middle">{esc(xv)}</text>'
            )
    for name, col, ys, emph in series:
        pts = [f"{fx(i):.1f},{fy(v):.1f}" for i, v in enumerate(ys) if v is not None]
        if len(pts) < 2:
            continue
        w = 2.4 if emph else 1.4
        op = 1.0 if emph else 0.45
        out.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="{w}" '
            f'opacity="{op}" stroke-linejoin="round"/>'
        )
        for i, v in enumerate(ys):
            if v is None:
                continue
            out.append(
                f'<g class="mark"><title>{esc(name)} · {esc(xs[i])} → {v:.4g}</title>'
                f'<circle cx="{fx(i):.1f}" cy="{fy(v):.1f}" r="{4 if emph else 3}" fill="{col}" '
                f'opacity="{op}" stroke="var(--panel)" stroke-width="1.5"/></g>'
            )
    if ylab:
        out.append(f'<text class="lbl2" x="{x0 - 44}" y="{y0 + 2}">{esc(ylab)}</text>')
    if xlab:
        out.append(f'<text class="lbl2" x="{(x0 + x1) / 2}" y="{height - 2}" text-anchor="middle">{esc(xlab)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return (
        "<div class='legend'>"
        + "".join(f"<span><i style='background:{c}'></i>{esc(n)}</span>" for n, c in items)
        + "</div>"
    )


def _embed(mp4: pathlib.Path, cache: pathlib.Path) -> str:
    """Re-encode a clip small enough to inline, and return it as a data URI.

    The published page cannot reach the filesystem, so a linked clip is a black box there. Inlining
    costs base64's 33% on top, which only works if the clip is small first: 768px at 10 fps and
    CRF 32 lands at ~1 MB (checked frame by frame that every HUD number — value, spread, chosen
    candidate, prefix, axis labels — is still legible; the video exists to be read). Re-encodes are
    cached by source digest so regenerating the page is cheap.
    """
    import imageio
    import imageio.v3 as iio

    key = hashlib.sha1(f"{mp4}:{mp4.stat().st_mtime_ns}:{mp4.stat().st_size}".encode()).hexdigest()[:16]
    small = cache / f"{key}.mp4"
    if not small.exists():
        cache.mkdir(parents=True, exist_ok=True)
        fr = iio.imread(mp4, plugin="pyav")
        tmp = small.with_suffix(".tmp.mp4")
        imageio.mimwrite(
            tmp,
            fr[::2],
            fps=10,
            codec="libx264",
            output_params=["-crf", "32", "-preset", "slow", "-vf", "scale=768:768"],
        )
        tmp.replace(small)
    return "data:video/mp4;base64," + base64.b64encode(small.read_bytes()).decode()


def videos_for(run_dir, root, *, budget=None, cache=None):
    """Clips for one run, ordered vla → rand → bon → prefix → critic so the comparison reads left to
    right. Linked by relative path: they play when the page is served from $CACHE_DIR, and the page
    says so rather than showing a silent black box if it is opened from anywhere else."""
    vids = collections.defaultdict(list)
    for mp4 in sorted((run_dir / "videos").glob("*.mp4")):
        m = re.match(r".*_(vla|rand|bon|prefix|critic)_t(\d+)_(succ|fail)\.mp4$", mp4.name)
        if m:
            vids[m.group(1)].append((int(m.group(2)), m.group(3) == "succ", mp4.relative_to(root).as_posix()))
    if not vids:
        return ""
    cards, n_emb = [], 0
    # A shared budget, not a per-run one: the comparison that matters is across MODES on
    # one run, so spending it all on the first run beats spreading it thin.
    left = budget[0] if budget else 0
    for mode in MODE_ORDER:
        for trial, ok, rel in sorted(vids.get(mode, [])):
            cls, txt = ("vok", "성공") if ok else ("vno", "실패")
            # Embed the FIRST clip of each mode: the point is the side-by-side across modes on one
            # scene, so one per mode beats several of one mode.
            src, tag = rel, ""
            if left > 0 and trial == min(t for t, _, _ in vids[mode]):
                try:
                    src = _embed(root / rel, cache)
                    n_emb += 1
                    left -= 1
                    budget[0] -= 1
                    tag = "<span class='vt'>임베드</span>"
                except Exception:
                    src = rel
            cards.append(
                f"<div class='vcard'><video controls preload='metadata' src='{src}'></video>"
                f"<div class='vcap'><span class='vmode'>{esc(mode)}</span>"
                f"<span class='vt'>trial {trial}</span>{tag}"
                f"<span class='vb {cls}'>{txt}</span></div></div>"
            )
    note = (
        "<p class='q' style='font-size:.8rem;color:var(--ink3)'>"
        + (
            f"모드별 첫 클립 {n_emb}개는 페이지에 <b>임베드</b>되어 어디서나 재생됩니다"
            " (768px·10fps·CRF32로 재인코딩 — HUD 숫자는 그대로 읽힙니다). 나머지는 "
            if n_emb
            else "영상은 "
        )
        + "<code>cd $CACHE_DIR &amp;&amp; python -m http.server 8800</code> 로 서빙할 때 재생됩니다. "
        "HUD: 우측 = 후보 16개 값 산포, 아래 = value trace(로그축)와 replan당 커밋 스텝.</p>"
    )
    return "<div class='vidgrid'>" + "".join(cards) + "</div>" + note


# 이 페이지는 코드가 바뀌는 동안 쌓인 결과를 함께 그린다. 어느 sweep이 어느 코드 상태에서
# 나왔는지 명시하지 않으면, 두 차트가 다른 이유를 독자가 알 수 없다.
SWEEP_NOTE = {
    "fix_main": "종료 부트스트랩을 <code>--terminal-uses-mc</code> 플래그로 우회하던 시점. "
    "이후 그 처리를 기본 동작으로 고쳤는데, 종료 지점에서 "
    "<code>mc_return == reward</code>가 오차 0으로 일치하므로 <b>수치는 동일</b>합니다 "
    "(6k step 두 런이 자릿수까지 같음을 확인).",
    "mcdist": "종료 부트스트랩이 고쳐진 뒤. <code>--terminal-uses-mc</code>는 삭제됐습니다.",
}


def run_desc(runs_root, sweep, run):
    """A run name is an abbreviation invented for a manifest. Expand it wherever it appears."""
    f = runs_root / sweep / run / "config.json"
    if not f.exists():
        return ""
    try:
        d = describe(json.loads(f.read_text()))
    except Exception:
        return ""
    return " · ".join(d) if d else "baseline (아래 기본 설정 그대로)"


def run_head(runs_root, sweep, run, extra=""):
    d = run_desc(runs_root, sweep, run)
    return f"<h3 style='font-size:.95rem;margin:1.4rem 0 .15rem'><code>{esc(run)}</code>{extra}</h3>" + (
        f"<p class='rd'>{esc(d)}</p>" if d else ""
    )


MODE_KO = {
    "vla": "vla — critic 없음 (기준)",
    "rand": "rand — 후보 무작위 선택",
    "bon": "bon — critic이 후보만 선택",
    "prefix": "prefix — critic이 커밋 길이만 선택",
    "critic": "critic — 후보+길이 결합 선택",
}
MODE_ORDER = ["vla", "rand", "bon", "prefix", "critic"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--generated", default="")
    ap.add_argument(
        "--embed-videos",
        type=int,
        default=0,
        help="총 이만큼의 클립을 data URI로 임베드 (첫 run의 모드별 첫 클립부터). " "클립당 약 1.3 MB. 0 = 링크만.",
    )
    args = ap.parse_args()
    root = args.root or pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
    out = args.out or root / "viz.html"
    runs_root = root / "critic_runs"
    rd = lambda p: (json.loads(p.read_text()) if p.exists() else None)  # noqa: E731

    P = []
    A = P.append
    A("<h1>ACRFT critic — 측정된 것 전부</h1>")
    A(f"<p class='sub'>{esc(args.generated)}</p>")
    A("<h2 style='margin-top:1.5rem'>0. 이 페이지의 약자</h2>")
    A("<p class='q'>아래 표기가 차트 전체에서 반복됩니다.</p>")
    A("<h3 style='font-size:.92rem;margin:1.2rem 0 .3rem'>rollout 모드 — 무엇을 critic이 고르는가</h3>")
    A(
        "<div class='scroll'><table><thead><tr><th>표기</th><th>후보 선택</th><th>커밋 길이 선택</th>"
        "<th>무엇을 알려주나</th></tr></thead><tbody>"
        "<tr><td><code>vla</code></td><td>정책의 첫 샘플</td><td>고정 (16스텝 전부)</td>"
        "<td><b>기준</b> — critic을 아예 안 씀</td></tr>"
        "<tr><td><code>rand</code></td><td><b>무작위</b></td><td>고정</td>"
        "<td>'다른 샘플을 고르는 것'만으로 이득이 있는가</td></tr>"
        "<tr><td><code>bon</code></td><td><b>critic</b>이 선택</td><td>고정</td>"
        "<td>critic의 후보 순위가 무작위보다 나은가</td></tr>"
        "<tr><td><code>prefix</code></td><td>첫 샘플 고정</td><td><b>critic</b>이 선택</td>"
        "<td>적응적 커밋 길이만으로 이득이 있는가</td></tr>"
        "<tr><td><code>critic</code></td><td><b>critic</b></td><td><b>critic</b></td>"
        "<td>실제 배포 규칙 — 둘을 결합한 arg-max</td></tr>"
        "</tbody></table></div>"
    )
    A(
        "<p class='q' style='font-size:.85rem'>다섯 모드가 <b>같은 scene</b>(seed 고정)에서 돌므로 짝지어 비교됩니다. "
        "<code>critic</code>이 <code>vla</code>를 이기는데 <code>bon</code>은 못 이기면 이득의 출처가 "
        "액션 선택이 아니라 커밋 길이입니다.</p>"
    )

    A("<h3 style='font-size:.92rem;margin:1.6rem 0 .3rem'>지표</h3>")
    A(
        "<div class='scroll'><table><thead><tr><th>표기</th><th>무엇을 재나</th><th>기준값</th>"
        "<th>방향</th></tr></thead><tbody>"
        "<tr><td><code>act_sens</code></td><td>상태 <b>내부</b> Q 분산 ÷ 상태 <b>간</b> Q 분산. "
        "Q의 변동 중 액션 때문인 몫</td><td>0 = 액션 완전 무시</td><td>클수록 좋음</td></tr>"
        "<tr><td><code>rank_cand</code></td><td>같은 상태에서 시연 chunk가 정책 후보보다 높은 Q를 받는 비율</td>"
        "<td><b>우연 0.5</b></td><td>클수록 좋음</td></tr>"
        "<tr><td><code>ρ(Q, prefix)</code></td><td>커밋 길이가 길어질 때 Q가 오르는가 내리는가 (순위상관)</td>"
        "<td>0 = 무관</td><td>음수 = 짧게 커밋하는 편향</td></tr>"
        "<tr><td><code>평균 커밋</code></td><td>replan마다 실제로 실행한 스텝 수 (최대 16)</td>"
        "<td>—</td><td>클수록 길게 커밋</td></tr>"
        "<tr><td><code>Q</code></td><td>critic이 매긴 값. 여기선 <code>γ^(성공까지 남은 스텝)</code>을 학습</td>"
        "<td>0~1</td><td>1에 가까울수록 목표 근처</td></tr>"
        "</tbody></table></div>"
    )

    A("<h3 style='font-size:.92rem;margin:1.6rem 0 .3rem'>run 이름</h3>")
    A(
        f"<p class='q' style='font-size:.85rem'><b>baseline</b> = {BASELINE_NOTE}.<br>"
        "각 run 이름 아래에 <b>baseline과 다른 점만</b> 적혀 있습니다 — 한 번에 하나씩만 바꾸는 설계라, "
        "적히지 않은 설정은 전부 baseline과 같습니다. 자주 나오는 약자: "
        "<code>sca</code>=스칼라 Q, <code>hlg</code>=HL-Gauss 분포형, <code>mcf</code>=MC 하한, "
        "<code>upc</code>=prefix 커버리지 균일화, <code>bc</code>=부트스트랩 후보 수, "
        "<code>tn</code>=타깃 노이즈, <code>mg</code>=macro group, <code>g999</code>=γ 데이터셋.</p>"
    )
    A(
        "<p class='q' style='font-size:.85rem'><b>sweep 이름</b>: <code>fix_main</code> = 종료 부트스트랩을 "
        "플래그로 우회하던 시점의 16런. <code>mcdist</code> = 그 처리를 고친 뒤 돌린 "
        "MC하한 × 분포형 × 커버리지 2×2×2 교차 8런.</p>"
    )

    A(
        "<div class='note'><b>결과마다 얻은 시각이 제목 오른쪽에 붙어 있습니다.</b> "
        "이 조사는 코드를 고쳐가며 진행됐으므로, 시각이 다르면 코드 상태도 다를 수 있습니다 — "
        "해당하는 곳에 어느 상태였는지 적어뒀습니다.</div>"
    )
    A(
        "<div class='note'><b>이 페이지는 결론을 내지 않습니다.</b> 측정값과 각 축의 기준값(우연·영점)만 그립니다. "
        "두 개의 중심 지표는 해석이 모호하며, 그 모호함을 해당 차트 옆에 적어뒀습니다.</div>"
    )

    # ---- 1. rollout 성공률 -----------------------------------------------------------------------
    A("<h2>1. rollout 성공률 — 실제 목적함수" + stamp(*runs_root.glob("*/*/rollout/*.json")) + "</h2>")
    A(
        "<p class='q'>critic이 고른 것이 VLA가 그냥 낸 것보다 나은가. 다섯 모드가 <b>동일 scene</b>(seed 고정)에서 "
        "돌아 짝지어 비교됩니다. 오차막대는 95% Wilson 구간.</p>"
    )
    any_roll = False
    _budget = [args.embed_videos]
    for sweep_run in sorted(runs_root.glob("*/*/rollout")):
        run = sweep_run.parent.name
        agg = collections.defaultdict(lambda: [0, 0])
        for f in sorted(sweep_run.glob("*.json")):
            for m, v in (rd(f) or {}).items():
                if isinstance(v, dict) and "successes" in v:
                    agg[m][0] += v["successes"]
                    agg[m][1] += v["num_trials"]
        if not agg:
            continue
        any_roll = True
        rows = [
            (MODE_KO.get(m, m), agg[m][0], agg[m][1], f"{m}: {agg[m][0]}/{agg[m][1]}") for m in MODE_ORDER if m in agg
        ]
        base = agg.get("vla")
        nullv = (base[0] / base[1]) if base and base[1] else None
        A(run_head(runs_root, sweep_run.parent.parent.name, run, stamp(*sweep_run.glob("*.json"))))
        A(
            "<figure>"
            + bars_with_ci(rows, title_null=nullv, null_label="vla")
            + "<figcaption>점선 = <code>vla</code>의 성공률. 오른쪽 숫자는 성공/시도.</figcaption></figure>"
        )
        A(videos_for(sweep_run.parent, root, budget=_budget, cache=root / ".viz_cache"))
    if not any_roll:
        A("<p class='q'>아직 rollout 결과가 없습니다.</p>")

    # ---- 2. 짝지은 승패 -------------------------------------------------------------------------
    A("<h2>2. 같은 scene에서 누가 이겼나" + stamp(*runs_root.glob("*/*/rollout/*.json")) + "</h2>")
    A(
        "<p class='q'>성공률 차이는 표본에 흔들립니다. 같은 scene을 짝지어 <b>한쪽만 성공한 경우</b>만 세면 "
        "훨씬 적은 표본으로 방향이 보입니다.</p>"
    )
    for sweep_run in sorted(runs_root.glob("*/*/rollout")):
        run = sweep_run.parent.name
        tr = {}
        for f in sorted(sweep_run.glob("*.json")):
            for m, v in (rd(f) or {}).items():
                if isinstance(v, dict) and "trials" in v:
                    for t in v["trials"]:
                        tr.setdefault((f.stem, t["trial"]), {})[m] = t["success"]
        if not tr:
            continue
        rows = []
        for m in MODE_ORDER:
            if m == "vla":
                continue
            a = sum(1 for d in tr.values() if d.get("vla") and not d.get(m))
            b = sum(1 for d in tr.values() if d.get(m) and not d.get("vla"))
            if a or b:
                rows.append(
                    (
                        MODE_KO.get(m, m).split(" — ")[0] + " — " + MODE_KO.get(m, m).split(" — ")[-1],
                        a,
                        b,
                        f"{m}: vla만 {a}, {m}만 {b}, 나머지 {len(tr) - a - b}개는 동일",
                    )
                )
        if rows:
            A(
                run_head(
                    runs_root,
                    sweep_run.parent.parent.name,
                    run,
                    f" <span style='color:var(--ink3);font-weight:400'>· {len(tr)} scene</span>",
                )
            )
            A(
                "<figure>"
                + diverging(rows)
                + "<figcaption>양쪽 다 성공하거나 다 실패한 scene은 세지 않습니다(정보가 없음).</figcaption></figure>"
            )

    # ---- 3. 값함수 -------------------------------------------------------------------------------
    vb, vt = rd(runs_root / "vvd_base.json"), rd(runs_root / "vvd_tmc.json")
    if vb and vt:
        A(
            "<h2>3. 값함수는 목표까지의 거리를 아는가"
            + stamp(runs_root / "vvd_base.json", runs_root / "vvd_tmc.json")
            + "</h2>"
        )
        A("<p class='q'>참값은 <code>γ^(남은 스텝)</code>입니다. 부트스트랩의 종료 처리를 고치기 전/후.</p>")
        xs = [f"{b['lo']}–{b['hi'] if b['hi'] < 10**6 else '∞'}" for b in vb["bins"]]
        A(legend([("참값 γ^d", "var(--s3)"), ("수정 후", "var(--s1)"), ("수정 전", "var(--s2)")]))
        A(
            "<figure>"
            + lines(
                [
                    ("참값 γ^d", "var(--s3)", [b["mc"] for b in vb["bins"]], True),
                    ("수정 후", "var(--s1)", [b["q"] for b in vt["bins"]], True),
                    ("수정 전", "var(--s2)", [b["q"] for b in vb["bins"]], True),
                ],
                xs,
                logy=True,
                ylab="Q (로그)",
                xlab="목표까지 남은 스텝",
            )
            + "<figcaption>로그축. 수정 전 곡선이 오른쪽으로 <b>올라가는</b> 것이 결함의 서명입니다 — "
            "목표에서 멀수록 값이 높다는 뜻.</figcaption></figure>"
        )

    # ---- 4. prefix 프로파일 ----------------------------------------------------------------------
    for sw in ("fix_main", "mcdist"):
        prof = rd(runs_root / f"prefix_profile_{sw}.json")
        if not prof:
            continue
        A(f"<h2>4. 커밋 길이 — <code>{esc(sw)}</code>" + stamp(runs_root / f"prefix_profile_{sw}.json") + "</h2>")
        if sw in SWEEP_NOTE:
            A(f"<p class='q' style='font-size:.82rem'>{SWEEP_NOTE[sw]}</p>")
        A(
            "<p class='q'>배포는 <code>(후보, prefix)</code> 결합 arg-max로 커밋 길이를 정합니다. "
            "Q가 prefix에 대해 단조 감소하면 항상 짧게 커밋합니다.</p>"
        )
        names = sorted(prof)
        xs = [str(p) for p in prof[names[0]]["prefixes"]]
        ser = []
        for n in names:
            a = prof[n].get("all") or {}
            if a.get("mean_q_by_prefix"):
                emph = n in ("upc", "mcf_sca_upc", "base", "s_sca", "soft")
                col = "var(--s2)" if "upc" in n else ("var(--s1)" if n in ("base", "s_sca") else "var(--s3)")
                ser.append((n, col, a["mean_q_by_prefix"], emph))
        if ser:
            A(legend([("baseline", "var(--s1)"), ("upc (커버리지 균일화)", "var(--s2)"), ("기타 변종", "var(--s3)")]))
            A(
                "<figure>"
                + lines(ser, xs, ylab="평균 Q", xlab="prefix 길이 (스텝)")
                + "<figcaption>기울기가 음수 = 짧은 커밋 선호. <code>upc</code>가 평탄해지는지 보세요.</figcaption></figure>"
            )
        rows = [
            [
                n,
                run_desc(runs_root, sw, n),
                f"{100 * (prof[n]['all'] or {}).get('frac_shortest', 0):.1f}%",
                f"{(prof[n]['all'] or {}).get('mean_exec_steps', 0):.2f}",
                f"{(prof[n]['all'] or {}).get('spearman_q_vs_prefix', 0):+.2f}",
            ]
            for n in names
        ]
        A(
            "<div class='scroll'><table><thead><tr><th>run</th><th>baseline과 다른 점</th>"
            "<th>최단 prefix를<br>고른 비율</th><th>평균 커밋<br>(16 중 몇 스텝)</th>"
            "<th>ρ(Q, prefix)<br>−1이면 단조감소</th></tr></thead><tbody>"
            + "".join(
                "<tr><td><code>"
                + esc(r[0])
                + "</code></td><td style='white-space:normal;font-size:.8rem'>"
                + esc(r[1])
                + "</td>"
                + "".join(f"<td class='num'>{esc(c)}</td>" for c in r[2:])
                + "</tr>"
                for r in rows
            )
            + "</tbody></table></div>"
        )

    # ---- 5. 오프라인 지표 ------------------------------------------------------------------------
    A("<h2>5. 오프라인 지표 — 그리고 왜 미덥지 못한가" + stamp(*runs_root.glob("*/*/diag.json")) + "</h2>")
    A(
        "<div class='note'><b>이 두 지표는 'critic이 눈이 멀었다'와 '볼 것이 없다'를 구분하지 못합니다.</b><br>"
        "<code>act_sens</code> = 상태 내부 Q 분산 ÷ 상태 간 분산. 0이면 Q가 액션을 무시. "
        "그런데 후보들의 진짜 가치가 모두 같다면 <b>완벽한 critic도 0</b>을 냅니다.<br>"
        "<code>rank_cand</code> = 시연 chunk가 후보보다 높은 Q를 받는 비율, 우연 0.5. "
        "그런데 VLA는 그 시연을 모방하도록 학습됐고, 실측상 시연은 후보 구름 <b>안</b>에 있습니다 "
        "(후보끼리 거리 0.611 vs 후보↔시연 0.706). 즉 정답이 '더 낫다'가 아니라 '같다'일 수 있고, "
        "그러면 <b>0.5가 올바른 동작</b>입니다.</div>"
    )
    for sw in ("fix_main", "mcdist"):
        d = runs_root / sw
        if not d.is_dir():
            continue
        rows = []
        for r in sorted(p for p in d.iterdir() if p.is_dir()):
            dg = rd(r / "diag.json")
            if dg:
                rows.append((r.name, dg.get("action_sensitivity"), dg.get("ranking_accuracy_demo_vs_candidate")))
        if not rows:
            continue
        A(f"<h3 style='font-size:.95rem;margin:1.4rem 0 .15rem'><code>{esc(sw)}</code> sweep</h3>")
        if sw in SWEEP_NOTE:
            A(f"<p class='q' style='font-size:.82rem'>{SWEEP_NOTE[sw]}</p>")
        mx = max(x[1] or 0 for x in rows) or 1
        A(
            "<figure>"
            + bars_with_ci(
                [
                    (
                        f"{n} — {run_desc(runs_root, sw, n)[:46]}",
                        int(round((a or 0) / mx * 1000)),
                        1000,
                        f"{n}: act_sens {a:.5f} · {run_desc(runs_root, sw, n)}",
                    )
                    for n, a, _ in rows
                ],
                title_null=0.0,
                null_label="0 = 액션 무시",
            )
            + "<figcaption>act_sens, 최댓값 기준 정규화. 오른쪽 숫자는 무시하고 막대 길이만 보세요 — "
            f"가장 큰 값이 {mx:.4f}입니다 (1.0이면 액션이 상태만큼 중요).</figcaption></figure>"
        )
        A(
            "<figure>"
            + bars_with_ci(
                [
                    (
                        f"{n} — {run_desc(runs_root, sw, n)[:46]}",
                        int(round((b or 0) * 1000)),
                        1000,
                        f"{n}: rank_cand {b:.3f} · {run_desc(runs_root, sw, n)}",
                    )
                    for n, _, b in rows
                ],
                title_null=0.5,
                null_label="우연 0.5",
            )
            + "<figcaption>rank_cand. 점선이 우연(0.5)입니다.</figcaption></figure>"
        )

    # ---- 6. 후보 기하 ---------------------------------------------------------------------------
    A("<h2>6. 애초에 고를 것이 있는가 — 후보들의 기하</h2>")
    A("<p class='q'>순위를 매기려면 후보가 서로 달라야 하고, 시연이 그중 특별해야 합니다.</p>")
    A(
        "<figure>"
        + bars_with_ci(
            [
                ("상태 내부, 후보끼리", 611, 5460, "평균 L2 거리 0.611"),
                ("후보 ↔ 같은 상태의 시연", 706, 5460, "평균 L2 거리 0.706"),
                ("상태 간, 시연끼리 (기준)", 5460, 5460, "평균 L2 거리 5.460"),
            ],
            title_null=None,
        )
        + "<figcaption>액션 chunk 사이의 평균 L2 거리, 상태 간 거리로 정규화. "
        "<b>시연이 후보 구름 밖에 있지 않습니다</b> — 후보끼리(0.611)와 후보↔시연(0.706)이 거의 같습니다. "
        "VLA가 그 시연을 모방하도록 학습됐으니 당연한 결과이고, <code>rank_cand</code>의 전제가 무너지는 지점입니다.</figcaption></figure>"
    )

    # ---- 7. prefix 타깃 천장 ---------------------------------------------------------------------
    pb = rd(runs_root / "prefix_bias_noprop.json")
    if pb:
        A("<h2>7. prefix별 타깃 천장 — 커버리지 비대칭" + stamp(runs_root / "prefix_bias_noprop.json") + "</h2>")
        A(
            "<p class='q'>종료를 넘는 prefix는 transition을 만들지 못합니다. 그래서 prefix h가 받을 수 있는 "
            "최대 타깃이 정확히 γ^h로 묶입니다.</p>"
        )
        A(
            "<figure>"
            + lines(
                [("최대 타깃 (=γ^h)", "var(--s1)", [r["max_mc"] for r in pb["per_prefix"]], True)],
                [str(r["prefix"]) for r in pb["per_prefix"]],
                ylab="최대 타깃",
                xlab="prefix 길이 (스텝)",
            )
            + "<figcaption>긴 prefix head는 값 범위의 상단을 구조적으로 못 봅니다 — 그것도 목표 근처에서. "
            "<code>upc</code>는 이 비대칭을 제거하는 변종입니다 (§4).</figcaption></figure>"
        )

    out.write_text(
        f"<title>ACRFT critic — 측정값</title>\n<style>{CSS}</style>\n<main>{''.join(P)}</main>\n", encoding="utf-8"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
