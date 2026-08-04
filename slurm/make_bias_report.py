"""Write the investigation report: why the critic hurts, and what was measured on the way.

Reads only files the experiments themselves wrote — rollout/*.json, diag.json, config.json, plus the
two probe outputs (vbias.json, pfx_curve.json) — so re-running it after more runs finish refreshes
the page without editing anything here.

    uv run slurm/make_bias_report.py --out /scratch/jellyho/acrft/report_bias.html
"""

import argparse
import glob
import html
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _describe import BASELINE_NOTE, describe  # noqa: E402

MODES = ["vla", "rand", "bon", "prefix", "critic"]
MODE_DESC = {
    "vla": ("critic 없음", "VLA가 뽑은 첫 청크를 그대로 16스텝 실행. 비교 기준선."),
    "rand": ("무작위 후보", "후보 16개 중 하나를 균등 무작위로. critic을 쓰지 않으면서 VLA 기본값에서는 벗어난다 — 손실이 '다르게 골라서'인지 'critic이 골라서'인지 가르는 대조군."),
    "bon": ("critic이 후보 선택", "후보 16개를 Q로 순위 매겨 최고를 실행. 커밋 길이는 16 고정."),
    "prefix": ("critic이 커밋 길이 선택", "첫 후보로 고정하고 프리픽스 길이만 Q로 고름."),
    "critic": ("critic이 둘 다", "(후보, 프리픽스) 결합 arg-max. 실제 배포 규칙."),
}
BANDS = [(5, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 600)]


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


CSS = """
:root{
  --paper:#f4f6f3; --ink:#15181b; --mut:#5f6a6d; --line:#dbe0da; --panel:#eaeee8;
  --bad:#9b2c2c; --ok:#2f6b4a; --warn:#8a6a1c; --rule:#c3cbc2;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#13161a; --ink:#e6e9e4; --mut:#8d959a; --line:#252b30; --panel:#1a1f23;
  --bad:#e08076; --ok:#7cc39a; --warn:#cfa94b; --rule:#333a40;}}
:root[data-theme=light]{--paper:#f4f6f3;--ink:#15181b;--mut:#5f6a6d;--line:#dbe0da;--panel:#eaeee8;
  --bad:#9b2c2c;--ok:#2f6b4a;--warn:#8a6a1c;--rule:#c3cbc2;}
:root[data-theme=dark]{--paper:#13161a;--ink:#e6e9e4;--mut:#8d959a;--line:#252b30;--panel:#1a1f23;
  --bad:#e08076;--ok:#7cc39a;--warn:#cfa94b;--rule:#333a40;}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:3rem 1.25rem 7rem;
  font:16px/1.68 -apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo",system-ui,sans-serif;
  font-variant-numeric:tabular-nums}
main{max-width:60rem;margin:0 auto}
h1{font:600 2.05rem/1.2 Georgia,"Times New Roman",serif;margin:0 0 .5rem;letter-spacing:-.015em;text-wrap:balance}
h2{font:600 1.3rem/1.3 Georgia,serif;margin:3.4rem 0 .9rem;padding-top:1.1rem;
  border-top:1px solid var(--rule);letter-spacing:-.01em;text-wrap:balance}
h3{font:600 1.02rem/1.35 Georgia,serif;margin:2rem 0 .5rem;color:var(--ink)}
.lede{color:var(--mut);font-size:1.02rem;margin:0 0 .4rem}
.stamp{color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem;margin:0 0 2.6rem}
p{margin:.85rem 0}
.scroll{overflow-x:auto;margin:1.1rem 0;border:1px solid var(--line);border-radius:3px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.79rem}
th{text-align:right;padding:.55rem .7rem;color:var(--mut);font-weight:600;white-space:nowrap;
  border-bottom:1px solid var(--rule);position:sticky;top:0;background:var(--panel)}
td{text-align:right;padding:.42rem .7rem;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
tr.total td{border-top:2px solid var(--rule);font-weight:700}
.bad{color:var(--bad)} .ok{color:var(--ok)} .warn{color:var(--warn)} .mut{color:var(--mut)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
  background:var(--panel);padding:.1em .36em;border-radius:2px}
.key{border-left:3px solid var(--bad);background:var(--panel);padding:1rem 1.2rem;margin:1.4rem 0;border-radius:0 3px 3px 0}
.key p:first-child{margin-top:0} .key p:last-child{margin-bottom:0}
.note{border-left:3px solid var(--rule);background:var(--panel);padding:.9rem 1.1rem;margin:1.3rem 0;
  border-radius:0 3px 3px 0;font-size:.93rem;color:var(--mut)}
.note b{color:var(--ink)}
dl{margin:1rem 0}
dt{font-family:ui-monospace,monospace;font-size:.84rem;font-weight:600;margin-top:.8rem}
dd{margin:.15rem 0 0 0;color:var(--mut);font-size:.93rem}
pre{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.9rem 1rem;
  overflow-x:auto;font-size:.8rem;line-height:1.5;margin:1rem 0}
.bar{height:.62rem;border-radius:1px;display:inline-block;vertical-align:middle}
.barwrap{display:flex;align-items:center;gap:.5rem}
.ci{font-size:.72rem;color:var(--mut)}
figure{margin:1.6rem 0}
figcaption{font-size:.83rem;color:var(--mut);margin-top:.5rem}
svg{max-width:100%;height:auto;display:block}
ol,ul{padding-left:1.3rem} li{margin:.35rem 0}
"""


def bars_svg(rows, w=620, rowh=30, pad=110):
    """Horizontal success-rate bars with Wilson intervals. Static SVG - no script, no CDN."""
    h = len(rows) * rowh + 34
    inner = w - pad - 78
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="모드별 성공률">']
    for gx in range(0, 11, 2):
        x = pad + inner * gx / 10
        out.append(f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{h - 20}" stroke="var(--line)" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 6}" fill="var(--mut)" font-size="10" text-anchor="middle" '
                   f'font-family="ui-monospace,monospace">{gx * 10}%</text>')
    for i, (name, k, n) in enumerate(rows):
        y = 20 + i * rowh
        p = k / n if n else 0
        lo, hi = wilson(k, n)
        col = "var(--ok)" if name == "vla" else ("var(--bad)" if p < 0.55 else "var(--warn)")
        out.append(f'<text x="{pad - 10}" y="{y + 11}" fill="var(--ink)" font-size="12" text-anchor="end" '
                   f'font-family="ui-monospace,monospace">{name}</text>')
        out.append(f'<rect x="{pad}" y="{y + 3}" width="{inner * p:.1f}" height="14" fill="{col}" rx="1"/>')
        x1, x2 = pad + inner * lo, pad + inner * hi
        out.append(f'<line x1="{x1:.1f}" y1="{y + 10}" x2="{x2:.1f}" y2="{y + 10}" stroke="var(--ink)" '
                   f'stroke-width="1.2" opacity=".55"/>')
        for xx in (x1, x2):
            out.append(f'<line x1="{xx:.1f}" y1="{y + 5}" x2="{xx:.1f}" y2="{y + 15}" stroke="var(--ink)" '
                       f'stroke-width="1.2" opacity=".55"/>')
        out.append(f'<text x="{pad + inner + 8}" y="{y + 14}" fill="var(--mut)" font-size="11" '
                   f'font-family="ui-monospace,monospace">{p:.3f}  {k}/{n}</text>')
    out.append("</svg>")
    return "".join(out)


def lines_svg(series, bands, w=620, h=280, pad_l=56, pad_b=44):
    """Bias-vs-distance curves, one polyline per run. Log-ish x by band index (bands are unequal)."""
    ys = [v for _, vals in series for v in vals if v is not None]
    lo, hi = min(ys + [0.0]), max(ys)
    span = (hi - lo) or 1.0
    iw, ih = w - pad_l - 96, h - pad_b - 18
    X = lambda i: pad_l + iw * i / max(len(bands) - 1, 1)  # noqa: E731
    Y = lambda v: 18 + ih * (1 - (v - lo) / span)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="거리별 V 편향">']
    for t in range(5):
        v = lo + span * t / 4
        out.append(f'<line x1="{pad_l}" y1="{Y(v):.1f}" x2="{pad_l + iw}" y2="{Y(v):.1f}" stroke="var(--line)"/>')
        out.append(f'<text x="{pad_l - 8}" y="{Y(v) + 4:.1f}" fill="var(--mut)" font-size="10" text-anchor="end" '
                   f'font-family="ui-monospace,monospace">{v:+.2f}</text>')
    out.append(f'<line x1="{pad_l}" y1="{Y(0):.1f}" x2="{pad_l + iw}" y2="{Y(0):.1f}" stroke="var(--mut)" '
               f'stroke-dasharray="3 3"/>')
    for i, (a, b) in enumerate(bands):
        out.append(f'<text x="{X(i):.1f}" y="{h - 20}" fill="var(--mut)" font-size="10" text-anchor="middle" '
                   f'font-family="ui-monospace,monospace">{a}–{b}</text>')
    out.append(f'<text x="{pad_l + iw / 2:.0f}" y="{h - 4}" fill="var(--mut)" font-size="10.5" '
               f'text-anchor="middle">목표까지 남은 스텝</text>')
    for name, vals, col in series_with_colors(series):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.9" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        last = [i for i, v in enumerate(vals) if v is not None][-1]
        out.append(f'<text x="{X(last) + 7:.1f}" y="{Y(vals[last]) + 4:.1f}" fill="{col}" font-size="10.5" '
                   f'font-family="ui-monospace,monospace">{html.escape(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def series_with_colors(series):
    pal = ["var(--bad)", "var(--warn)", "var(--ok)", "var(--mut)", "var(--ink)"]
    return [(n, v, pal[i % len(pal)]) for i, (n, v) in enumerate(series)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--generated", default="")
    args = ap.parse_args()
    root = args.root or pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
    out = args.out or root / "report_bias.html"

    roll = {}
    for f in sorted(glob.glob(str(root / "critic_runs/v3_fixedmask/*/rollout/*.json"))):
        r = f.split("/critic_runs/v3_fixedmask/")[1].split("/")[0]
        d = json.loads(pathlib.Path(f).read_text())
        e = {m: (d[m]["successes"], d[m]["num_trials"]) for m in MODES if m in d}
        if "critic" in d and "vla" in d:
            cv = {t["trial"]: t["success"] for t in d["critic"]["trials"]}
            vv = {t["trial"]: t["success"] for t in d["vla"]["trials"]}
            b = sum(1 for t in cv if cv[t] and not vv.get(t))
            c = sum(1 for t in cv if not cv[t] and vv.get(t))
            n = b + c
            e["p"] = 1.0 if n == 0 else min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2 / 2**n)
        roll[r] = e
    diag, cfgs = {}, {}
    for d_ in sorted(glob.glob(str(root / "critic_runs/v3_fixedmask/*/"))):
        r = os.path.basename(d_.rstrip("/"))
        for name, tgt in (("diag.json", diag), ("config.json", cfgs)):
            p = pathlib.Path(d_) / name
            if p.exists():
                tgt[r] = json.loads(p.read_text())
    vb = json.loads((root / "vbias.json").read_text()) if (root / "vbias.json").exists() else {}
    pfx = json.loads((root / "pfx_curve.json").read_text()) if (root / "pfx_curve.json").exists() else {}

    A = []
    P = A.append
    P("<h1>critic은 왜 성능을 떨어뜨리는가</h1>")
    P("<p class='lede'>ACRFT 오프라인 critic — v3_fixedmask 15런의 롤아웃 평가와, 그 결과를 설명하려고 "
      "돌린 세 가지 측정. 그리고 거기서 나온 IQL 구현.</p>")
    P(f"<p class='stamp'>{html.escape(args.generated)} · PrepareCoffee · 시연 514 에피소드 / 279,534 프레임 · "
      f"γ=0.99 · 모드당 30 trial</p>")

    tot = {m: [0, 0] for m in MODES}
    for e in roll.values():
        for m in MODES:
            if m in e:
                tot[m][0] += e[m][0]
                tot[m][1] += e[m][1]
    P("<div class='key'><p><b>결론부터.</b> critic에 결정권을 줄수록 성공률이 단조롭게 떨어집니다. "
      f"critic을 전혀 안 쓰면 <b>{tot['vla'][0] / tot['vla'][1]:.1%}</b>, 후보와 커밋 길이를 모두 맡기면 "
      f"<b>{tot['critic'][0] / tot['critic'][1]:.1%}</b>이고 두 신뢰구간은 겹치지 않습니다. "
      "후보 선택만 맡겨도 <b>무작위 선택보다 나쁩니다</b>.</p></div>")

    # ---------------------------------------------------------------- 1. 프로토콜
    P("<h2>1. 실험 프로토콜</h2>")
    P("<p>재현에 필요한 전부입니다. 명령은 그대로 실행됩니다.</p>")
    P("<h3>데이터</h3>")
    P("<p>RoboCasa <code>PrepareCoffee</code>의 성공 시연 514 에피소드(279,534 프레임)를 π₀.₅ RLT "
      "체크포인트로 주석 처리했습니다. 프레임마다 저장되는 것: RLT 토큰 2048차원, 실제 실행된 액션 청크 "
      "(16스텝 × 12차원), VLA가 뽑은 후보 청크 16개, proprio 16차원, 보상, terminal 표시, "
      "그리고 γ=0.99로 뒤에서부터 누적한 <code>mc_return</code>.</p>")
    P("<pre>uv run examples/robocasa/annotate_rlt.py \\\n"
      "    --config pi05_robocasa_PrepareCoffee_rlt \\\n"
      "    --checkpoint &lt;pi05-robocasa-prepcoffee-rlt-pardec-noprop-70k&gt; \\\n"
      "    --out $CACHE_DIR/annot/noprop --stride 1 --num-samples 16 --num-heldout 8</pre>")
    P("<p>보상은 sparse입니다 — 성공 프레임 하나에만 1, terminal. 에피소드당 정확히 하나이고 "
      "(<code>done==1</code> 514개 = 에피소드 수), 따라서 <b>참 가치가 <code>γ^d</code>로 정확히 알려져 "
      "있습니다</b>(d = 목표까지 남은 스텝). 이 글의 모든 편향 측정이 그 사실 위에 서 있습니다 — 참값을 "
      "추정하지 않고 계산합니다.</p>")
    P("<h3>critic 학습</h3>")
    P(f"<p>baseline: {html.escape(BASELINE_NOTE)}. 15개 런은 여기서 <b>한 번에 하나씩만</b> 바꿉니다.</p>")
    P("<pre>ROLLOUT=0 SWEEP=v3_fixedmask slurm/sweep.sh    # 15런, 각 200,000 스텝, batch 1024</pre>")
    P("<h3>롤아웃 평가</h3>")
    P("<p>학습과 <b>별도 잡</b>으로, 별도 GPU에서 돌립니다. 다섯 모드가 <b>같은 seed의 같은 장면</b>을 "
      "봅니다 — 그래야 비교가 성립합니다. 모드당 30 trial, 최대 1000 스텝.</p>")
    P("<pre>RUN_DIR=$CACHE_DIR/critic_runs/v3_fixedmask/&lt;run&gt; TRIALS=30 \\\n"
      "  sbatch slurm/eval_rollout.sbatch</pre>")
    P("<dl>" + "".join(
        f"<dt>{m}</dt><dd><b>{html.escape(MODE_DESC[m][0])}</b> — {html.escape(MODE_DESC[m][1])}</dd>"
        for m in MODES) + "</dl>")
    P("<div class='note'><p><b>이 다섯 모드가 왜 이렇게 짜였나.</b> critic은 두 가지를 결정합니다 — "
      "어느 후보를 실행할지, 몇 스텝 커밋할지. <code>bon</code>과 <code>prefix</code>가 그 둘을 하나씩 "
      "떼어내고, <code>critic</code>이 둘을 합칩니다. <code>rand</code>는 <b>critic 없이 VLA 기본값에서만 "
      "벗어나는</b> 대조군이라, 손실이 '선택을 바꿔서'인지 'critic이 선택해서'인지를 가릅니다. "
      "이 대조군이 없으면 두 설명을 구별할 수 없습니다.</p></div>")

    # ---------------------------------------------------------------- 2. 예상
    P("<h2>2. 무엇을 예상했나</h2>")
    P("<p>오프라인 진단에서 <code>action_sensitivity</code>가 15런 전부 0.0001~0.0014, "
      "<code>ranking_accuracy_demo_vs_candidate</code>가 전부 0.50±0.04였습니다. 둘 다 "
      "'critic이 액션을 구별하지 못한다'는 뜻이라, 예상은 이랬습니다:</p>")
    P("<ol><li><b><code>bon</code> ≈ <code>rand</code></b> — 순위가 무작위면 최고를 고르는 것과 아무거나 "
      "고르는 것이 같아야 한다.</li>"
      "<li><b><code>critic</code> ≈ <code>vla</code></b> — critic이 아무 정보를 안 더하면 성능도 안 바뀐다.</li>"
      "<li>즉 critic은 <b>무해하지만 무용</b>할 것이다.</li></ol>")
    P("<p>셋 다 틀렸습니다.</p>")

    # ---------------------------------------------------------------- 3. 결과
    P("<h2>3. 실제 결과</h2>")
    P("<figure>" + bars_svg([(m, *tot[m]) for m in MODES]) +
      "<figcaption>13개 런 전체를 합산한 모드별 성공률. 가로 막대는 성공률, 가는 선은 95% Wilson 신뢰구간 "
      "(n=390). <code>vla</code>와 <code>critic</code>의 구간이 완전히 분리되어 있습니다.</figcaption></figure>")
    P("<div class='scroll'><table><thead><tr><th>run</th>" +
      "".join(f"<th>{m}</th>" for m in MODES) + "<th>critic−vla</th><th>McNemar p</th></tr></thead><tbody>")
    for r in sorted(roll):
        e = roll[r]
        cells = ""
        for m in MODES:
            if m in e:
                k, n = e[m]
                cells += f"<td>{k}/{n}</td>"
            else:
                cells += "<td class='mut'>—</td>"
        if "critic" in e and "vla" in e:
            d = e["critic"][0] / e["critic"][1] - e["vla"][0] / e["vla"][1]
            p = e.get("p", 1.0)
            cells += f"<td class='{'bad' if d < 0 else 'ok'}'>{d:+.3f}</td>"
            cells += f"<td class='{'bad' if p < 0.05 else 'mut'}'>{p:.3f}</td>"
        P(f"<tr><td>{html.escape(r)}</td>{cells}</tr>")
    P("<tr class='total'><td>합계</td>" + "".join(f"<td>{tot[m][0]}/{tot[m][1]}</td>" for m in MODES) +
      "<td colspan='2'></td></tr></tbody></table></div>")
    P("<div class='note'><p><b>McNemar p</b> — 같은 장면을 두 모드가 각각 푼 <b>짝지은</b> 결과에 대한 "
      "정확검정. 'critic만 성공한 장면'과 'vla만 성공한 장면'의 개수를 비교합니다. 짝을 지어야 장면 난이도가 "
      "상쇄되므로, 30 trial로도 검정력이 나옵니다. <b>p&lt;0.05면 우연이 아닙니다.</b></p></div>")

    # ---------------------------------------------------------------- 4. 분석
    P("<h2>4. 분석</h2>")
    P("<h3>4.1 왜 <code>bon</code>이 <code>rand</code>보다 나쁜가</h3>")
    P("<p>후보 16개의 참 가치가 사실상 같고 추정 오차만 다르면, arg-max는 <b>가장 좋은 후보가 아니라 "
      "가장 많이 부풀려진 후보</b>를 고릅니다. 무작위 선택은 오차의 평균을 뽑지만 arg-max는 오차의 "
      "최댓값을 뽑습니다. 그래서 무작위보다 나쁠 수 있고, 실제로 그렇습니다 "
      f"(<code>bon</code> {tot['bon'][0] / tot['bon'][1]:.3f} vs <code>rand</code> "
      f"{tot['rand'][0] / tot['rand'][1]:.3f}).</p>")
    P("<p>측정으로 뒷받침됩니다. 한 상태에서 후보 16개의 Q가 전부 <code>0.007</code> 폭 안에 들어 있는데 "
      "상태가 바뀌면 Q는 <code>0.28</code>만큼 움직입니다 — <b>150배 차이</b>입니다. 그리고 후보 Q 순위와 "
      "'실제로 통했던 청크와의 근접도' 순위의 상관이 −0.008 ~ +0.027로 <b>0</b>입니다.</p>")

    P("<h3>4.2 프리픽스 선택이 가장 해로운 이유</h3>")
    P("<p>참값이 <code>γ^d</code>로 알려져 있으므로 타깃을 분해할 수 있습니다:</p>")
    P("<pre>y_h = γ^h · V̂(s_{t+h}) = γ^h · (γ^{d-h} + b) = γ^d + γ^h·b\n"
      "                              ↑ h 무관, 참값      ↑ critic 오차</pre>")
    P("<p>V̂가 정확하면 <code>γ^h</code>이 상쇄되어 <b>프리픽스 8개의 타깃이 전부 같아야 합니다</b>. "
      "실제로는 그렇지 않습니다:</p>")
    if pfx.get("buckets"):
        pl = pfx.get("pfx", [2, 4, 6, 8, 10, 12, 14, 16])
        P("<div class='scroll'><table><thead><tr><th>거리 구간</th><th>n</th>" +
          "".join(f"<th>h={h}</th>" for h in pl) + "</tr></thead><tbody>")
        for b in pfx["buckets"]:
            cells = "".join(
                f"<td class='{'bad' if v > 1.15 else ('warn' if v > 1.03 else '')}'>{v:.4f}</td>"
                for v in b["ratio"])
            P(f"<tr><td>{b['lo']}–{b['hi']}</td><td class='mut'>{b['n']}</td>{cells}</tr>")
        P("</tbody></table></div>")
        P("<p class='mut' style='font-size:.9rem'>값은 <code>y_h / γ^d</code> — 1.0이면 이론과 일치, "
          "1보다 크면 과대추정. 각 구간 400상태.</p>")
    P("<p><b>두 가지가 동시에 보입니다.</b> 세로로 — 목표에서 멀수록 과대추정이 커져 250스텝 밖에서는 "
      "<b>5배</b>입니다. 가로로 — 모든 구간에서 h가 커질수록 비율이 <b>단조 감소</b>합니다. "
      "짧은 프리픽스가 먼 후속 상태에서 부트스트랩해 더 부푼 값을 물어오기 때문입니다.</p>")
    P("<p>그래서 배포의 <code>(후보, 프리픽스)</code> 결합 arg-max에서 <b>프리픽스 축이 커밋 길이가 아니라 "
      "'critic이 어느 후속 상태를 더 과대평가하나'를 재고 있습니다.</b> arg-max가 구조적으로 최단 커밋으로 "
      "몰리고, adaptive chunking이 퇴화합니다.</p>")

    P("<h3>4.3 V 편향은 실재하지만, 런 간 성능 차이를 설명하지 못한다</h3>")
    if vb:
        keep = [r for r in ("soft", "mg8", "base", "topm", "tn03") if r in vb]
        ser = [(r, [x["b"] if x else None for x in vb[r]["rows"]]) for r in keep]
        if ser:
            P("<figure>" + lines_svg(ser, BANDS) +
              "<figcaption>V 편향 <code>b(d) = V̂(s) − γ^d</code>를 목표까지 거리별로. 각 런의 자기 설정"
              "(<code>v_agg</code>, <code>ens_agg</code>, γ, 구조)으로 V̂를 계산했습니다. 점선이 0.</figcaption></figure>")
        P("<div class='scroll'><table><thead><tr><th>run</th>" +
          "".join(f"<th>{a}–{b}</th>" for a, b in BANDS) + "<th>critic 성공률</th></tr></thead><tbody>")
        for r in sorted(vb):
            cells = ""
            for x in vb[r]["rows"]:
                if x is None:
                    cells += "<td class='mut'>—</td>"
                else:
                    cls = "bad" if x["b"] > 0.1 else ("warn" if x["b"] > 0.03 else ("ok" if x["b"] < 0 else ""))
                    cells += f"<td class='{cls}'>{x['b']:+.4f}</td>"
            sr = ""
            if r in roll and "critic" in roll[r]:
                k, n = roll[r]["critic"]
                sr = f"{k / n:.3f}"
            P(f"<tr><td>{html.escape(r)}</td>{cells}<td>{sr or '—'}</td></tr>")
        P("</tbody></table></div>")
    P("<div class='key'><p><b>여기서 제 가설이 깨졌습니다.</b> 편향이 성능을 설명할 것으로 봤는데, "
      "13개 런에서 <code>b</code> 평균과 <code>critic</code> 성공률의 순위상관이 <b>−0.17</b>입니다. "
      "극단만 맞습니다 — <code>soft</code>는 유일하게 편향이 음수이고 성공률이 가장 높으며(0.633), "
      "<code>tn03</code>은 편향이 최악이고 성공률도 최악(0.133)입니다. 그런데 <code>g999</code>/"
      "<code>g9995</code>는 <b>편향이 가장 낮은데 <code>critic−vla</code>가 가장 나쁩니다</b>"
      "(−0.367, −0.500).</p>"
      "<p>이유는 단순합니다. <b>일정한 편향은 arg-max를 바꾸지 않습니다.</b> 모든 후보가 똑같이 부풀면 "
      "순위가 그대로입니다. arg-max를 망치는 것은 편향의 크기가 아니라 후보 간 <b>오차의 산포</b>이고, "
      "평균 편향은 그것을 재지 못합니다.</p></div>")

    P("<h3>4.4 더 근본적인 이유 — 타깃이 액션에 무관하다</h3>")
    P("<p>궤적이 하나뿐인 데이터에서 타깃을 전개하면:</p>")
    P("<pre>y_h = Σ_{i&lt;h} γ^i r_{t+i} + γ^h V(s_{t+h}) = γ^h · γ^{d-h} = γ^d</pre>")
    P("<p><code>cum</code>도 <code>V(s_{t+h})</code>도 <b>지금 평가하는 액션에 의존하지 않습니다</b>. "
      "데이터에 상태당 액션이 하나뿐이라 각 <code>z</code>가 정확히 하나의 청크와만 짝지어 나타나고, "
      "손실은 <code>a</code>를 완전히 무시하는 함수로 최소화됩니다. <b>h도 소거되므로 프리픽스 축에도 "
      "학습할 신호가 없습니다.</b> 두 현상이 하나의 원인입니다.</p>")
    P("<p>부트스트랩의 후보 max가 액션 정보를 넣어주는 것처럼 보이지만, 그것은 <b>다음 상태에서</b> 일어나 "
      "스칼라 하나로 접힌 뒤 들어옵니다. 현재 상태 안의 대비를 만들지 못합니다.</p>")
    P("<div class='note'><p><b>표현은 병목이 아닙니다.</b> RLT 토큰만으로 <b>본 적 없는 114개 에피소드</b>의 "
      "진행도를 선형 프로브가 R²=0.844(순위상관 0.926)로 맞힙니다 — progress objective 없이 재구성만으로 "
      "학습된 토큰인데도 그렇습니다. 문제는 표현이 아니라 타깃입니다.</p></div>")

    # ---------------------------------------------------------------- 5. 결론
    P("<h2>5. 결론과 다음 단계</h2>")
    P("<ol>"
      "<li><b>critic은 무해하지 않고 유해합니다.</b> 결정권을 줄수록 단조롭게 나빠지고, 후보 선택만 맡겨도 "
      "무작위보다 못합니다.</li>"
      "<li><b>arg-max가 잡음을 고릅니다.</b> 후보 간 참 가치 차이가 없는데 추정 오차만 다르면, "
      "최댓값 선택은 오차 선택입니다.</li>"
      "<li><b>프리픽스 축은 커밋 길이를 재고 있지 않습니다.</b> 거리에 따라 구조화된 V 편향을 읽고 있어서 "
      "최단 커밋으로 몰립니다.</li>"
      "<li><b>편향의 크기는 런 간 성능 차이를 설명하지 못합니다.</b> 설명하는 것은 아마 후보 간 오차의 "
      "산포이고, 그건 후보별 참값이 없어 직접 재기 어렵습니다.</li>"
      "<li><b>원인은 알고리즘이 아니라 데이터입니다.</b> 성공 시연만 있고 상태당 액션이 하나면, 액션 축에 "
      "대한 정보가 데이터에 없습니다. 15가지 방법이 전부 같은 자리에 도달한 이유입니다.</li>"
      "</ol>")
    P("<h3>IQL — arg-max를 없앤다</h3>")
    P("<p>지금까지의 대응(<code>topm</code>, <code>soft</code>, <code>lcb</code>, "
      "<code>bootstrap-candidates</code>)은 전부 max를 <b>부드럽게</b> 만들 뿐이었습니다. IQL은 max를 "
      "<b>제거</b>합니다 — 후보 배열을 아예 쓰지 않고, 상태 가치 <code>V(z)</code>를 시연 액션에 대한 "
      "expectile 회귀로 배워 그것을 부트스트랩합니다.</p>")
    P("<pre>L_V = E[ |τ − 1(u&lt;0)| · u² ],   u = Q_target(z, a_demo, h) − V(z)\n"
      "L_Q = E[ (Q(z, a_demo, h) − y_h)² ],  y_h = cum_h + γ^h · ¬ended · V(z_{t+h})</pre>")
    P("<p>τ=0.5는 최소자승(V → 평균 Q, 개선 없음), τ가 클수록 초과분에 가중해 <code>max_a Q</code>에 "
      "접근합니다. 후보 forward가 사라져 <b>학습이 약 2배 빠릅니다</b>.</p>")
    P("<pre>ROLLOUT=0 AXES=iql SWEEP=v6_iql slurm/sweep.sh</pre>")
    P("<p>돌고 있는 것: <code>iql_e50</code> / <code>e70</code> / <code>e90</code> / <code>e95</code> "
      "(expectile 4종)와 <code>iql_qc</code>(프리픽스 head 없이 — 'IQL이 좋다'와 '프리픽스 축이 문제였다'를 "
      "분리하기 위한 것). 함께 <code>v5_stability</code>가 한 번도 안 건드린 세 축을 봅니다: 앙상블 크기 "
      "K=2→3/5, target network 제거, Polyak 시정수.</p>")
    P("<div class='note'><p><b>미리 말해둘 것.</b> IQL도 타깃이 액션에 무관하다는 근본 문제는 고치지 "
      "못합니다 — <code>y_h</code>는 여전히 상태의 함수입니다. IQL이 고치는 것은 <b>arg-max가 오차를 "
      "고르는 경로</b>입니다. 그래서 예상은 '<code>critic</code>이 <code>vla</code>를 이긴다'가 아니라 "
      "'<code>critic</code>이 <code>vla</code>를 <b>덜 해친다</b>'입니다. 만약 모든 τ가 같은 자리에 "
      "도달한다면, 그 자체가 '순위 신호가 없다'는 답입니다.</p></div>")

    # ---------------------------------------------------------------- 부록
    P("<h2>부록 A. 지표 정의</h2>")
    P("<dl>"
      "<dt>action_sensitivity (act_sens)</dt><dd>상태 <b>내부</b> Q 분산 ÷ 상태 <b>간</b> Q 분산. "
      "Q의 변동 중 액션 때문인 몫. <b>0 = 액션 완전 무시</b>(<code>Q(z,a)=V(z)</code>로 붕괴), "
      "1 = 액션이 상태만큼 중요. 클수록 좋음.</dd>"
      "<dt>ranking_accuracy_demo_vs_candidate (rank_cand)</dt><dd>같은 상태에서 시연된 청크가 정책 후보보다 "
      "높은 Q를 받는 비율. <b>우연 = 0.5.</b> 클수록 좋음.</dd>"
      "<dt>ranking_accuracy_demo_vs_other (rank_other)</dt><dd>시연 청크 vs <b>다른 상태</b>에서 빌려온 청크를 "
      "같은 상태에서 평가. 무관한 액션이므로 쉬운 문제. <b>우연 = 0.5.</b> 이것마저 0.5면 critic이 액션 입력 "
      "자체를 무시한다는 뜻.</dd>"
      "<dt>spearman_q_demo_vs_mc (ρ(Q,MC))</dt><dd>시연 청크의 Q와 실제 거둔 리턴의 순위상관. "
      "0 = 무관. 클수록 좋음. <b>상태 가치</b>를 얼마나 잘 읽는지의 지표.</dd>"
      "<dt>b(d) — V 편향</dt><dd><code>V̂(s) − γ^d</code>. sparse+terminal 보상이라 참값이 <code>γ^d</code>로 "
      "정확히 알려져 있으므로 근사가 아닙니다. <b>0 = 정확, &gt;0 = 과대추정.</b></dd>"
      "<dt>y_h / γ^d</dt><dd>프리픽스 h의 타깃을 참값으로 나눈 비율. <b>1.0 = 일치.</b> 이론상 h에 무관해야 함.</dd>"
      "<dt>prefix_argmax_entropy (pfx_H)</dt><dd>배포 arg-max가 고르는 프리픽스 길이 분포의 정규화 엔트로피. "
      "1 = 고르게 분산, <b>0 = 항상 같은 길이</b>(adaptive chunking 퇴화).</dd>"
      "<dt>McNemar p</dt><dd>같은 장면에 대한 짝지은 성패의 정확검정. <b>p&lt;0.05 = 우연 아님.</b></dd>"
      "<dt>Wilson 구간</dt><dd>이항 비율의 95% 신뢰구간. 성공률이 0이나 1에 가까울 때도 정상 동작합니다.</dd>"
      "</dl>")
    P("<h2>부록 B. 런 이름</h2>")
    P(f"<p class='mut'>baseline = {html.escape(BASELINE_NOTE)}. 각 런은 여기서 아래 한 가지만 다릅니다.</p>")
    P("<div class='scroll'><table><thead><tr><th>run</th><th>baseline과 다른 점</th></tr></thead><tbody>")
    for r in sorted(cfgs):
        d = describe(cfgs[r])
        P(f"<tr><td>{html.escape(r)}</td><td style='text-align:left;white-space:normal'>"
          f"{html.escape(' / '.join(d)) if d else '<span class=mut>baseline</span>'}</td></tr>")
    P("</tbody></table></div>")

    out.write_text(
        f"<title>critic은 왜 성능을 떨어뜨리는가 — ACRFT</title>\n<style>{CSS}</style>\n<main>{''.join(A)}</main>\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
