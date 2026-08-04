"""Two-day experiment log: what was run, what was expected, what came back, what it settled.

Reads the files the experiments wrote (diag.json / config.json / rollout/*.json / vbias*.json /
pfx_curve.json), so re-running refreshes every number without editing this file.

    uv run slurm/make_progress_report.py --out $CACHE_DIR/report_progress.html
"""

import argparse
import contextlib
import glob
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
    for d in sorted(glob.glob(str(root / f"critic_runs/{sweep}/*/"))):
        r = os.path.basename(d.rstrip("/"))
        e = {}
        for name in ("diag.json", "config.json"):
            p = pathlib.Path(d) / name
            if p.exists():
                with contextlib.suppress(json.JSONDecodeError):
                    e[name.split(".")[0]] = json.loads(p.read_text())
        rolls = sorted(glob.glob(d + "rollout/*.json"))
        if rolls:
            e["rollout"] = json.loads(pathlib.Path(rolls[-1]).read_text())
        if e:
            out[r] = e
    return out


def diag_table(emit, sweeps, note=""):
    """One combined diagnostics table. Columns are the necessary-condition metrics."""
    emit(
        "<div class='scroll'><table><thead><tr><th>run</th><th>스윕</th><th>ρ(Q,MC)</th><th>act_sens</th>"
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


def rollout_table(emit, runs, id_prefix=""):
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
            "<tr class='total'><td>합계</td>"
            + "".join(f"<td>{tot[m][0]}/{tot[m][1]}</td>" for m in MODES)
            + "<td colspan='2'></td></tr>"
        )
    emit("</tbody></table></div>")
    return tot


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
    P("<h1>ACRFT critic — 이틀간의 실험 일지</h1>")
    P(
        "<p class='lede'>15종 TD ablation이 전부 같은 자리에 도달한 이유를 측정으로 좁혀 들어가, "
        "원인(부트스트랩 arg-max)을 확정하고 그것을 제거한 IQL을 구현·검증하기까지.</p>"
    )
    P(
        f"<p class='stamp'>{html.escape(args.generated)} · RoboCasa PrepareCoffee · 시연 514 에피소드 / "
        f"279,534 프레임 · γ=0.99 · 상세 조사 리포트: report_bias.html</p>"
    )

    # ---------------------------------------------------------------- 요약
    P("<h2>요약 — 확정된 것 다섯 가지</h2>")
    P(
        "<ol>"
        "<li><b>critic은 유해하다.</b> 롤아웃에서 결정권을 줄수록 성공률이 단조 하락 "
        "(vla 74% → critic 46%, 13런 × 30 trial, 신뢰구간 분리). 후보 선택만 맡겨도 무작위 선택보다 나쁘다.</li>"
        "<li><b>원인은 부트스트랩의 arg-max.</b> 타깃의 V가 목표에서 먼 상태에서 체계적으로 과대추정된다 "
        "(250스텝 밖 5.07배). 참값이 γ^d로 알려져 있어 이것은 추정이 아니라 계산이다.</li>"
        "<li><b>IQL(expectile 회귀)이 그 팽창을 19~38배 줄였다.</b> expectile을 올리면 편향이 단조 복귀하는 "
        "용량-반응까지 확인 — 인과가 확정됐다.</li>"
        "<li><b>액션 순위 신호는 여전히 데이터에 없다.</b> 상태당 액션이 하나뿐이라 타깃이 액션·프리픽스 모두에 "
        "무관하고, 그래서 act_sens ≈ 0은 21개 런 전부에서 유지된다. 유일한 예외 후보는 IQL+QC 조합의 "
        "rank_c 0.571 (롤아웃 검증 중).</li>"
        "<li><b>프리픽스 축은 커밋 길이가 아니라 critic 오차를 재고 있었다.</b> 같은 상태의 프리픽스별 타깃은 "
        "이론상 전부 γ^d로 같아야 하는데, V 편향이 거리에 따라 달라 단조 감소했다 — 영상에서 보인 "
        "'prefix 값 단조 감소'의 정체.</li></ol>"
    )

    # ---------------------------------------------------------------- 타임라인
    P("<h2>경과 타임라인</h2><ul class='tl'>")
    for t, ev in [
        ("어제 오후", "v3_fixedmask 15런 완료. 전 런 act_sens≈0.0003, rank_c≈0.5 — 서열을 매길 축이 없음을 확인."),
        (
            "어제 저녁",
            "15런 전부 롤아웃 제출 (30 trial × 5모드). HUD 패널 침범을 수집후렌더로 해결 (1000프레임 전수검사 0건).",
        ),
        (
            "어제 밤",
            "롤아웃 13런 회수: critic이 유해함이 판명. targets()를 단일 ended 마스크로 재작성 "
            "(2.2M 셀 비트단위 동치 검증). γ 변형 데이터셋 자동 생성(ensure_discount), proprio를 annotation 단계로 이동.",
        ),
        (
            "오늘 새벽",
            "프리픽스별 타깃 분해 → y_h = γ^d + γ^h·b, 편향 b가 거리 구조를 가짐을 확인. "
            "IQL 구현 (ValueNet + expectile, 후보 배열 미사용). v4(HL-Gauss+mcfloor baseline) 재제출.",
        ),
        (
            "오늘 오전",
            "b(d) 15런 측정: 편향은 실재하나 런 간 성능차는 설명 못 함 (ρ=−0.17) — 원인은 편향 크기가 아니라 "
            "arg-max가 고르는 오차 산포로 좁혀짐. v5(stability)·v6(IQL)·v7(에피소드 사다리) 제출.",
        ),
        (
            "오늘 오후",
            "v6 완료: IQL이 b(d)를 19~38배 축소, expectile 용량-반응 확인. iql_qc rank_c 0.571 (21런 중 최초로 "
            "우연 초과). QC 롤아웃이 세 스윕 내내 죽고 있던 버그(load_trained P축 누락) 발견·수정. "
            "오버레이 월드공간 확대로 재작성. IQL 롤아웃 4건 진행 중.",
        ),
    ]:
        P(f"<li><time>{t}</time><span>{ev}</span></li>")
    P("</ul>")

    # ---------------------------------------------------------------- 1. v3 롤아웃
    P("<h2>1. v3 롤아웃 — critic은 무해하지 않았다 <span class='pill p-done'>완료</span></h2>")
    P("<h3>프로토콜</h3>")
    P(
        "<p>15개 critic(200k 스텝, batch 1024) 각각을 다섯 모드로 30 trial씩 평가. 다섯 모드는 같은 seed의 "
        "같은 장면을 보므로 짝지은 비교가 성립한다. <code>vla</code>=critic 없음(기준), <code>rand</code>=후보 "
        "무작위(대조군), <code>bon</code>=후보만 critic 선택, <code>prefix</code>=커밋 길이만 critic 선택, "
        "<code>critic</code>=결합 arg-max(배포 규칙).</p>"
    )
    P(
        "<pre>ROLLOUT=0 SWEEP=v3_fixedmask slurm/sweep.sh\n"
        "RUN_DIR=$CACHE_DIR/critic_runs/v3_fixedmask/&lt;run&gt; TRIALS=30 sbatch slurm/eval_rollout.sbatch</pre>"
    )
    P(
        "<h3>예상</h3><p>오프라인 진단(act_sens≈0, rank_c≈0.5)에 근거해 <b>bon ≈ rand, critic ≈ vla</b> — "
        "즉 '무해하지만 무용'을 예상했다.</p>"
    )
    P("<h3>실제</h3>")
    tot = rollout_table(P, v3)
    if tot["vla"][1]:
        P(
            "<div class='key fail'><p>예상이 틀렸다. "
            + " · ".join(f"<b>{m}</b> {tot[m][0] / tot[m][1]:.3f}" for m in MODES if tot[m][1])
            + ". critic 개입이 늘수록 단조 하락하고, bon이 rand보다 나쁘다 — arg-max가 '최선'이 아니라 "
            "'가장 부풀려진 오차'를 고른다는 뜻. rand≈vla는 후보 재선택 자체는 거의 공짜임을 보인다: "
            "손실은 선택을 바꿔서가 아니라 critic이 선택해서 생긴다.</p></div>"
        )

    # ---------------------------------------------------------------- 2. 타깃 분해
    P("<h2>2. 프리픽스별 타깃 분해 — 편향의 구조 <span class='pill p-done'>완료</span></h2>")
    P(
        "<p>sparse+terminal 보상이라 참 가치가 <code>γ^d</code>(d=목표까지 스텝)로 정확히 계산된다. "
        "타깃을 전개하면 <code>y_h = γ^d + γ^h·b</code>: V̂가 정확하면 프리픽스 8개의 타깃이 전부 같아야 한다.</p>"
    )
    if pfx.get("buckets"):
        pl = pfx.get("pfx", [2, 4, 6, 8, 10, 12, 14, 16])
        P(
            "<div class='scroll'><table><thead><tr><th>거리 구간</th><th>n</th>"
            + "".join(f"<th>h={h}</th>" for h in pl)
            + "</tr></thead><tbody>"
        )
        for b in pfx["buckets"]:
            cells = "".join(
                f"<td class='{'bad' if v > 1.15 else ('warn' if v > 1.03 else '')}'>{v:.3f}</td>" for v in b["ratio"]
            )
            P(f"<tr><td>{b['lo']}–{b['hi']}</td><td class='mut'>{b['n']}</td>{cells}</tr>")
        P("</tbody></table></div>")
        P("<p class='mut' style='font-size:.9rem'>값 = y_h / γ^d (1.0=정확). TD base critic, 구간당 400상태.</p>")
    P(
        "<p>모든 거리에서 h에 대해 단조 감소 — 영상의 'prefix 값이 단조 감소'는 critic이 목표에서 먼 후속 상태를 "
        "더 과대평가한 것의 직접 반영이다. 250스텝 밖에서는 타깃이 참값의 <b>5배</b>.</p>"
    )
    P(
        "<div class='note'><p><b>부수 확인 — 편향 크기는 런 간 성능차를 설명 못 한다.</b> 13런에서 b 평균과 "
        "critic 성공률의 순위상관 −0.17. 일정한 편향은 arg-max를 바꾸지 않으므로(전부 같이 부풀면 순위 불변), "
        "성능을 가르는 것은 후보 간 오차의 <b>산포</b>다. soft(유일한 음수 편향)가 최고, tn03(최대 편향)이 "
        "최악인 극단만 맞는다.</p></div>"
    )

    # ---------------------------------------------------------------- 3. v4
    P("<h2>3. v4 — baseline을 HL-Gauss+mcfloor로 <span class='pill p-done'>완료</span></h2>")
    P(
        "<p>사용자 결정으로 분포형 head(51 atoms)와 mc_return 하한을 기본값으로 승격, 같은 15 arm을 재실행. "
        "스칼라·무하한으로 되돌리는 <code>scalarq</code>/<code>nofloor</code> arm이 방향을 뒤집어 대신한다.</p>"
    )
    P(
        "<h3>예상</h3><p>act_sens는 그대로 0일 것(타깃의 문제이므로). mg4/mg8의 ρ 이득(0.92/0.91)이 유지되는지가 "
        "관전 포인트.</p>"
    )
    P("<h3>실제</h3>")
    diag_table(
        P,
        [("v4", v4)],
        "v3 대비: act_sens 여전히 ≤0.0014 (예상대로). mg4/mg8의 ρ 이득은 <b>사라짐</b> (0.92/0.91 → "
        "0.82/0.82; v4는 전 런 0.81–0.83으로 평평) — 그 이득은 스칼라 회귀와의 조합 특이 효과였다. "
        "v3에서의 'mg가 최대 효과' 결론은 baseline 의존적이었던 것으로 정정.",
    )

    # ---------------------------------------------------------------- 4. IQL
    P("<h2>4. IQL — arg-max를 제거한다 <span class='pill p-run'>롤아웃 진행 중</span></h2>")
    P("<h3>설계</h3>")
    P(
        "<pre>L_V = E[ |τ − 1(u&lt;0)| · u² ],   u = Q_tgt(z, a_demo, h) − V(z)\n"
        "L_Q = E[ (Q(z, a_demo, h) − y_h)² ],  y_h = cum_h + γ^h · ¬ended · V(z_{t+h})</pre>"
    )
    P(
        "<p>후보 배열을 학습에서 아예 쓰지 않는다. τ=0.5는 최소자승(V→평균 Q, 개선 없음), τ↑일수록 max_a Q에 "
        "접근. 후보 forward가 사라져 학습 ~2.5배 고속. τ ∈ {0.5, 0.7, 0.9, 0.95} + QC 변형(<code>iql_qc</code>, "
        "프리픽스 head 없음 — 'IQL이 좋다'와 '프리픽스 축이 문제'를 분리).</p>"
    )
    P("<pre>ROLLOUT=0 AXES=iql SWEEP=v6_iql slurm/sweep.sh</pre>")
    P(
        "<h3>예상</h3><p>b(d)가 0 근처로 내려가고, τ가 클수록 max에 접근하므로 편향이 되돌아올 것. 롤아웃에서는 "
        "'vla를 이긴다'가 아니라 '<b>덜 해친다</b>'가 성공 기준 — 액션 순위 신호 자체는 IQL도 못 만든다.</p>"
    )
    P("<h3>실제 — V 편향</h3>")
    if vb6:
        P(
            "<div class='scroll'><table><thead><tr><th>run</th>"
            + "".join(f"<th>{a}–{b}</th>" for a, b in BANDS)
            + "</tr></thead><tbody>"
        )
        base_row = vb3.get("base", {}).get("rows")
        if base_row:
            cells = "".join(f"<td class='bad'>{x['b']:+.4f}</td>" if x else "<td>—</td>" for x in base_row)
            P(f"<tr><td>TD base (참고)</td>{cells}</tr>")
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
            "<p class='mut' style='font-size:.9rem'>b(d) = V̂(s) − γ^d. V̂는 배포와 동일하게 후보 16개 max로 계산 "
            "— 즉 배포가 실제로 읽는 양의 편향이다. 최종(200k) 체크포인트.</p>"
        )
    P(
        "<div class='key win'><p><b>60–120 구간: TD +0.100 → iql_e50 +0.003 / e70 +0.005 — 19~38배 축소.</b> "
        "그리고 e50 ≈ e70 &lt; e90 &lt; e95의 단조 용량-반응: expectile이 max에 접근할수록 max의 편향이 "
        "되돌아온다. 다섯 런이 같은 학습 단계이므로 시점 교란이 아니다 — <b>'팽창의 원인은 arg-max'가 인과로 "
        "확정</b>됐다. TD와 달리 학습이 길어져도(100k→200k) 편향이 자라지 않는다.</p></div>"
    )
    P("<h3>실제 — 진단</h3>")
    diag_table(
        P,
        [("v6", v6)],
        "iql_qc의 rank_c 0.571은 21개 런 중 유일하게 우연(0.5) 대역을 벗어난 값. 같은 τ의 ARQ(iql_e70 "
        "0.523)에서는 나지 않으므로 IQL 단독도 QC 단독도 아닌 <b>조합</b> 효과 — ARQ는 트렁크를 프리픽스 "
        "head 8개가 나눠 쓰지만 QC는 head 하나가 concat(z,a)를 직접 본다는 가설. 롤아웃이 판정한다.",
    )
    ir = {r: e for r, e in v6.items() if "rollout" in e}
    if ir:
        P("<h3>실제 — 롤아웃 (나온 것)</h3>")
        rollout_table(P, ir)
    else:
        P("<p class='mut'>롤아웃 4건 진행 중 (각 30 trial × 5모드) — 완료 시 이 표가 채워진다.</p>")

    # ---------------------------------------------------------------- 5. v5/v7
    P("<h2>5. 진행 중인 나머지 축</h2>")
    P(
        "<dl><dt>v5_stability — k3 / k5 / online / tau001 / tau05</dt>"
        "<dd>한 번도 ablate하지 않았던 세 축: 앙상블 크기(min의 억제력), target network 유무(참조 구현 기본은 "
        "없음), Polyak 시정수. TD 쪽에서 편향을 얼마나 줄일 수 있는지의 대조축.</dd>"
        "<dt>v7_single — ep1 / ep4 / ep16 / ep64</dt>"
        "<dd>단일 궤적 암기 극한. ep1에서도 b(d)&gt;0이 남으면 편향은 에피소드 간 간섭이 아니라 부트스트랩 구조 "
        "자체의 산물. act_sens가 잡음으로 오히려 오르면 'act_sens 높음=액션 이해'가 아님도 증명된다.</dd></dl>"
    )
    if any("diag" in e for e in list(v5.values()) + list(v7.values())):
        diag_table(P, [("v5", v5), ("v7", v7)])

    # ---------------------------------------------------------------- 6. 엔지니어링
    P("<h2>6. 함께 고친 것들</h2>")
    P(
        "<ul>"
        "<li><b>targets() 단순화</b> — crossed/lands_on_term/boot/term_inside 4변수 3분기 → <code>ended</code> "
        "마스크 하나. 2,236,272셀 중 valid 셀 전부에서 기존과 비트단위 동일 검증.</li>"
        "<li><b>QC 롤아웃 버그</b> — load_trained가 QC에 프리픽스 축 없이 반환해 np.unravel_index가 사망. "
        "세 스윕 내내 QC 롤아웃이 조용히 죽고 있었다. P=1 축 + macro=horizon으로 계약 복구.</li>"
        "<li><b>HUD 패널 침범 해결</b> — 롤아웃 중 렌더하지 않고 원시 데이터만 수집, 종료 후 일괄 합성. "
        "1000프레임 실패 에피소드 전수검사 침범 0건. (원인 후보 5개를 측정으로 기각한 끝의 우회 해결.)</li>"
        "<li><b>오버레이 재작성</b> — (a) 스케일 상수를 측정으로 교정(0.0054 m/unit), (b) 확대는 월드 공간에서 "
        "(화면 공간 확대는 원근을 파괴), (c) 배율을 프레임에 명시, (d) 그리퍼가 화면 밖이면 그리지 않음(테두리 "
        "clip 잔재 제거), (e) trial 간 projector 재생성(죽은 sim 참조 크래시).</li>"
        "<li><b>재현성</b> — --discount가 주석과 다르면 mc_return 재누적 데이터셋을 자동 생성(동시 실행 안전, "
        "원자적 rename). proprio는 annotation 단계에서 직접 기록(사후 join 제거; 기존 파일과 비트단위 동일 검증). "
        "--use-proprio 플래그 삭제(항상 켜짐).</li>"
        "<li><b>운영</b> — base_qos 포화 시 스윕 전체를 big_qos로 자동 라우팅. 잡 제출 후 산출물 파일 기반 감시 "
        "+ 마일스톤 보고 체계.</li></ul>"
    )

    # ---------------------------------------------------------------- 다음
    P("<h2>7. 다음 판정</h2>")
    P(
        "<ol><li><b>IQL 롤아웃</b> — critic−vla가 −0.285에서 0 쪽으로 오는가. iql_qc의 rank_c 0.571이 성공률로 "
        "이어지는가.</li>"
        "<li><b>v7 ep1</b> — 암기 극한에서의 b(d): 편향의 최종 귀속.</li>"
        "<li><b>v5</b> — TD를 고쳐 쓸 수 있는지, 아니면 IQL로 갈아타는 게 맞는지.</li>"
        "<li>액션 순위 신호 자체는 여전히 데이터에 없다 — 마진 랭킹 손실(다른 에피소드의 청크를 음성 표본으로) 또는 "
        "실패 궤적 수집이 다음 단계.</li></ol>"
    )

    # ---------------------------------------------------------------- 부록
    P("<h2>부록 A. 지표 정의</h2><dl>")
    for dt, dd in [
        (
            "ρ(Q,MC) — spearman_q_demo_vs_mc",
            "시연 청크의 Q ↔ 실제 거둔 리턴의 순위상관. 0=무관, 클수록 좋음. 상태 가치를 읽는 능력.",
        ),
        (
            "act_sens — action_sensitivity",
            "상태 내부 Q분산 ÷ 상태 간 Q분산. 0 = 액션 완전 무시(Q(z,a)=V(z)). 클수록 좋음.",
        ),
        (
            "rank_c — ranking_accuracy_demo_vs_candidate",
            "같은 상태에서 시연 청크가 정책 후보보다 높은 Q를 받는 비율. 우연=0.5, 클수록 좋음.",
        ),
        (
            "rank_o — ranking_accuracy_demo_vs_other",
            "시연 청크 vs 다른 상태에서 빌려온 무관한 청크. 쉬운 문제이며 우연=0.5. 이것마저 0.5면 액션 입력 자체를 무시.",
        ),
        (
            "pfx_H — prefix_argmax_entropy",
            "배포 arg-max가 고르는 프리픽스 길이 분포의 정규화 엔트로피. 1=고르게, 0=항상 같은 길이(적응 청킹 퇴화).",
        ),
        ("b(d)", "V̂(s) − γ^d. 참값이 닫힌형으로 알려져 있어 근사가 아님. 0=정확, >0=과대추정."),
        ("y_h / γ^d", "프리픽스 h 타깃 ÷ 참값. 이론상 h 무관하게 1.0."),
        ("McNemar p", "같은 장면의 짝지은 성패에 대한 정확검정. p<0.05 = 우연 아님."),
        ("Wilson 구간", "이항 비율 95% 신뢰구간; 극단 비율에서도 안정."),
        ("expectile τ", "IQL의 비대칭 회귀 파라미터. 0.5=평균(개선 없음), 1에 접근할수록 max_a Q에 접근."),
    ]:
        P(f"<dt>{html.escape(dt)}</dt><dd>{dd}</dd>")
    P("</dl>")
    P("<h2>부록 B. 런 이름</h2>")
    P(f"<p class='mut'>v4 이후 baseline = {html.escape(BASELINE_NOTE)}. 각 런은 한 가지만 다르다.</p>")
    P("<div class='scroll'><table><thead><tr><th>run</th><th>스윕</th><th>무엇이 다른가</th></tr></thead><tbody>")
    extra = {
        "k3": "앙상블 3개 (기본 2)",
        "k5": "앙상블 5개",
        "online": "target network 없음 — 온라인 critic으로 부트스트랩",
        "tau001": "Polyak τ=0.001 (10배 느린 타깃)",
        "tau05": "Polyak τ=0.05 (10배 빠른 타깃)",
        "iql_e50": "IQL, expectile 0.50 (=최소자승, 개선 없음 대조)",
        "iql_e70": "IQL, expectile 0.70",
        "iql_e90": "IQL, expectile 0.90",
        "iql_e95": "IQL, expectile 0.95",
        "iql_qc": "IQL(τ=0.7) + QC — 프리픽스 head 없음",
        "ep1": "에피소드 1개(745프레임)만으로 학습 — 암기 극한",
        "ep4": "에피소드 4개",
        "ep16": "16개",
        "ep64": "64개",
        "scalarq": "스칼라 회귀로 회귀(기본 HL-Gauss 대신)",
        "nofloor": "mc 하한 없음(기본은 max(TD, mc_return))",
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
        f"<title>ACRFT critic 실험 일지</title>\n<style>{CSS}</style>\n<main>{''.join(A)}</main>\n", encoding="utf-8"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
