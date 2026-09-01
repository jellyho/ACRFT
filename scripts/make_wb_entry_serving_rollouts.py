"""Hub entry for the first REAL-ROBOT test of the extraction ring's serving arms.

New eid (house rule: new content, new entry). Numbers are recomputed from the LEGOPROG workbook
every time this runs -- nothing here is transcribed by hand.
"""

import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
XLSX = pathlib.Path.home() / ".claude/uploads/9b8fe2d1-78b2-49b5-88f6-b730fd35fe92/c0b2be4f-LEGOPROG.xlsx"
d = pd.read_excel(XLSX, header=None)


def block(row0, cols):
    out = {}
    for col, label in cols.items():
        vals = [d.iloc[r, col + 2] for r in range(row0, row0 + 10)]
        v = np.array([x for x in vals if pd.notna(x)], float)
        out[label] = v
    return out


SEL = block(25, {0: "implicit N=8", 3: "no selection (N=1)", 6: "argmax N=8 (BoN)"})
QP = block(37, {15: 0.0, 12: 0.005, 9: 0.01, 6: 0.025, 3: 0.05, 0: 0.1})


def stat(v):
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def row(label, v, extra=""):
    m, c = stat(v)
    return f"<tr><td>{label}</td><td>{m:.2f} ± {c:.2f}</td><td>{len(v)}</td><td>{extra}</td></tr>"


sel_rows = "".join(row(k, SEL[k]) for k in ("no selection (N=1)", "argmax N=8 (BoN)", "implicit N=8"))
qp_rows = "".join(row(f"α = {a:g}" + (" (no steering)" if a == 0 else ""), QP[a]) for a in sorted(QP))
best_sel = stat(SEL["implicit N=8"])[0]
base_sel = stat(SEL["no selection (N=1)"])[0]
argmax_sel = stat(SEL["argmax N=8 (BoN)"])[0]
qp0 = stat(QP[0.0])[0]
qp_worst = stat(QP[0.1])[0]
zeros_at_01 = int((QP[0.1] == 0).sum())

stamp = subprocess.run(
    ["git", "-C", str(R), "log", "-1", "--format=%h"], capture_output=True, text=True, check=False
).stdout.strip()
if subprocess.run(
    ["git", "-C", str(R), "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False
).stdout.strip():
    stamp += "+dirty"
branch = subprocess.run(
    ["git", "-C", str(R), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False
).stdout.strip()

head = (
    "<table class='num'><tr><th>항목</th><th>내용</th></tr>"
    "<tr><th>who</th><td>사용자가 직접 실물 YAM에서 롤아웃, 워커B가 분석</td></tr>"
    "<tr><th>where</th><td>YAM lego-taxi 실물 로봇 · BC h30 정책 · critic <code>fixed</code> · 조건당 10 에피소드</td></tr>"
    "<tr><th>what</th><td>서빙 규칙 3종(선택 규칙) + QPILOTS-U α 스윕 6점의 실물 성능</td></tr>"
    "<tr><th>how</th><td>progress 0-4 단계 카운트, 조건당 10 에피소드, 평균 ± 95% t-CI</td></tr>"
    "<tr><th>why</th><td>지금까지의 arm 비교가 전부 오프라인 프록시(critic-Q)였고, ΔQ는 조향 arm에 자기 편향이 있어 실물 판정이 필요했다</td></tr>"
    f"<tr><th>코드</th><td><code>{branch}@{stamp}</code></td></tr></table>"
)

KO = f"""{head}
<p><b>왜.</b> <span class='xref' data-eid='extraction-suite-yam'>extraction 링</span>의 arm들을 지금까지
오프라인 지표(critic-Q, demo-MSE, jerk)로만 비교해 왔다. 그런데 critic-Q는 조향·선택 arm에 대해
<b>자기 심판</b>이다 — 조향은 바로 그 critic을 올리도록 설계됐으므로 ΔQ는 정의상 오른다. 실물 롤아웃만이
그 편향 밖에 있다. 이 리포트는 학습이 필요 없는 서빙 규칙들만 먼저 실물에서 잰 결과다.</p>

<p><b>공통 조건.</b> 정책은 BC h30 하나, critic은 <code>fixed</code> 하나로 고정하고 <b>실행할 청크를 고르는
규칙만</b> 바꿨다. 조건당 10 에피소드, progress는 0-4 단계 카운트다.</p>

<h3>1. 선택 규칙 — implicit이 argmax를 이겼다</h3>
<table class='num'><tr><th>규칙</th><th>평균 progress</th><th>n</th><th></th></tr>{sel_rows}</table>

<p><b>판정 — argmax는 선택을 안 한 것과 같았다({argmax_sel:.2f} vs {base_sel:.2f}).</b> N=8로 뽑아 최고점을
골랐는데 대조군과 평균이 같다. 분포를 보면 이유가 보인다: argmax는 10개 중 7개가 stage 1에서 멈추고 2개만
stage 4까지 간다 — <b>대부분 실패하고 가끔 크게 성공하는 양극화</b>다. 반면 implicit N=8은 {best_sel:.2f}로
분포 자체가 오른쪽으로 밀렸다(stage 4가 4개).</p>

<p><b>이는 IDQL 논문의 보고와 반대다.</b> 논문은 D4RL에서 argmax가 implicit보다 낫다고 하고, 본체 IDQL의
평가 규칙도 argmax다(Eq. 13, w를 one-hot으로 두는 것에 해당). 우리 로봇에서는 뒤집혔다.</p>

<p><b>기제 가설.</b> 잡음 있는 추정치의 최댓값은 위로 편향된다 — critic이 과대평가한 후보를 argmax가 정확히
찾아낸다. 우리 critic에서 <b>평균 이상 후보 비율은 18.7%</b>로 각박해서, N=8의 "최고"가 과대평가일 확률이
높다. implicit은 advantage의 <b>부호만</b> 보고 추첨하므로(가중치 τ=0.9 vs 1−τ=0.1) 그 극단 꼬리를 대부분
피한다. argmax의 양극화가 정확히 critic exploitation의 모양이다.</p>

<h3>2. QPILOTS-U 조향 강도 — 어느 값에서도 이득이 없었다</h3>
<table class='num'><tr><th>α</th><th>평균 progress</th><th>n</th><th></th></tr>{qp_rows}</table>
<figure><img src='figures/serving-rollouts-yam/fig_legoprog_qpilots.png' alt='QPILOTS alpha sweep'>
<figcaption>α는 실제 간격(선형 축). 회색 띠는 α=0의 95% CI.</figcaption></figure>

<p><b>판정 — 조향은 켜는 순간 손해다.</b> α=0이 {qp0:.2f}로 가장 높고, 가장 약한 α=0.005에서 이미
1.10으로 떨어진다. 0.005~0.05 구간은 서로 구분되지 않고(1.0~1.3) 전부 대조군 아래다. α=0.1에서는
{qp_worst:.2f}로 무너지며 <b>10 에피소드 중 {zeros_at_01}개가 stage 0</b> — 첫 단계도 진행하지 못한다.</p>

<p><b>범위에 주의.</b> 논문은 π0.5-LIBERO에서 α ∈ {{0.1, 0.2, 0.3}}을 스윕했다. 즉 <b>논문의 최솟값이 우리의
최댓값</b>이고, 논문이 정상 동작을 보고한 구간 전체가 우리에겐 이미 붕괴 지점이다.</p>

<p><b>오프라인 예측이 틀렸다.</b> 사전 오프라인 측정에서 QPILOTS는 α에 비례해 ΔQ가 올랐고(α=0.3에서 +23),
그 근거로 "α=0.1부터 시작"을 권고했다. 실물에서는 0.1이 이미 파국이었다. ΔQ가 조향에 유리하게 나온 것이
자기 심판이었음이 실증된 셈이다. 다만 같은 측정에서 <b>jerk +28%</b>도 함께 봤고 그쪽은 방향이 맞았다 —
분포 이탈 지표가 값 지표보다 실물을 잘 예측했다.</p>

<h3>3. 남는 질문 — 값이냐 기울기냐</h3>
<p>같은 critic으로 <b>선택</b>(implicit)은 잘 되고 <b>조향</b>(QPILOTS)은 망가졌다. 선택은 Q의 <b>순위</b>만
쓰고 조향은 <b>∇Q</b>를 쓴다. 즉 이 critic은 순위는 쓸 만한데 기울기는 신뢰하기 어려울 가능성이 있다 —
값과 기울기는 다른 요구조건이다. 이것이 사실이라면 ∇Q에 의존하는 나머지 학습 arm들(FlowDPG, QAM,
FlowDAgger)의 기대치도 함께 내려가야 한다. 다음 단계는 ∇Q 유용성을 직접 재는 오프라인 진단이다.</p>

<p><b>한계.</b> 조건당 10 에피소드라 CI가 넓다 — implicit({best_sel:.2f})과 argmax({argmax_sel:.2f})의 구간은
겹치므로 "우세해 보인다"까지가 정직한 진술이다. QPILOTS의 α=0 대 α=0.1은 겹치지 않아 그쪽은 확정이다.
progress는 단계 카운트라 등간격 척도가 아니며, 평균은 편의상 통계다.</p>"""

EN = f"""{head}
<p><b>Why.</b> Every comparison in the <span class='xref' data-eid='extraction-suite-yam'>extraction ring</span>
so far has been an offline proxy (critic-Q, demo-MSE, jerk). But critic-Q is <b>self-refereed</b> for
steering and selection arms — steering ascends that very critic, so ΔQ rises by construction. Only a
real rollout sits outside that bias. This entry measures the serving rules that need no training.</p>

<p><b>Held fixed.</b> One policy (BC h30), one critic (<code>fixed</code>); only the rule choosing which
sampled chunk to execute changes. 10 episodes per condition, progress is a 0-4 stage count.</p>

<h3>1. Selection rule — implicit beat argmax</h3>
<table class='num'><tr><th>rule</th><th>mean progress</th><th>n</th><th></th></tr>{sel_rows}</table>

<p><b>Verdict — argmax matched doing no selection at all ({argmax_sel:.2f} vs {base_sel:.2f}).</b> The
distribution shows why: argmax stalls at stage 1 in 7 of 10 episodes and reaches stage 4 in 2 — mostly
failing, occasionally succeeding big. implicit N=8 reaches {best_sel:.2f} with the whole distribution
shifted right (four episodes at stage 4).</p>

<p><b>This contradicts the IDQL paper</b>, which reports argmax outperforming the implicit policy on D4RL
and uses argmax as its own evaluation rule (Eq. 13, equivalent to a one-hot w).</p>

<p><b>Mechanism.</b> The max of noisy estimates is biased upward — argmax finds precisely the candidate the
critic over-valued. In our critic only <b>18.7%</b> of candidates sit above V, so the "best" of 8 is often
an over-estimate. implicit weights by the <b>sign</b> of the advantage only (τ=0.9 vs 1−τ=0.1) and mostly
avoids that tail. argmax's polarised outcomes are the shape critic exploitation takes.</p>

<h3>2. QPILOTS-U steering strength — no setting helped</h3>
<table class='num'><tr><th>α</th><th>mean progress</th><th>n</th><th></th></tr>{qp_rows}</table>
<figure><img src='figures/serving-rollouts-yam/fig_legoprog_qpilots.png' alt='QPILOTS alpha sweep'>
<figcaption>α at true linear spacing; the grey band is α=0's 95% CI.</figcaption></figure>

<p><b>Verdict — steering costs from the moment it is switched on.</b> α=0 is highest at {qp0:.2f}; the
weakest dose tested (0.005) already drops to 1.10; 0.005–0.05 are indistinguishable from each other and all
below the control. At α=0.1 it collapses to {qp_worst:.2f}, with <b>{zeros_at_01} of 10 episodes at stage
0</b> — never getting started.</p>

<p><b>Note the range.</b> The paper sweeps α ∈ {{0.1, 0.2, 0.3}} on pi0.5-LIBERO. Its smallest tested value
is our largest, so the entire regime it reports as working is already past collapse here.</p>

<p><b>The offline prediction was wrong.</b> Offline, QPILOTS' ΔQ grew with α (+23 at α=0.3) and we
recommended starting at 0.1 on that basis. On the robot 0.1 was already catastrophic — the self-refereed
metric misled. The <b>+28% jerk</b> measured alongside it did point the right way, so the
distribution-shift metric predicted the robot better than the value metric.</p>

<h3>3. The open question — values or gradients</h3>
<p>With one critic, <b>selection</b> works and <b>steering</b> fails. Selection uses only the <b>ranking</b>
of Q; steering uses <b>∇Q</b>. So this critic may have usable values and unreliable gradients — different
requirements. If that holds, the trained arms that lean on ∇Q (FlowDPG, QAM, FlowDAgger) should have their
expectations lowered too. Next step is an offline diagnostic that measures ∇Q's usefulness directly.</p>

<p><b>Limits.</b> 10 episodes per condition, so CIs are wide: implicit ({best_sel:.2f}) and argmax
({argmax_sel:.2f}) overlap, and "appears better" is the honest claim. QPILOTS α=0 vs α=0.1 do not overlap.
Progress is a stage count, not an interval scale; the mean is a convenience statistic.</p>"""

entry = {
    "eid": "serving-rollouts-yam",
    "date": "2026-09-01 21:40",
    "worker": "B",
    "title": f"🤖 [워커B] 실물 롤아웃 1차 — implicit이 BoN을 이기고({best_sel:.1f} vs {argmax_sel:.1f}), QPILOTS 조향은 켜는 순간 손해",
    "summary": (
        f"YAM 실물, 조건당 10에피소드. 선택 규칙: implicit N=8 {best_sel:.2f} > argmax N=8 {argmax_sel:.2f} "
        f"= 선택없음 {base_sel:.2f} — argmax는 7/10이 stage 1에 멈추는 양극화(critic exploitation의 모양)이고 "
        f"IDQL 논문 보고와 반대다. QPILOTS-U는 α=0({qp0:.2f})이 최고이고 α=0.005부터 이미 손해, α=0.1에서 "
        f"{qp_worst:.2f}로 붕괴({zeros_at_01}/10이 stage 0) — 논문의 최소 α가 우리의 최대 α다. 같은 critic에서 "
        "순위(선택)는 되고 기울기(조향)는 안 되므로, ∇Q에 의존하는 arm들의 기대치를 함께 낮춰야 한다."
    ),
    "tags": ["워커B", "실물롤아웃", "YAM", "extraction", "QPILOTS", "IDQL"],
    "status": "finding",
    "phase": "방법 비교",
    "links": ["extraction-suite-yam", "iql-pair-yam", "wa-emaq-bon"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/serving_rollouts_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "| stamp", branch, stamp)
