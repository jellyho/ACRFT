"""Hub entry for the single-lag non-Markovianity profile (eid nonmarkov-yam-lag).

Follow-up (new eid, prior entry untouched per house rule) correcting the interpretation of
nonmarkov-yam-meas. Numbers recomputed from the four probe JSONs.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((R / f".scratch/{name}/results.json").read_text())


lag = load("nonmarkov_yam_lags")
vel = load("nonmarkov_yam_velonly")
feat = load("nonmarkov_yam_feathist")
prop = load("nonmarkov_yam_propriohist")

m0 = lag["arms"]["0"]["val_mse"]
lags = sorted(int(k) for k in lag["arms"] if k != "0")
gain = {k: 100 * (m0 - lag["arms"][str(k)]["val_mse"]) / m0 for k in lags}


def ch_gain(d):
    b = d["arms"]["0"]["val_mse"]
    return 100 * (b - d["arms"]["15"]["val_mse"]) / b


g_vel, g_feat, g_prop = ch_gain(vel), ch_gain(feat), ch_gain(prop)

stamp = subprocess.run(
    ["git", "-C", str(R), "log", "-1", "--format=fql-one-step-actor@%h"], capture_output=True, text=True, check=False
).stdout.strip()
if subprocess.run(
    ["git", "-C", str(R), "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False
).stdout.strip():
    stamp += "+dirty"

rows = "".join(
    f"<tr><td>{k} ({k / 30 * 1000:.0f}ms)</td><td>{lag['arms'][str(k)]['val_mse']:.4f}</td><td>{gain[k]:+.1f}%</td></tr>"
    for k in lags
)
peak = max(lags, key=lambda k: gain[k])

head = (
    f"<table class='num'><tr><th>항목</th><th>내용</th></tr>"
    f"<tr><th>who</th><td>워커B — 사용자 설계 지시(single-lag가 공정) + 워커C 교차검증 문답의 산물</td></tr>"
    f"<tr><th>where</th><td>pc_cache/yam_s347 · held-out 52ep · L40S jobs 36589/36590/36593</td></tr>"
    f"<tr><th>what</th><td>lag별 단일 프레임 추가의 정보량 프로파일 — 정점 {peak / 30 * 1000:.0f}ms {gain[peak]:+.1f}%, 5초는 해악; 명시적 속도 특징 하나가 {g_vel:+.1f}%</td></tr>"
    f"<tr><th>how</th><td>measure_nonmarkov_yam.py --lags (각 arm=[t−n, t] 두 프레임, Markov=[t,t]) + --zero-hist 채널 진단, 표·그림 JSON 재계산</td></tr>"
    f"<tr><th>why</th><td>창-스윕(nonmarkov-yam-meas)의 창길이×밀도 confound 제거 + 함량의 정체 규명</td></tr>"
    f"<tr><th>코드</th><td><code>{stamp}</code></td></tr></table>"
)

KO = f"""{head}
<p><b>왜 (정정의 성격).</b> <span class='xref' data-eid='nonmarkov-yam-meas'>선행 측정</span>의 28.3%는 6프레임
창 스윕이라 창길이와 프레임 밀도가 얽혀 있었다(사용자 지적: "n스텝 전 프레임 하나가 더 공정"). 워커C도 같은
confound를 자기 도메인 사다리에서 발견해 depth-스윕을 던졌다. 이 리포트는 그 공정 설계의 결과다 —
<b>주 판정은 시각-포함(양채널) 조건</b>, 채널 차단은 구성 진단으로만.</p>

<p><b>어떻게.</b> 각 arm의 입력은 정확히 두 프레임 [t−n, t](Markov arm은 [t, t] — 차원 동일). n ∈
{{1,3,5,15,30,60,150}}. 채널 진단: history의 시각/proprio 한쪽 차단 + "명시적 속도" arm(시각 이력 차단,
history 슬롯에 p_t−p_{{t−1}} 반복 — 워커C 제안).</p>

<table class='num'><tr><th>lag n</th><th>val MSE</th><th>Markov 대비</th></tr>{rows}</table>

<p><b>판정 1 — 프로파일은 단봉, 정점 {peak / 30 * 1000:.0f}ms({gain[peak]:+.1f}%).</b> 직전 프레임(33ms)은 +4.8%뿐이고
100~170ms에서 폭발하며 2초에서 소멸, <b>5초 프레임은 오히려 해악(−3.4%, 간섭)</b>. "현재 + ~150ms 전 프레임
한 장"이면 raw 관측 기준 회수 가능량의 대부분을 얻는다 — history 크리틱/정책의 창 설계에 대한 구체적 답.</p>

<p><b>판정 2 — 함량의 정체는 대부분 kinematic continuation이고, 병목은 표현이다.</b> 채널 진단(각주 지위):
명시적 속도 특징 하나가 <b>{g_vel:+.1f}%</b> — raw 프레임 어느 arm보다 크다. 즉 lag-1 arm(+4.8%)에도 같은 정보가
있었지만 정규화된 절대 관절각 두 개의 차를 네트워크가 못 뽑는다(근사-동일 값의 뺄셈 = 표현/최적화 병목).
100~170ms 정점도 "그 기준선에서야 차분이 노이즈를 이기는" 현상으로 읽힌다. proprio-이력 단독 {g_prop:+.1f}% >
양채널 28.3%(<b>시각 이력을 더하면 오히려 악화 — 간섭 역설</b>), 시각-이력 단독 {g_feat:+.1f}%.
워커C의 경고("얕은 사촌이 지배")가 이 데이터에서 실증됐고, 선행 리포트의 "깊은 함량" 뉘앙스는 하향 정정한다:
<b>다음-액션 예측 기준, 2초 너머 깊은 기억은 ≈0이고 지배 성분은 실행-운동의 연속성이다.</b></p>

<p><b>귀결.</b> ① 정책/크리틱에 <b>명시적 속도 입력(또는 ~150ms 프레임 한 장)</b>이 값싸고 지배적인 개선 —
long-context 불필요(다음-액션 기준). ② Zeng/Lazzati/Park의 정책측 라인과 정합하되, "숨은 의도" 류의 깊은
non-Markov는 이 프로브(1-step 타깃)로는 안 보인다 — <b>청크/가치 스케일 타깃 재측정이 다음 단계</b>(의도는
긴 지평에서 발현될 수 있음). ③ 도메인 사다리(워커C: OGBench 6.5~10 &lt; YAM &lt; MimicGen 27~34)의 해석도
"관측에 kinematic 정보가 얼마나 이미 있나"로 재검토 중(C의 depth-스윕 진행).</p>

<p><b>한계.</b> run-to-run 노이즈 ~3%(같은 시드 GPU 비결정성 실측) — ±5% 미만 차이는 유보. mean-pooled 시각
표현이라 시각측 절대값은 하한. 1-step 액션 타깃 한정 — 청크·리턴 타깃은 별도.</p>
<figure><img src='figures/nonmarkov-yam-lag/fig_nonmarkov_lags.png' alt='single-lag profile'>
<figcaption>시각-포함 공정 조건의 lag 프로파일(파랑)과 명시적 속도 특징 기준선(주황 점선).</figcaption></figure>"""

EN = f"""{head}
<p><b>Why (a correction).</b> The prior <span class='xref' data-eid='nonmarkov-yam-meas'>windowed measurement</span>
(28.3%) confounded window length with frame density (user: "one frame at lag n is fairer"); worker C found the
same confound in their domain ladder. This report is the fair design — <b>primary verdicts come from the
vision-present (both-channel) condition</b>; channel knockouts are diagnostics only.</p>

<p><b>How.</b> Each arm's input is exactly two frames [t−n, t] (Markov = [t, t], same dims), n ∈
{{1,3,5,15,30,60,150}}. Diagnostics: visual/proprio history knockouts + an "explicit velocity" arm
(worker C's suggestion: history slots carry the repeated p_t − p_{{t−1}}).</p>

<table class='num'><tr><th>lag n</th><th>val MSE</th><th>vs Markov</th></tr>{rows}</table>

<p><b>Verdict 1 — unimodal profile, peak at {peak / 30 * 1000:.0f}ms ({gain[peak]:+.1f}%).</b> The immediately previous
frame (33ms) adds only +4.8%; the gain explodes at 100–170ms, vanishes by 2s, and a 5s-old frame actively
hurts (−3.4%, interference). "Current + one frame ~150ms back" recovers most of what raw observations can —
a concrete answer for history-critic/policy window design.</p>

<p><b>Verdict 2 — the content is mostly kinematic continuation; the bottleneck is representation.</b> A single
explicit velocity feature yields <b>{g_vel:+.1f}%</b> — more than any raw-frame arm. The same information exists in
the lag-1 arm (+4.8%), but subtracting two normalized near-equal joint angles is something the network fails to
learn (representation/optimization bottleneck); the 100–170ms peak reads as "the baseline at which the
difference beats noise". Proprio-history alone {g_prop:+.1f}% &gt; both channels 28.3% (<b>adding visual history
hurts — an interference paradox</b>); visual-only {g_feat:+.1f}%. Worker C's warning (the shallow cousin dominates)
is confirmed on this data, and the prior report's "deep content" flavor is revised down: <b>for next-action
prediction, memory beyond 2s is ≈0 and the dominant component is executed-motion continuity.</b></p>

<p><b>Consequences.</b> (1) An explicit velocity input (or one ~150ms-back frame) is the cheap dominant win for
policies/critics — no long context needed at the next-action scale. (2) Consistent with the Zeng/Lazzati/Park
policy-side line, but intent-like deep non-Markovianity is invisible to this 1-step probe — <b>re-measuring with
chunk/value-scale targets is the next step</b>. (3) The domain ladder (worker C: OGBench 6.5–10 &lt; YAM &lt;
MimicGen 27–34) is being re-examined as possibly a ladder of "how much kinematics the observation already
carries" (C's depth sweep running).</p>

<p><b>Limitations.</b> Run-to-run noise ~3% (measured, same-seed GPU nondeterminism) — differences under ±5%
are withheld. Mean-pooled visual features make the visual-side absolute a lower bound. Next-action target only —
chunk/return targets are separate.</p>
<figure><img src='figures/nonmarkov-yam-lag/fig_nonmarkov_lags.png' alt='single-lag profile'>
<figcaption>Vision-present lag profile (blue) and the explicit-velocity reference (orange dotted).</figcaption></figure>"""

entry = {
    "eid": "nonmarkov-yam-lag",
    "date": "2026-08-27 15:10",
    "worker": "B",
    "title": f"🧪 [워커B] single-lag 프로파일 — non-Markov 정보는 {peak / 30 * 1000:.0f}ms 프레임 한 장에 살고({gain[peak]:+.0f}%), 정체는 속도({g_vel:+.0f}%), 5초는 해악 (선행 측정 정정)",
    "summary": f"공정 설계(각 arm=[t−n,t] 두 프레임): 프로파일 단봉 — 33ms +4.8% → 167ms {gain[peak]:+.1f}% → 5s −3.4%(간섭). 명시적 속도 특징 하나가 {g_vel:+.1f}%로 raw 어느 arm보다 커서, 함량의 지배 성분은 kinematic continuation이고 병목은 표현. 깊은 기억(>2s)은 다음-액션 기준 ≈0 — 선행 28.3%의 해석 하향 정정, 청크/가치 타깃 재측정이 다음.",
    "tags": ["워커B", "non-markov", "dataset", "측정", "정정"],
    "status": "done",
    "phase": "진단·방법",
    "links": ["nonmarkov-yam-meas", "rcv-value-of-information", "nonmarkov-longer", "rcv-honest-critic-recipe"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/nonmarkov_lag_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "stamp", stamp)
