"""Hub entry: the critic's arg-max bias is a function of how wide the candidate set is.

Reconciles this probe with q-landscape-ood, which measured the SAME bias against the SAME anchor
over the BC sampler's own draws and found it small (+1.88 +- 0.62 at N=16). Both are points on one
curve. Every number is recomputed from .scratch/extraction/diag_selection_bias.json.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
D = json.loads((R / ".scratch/extraction/diag_selection_bias.json").read_text())
CC = json.loads((R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k/config.json").read_text())


def _git(*args):
    return subprocess.run(["git", "-C", str(R), *args], capture_output=True, text=True, check=False).stdout.strip()


stamp = _git("log", "-1", "--format=%h") + ("+dirty" if _git("status", "--porcelain", "-uno") else "")
branch = _git("rev-parse", "--abbrev-ref", "HEAD")

N = D["n"]
AM, FR, LC = D["argmax_margin"], D["argmax_beats_executed_frac"], D["lcb_margin"]
i8, i16 = N.index(8), N.index(16)
BON_PRIOR = 1.88  # q-landscape-ood, N=16, over the BC sampler's own draws
SIGMA_BC = 0.0417  # per-dim RMS distance between two BC draws; see the correction block
RATIO = AM[i16] / BON_PRIOR
import numpy as np  # noqa: E402

SLOPE = float(np.polyfit(np.log(D["width_spread_rms"]), np.log(D["width_argmax_bias"]), 1)[0])
_c = np.polyfit(np.log(D["width_spread_rms"]), np.log(D["width_argmax_bias"]), 1)
PRED_AT_BC = float(np.exp(_c[1] + _c[0] * np.log(SIGMA_BC)))
OVERSHOOT = BON_PRIOR / PRED_AT_BC
NARROW = D["width_argmax_bias"][0]
WIDEST = D["width_spread_rms"][-1] / SIGMA_BC

bias_tbl = (
    "<table class='num'><tr><th>N</th>"
    + "".join(f"<th>{n}</th>" for n in N)
    + "</tr><tr><th>arg-max Q − Q(실행됨)</th>"
    + "".join(f"<td>{v:+.1f}</td>" for v in AM)
    + "</tr><tr><th>실행된 청크를 이긴 비율</th>"
    + "".join(f"<td>{v:.3f}</td>" for v in FR)
    + "</tr><tr><th>LCB(β=1, K=2)로 골랐을 때</th>"
    + "".join(f"<td>{v:+.1f}</td>" for v in LC)
    + "</tr></table>"
)
bias_tbl_en = (
    bias_tbl.replace("arg-max Q − Q(실행됨)", "arg-max Q − Q(executed)")
    .replace("실행된 청크를 이긴 비율", "fraction beating the executed chunk")
    .replace("LCB(β=1, K=2)로 골랐을 때", "selected by LCB (β=1, K=2)")
)

wrows = "".join(
    f"<tr><td>{'다른 상태' if d < 0 else f'±{d} 프레임'}</td><td>{s:.4f}</td><td>{b:+.1f}</td></tr>"
    for d, s, b in zip(D["width_deltas"], D["width_spread_rms"], D["width_argmax_bias"], strict=True)
)
width_tbl = (
    "<table class='num'><tr><th>후보를 어디서 뽑았나</th><th>실행된 청크와의 거리 (차원당 RMS)</th>"
    f"<th>arg-max 편향 (N={D['width_n']})</th></tr>{wrows}</table>"
)
width_tbl_en = (
    width_tbl.replace("후보를 어디서 뽑았나", "where the candidates came from")
    .replace("실행된 청크와의 거리 (차원당 RMS)", "distance from the executed chunk (per-dim RMS)")
    .replace("arg-max 편향", "arg-max bias")
    .replace("다른 상태", "another state")
    .replace(" 프레임", " frames")
)

FIG = "<figure><img src='figures/argmax-width/fig_selection_bias.png' alt='selection bias vs candidate width'><figcaption>{cap}</figcaption></figure>"

CORR_KO = """<div style='border-left:4px solid #c44;background:#fff5f5;padding:10px 14px;margin:12px 0'>
<b>정정 (2026-09-02, 게시 당일).</b> 이 글의 초판은 실물 결과를 <b>"BoN이 실패했고 확률적 추첨이 이겼다"</b>로
서술했다. 원자료로 run-level 통계를 다시 내면 그렇게 말할 수 없다 — 조건당 n=10, 0–4 정수 척도:
<br><code>무선택 1.70 ± 0.59 &nbsp;|&nbsp; argmax BoN 1.70 ± 0.78 &nbsp;|&nbsp; 추첨 2.70 ± 0.78</code>
<br>BoN vs 무선택 <b>Δ=+0.00, 95% CI [−1.05, +1.05], p=1.000</b> — "실패"가 아니라 <b>신호가 없을 때
예측되는 바로 그 결과</b>이며, 설명이 필요한 현상이 아니다.
추첨 vs 무선택 <b>Δ=+1.00, 95% CI [−0.05, +2.05], p=0.060</b> — <b>유의하지 않다.</b>
따라서 <b>세 조건은 통계적으로 서로 구분되지 않으며</b>, "덜 고르는 규칙이 이겼다"는 아래 문장들은
유의하지 않은 차이에 기댄 것이므로 취소한다. Δ=1.0을 잡으려면 조건당 30–40 에피소드가 필요하다.
<br><b>이 글의 오프라인 측정(판별 정확도 0.566, 폭-편향 멱함수, 앙상블 4.7%)은 로봇 결과와 독립적으로
성립하며 영향받지 않는다.</b> 영향받는 것은 그것을 실물 결과와 잇는 해석뿐이다.
</div>"""
CORR_EN = """<div style='border-left:4px solid #c44;background:#fff5f5;padding:10px 14px;margin:12px 0'>
<b>Correction (2026-09-02, same day as publication).</b> The first version of this entry described the
robot result as <b>"best-of-N failed and the stochastic lottery won"</b>. Recomputing run-level
statistics from the raw data does not support that — n=10 per condition, integer 0–4 scale:
<br><code>no selection 1.70 ± 0.59 &nbsp;|&nbsp; argmax BoN 1.70 ± 0.78 &nbsp;|&nbsp; lottery 2.70 ± 0.78</code>
<br>BoN vs no selection: <b>Δ=+0.00, 95% CI [−1.05, +1.05], p=1.000</b> — not a failure but
<b>exactly what no signal predicts</b>, and therefore nothing that needs explaining.
Lottery vs no selection: <b>Δ=+1.00, 95% CI [−0.05, +2.05], p=0.060</b> — <b>not significant.</b>
<b>All three conditions are statistically indistinguishable</b>, so the sentences below that lean on
"the rule that selects less won" are withdrawn. Detecting Δ=1.0 needs 30–40 episodes per condition.
<br><b>The offline measurements in this entry (0.566 discrimination, the width–bias power law, the
4.7% ensemble ratio) hold independently of the robot result and are unaffected.</b> What is affected
is only the interpretation that joins them to it.
</div>"""

KO = f"""{CORR_KO}<table class='num'><tr><th>항목</th><th>내용</th></tr>
<tr><th>who</th><td>워커B</td></tr>
<tr><th>when</th><td>2026-09-02</td></tr>
<tr><th>where</th><td>CPU only — 캐시된 DINOv2 feature(<code>pc_cache/yam_s347</code>, 93.8만 프레임) 위. GPU·로봇 불필요</td></tr>
<tr><th>what</th><td>critic의 arg-max 편향이 <b>후보 집합의 폭</b>의 함수임을 실측하고, 그 축 위에서
<span class='xref' data-eid='q-landscape-ood'>q-landscape-ood</span>의 BoN 편향 +1.88과 이 프로브의
{AM[i16]:+.0f}을 화해시킴. 부수적으로 critic의 <b>상태 판별력이 {D["ranking_accuracy"]:.3f}</b>(우연 0.5)</td></tr>
<tr><th>how</th><td><code>scripts/diag_critic_selection_bias.py</code> — 각 상태에서 (a) 실제로 실행된 청크와
(b) 통제된 거리만큼 떨어진 실제 청크 N개를 같은 critic으로 채점. 앵커·단위는 q-landscape-ood와 동일</td></tr>
<tr><th>why</th><td>실물에서 BoN(N=8)이 무선택과 동률이었다. 원인이 "arg-max가 과대평가를 고른다"인지
"그 폭에서는 고를 신호 자체가 없다"인지 가른다 — 처방이 정반대다</td></tr>
<tr><th>코드</th><td><code>{branch}@{stamp}</code></td></tr></table>

<p><b>배경.</b> 우리 patch critic은 사람 시연 347에피소드 위에서 IQL expectile로 학습된다. IQL은 설계상
데이터셋 밖의 액션을 <b>한 번도 평가하지 않으므로</b>, 정책이 제안하는 청크의 Q에는 학습 중 아무 제약이 걸리지
않는다. 그런데 서빙에서는 바로 그런 청크 N개를 놓고 arg-max를 시킨다.
<span class='xref' data-eid='serving-rollouts-yam'>실물 롤아웃 1차</span>에서 이 best-of-N(N=8)은
<b>무선택과 정확히 동률</b>(평균 진행도 1.70 vs 1.70)이었고, 같은 후보에 대한 확률적 expectile 추첨만
2.70을 냈다.</p>

<h3>먼저: 이 프로브 하나만 보면 결론을 틀리게 낸다</h3>
<p>후보를 <b>다른 상태에서 실행된 청크</b>로 잡고 arg-max 편향을 재면 아래처럼 나온다. 앵커는
<b>그 상태에서 실제로 실행된 청크</b>다 — 대부분 성공 에피소드에서 뽑혔으므로 목표에 도달한 것이 확인된 행동이고,
보상이 cost-to-goal이므로 critic이 이보다 크게 나은 값을 내놓아서는 안 된다.</p>
{bias_tbl}
<p>N=8에서 <b>{FR[i8] * 100:.1f}%의 상태</b>에서 이식된 청크가 실행된 청크를 이기고, 마진은 {AM[i8]:+.1f}이다.
γ={CC["discount"]}의 cost-to-goal이라 <b>value 1단위 ≈ 제어 1스텝</b>이므로 이는 "목표에
{AM[i8] / 30:.1f}초 더 가깝다"는 주장이다. 곡선은 N=32까지 <b>포화하지 않는다</b>.</p>
<p><b>그런데 이 숫자로 실물 BoN을 설명하면 틀린다.</b>
<span class='xref' data-eid='q-landscape-ood'>q-landscape-ood</span>는 같은 앵커·같은 단위로,
<b>BC 정책이 실제로 뽑은 draw</b> 위에서 같은 편향을 재서 <b>+{BON_PRIOR} ± 0.62 (N=16)</b>를 얻었다.
이 프로브의 N=16 값은 {AM[i16]:+.1f} — <b>{RATIO:.0f}배</b> 차이다. 두 숫자 모두 옳다. 다른 것은
<b>후보 집합</b>이다.</p>

<h3>화해: 편향은 후보 폭의 함수다</h3>
<p>그래서 폭을 통제해서 다시 쟀다. 후보를 <b>같은 에피소드에서 ±δ 프레임 떨어진 지점의 실제 청크</b>로 잡으면
δ가 액션 공간에서의 거리를 조절한다. 거리는 정규화 액션 공간의 <b>차원당 RMS</b>로 보고하여
q-landscape-ood가 보고한 BC 샘플러 자체의 폭 σ≈{SIGMA_BC}과 같은 축에 놓았다.</p>
{width_tbl}
{FIG.format(cap="왼쪽: 이식 후보에 대한 arg-max 편향 — 포화하지 않는다. 오른쪽: 같은 편향을 후보 폭의 함수로. 음영이 BC 샘플러 자신의 폭이다.")}
<p><b>깨끗한 멱함수다.</b> log-log 기울기 <b>{SLOPE:.2f}</b> — 편향은 후보 폭에 거의 제곱으로 자라며,
폭 25배 구간에서 편향이 275배 움직인다. 그리고 best-of-N이 서빙에서 뻗을 수 있는 거리는
<b>샘플러 자신의 노이즈</b>가 전부다: σ_BC≈{SIGMA_BC}는 이 스윕의 가장 좁은 점(±1 프레임, {D["width_spread_rms"][0]:.4f})
보다도 좁다. 그 폭에서 이 프로브의 편향은 <b>{NARROW:+.1f}</b>, q-landscape-ood의 측정은 <b>+{BON_PRIOR}</b> —
둘 다 <b>value 1~2단위, 즉 {BON_PRIOR / 30:.2f}초</b>다. 진행도가 0–4단계인 과제에서 이는 아무것도 아니다.
반면 넓은 집합에서는 {AM[i16]:+.0f}. <b>두 자릿수 차이가 나는 것은 critic이 아니라 후보 집합이다.</b></p>
<p><b>[취소됨 — 아래 정정 참조]</b> <span style='opacity:.55'> 멱함수를 σ_BC까지 외삽하면 {PRED_AT_BC:+.2f}인데 q-landscape-ood의 실측은
+{BON_PRIOR}로 <b>{OVERSHOOT:.0f}배 위</b>다. 자릿수는 맞지만 정확히 맞지는 않으며, 그럴듯한 이유가 있다 —
이 스윕의 좁은 후보는 <b>시간축으로 밀린 실제 액션</b>이라 데모 다양체 위에 얌전히 놓이는 반면, BC draw는
<b>생성 모델의 샘플</b>이라 같은 거리에서도 데이터가 가보지 않은 방향으로 벗어날 수 있다. 즉
<b>거리당 적대성이 더 높다</b>는 가설이고, 정책 샘플 뱅크가 도착하면 같은 스크립트로 직접 검증된다.</span></p>
<div style='border-left:4px solid #2a7;background:#f2fbf6;padding:10px 14px;margin:12px 0'>
<b>정정 2 (2026-09-02) — 위 잔차는 존재하지 않았다. 축이 틀렸다.</b>
워커B(ACRFT-WS)가 단위를 지적했고, 그쪽 프로브의 frozen 출력으로 직접 계산해 확인했다.
프로브가 보고하는 σ=0.009는 <code>bc.std(axis=0).mean()</code>(<code>probe_q_landscape.py:177</code>),
즉 <b>좌표별 std의 평균</b>이다. 이 글의 축은 <code>norm(Δ)/√(H·AD)</code>, 즉 <b>차원당 RMS</b>이고,
좌표가 이질적이면(30스텝 청크가 팔 관절과 그리퍼를 섞으므로 당연히 그렇다) Jensen 부등식으로
평균-of-std &lt; RMS-of-std 이다. 프로브의 <code>q_landscape.json.gz</code>에서 되돌리면:
<br><code>pc_sigma=[0.4565, 0.2205], pc_var_frac=[0.544, 0.154] → 총분산 0.4565²/0.544 = 0.383
→ 좌표당 RMS std = √(0.383/420) = 0.0295 → 두 draw 사이 거리 = √2×0.0295 = <b>0.0417</b></code>
<br>즉 BC 구름은 이 축에서 <b>0.042</b>에 있다. 거기서 멱함수는 <b>+1.74</b>를 주고, 프로브의 독립 실측은
<b>+1.88 ± 0.62</b> — <b>8% 이내로 일치</b>한다. 14배 잔차도, 그것을 설명하려던 "BC draw가 거리당 더
적대적"이라는 가설도 <b>필요 없다</b>. 마커가 엉뚱한 자리에 찍혀 있었을 뿐이다.
<br>결론 방향은 그대로다 — 서빙 폭에서의 편향은 여전히 <b>제어 1~2스텝(≈0.06초)</b>이고, 넓은 집합의
+165와 두 자릿수 차이다. 달라진 것은 두 독립 측정이 이제 <b>어긋나지 않는다</b>는 점이다.
</div>

<h3>그래서 실물 BoN이 실패한 이유는 과대평가가 아니다</h3>
<p>처방이 갈리는 지점이다. 편향이 문제라면 답은 <b>더 강한 비관성</b>(앙상블·LCB·CQL)이다. 그러나 서빙 폭에서
편향이 +{BON_PRIOR}에 불과하다면, N=8 arg-max가 무선택과 동률인 이유는 <b>그 폭에서 고를 신호가 없기 때문</b>이다.
다만 위 정정대로 <b>실물 데이터는 아직 이 해석을 지지하지도 반박하지도 못한다</b>(세 조건 모두 CI가 겹친다). 이 문단은 오프라인 측정이 시사하는 <b>가설</b>이며, 판정은 에피소드 수를 늘린 뒤에 내린다.</p>
<p><b>판별력을 직접 재면 이 해석이 지지된다.</b> 위 표의 후보는 "다른 상태에서 실행된 청크"로,
BoN이 마주하는 것보다 <b>{WIDEST:.0f}배 넓은</b> — 즉 훨씬 <b>쉬운</b> — 판별 과제다.
그런데도 후보 하나씩 볼 때 critic의 정확도는 <b>{D["ranking_accuracy"]:.3f}</b>(우연 0.5), 후보당 마진 중앙값은
{D["per_candidate_margin_median"]:+.1f}에 불과하다. 이렇게 쉬운 과제에서 우연보다 겨우 나은 critic이,
σ≈{SIGMA_BC}짜리 구름 안에서 후보를 가려낼 이유가 없다.</p>

<h3>앙상블은 이 불확실성을 측정하고 있지 않다</h3>
<p>상태당 앙상블 불일치(K=2)는 <b>{D["ensemble_disagreement_std"]:.1f}</b>인데, arg-max가 착취하는 후보 간
spread는 <b>{D["within_state_candidate_std"]:.1f}</b>다 — <b>{D["disagreement_to_spread_ratio"] * 100:.1f}%</b>만 보고 있다.
LCB(β=1)를 걸어도 N=8에서 {AM[i8]:+.1f} → {LC[i8]:+.1f}, {(1 - LC[i8] / AM[i8]) * 100:.0f}%밖에 못 깎는다.
<b>β를 올려서 될 일이 아니라, 측정 자체가 없다.</b> (그리고 K=2에서는 <code>mean − 1σ</code>가
<code>min</code>과 <b>항등</b>임을 확인했다 — 부동소수점 오차 4.4e-16. 위 LCB 곡선은 독립 추정량이 아니라
min 그 자체이고, β는 K=2에서 자유도가 아니다. 워커B(ACRFT-WS)의 지적.) 이는
<span class='xref' data-eid='critic-detail-survey'>구현 디테일 서베이</span>의 가설 2(∇Q를 쓰는 방법은 전부 앙상블 10,
우리는 2)에 처음으로 숫자를 붙인 것이다.</p>

<h3>이에 따라 만든 것 (PR #16)</h3>
<ul>
<li><b><code>CriticQ.q_lcb</code> / <code>q_disagreement</code></b> — min-over-K는 K가 커질수록 이유 없이
비관적이 되어, 앙상블 <b>크기</b>(불일치를 얼마나 잘 재나)와 비관 <b>강도</b>가 한 손잡이에 묶여 있었다. LCB가 분리한다.</li>
<li><b><code>train_patch_critic_cached.py --alpha-cql</code></b> — IQL에 구조적으로 없는 보수항.
네거티브는 배치 내 <b>wrong-state 청크</b>(위 0.566 통계를 <b>직접</b> 학습 신호로 바꾼다), U(−1,1), 또는 동결 BC 뱅크.
Cal-QL 캘리브레이션(<code>--calql</code>)은 OOD Q를 궤적의 MC 리턴으로 하한한다 —
<code>nakamotoo/Cal-QL</code>이 정책 샘플에만 적용하고 랜덤에는 적용하지 않는 그 분기를 따랐다.</li>
<li><b><code>Pi0.sample_n_actions_batched</code></b> — 오프라인 뱅크용 배치 샘플러. Euler 루프를 스크립트로
복사하는 대신 모델 API로. 상태 i의 후보가 상태 i의 prefix를 쓰는지 테스트로 고정했다.</li>
</ul>
<p>critic 재학습 4종을 큐에 넣었다(K=10 min / K=10 mean / α_cql=10 / α_cql=30). 전부 배포 중인
<code>fixed_tau9_min_200k</code> 레시피에서 <b>한 인자만</b> 다르므로 기존 체크포인트가 대조군이다.</p>

<h3>한계 — 확정과 잠정의 구분</h3>
<ul>
<li><b>확정.</b> 표의 모든 수치는 critic {D["n_states"]}개 상태에서 재현 가능하게 계산됐고, 스크립트가 리포에 있다.</li>
<li><b>잠정 — 가장 큰 한계.</b> 이 프로브의 후보는 <b>실제 정책 draw가 아니라 이식된 시연 청크</b>다.
폭-편향 곡선은 두 측정을 잇지만, BC draw 위에서 이 곡선을 직접 재려면 정책 샘플 뱅크가 필요하다.
그 잡(<code>sample_policy_chunks.py</code>)은 큐에 있고, 도착하면 같은 스크립트로 다시 돌린다.</li>
<li>σ_BC≈{SIGMA_BC}는 q-landscape-ood에서 인용한 값이고, 그쪽은 평균으로부터의 std이며 이쪽은 두 청크 사이의
거리라 √2 정도의 차이가 있을 수 있다. 축의 위치가 아니라 <b>자릿수</b>가 논지다.</li>
<li>critic 1종({CC["horizon"]}스텝 청크, macro={CC["macro_group_size"]}, K={CC["num_critics"]}).
q-landscape-ood는 9종에서 과대추정이 보편적임을 보였지만, 여기 판별력 수치는 배포 중인 1종에 대한 것이다.</li>
<li>"신호가 없어서 동률"이라는 해석은 <b>추론</b>이다. 반증 방법이 있다: α_cql 런이 0.566을 끌어올렸는데도
실물 BoN이 여전히 동률이면 이 해석이 틀린 것이다.</li>
</ul>"""

EN = f"""{CORR_EN}<table class='num'><tr><th>field</th><th></th></tr>
<tr><th>who</th><td>worker B</td></tr>
<tr><th>when</th><td>2026-09-02</td></tr>
<tr><th>where</th><td>CPU only, over cached DINOv2 features (<code>pc_cache/yam_s347</code>, 938k frames). No GPU, no robot.</td></tr>
<tr><th>what</th><td>Measured that the critic's arg-max bias is a function of <b>how wide the candidate set is</b>, and used that axis to reconcile <span class='xref' data-eid='q-landscape-ood'>q-landscape-ood</span>'s BoN bias of +{BON_PRIOR} with this probe's {AM[i16]:+.0f}. Incidentally: the critic's state discrimination is <b>{D["ranking_accuracy"]:.3f}</b> against a chance of 0.5.</td></tr>
<tr><th>how</th><td><code>scripts/diag_critic_selection_bias.py</code> — at each state, score (a) the chunk actually executed there against (b) N real chunks at a controlled distance. Same anchor and units as q-landscape-ood.</td></tr>
<tr><th>why</th><td>On the robot, best-of-N (N=8) tied with no selection. Whether the cause is "arg-max selects over-estimates" or "there is nothing to select on at that width" points at opposite fixes.</td></tr>
<tr><th>code</th><td><code>{branch}@{stamp}</code></td></tr></table>

<p><b>Background.</b> Our patch critic is trained by IQL expectile on 347 human demonstration episodes.
IQL by construction <b>never evaluates an action outside the dataset</b>, so nothing during training
constrains Q on a chunk the policy proposes. At serving we then ask it to arg-max over exactly such
chunks. In <span class='xref' data-eid='serving-rollouts-yam'>the first real-robot rollouts</span> that
best-of-N (N=8) <b>tied exactly with no selection</b> (1.70 vs 1.70 mean progress), while a stochastic
expectile lottery over the same candidates reached 2.70.</p>

<h3>First: this probe alone gives the wrong conclusion</h3>
<p>Take the candidates to be <b>chunks executed at other states</b> and the arg-max bias looks like this.
The anchor is <b>the chunk actually executed at that state</b> — drawn mostly from successful episodes,
so it is an action confirmed to reach the goal, and since the reward is cost-to-goal the critic should
not be able to beat it by much.</p>
{bias_tbl_en}
<p>At N=8 a transplanted chunk beats the executed one in <b>{FR[i8] * 100:.1f}% of states</b>, by {AM[i8]:+.1f}.
At cost-to-goal with γ={CC["discount"]}, <b>one value unit is about one control step</b>, so that is a claim
of being {AM[i8] / 30:.1f} s closer to the goal. The curve <b>does not saturate</b> through N=32.</p>
<p><b>But explaining the robot's BoN result with this number would be wrong.</b>
<span class='xref' data-eid='q-landscape-ood'>q-landscape-ood</span> measured the same bias against the same
anchor in the same units over the <b>BC policy's own draws</b> and got <b>+{BON_PRIOR} ± 0.62 (N=16)</b>.
This probe's N=16 value is {AM[i16]:+.1f} — a factor of <b>{RATIO:.0f}</b>. Both are correct. What differs is
the <b>candidate set</b>.</p>

<h3>Reconciliation: the bias is a function of candidate width</h3>
<p>So we controlled the width. Taking candidates to be <b>real chunks executed ±δ frames away in the same
episode</b> makes δ a dial on action-space distance. Distance is reported as <b>per-dimension RMS</b> in
normalized action units, putting it on the same axis as the BC sampler's own spread σ≈{SIGMA_BC}, as
reported by q-landscape-ood.</p>
{width_tbl_en}
{FIG.format(cap="Left: arg-max bias over transplanted candidates — it does not saturate. Right: the same bias as a function of candidate width; the shaded band is the BC sampler's own spread.")}
<p><b>It is a clean power law.</b> Log-log slope <b>{SLOPE:.2f}</b> — the bias grows almost quadratically in
candidate width, moving 275x across a 25x range of width. And what best-of-N can reach at serving is
<b>the sampler's own noise</b>: σ_BC≈{SIGMA_BC} is narrower than the narrowest point of this sweep
(±1 frame, {D["width_spread_rms"][0]:.4f}). At that width this probe's bias is <b>{NARROW:+.1f}</b> and
q-landscape-ood's measurement is <b>+{BON_PRIOR}</b> — both one or two value units, i.e. {BON_PRIOR / 30:.2f} s.
On a task scored 0–4 that is nothing. Over the wide set it is {AM[i16]:+.0f}. <b>What differs by two orders of
magnitude is the candidate set, not the critic.</b></p>
<p><b>[WITHDRAWN — see the correction below]</b> <span style='opacity:.55'> Extrapolating the power law to σ_BC gives {PRED_AT_BC:+.2f}, while
q-landscape-ood measured +{BON_PRIOR} — <b>{OVERSHOOT:.0f}x higher</b>. The order of magnitude agrees, the exact
value does not, and there is a plausible reason: the narrow candidates in this sweep are <b>real actions shifted
in time</b>, which stay politely on the demonstration manifold, whereas BC draws are <b>samples from a generative
model</b> and can leave it in directions the data never took, at the same distance. That is a hypothesis —
<b>higher adversariality per unit distance</b> — and the policy-sample bank tests it directly with this same script.</p>

<h3>So the robot's BoN failure is not over-estimation</h3>
<p>This is where the prescriptions diverge. If bias were the problem, the answer is <b>more pessimism</b>
(ensembles, LCB, CQL). But if the bias at serving width is only +{BON_PRIOR}, then N=8 arg-max tying with no
selection means <b>there is no signal to select on at that width</b>. Per the correction above, though, <b>the robot data can neither support nor refute this yet</b> — all three
conditions have overlapping CIs. This paragraph is a <b>hypothesis</b> suggested by the offline measurements;
the verdict waits on more episodes.</p>
<p><b>Measuring the discrimination directly supports that reading.</b> The candidates above are chunks
executed at OTHER states — a task <b>{WIDEST:.0f}x wider</b>, and therefore far
<b>easier</b>, than what BoN faces. Even so, per candidate the critic is right only <b>{D["ranking_accuracy"]:.3f}</b>
of the time (chance 0.5), with a median margin of {D["per_candidate_margin_median"]:+.1f}. A critic barely above
chance on a task that easy has no reason to separate candidates inside a σ≈{SIGMA_BC} cloud.</p>

<h3>The ensemble is not measuring this uncertainty</h3>
<p>Per-state ensemble disagreement (K=2) is <b>{D["ensemble_disagreement_std"]:.1f}</b>, while the spread across
candidates that the arg-max exploits is <b>{D["within_state_candidate_std"]:.1f}</b> — it sees
<b>{D["disagreement_to_spread_ratio"] * 100:.1f}%</b> of it. LCB at β=1 moves N=8 from {AM[i8]:+.1f} to {LC[i8]:+.1f},
only {(1 - LC[i8] / AM[i8]) * 100:.0f}%. <b>Not a β to tune — a measurement that is missing.</b> (And at K=2, <code>mean − 1σ</code> is
<b>identically</b> <code>min</code> — verified to 4.4e-16. The LCB curve above is therefore the min curve, not an
independent estimator, and β is not a degree of freedom at K=2. Pointed out by worker B, ACRFT-WS.) This is the first
number attached to hypothesis 2 of the
<span class='xref' data-eid='critic-detail-survey'>implementation-detail survey</span> (every method that uses
∇Q runs a 10-ensemble; we ran 2).</p>

<h3>What was built as a result (PR #16)</h3>
<ul>
<li><b><code>CriticQ.q_lcb</code> / <code>q_disagreement</code></b> — min-over-K grows more pessimistic with K
for reasons unrelated to uncertainty, tying ensemble <b>size</b> to pessimism <b>strength</b>. LCB separates them.</li>
<li><b><code>train_patch_critic_cached.py --alpha-cql</code></b> — the conservative term IQL structurally lacks.
Negatives from in-batch <b>wrong-state chunks</b> (which turns the 0.566 statistic above <b>directly</b> into a
training signal), U(−1,1), or a frozen-BC bank. Cal-QL calibration (<code>--calql</code>) lower-bounds OOD Q by
the trajectory's MC return, following <code>nakamotoo/Cal-QL</code>, which applies it to policy-sampled actions
and not to random ones.</li>
<li><b><code>Pi0.sample_n_actions_batched</code></b> — the batched sampler the offline bank needs, in the model
rather than hand-copied into a script, with a test pinning that state i's candidates use state i's prefix.</li>
</ul>
<p>Four critic retrains are queued (K=10 min / K=10 mean / α_cql=10 / α_cql=30), each differing from the deployed
<code>fixed_tau9_min_200k</code> recipe in <b>exactly one factor</b>, so the existing checkpoint is the control.</p>

<h3>Limits — what is settled and what is not</h3>
<ul>
<li><b>Settled.</b> Every number in the tables is computed reproducibly over {D["n_states"]} states; the script is in the repo.</li>
<li><b>Provisional, and the largest limit.</b> The candidates here are <b>transplanted demonstration chunks, not
actual policy draws</b>. The width-bias curve connects the two measurements, but measuring it directly on BC draws
needs the policy-sample bank. That job (<code>sample_policy_chunks.py</code>) is queued; the same script reruns on it.</li>
<li>σ_BC≈{SIGMA_BC} is quoted from q-landscape-ood, where it is a std from the mean while ours is a distance between
two chunks — a factor of about √2. The argument is about <b>orders of magnitude</b>, not the exact position.</li>
<li>One critic ({CC["horizon"]}-step chunk, macro={CC["macro_group_size"]}, K={CC["num_critics"]}). q-landscape-ood showed
over-estimation is universal across 9 critics, but the discrimination numbers here are for the deployed one.</li>
<li>"Tied because there is no signal" is an <b>inference</b>. It is falsifiable: if an α_cql run lifts 0.566 and robot
BoN still ties, the reading is wrong.</li>
</ul>"""

entry = {
    "eid": "argmax-width",
    "worker": "B",
    "date": "2026-09-02 04:10",
    "status": "finding",  # offline part settled; the robot-side reading is explicitly provisional
    "title": "🤖 [워커B] arg-max 편향은 후보 폭의 함수다 — BoN은 그 폭이 없고, critic은 상태를 0.57로만 가린다",
    "summary": (
        f"같은 앵커·같은 단위인데 q-landscape-ood의 BoN 편향(+{BON_PRIOR}, N=16)과 이 프로브({AM[i16]:+.0f}, N=16)가 "
        f"{RATIO:.0f}배 달랐다. 후보 폭을 통제해 재보니 둘은 한 곡선의 두 점이었다 — 편향은 후보가 실행된 행동에서 "
        "얼마나 멀리 뻗느냐에 비례하고, best-of-N이 뻗을 수 있는 거리는 샘플러 자신의 노이즈뿐이다. 따라서 실물에서 "
        "BoN이 무선택과 동률이었던 것은 과대평가가 아니라 그 폭에 신호가 없어서일 가능성이 높다(확률적 추첨이 "
        f"이긴 것과 정합). 판별력을 직접 재면 {D['ranking_accuracy']:.3f} — '다른 상태에서 실행된 청크'라는 훨씬 쉬운 "
        f"과제에서도 우연(0.5)보다 겨우 낫다. 앙상블 K=2는 착취되는 spread의 {D['disagreement_to_spread_ratio'] * 100:.1f}%만 "
        "측정 중이라 LCB가 듣지 않는다. LCB read·CQL/Cal-QL 보수항·배치 샘플러를 구현하고 critic 재학습 4종을 큐에 넣음."
    ),
    "tags": ["워커B", "critic", "OOD", "BoN", "진단", "CQL"],
    "phase": "진단·방법",
    "links": [
        "q-landscape-ood",
        "serving-rollouts-yam",
        "critic-detail-survey",
        "vla-rl-survey",
        "extraction-suite-yam",
    ],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/argmax_bias_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print(f"wrote {out}")
print(
    f"  ranking accuracy {D['ranking_accuracy']:.3f} | N=16 bias {AM[i16]:+.1f} vs prior BoN +{BON_PRIOR} = {RATIO:.0f}x"
)
print(f"  widths: {[round(x, 4) for x in D['width_spread_rms']]}")
print(f"  biases: {[round(x, 1) for x in D['width_argmax_bias']]}")
