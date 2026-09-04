"""Hub entry: how VLA-scale RL work actually couples a value function to a big policy.

Companion to critic-detail-survey, which read the small-scale offline-RL codebases. This one
looks at the VLA-scale literature, where the question we are stuck on -- a frozen offline critic
scoring a long action chunk -- is answered very differently by everyone who has published on it.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
cc = json.loads((R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k/config.json").read_text())
OURS = cc["horizon"] * cc["action_dim"]

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

TABLE = f"""<div class='tblwrap'><table class='num'>
<tr><th>연구</th><th>베이스 정책</th><th>가치함수가 채점하는 것</th><th>critic 학습</th><th>추출/선택 방식</th><th>평가</th></tr>
<tr><td><b>V-GPS</b><br><span class='sub'>2410.13816</span></td><td>OpenVLA·Octo 등 5종 (동결)</td><td><b>단일 7차원 액션</b></td><td>오프라인 <b>Cal-QL</b> 1M 스텝 (IQL τ=0.7도 검증), ResNet-34+FiLM, clipped double Q</td><td>K=10~50 재순위. <b>실물은 K=50 + argmax(β→0)</b>; 온도 softmax β∈{{0, 0.1, 1.0}}은 시뮬 스윕 (App. B.2)</td><td>실물 WidowX 6태스크 + SIMPLER</td></tr>
<tr><td><b>Q-VGM</b><br><span class='sub'>2606.08015</span></td><td><b>π0.5</b> (VLM 동결, action expert만)</td><td><b>청크 H=5</b> (7-DoF → 35차원)</td><td>오프라인 Cal-QL 앙상블, LayerNorm, <b>액션을 매 은닉층에 재결합</b>, 이후 동결</td><td>∇Q를 velocity로 변환: 1-step 투영 → J회 ascent → <b>keep-best, 실패 시 base 액션으로 폴백</b> → residual matching. BPTT 없음</td><td>LIBERO 40 + RoboTwin 10 + 실물 2태스크</td></tr>
<tr><td><b>RoboMonkey</b><br><span class='sub'>2506.17811</span></td><td>OpenVLA·CogACT·Octo·SpatialVLA</td><td>—</td><td><b>Q함수 없음.</b> 7B VLM <b>선호 학습 verifier</b></td><td>샘플 + 가우시안 섭동 + 다수결 → verifier 선택. 샘플 수에 대한 <b>거듭제곱 스케일링 법칙</b> (1만 샘플에서 RMSE −59.3%)</td><td>V-GPS보다 우수하다고 보고</td></tr>
<tr><td><b>RIPT-VLA</b><br><span class='sub'>2505.17016</span></td><td>VLA 사후학습</td><td>—</td><td><b>가치함수 자체를 쓰지 않음</b> (희소 이진 보상)</td><td>정책 경사 (leave-one-out 이점)</td><td>시뮬</td></tr>
<tr><td><b>ConRFT</b></td><td>VLA (consistency policy)</td><td>단일 액션</td><td>오프라인 BC+Q로 시작해 <b>온라인 파인튜닝</b></td><td>일관성 목적으로 통합</td><td>실물</td></tr>
<tr><td><b>QPILOTS-U</b><br><span class='sub'>2606.14801</span></td><td>π0.5 (동결)</td><td>청크 H=5</td><td><b>온라인 5e5 스텝, SARSA 타깃</b>, 앙상블 J=10 + 비관 ρ=0.5</td><td>ODE 스텝마다 ∇Q 주입 (학습 0)</td><td>LIBERO 시뮬</td></tr>
<tr><td><b>우리</b></td><td>π0.5 (동결)</td><td><b>청크 H=30 → {OURS}차원</b></td><td>오프라인 IQL(expectile 0.9), 앙상블 <b>2</b>, 액션은 <b>입력에서 1회</b> 토큰화, 이후 동결</td><td>BoN argmax / implicit 추첨 / QPILOTS 조향</td><td><b>실물 YAM</b></td></tr>
</table></div>"""

KO = f"""<table class='num'><tr><th>항목</th><th>내용</th></tr>
<tr><th>who</th><td>워커B — 사용자 요청("VLA같은 큰 모델과 RL 결합 연구 위주로 검색")</td></tr>
<tr><th>where</th><td>arXiv: V-GPS 2410.13816, Q-VGM 2606.08015, RoboMonkey 2506.17811, RIPT-VLA 2505.17016, QPILOTS 2606.14801, ConRFT</td></tr>
<tr><th>what</th><td>VLA 스케일에서 가치함수를 큰 정책에 어떻게 붙이는가 — 채점 대상·학습 방식·선택 규칙의 비교</td></tr>
<tr><th>how</th><td>논문 본문/HTML에서 구현 디테일을 추출. 우리 값은 critic config.json에서 재계산</td></tr>
<tr><th>why</th><td>우리 실물 결과(조향 붕괴·argmax 무효)가 이 문헌에서 어디에 위치하는지 확인</td></tr>
<tr><th>코드</th><td><code>{branch}@{stamp}</code></td></tr></table>

<div class='missing'><b>정정 (게시 직후, 사용자 지적).</b> 초판의 "발견 1"은 <b>"아무도 우리처럼 하지 않는다"</b>로
읽히게 썼는데 과했다. <b>동결 오프라인 critic + BoN은 여러 곳에서 한다</b> — 바로 이 표의 V-GPS(Cal-QL 동결,
K=50 재순위)가 그렇고, 우리 자신의 <span class='xref' data-eid='deas'>DEAS 재현</span>도 그렇다. DEAS는
VLA + 액션 시퀀스(청크) + 분포형 가치 + BoN이라 <b>청크 단위 BoN의 직접 선례</b>다. 정확한 진술은
"BoN을 하는 곳이 없다"가 아니라 <b>"우리만큼 큰 채점 대상(H=30, {OURS}차원)을 쓰는 곳이 없다"</b>이다.
아래 본문은 그 좁은 주장으로 고쳐 읽어야 한다.</div>

<p><b>그리고 더 중요한 재구성.</b> 사용자 지적대로, 이 논문들은 형태는 달라도 <b>대부분 critic을 쓰고 있고
그들에게는 그 critic이 쓸 만했다</b>. 그러므로 질문은 "critic을 쓰는가"가 아니라 <b>"무엇이 critic을 믿을 만하게
만드는가"</b>이다. 다만 '믿을 만하다'가 생각보다 약할 수 있다는 내부 증거가 하나 있다 — 우리
<span class='xref' data-eid='deas'>DEAS 재현</span>의 결론은 <b>"critic은 VLA와 동률(못 이기고 안 해침)"</b>이었고,
지난 '유해' 판정은 n=25 노이즈였다. 즉 <b>청크 BoN이 베이스를 못 이기는 것은 이번이 처음이 아니라 우리
계보에서 재현되는 패턴</b>이다. 오늘의 argmax ≈ 선택없음(1.70 vs 1.70)은 그 연장선에 있다.</p>

<p><b>이 리포트의 질문.</b> <span class='xref' data-eid='critic-detail-survey'>앞선 서베이</span>는 소규모
오프라인 RL 코드베이스(IDQL·QAM·CFGRL·LPS)를 읽었다. 그 결론은 "액션 차원과 앙상블이 다르다"였는데,
그 논문들은 모두 <b>proprio 기반 시뮬</b>이라 VLA와는 거리가 있다. 이번에는 <b>실제로 대형 VLA에
가치함수를 붙인 연구들</b>만 본다.</p>

{TABLE}

<h3>발견 1 — critic은 다들 쓴다. 다만 <b>채점 대상</b>이 우리만 크다</h3>
<p>동결 오프라인 critic으로 BoN을 하는 것 자체는 흔하다(V-GPS, DEAS). 표를 세로로 읽을 때 우리만 다른 칸은
<b>Q가 채점하는 대상의 크기</b>다 — 성공을 보고한 연구들은 다음 중 하나로 그 부담을 줄인다:</p>
<ul>
<li><b>채점 대상을 작게</b> — V-GPS는 단일 7차원 액션, Q-VGM·QPILOTS는 H=5 청크. 우리는 H=30, {OURS}차원이다.</li>
<li><b>critic을 온라인으로</b> — QPILOTS(5e5 스텝 SARSA), ConRFT(오프라인 후 온라인).</li>
<li><b>Q함수를 아예 버리고 선호 학습 verifier로</b> — RoboMonkey의 7B VLM verifier. TD 부트스트랩이 없으니
OOD 외삽 문제 자체가 사라진다. 그리고 <b>V-GPS(Q함수)보다 낫다고 보고</b>한다.</li>
<li><b>가치함수를 안 쓴다</b> — RIPT-VLA는 희소 이진 보상 + 정책 경사.</li>
</ul>
<p>우리는 이 넷 중 <b>어느 것도 택하지 않았다</b>: {OURS}차원 채점 대상, 동결 오프라인 critic, 실물 평가.
각각은 선례가 있지만 <b>이 조합은 없다</b>.</p>

<h3>발견 2 — Q-VGM이 우리 결과를 독립적으로 재현했다</h3>
<p>Q-VGM은 우리와 <b>같은 베이스(π0.5), 같은 동결 방식(VLM 동결, action expert만 학습)</b>을 쓴다.
그들이 보고한 것:</p>
<ul>
<li><b>Diffusion-QL(=우리가 뺀 DQL)은 SFT 아래로 떨어진다</b> — LIBERO 69.5% vs SFT 75.0%. 우리는
메모리 때문에 뺐는데, 성능 근거로도 뺄 만했다는 독립 확인이다.</li>
<li>"다단계 디노이징을 통과하는 역전파는 <b>VLA 스케일에서 수치적으로 불안정</b>하다" — QAM·FlowDPG가
BPTT를 피하는 이유를 같은 언어로 말한다.</li>
<li>"critic은 실행 가능한 <b>깨끗한</b> 액션을 평가하는데 flow 정책은 <b>노이즈 낀</b> 중간 상태를 지난다" —
QPILOTS의 Tweedie 투영과 같은 문제 인식이다.</li>
</ul>

<h3>발견 3 — 우리에게 없는 구현 디테일 두 가지</h3>
<p><b>① 액션을 매 은닉층에 재결합한다(Q-VGM).</b> 그들의 critic은 액션을 <b>모든 층에서</b> 다시 concat해
"액션 민감도를 보존"한다. 우리 critic은 액션을 <b>입력에서 한 번</b> 토큰으로 만들고 3층 트랜스포머에
흘려보낸다. 층이 깊어질수록 액션 정보가 관측에 묻히면 ∇Q가 무뎌지는데, 이는 <b>순위(선택)에는 거의
영향이 없고 기울기(조향)에 직접 타격</b>이다 — 우리 실물 결과의 비대칭과 정확히 같은 방향이다.</p>
<p><b>② 실패 시 폴백(Q-VGM).</b> 그들은 ascent 후 <b>keep-best</b>를 하고, 개선이 없으면 <b>base 액션으로
되돌린다</b>. 즉 조향은 "더 나아지면 채택"이지 무조건 적용이 아니다. QPILOTS에도 우리 구현에도 이
안전밸브가 없다 — 우리 α 스윕에서 조향이 <b>켜는 순간부터</b> 손해였던 것(1.80 → 1.10)은 개선이 없을 때
되돌릴 방법이 없었기 때문일 수 있다.</p>

<h3>발견 4 — 선택 규칙 <span class='sub'>(정정됨)</span></h3>
<div class='missing'><b>정정 (2026-09-04).</b> 초판은 <b>"V-GPS는 온도 softmax를 쓴다 … 우리 결과는 이
처방을 지지한다"</b>고 썼다. <b>논문을 다시 읽으니 틀렸고, 부호가 반대다.</b> App. B.2:
<i>"In the real-world evaluations with the Cal-QL value function, we used K = 50 and we found selecting the
action <b>greedily by setting β → 0</b> leads to satisfactory results. In simulation, we swept over
K = {{10, 50}} and β = {{0, 0.1, 1.0}} and report the <b>best result for each policy</b>."</i>
<br>즉 <b>V-GPS의 실물 규칙은 argmax</b>이고, β 스윕은 시뮬레이션 한정이며 거기서도 "정책별 최고치 보고"라
soft가 greedy를 이겼다는 근거가 아니다. 우리 실물 결과를 지지하기는커녕 <b>반대 방향의 증거</b>다 —
실물 하드웨어에서 K=50 argmax가 만족스럽게 작동했다는 보고이므로.
<br>그리고 우리 쪽 근거도 부족했다: argmax(1.70)와 무선택(1.70)이 <b>정확히 같고</b>, 추첨(2.70)의 우위는
p=0.06–0.097로 <b>유의하지 않다</b>(n=10). <span class='xref' data-eid='argmax-width'>argmax-width</span>의
정정 참조. "우리 결과는 이 처방을 지지한다"는 문장을 철회한다.</div>
<p><b>남는 것은 K다.</b> V-GPS는 실물에서 <b>K=50</b>, IDQL 기본값은 64인데 <b>우리는 N=8</b>이었다. 선택
규칙이 아니라 <b>후보 수</b>가 우리와 이 문헌 사이의 확인된 차이다.</p>
<p>또 RoboMonkey는 <b>샘플 수에 대한 거듭제곱 스케일링 법칙</b>을 보고한다(1만 샘플에서 RMSE −59.3%).
우리 N=8은 그 곡선의 맨 왼쪽 끝이다.</p>


<h3>발견 5 — 진짜 actor-critic으로 <b>학습</b>하는 연구는 생각보다 적다</h3>
<p>가치신호를 어디에 쓰는지로 나누면 지형이 달라진다.</p>
<div class='tblwrap'><table class='num'>
<tr><th>가치신호의 쓰임</th><th>연구</th><th>정책 파라미터를 Q로 갱신하나</th></tr>
<tr><td><b>추론 시점 선택만</b></td><td>V-GPS, RoboMonkey, QPILOTS-U, IDQL, <b>DEAS</b>, 우리 bon/implicit</td><td>아니오</td></tr>
<tr><td><b>상태값·진척도로 재가중</b></td><td>AWR 계열, <b>RECAP(π0.6 방식)</b></td><td>간접 — 가치가 <b>상태/진척</b> 수준이고 현재정책 action-value가 아님</td></tr>
<tr><td><b>가치함수 없음</b></td><td>RIPT-VLA, SimpleVLA-RL</td><td>아니오 (희소 보상 정책경사)</td></tr>
<tr><td><b>진짜 action-value actor-critic</b></td><td>PA-RL, Q-VGM, ConRFT, Diffusion-QL</td><td><b>예</b></td></tr>
</table></div>
<p>마지막 칸이 눈에 띄게 얇다. 그리고 그 안에서도 <b>Diffusion-QL은 실패로 보고</b>되고(Q-VGM: LIBERO 69.5% vs
SFT 75.0%), Q-VGM 자신은 BPTT를 피해 <b>Q로 만든 타깃을 residual로 회귀</b>하는 우회로를 쓴다. 즉 "Q의 기울기로
큰 정책을 직접 미는" 순수한 형태는 VLA 스케일에서 아직 잘 되는 사례가 드물다.</p>
<p>특히 <b>π0.6의 RECAP이 action-value가 아니라 상태/진척 수준 가치</b>를 쓴다는 점은 시사적이다 — 프런티어
쪽이 Q(s,a)에서 <b>멀어지는</b> 방향으로 움직이고 있다. 우리 링은 반대로 Q(s,a) 위에 8개 arm을 세웠다.</p>
<h3>그래서 무엇을 바꿀 것인가 — 우선순위</h3>
<ol>
<li><b>채점 청크를 줄인다.</b> 문헌 전체가 H=1~5인데 우리만 H=30이다. macro_group_size를 줄여
H=5 프리픽스를 채점하는 critic은 이미 우리 구조로 학습 가능하다(g5 critic이 그 방향이었다). 가장 큰
격차이자 가장 값싼 실험이다.</li>
<li><b>앙상블 2 → 10.</b> ∇Q를 쓰는 연구는 전부 10 이상이고, 우리 critic은 멤버당 4.57M이라
K=10이 45.7M — 3.35B 정책의 1.4%다. 비용이 사실상 없다.</li>
<li><b>액션을 매 층에 재결합</b>(Q-VGM 방식)하는 critic 변형. 조향에만 선택적으로 이득이 있어야 하며,
그 비대칭 자체가 가설의 검증이 된다.</li>
<li><b>조향에 폴백을 넣는다</b> — 개선이 없으면 base로. 구현 몇 줄이고, 최악의 경우를 base 성능으로
막는다.</li>
<li><b>선택에 온도를 도입</b>(V-GPS β 스윕). 우리 implicit은 τ가 critic에 묶여 있어 자유도가 없는데,
온도 softmax는 연속적으로 조절 가능한 축을 준다.</li>
</ol>

<p><b>한계.</b> 표의 값은 각 논문의 본문/HTML에서 읽었고 코드로 확인한 것은 아니다(V-GPS·Q-VGM·
RoboMonkey는 공개 코드가 있으나 이번에는 받지 않았다). ConRFT 행은 검색 요약 수준이며 원문을 읽지
않았다 — 다른 행보다 신뢰도가 낮다. 발견 3의 인과("액션 재결합이 없어서 ∇Q가 무뎌졌다")는 가설이며
아직 측정되지 않았다.</p>"""

EN = f"""<table class='num'><tr><th>item</th><th>content</th></tr>
<tr><th>who</th><td>Worker B, at the user's request: survey the work that couples RL to large VLA models</td></tr>
<tr><th>where</th><td>arXiv: V-GPS 2410.13816, Q-VGM 2606.08015, RoboMonkey 2506.17811, RIPT-VLA 2505.17016, QPILOTS 2606.14801, ConRFT</td></tr>
<tr><th>what</th><td>How VLA-scale work attaches a value function to a large policy: what is scored, how the critic is trained, how the action is chosen</td></tr>
<tr><th>how</th><td>Details read from the papers' text/HTML; our own numbers recomputed from the critic config</td></tr>
<tr><th>why</th><td>To locate our real-robot result (steering collapsed, argmax was inert) within this literature</td></tr>
<tr><th>code</th><td><code>{branch}@{stamp}</code></td></tr></table>

<div class='missing'><b>Correction (posted shortly after publication, user-flagged).</b> The first version of
"Finding 1" read as <b>"nobody does what we do"</b>, which overstates it. <b>Frozen-offline-critic BoN is
common</b> — V-GPS in this very table (frozen Cal-QL, K=50 re-ranking) does it, and so does our own
<span class='xref' data-eid='deas'>DEAS reproduction</span>: VLA + action sequence (chunk) + distributional
value + BoN, a direct precedent for chunk-level BoN. The accurate claim is not "nobody does BoN" but
<b>"nobody scores a target as large as ours (H=30, {OURS} dims)"</b>. Read the section below in that narrower
sense.</p></div>

<p><b>A more important reframing.</b> As the user points out, these papers mostly DO use a critic, and for them
it was good enough. So the question is not "does anyone use a critic" but <b>"what makes a critic trustworthy"</b>.
One internal caveat on how strong "trustworthy" is: our own <span class='xref' data-eid='deas'>DEAS
reproduction</span> concluded the <b>critic tied with the VLA — neither beating nor harming it</b>, and the
earlier "critic is harmful" verdict was n=25 noise. So <b>chunk-level BoN failing to beat the base is not new
today; it is a pattern that reproduces in our line</b>. Today's argmax ≈ no-selection (1.70 vs 1.70) continues it.</p>

<p><b>The question.</b> The <span class='xref' data-eid='critic-detail-survey'>previous survey</span> read the
small-scale offline-RL codebases and concluded that action dimensionality and ensemble size differ. But those
papers are all proprioceptive simulation. This one looks only at work that actually attaches a value function
to a large VLA.</p>

{TABLE}

<h3>Finding 1 — everyone uses a critic; only our <b>scoring target</b> is large</h3>
<p>Frozen-offline-critic BoN is itself common (V-GPS, DEAS). Read down the columns and the cell where we differ
is <b>the size of what Q scores</b> — the works reporting success reduce that burden one of four ways:</p>
<ul>
<li><b>Score something small</b> — V-GPS a single 7-dim action; Q-VGM and QPILOTS an H=5 chunk. Ours is
H=30, {OURS} dimensions.</li>
<li><b>Train the critic online</b> — QPILOTS (5e5 steps, SARSA), ConRFT (offline then online).</li>
<li><b>Drop the Q-function for a preference-trained verifier</b> — RoboMonkey's 7B VLM verifier. No TD
bootstrap, so the OOD extrapolation problem does not arise; it reports beating V-GPS.</li>
<li><b>Use no value function at all</b> — RIPT-VLA: sparse binary reward and a policy gradient.</li>
</ul>
<p>We took <b>none of the four</b>: a {OURS}-dim scoring target, a frozen offline critic, and real hardware.
Each has precedent on its own; <b>the combination does not</b>.</p>

<h3>Finding 2 — Q-VGM independently reproduces our result</h3>
<p>Q-VGM uses <b>the same base (π0.5) and the same freezing (VLM frozen, action expert only)</b> as we do.
They report:</p>
<ul>
<li><b>Diffusion-QL — the DQL arm we dropped — falls below SFT</b>: 69.5% vs 75.0% on LIBERO. We dropped it
for memory; this is independent evidence it deserved dropping on performance too.</li>
<li>Backpropagating through the multi-step denoising chain is "numerically unstable at VLA scale" — the same
reason QAM and FlowDPG avoid BPTT.</li>
<li>"The critic evaluates executable clean actions while the flow policy evolves through noisy intermediate
states" — the same problem QPILOTS' Tweedie projection addresses.</li>
</ul>

<h3>Finding 3 — two implementation details we lack</h3>
<p><b>(1) The action is re-concatenated at every hidden layer (Q-VGM)</b>, explicitly "to preserve action
sensitivity". Our critic tokenises the action <b>once at the input</b> and lets three transformer layers mix
it with the patches. If action information washes out with depth, ∇Q blunts — which would barely affect
<b>ranking</b> and directly damage <b>gradients</b>: exactly the asymmetry our robot results show.</p>
<p><b>(2) A fallback when ascent fails (Q-VGM).</b> They <b>keep-best</b> after ascent and revert to the base
action if it did not improve. Steering is "adopt if better", not "always apply". Neither QPILOTS nor our
implementation has that valve — and our sweep showed steering costing <b>from the moment it is switched
on</b> (1.80 → 1.10), which is what having no way to revert looks like.</p>

<h3>Finding 4 — the selection rule <span class='sub'>(corrected)</span></h3>
<div class='missing'><b>Correction (2026-09-04).</b> The first version said <b>"V-GPS re-ranks with a
temperature softmax … our result supports it"</b>. <b>Re-reading the paper, that is wrong, and the sign is
backwards.</b> App. B.2: <i>"In the real-world evaluations with the Cal-QL value function, we used K = 50 and
we found selecting the action <b>greedily by setting β → 0</b> leads to satisfactory results. In simulation,
we swept over K = {{10, 50}} and β = {{0, 0.1, 1.0}} and report the <b>best result for each policy</b>."</i>
<br>So <b>V-GPS's real-robot rule IS arg-max</b>; the β sweep is simulation-only and reports a per-policy
best, which is not evidence that soft beat greedy. Far from supporting our reading, it is evidence the other
way — a report that K=50 arg-max worked satisfactorily on hardware.
<br>Our own side was thin too: arg-max (1.70) and no selection (1.70) are <b>exactly equal</b>, and the
lottery's edge (2.70) is <b>not significant</b> at p=0.06–0.097 with n=10 — see the correction in
<span class='xref' data-eid='argmax-width'>argmax-width</span>. The sentence "our result supports it" is
withdrawn.</div>
<p><b>What survives is K.</b> V-GPS runs <b>K=50</b> on hardware and IDQL's default is 64, while <b>we ran
N=8</b>. The confirmed difference between us and this literature is the number of candidates, not the rule
used to pick among them.</p>
<p>RoboMonkey additionally reports a <b>power-law scaling in the number of samples</b> (RMSE −59.3% at 10,000
samples). Our N=8 sits at the far-left end of that curve.</p>


<h3>Finding 5 — fewer works actually <b>train</b> with an action-value actor-critic than it seems</h3>
<p>Sorting by where the value signal is used changes the picture.</p>
<div class='tblwrap'><table class='num'>
<tr><th>use of the value signal</th><th>works</th><th>policy params updated by Q?</th></tr>
<tr><td><b>test-time selection only</b></td><td>V-GPS, RoboMonkey, QPILOTS-U, IDQL, <b>DEAS</b>, our bon/implicit</td><td>no</td></tr>
<tr><td><b>reweighting by state value / progress</b></td><td>AWR-family, <b>RECAP (π0.6-style)</b></td><td>indirect — the value is <b>state/progress</b> level, not a current-policy action value</td></tr>
<tr><td><b>no value function</b></td><td>RIPT-VLA, SimpleVLA-RL</td><td>no (sparse-reward policy gradient)</td></tr>
<tr><td><b>true action-value actor-critic</b></td><td>PA-RL, Q-VGM, ConRFT, Diffusion-QL</td><td><b>yes</b></td></tr>
</table></div>
<p>That last row is conspicuously thin — and inside it <b>Diffusion-QL is reported as failing</b> (Q-VGM: 69.5%
vs SFT 75.0%), while Q-VGM itself detours around BPTT by <b>regressing a residual onto Q-improved targets</b>.
The pure form — pushing a large policy directly with ∇Q — has few working examples at VLA scale.</p>
<p>Notably <b>π0.6's RECAP uses a state/progress-level value rather than an action value</b>, which suggests the
frontier is moving <b>away</b> from Q(s,a). Our ring built eight arms on top of Q(s,a).</p>
<h3>What to change, in order</h3>
<ol>
<li><b>Shorten the scored chunk.</b> The literature scores H=1–5; we score H=30. A critic over an H=5 prefix
is already trainable in our architecture. Biggest gap, cheapest experiment.</li>
<li><b>Ensemble 2 → 10.</b> Every gradient-using method uses ≥10; our critic is 4.57M per member, so K=10 is
45.7M — 1.4% of the 3.35B policy. Effectively free.</li>
<li><b>Re-concatenate the action at each layer</b> (Q-VGM's design). It should help steering selectively, and
that asymmetry is itself the test.</li>
<li><b>Add a fallback to steering</b> — revert to base when it does not improve. A few lines, and it bounds
the worst case at base performance.</li>
<li><b>Add a temperature to selection</b> (V-GPS's β). Our implicit rule has τ pinned to the critic and no
free knob; a temperature gives a continuous axis.</li>
</ol>

<p><b>Limits.</b> Table entries were read from each paper's text/HTML, not verified against code (V-GPS,
Q-VGM and RoboMonkey have public code that we did not pull). The ConRFT row is at search-summary level and
is less reliable than the others. The causal claim in Finding 3 is a hypothesis, not a measurement.</p>"""

entry = {
    "eid": "vla-rl-survey",
    "date": "2026-09-02 01:05",
    "worker": "B",
    "title": "📚 [워커B] VLA×RL 문헌 서베이 — 아무도 30스텝 청크를 동결 critic으로 argmax하지 않는다",
    "summary": (
        f"대형 VLA에 가치함수를 붙인 연구 6종 비교. 성공한 연구는 넷 중 하나를 택한다: 채점 대상을 작게(V-GPS 단일 "
        f"7차원, Q-VGM·QPILOTS H=5 청크 / 우리 H=30 {OURS}차원), critic을 온라인으로(QPILOTS 5e5 SARSA, ConRFT), "
        "Q함수 대신 선호 verifier(RoboMonkey 7B VLM — V-GPS보다 낫다고 보고), 또는 가치함수를 안 씀(RIPT-VLA). "
        "우리는 가장 어려운 조합을 골랐다. Q-VGM은 같은 π0.5 위에서 Diffusion-QL이 SFT 아래(69.5 vs 75.0)라고 "
        "보고해 우리의 DQL 제거를 독립 확인하고, 우리에게 없는 두 디테일을 쓴다 — 액션을 매 층 재결합, 조향 실패 시 "
        "base로 폴백. V-GPS의 온도 softmax는 우리 implicit 우세와 같은 처방이다."
    ),
    "tags": ["워커B", "서베이", "VLA", "RL", "critic", "문헌"],
    "status": "finding",
    "phase": "진단·방법",
    "links": ["argmax-width", "critic-detail-survey", "serving-rollouts-yam", "extraction-suite-yam", "deas"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/vla_rl_survey_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "| stamp", branch, stamp)
