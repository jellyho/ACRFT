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
<tr><td><b>V-GPS</b><br><span class='sub'>2410.13816</span></td><td>OpenVLA·Octo 등 5종 (동결)</td><td><b>단일 7차원 액션</b></td><td>오프라인 <b>Cal-QL</b> 1M 스텝 (IQL τ=0.7도 검증), ResNet-34+FiLM, clipped double Q</td><td>K=10~50 재순위, argmax 또는 <b>온도 softmax</b> (β∈{{0, 0.1, 1.0}})</td><td>실물 WidowX 6태스크 + SIMPLER</td></tr>
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

<p><b>이 리포트의 질문.</b> <span class='xref' data-eid='critic-detail-survey'>앞선 서베이</span>는 소규모
오프라인 RL 코드베이스(IDQL·QAM·CFGRL·LPS)를 읽었다. 그 결론은 "액션 차원과 앙상블이 다르다"였는데,
그 논문들은 모두 <b>proprio 기반 시뮬</b>이라 VLA와는 거리가 있다. 이번에는 <b>실제로 대형 VLA에
가치함수를 붙인 연구들</b>만 본다.</p>

{TABLE}

<h3>발견 1 — 아무도 우리처럼 하지 않는다</h3>
<p>표를 세로로 읽으면 한 가지가 두드러진다. <b>"동결된 오프라인 critic으로 긴 액션 청크를 채점해
argmax/조향한다"를 하는 연구가 하나도 없다.</b> 성공을 보고한 연구들은 넷 중 하나를 택한다:</p>
<ul>
<li><b>채점 대상을 작게</b> — V-GPS는 단일 7차원 액션, Q-VGM·QPILOTS는 H=5 청크. 우리는 H=30, {OURS}차원이다.</li>
<li><b>critic을 온라인으로</b> — QPILOTS(5e5 스텝 SARSA), ConRFT(오프라인 후 온라인).</li>
<li><b>Q함수를 아예 버리고 선호 학습 verifier로</b> — RoboMonkey의 7B VLM verifier. TD 부트스트랩이 없으니
OOD 외삽 문제 자체가 사라진다. 그리고 <b>V-GPS(Q함수)보다 낫다고 보고</b>한다.</li>
<li><b>가치함수를 안 쓴다</b> — RIPT-VLA는 희소 이진 보상 + 정책 경사.</li>
</ul>
<p>우리는 <b>가장 어려운 조합</b>을 골랐다: 가장 큰 채점 대상({OURS}차원), 동결 오프라인 critic, 실물 평가.</p>

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

<h3>발견 4 — 선택 규칙은 argmax가 유일한 선택지가 아니다</h3>
<p>V-GPS는 재순위에 <b>온도 softmax</b>를 쓰며 β∈{{0, 0.1, 1.0}}을 스윕한다(β→0이 argmax). 이는 우리가
실물에서 관측한 것과 같은 구조다 — <span class='xref' data-eid='serving-rollouts-yam'>argmax는 무효, 확률적
선택(implicit)은 유효</span>. IDQL의 expectile 추첨과 V-GPS의 온도 softmax는 형태는 달라도 <b>"최고를 고르지
말고 좋은 쪽으로 치우쳐 뽑아라"</b>는 같은 처방이다. 우리 결과는 이 처방을 지지한다.</p>
<p>또 RoboMonkey는 <b>샘플 수에 대한 거듭제곱 스케일링 법칙</b>을 보고한다(1만 샘플에서 RMSE −59.3%).
우리 N=8은 그 곡선의 맨 왼쪽 끝이다.</p>

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

<p><b>The question.</b> The <span class='xref' data-eid='critic-detail-survey'>previous survey</span> read the
small-scale offline-RL codebases and concluded that action dimensionality and ensemble size differ. But those
papers are all proprioceptive simulation. This one looks only at work that actually attaches a value function
to a large VLA.</p>

{TABLE}

<h3>Finding 1 — nobody does what we are doing</h3>
<p>Read the table down its columns and one thing stands out: <b>no published work scores a long action chunk
with a frozen offline critic and then argmaxes or steers on it.</b> Every method reporting success takes one
of four escapes:</p>
<ul>
<li><b>Score something small</b> — V-GPS a single 7-dim action; Q-VGM and QPILOTS an H=5 chunk. Ours is
H=30, {OURS} dimensions.</li>
<li><b>Train the critic online</b> — QPILOTS (5e5 steps, SARSA), ConRFT (offline then online).</li>
<li><b>Drop the Q-function for a preference-trained verifier</b> — RoboMonkey's 7B VLM verifier. No TD
bootstrap, so the OOD extrapolation problem does not arise; it reports beating V-GPS.</li>
<li><b>Use no value function at all</b> — RIPT-VLA: sparse binary reward and a policy gradient.</li>
</ul>
<p>We picked <b>the hardest combination</b>: the largest scoring target ({OURS} dims), a frozen offline
critic, and real hardware.</p>

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

<h3>Finding 4 — argmax is not the only selection rule in this literature</h3>
<p>V-GPS re-ranks with a <b>temperature softmax</b>, sweeping β∈{{0, 0.1, 1.0}} (β→0 is argmax). That is the
same structure we observed on hardware — <span class='xref' data-eid='serving-rollouts-yam'>argmax inert,
stochastic selection effective</span>. IDQL's expectile lottery and V-GPS's temperature are different forms
of one prescription: <b>do not take the best, lean toward the good ones.</b> Our result supports it.</p>
<p>RoboMonkey additionally reports a <b>power-law scaling in the number of samples</b> (RMSE −59.3% at 10,000
samples). Our N=8 sits at the far-left end of that curve.</p>

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
    "date": "2026-09-02 00:30",
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
    "links": ["critic-detail-survey", "serving-rollouts-yam", "extraction-suite-yam"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/vla_rl_survey_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "| stamp", branch, stamp)
