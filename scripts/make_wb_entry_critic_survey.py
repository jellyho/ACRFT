"""Hub entry: why BoN/steering works in the baselines' papers and not in ours.

An implementation-detail survey across the six reference codebases we ported from, read out of
the code rather than the papers' prose. New eid; the rollout evidence lives in
serving-rollouts-yam and is linked rather than restated.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
CRITIC = R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k/config.json"
cc = json.loads(CRITIC.read_text())
isp = cc["input_spec"]
OURS_DIM = cc["horizon"] * cc["action_dim"]

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
<tr><th>방법</th><th>critic 앙상블</th><th>critic LayerNorm</th><th>Q가 채점하는 액션 차원</th><th>관측</th><th>평가 환경</th><th>critic 학습</th></tr>
<tr><td>IDQL</td><td>2 <span class='sub'>(:92)</span></td><td><b>없음</b> <span class='sub'>(:170, MLP에 플래그 미전달)</span></td><td>3–8 (단일 액션)</td><td>proprio MLP</td><td>D4RL 시뮬</td><td>오프라인 IQL</td></tr>
<tr><td>CFGRL</td><td>2 <span class='sub'>(:47)</span></td><td>있음 <span class='sub'>(:56)</span></td><td>4–8 (단일 액션)</td><td>proprio</td><td>OGBench 시뮬</td><td>오프라인 IQL</td></tr>
<tr><td>LPS</td><td>2 <span class='sub'>(:512)</span></td><td>있음 <span class='sub'>(:507)</span></td><td>4–8 (단일 액션)</td><td>proprio</td><td>OGBench 시뮬</td><td>오프라인</td></tr>
<tr><td><b>QAM</b></td><td><b>10</b> <span class='sub'>(qam.py:416)</span></td><td>있음 <span class='sub'>(:415)</span></td><td>4–8 (단일 액션)</td><td>proprio</td><td>OGBench 시뮬</td><td>오프라인</td></tr>
<tr><td><b>QPILOTS-U</b></td><td><b>10</b> + 비관 ρ=0.5</td><td>(논문)</td><td>H=5 청크</td><td>CNN 64×64 듀얼뷰</td><td>LIBERO 시뮬</td><td><b>온라인 5e5 스텝, SARSA</b></td></tr>
<tr><td><b>우리</b></td><td><b>2</b></td><td>있음 (전 층)</td><td><b>{OURS_DIM}</b> ({cc["horizon"]}×{cc["action_dim"]} 청크)</td><td>동결 DINOv2 {isp["num_patches"]}패치×{isp["embed_dim"]}</td><td><b>실물 로봇</b></td><td>오프라인, 동결</td></tr>
</table></div>"""

KO = f"""<table class='num'><tr><th>항목</th><th>내용</th></tr>
<tr><th>who</th><td>워커B — 사용자 질문("baseline들은 왜 BoN/DDPG가 잘 됐나")에서 출발</td></tr>
<tr><th>where</th><td>로컬 클론 6종: IDQL, CFGRL, QAM, LPS, Diffusion-QL, FlowDAgger + QPILOTS 논문</td></tr>
<tr><th>what</th><td>critic 구현 디테일 비교 — 앙상블·정규화·액션 차원·관측·평가환경·critic 학습 방식</td></tr>
<tr><th>how</th><td>논문 산문이 아니라 <b>코드에서 파일:줄 단위로</b> 읽음. 우리 값은 critic config.json에서 재계산</td></tr>
<tr><th>why</th><td>같은 critic에서 선택(implicit)은 되는데 조향(QPILOTS)은 무너졌다 — 우리 세팅의 무엇이 다른가</td></tr>
<tr><th>코드</th><td><code>{branch}@{stamp}</code></td></tr></table>

<p><b>질문.</b> 우리가 이식한 baseline들은 대부분 IQL 계열 critic을 쓰고, 논문에서는 BoN·DDPG식 추출이
잘 작동한다. 그런데 <span class='xref' data-eid='serving-rollouts-yam'>우리 실물 롤아웃</span>에서는
argmax(BoN)가 선택을 안 한 것과 같았고 QPILOTS 조향은 켜는 순간 손해였다. OOD 과대평가로 고통받는
쪽은 우리다. 무엇이 다른가?</p>

<h3>먼저 기각된 가설: "우리 critic이 덜 정규화됐다"</h3>
<p>가장 먼저 의심한 것은 LayerNorm이었다(RLPD 계열에서 OOD 외삽을 억제하는 표준 처방). <b>틀렸다.</b>
우리 critic은 모든 블록에 LayerNorm이 있고(<code>patch_critic/critic.py:60,76,78,80</code>), 오히려
<b>IDQL의 critic에는 LayerNorm이 아예 없다</b> — <code>ddpm_iql_learner.py:170</code>에서 MLP를
만들 때 <code>use_layer_norm</code>을 넘기지 않아 기본값 False다. 우리가 그들보다 덜 정규화된 게 아니다.</p>

{TABLE}

<h3>가설 1 — 액션 차원이 두 자릿수 다르다 (가장 큰 격차)</h3>
<p>baseline들의 Q는 <b>3~8차원 단일 액션</b>을 채점한다. 우리 Q는 <b>{OURS_DIM}차원 청크</b>
({cc["horizon"]}스텝 × {cc["action_dim"]}관절)를 채점한다. 데이터 93.8만 프레임으로 {OURS_DIM}차원 공간에
대한 일반화를 요구하는 셈이고, 그 공간에서 N=8 샘플의 커버리지는 사실상 0이다. 데이터 다양체를 벗어난
지점에서 critic의 외삽을 제약하는 것이 아무것도 없으므로, argmax는 "가장 좋은 청크"가 아니라
<b>"가장 크게 과대평가된 청크"</b>를 찾을 확률이 높다. 실물에서 argmax가 10에피소드 중 7개는 stage 1에
멈추고 2개만 stage 4로 가는 양극화가 정확히 그 모양이다.</p>

<h3>가설 2 — ∇Q를 쓰는 방법은 전부 앙상블 10을 쓴다. 우리는 2다</h3>
<p>표에서 갈리는 지점이 뚜렷하다. <b>Q의 순위만 쓰는</b> 방법들(IDQL·CFGRL·LPS)은 앙상블 2로 충분하다.
반면 <b>∇Q를 쓰는</b> 방법들(QAM <code>num_qs=10</code>, QPILOTS <code>J=10</code> + ρ=0.5 비관)은
전부 10을 쓴다. 우리는 둘 다 2로 돌렸다.</p>
<p>이것이 우리 실물 결과와 정확히 맞아떨어진다 — <b>선택(순위)은 성공했고 조향(기울기)은 실패했다.</b>
값과 기울기는 다른 요구조건이고, 기울기 쪽이 훨씬 많은 앙상블을 요구한다. 두 방법을 같은 K=2 critic으로
돌린 것은 한쪽에만 불공정한 조건이었다.</p>

<h3>가설 3 — QPILOTS의 critic은 <b>온라인</b>이다 (제가 놓쳤던 것)</h3>
<p>논문 다이제스트에 기록해두고도 연결하지 못한 사실: QPILOTS의 π0.5-LIBERO 실험은 critic을
<b>온라인으로 5e5 환경 스텝</b> 학습하며, 타깃은 <b>버퍼에 저장된 실제 다음 액션</b>(SARSA)이다. 즉 그들의
조향용 critic은 <b>자기가 조향해 만들어낸 분포 위에서 계속 교정된다.</b> 조향이 정책을 데이터 밖으로 밀면
그 결과가 버퍼에 들어오고 critic이 그것을 배운다.</p>
<p>우리 critic은 완전히 오프라인이고 <b>동결</b>이다. 조향이 만들어낸 액션에 대해 영원히 교정되지 않는다.
같은 알고리즘이라도 이 둘은 다른 방법이다 — 우리가 재현한 것은 QPILOTS의 수식이지 그 학습 루프가 아니다.</p>

<h3>가설 4 — 시뮬 안에서의 평가 vs 실물</h3>
<p>D4RL·OGBench는 <b>데이터를 만든 바로 그 시뮬레이터</b>에서 평가한다. 임베디먼트 갭도, 카메라 변동도,
접촉 노이즈도 없다. 우리는 실물 로봇에서 평가하고 critic의 입력은 실제 카메라의 DINOv2 특징이다.
관측 분포 이동이 있는 상태에서 Q의 절대값은 학습 때와 같은 의미를 갖지 않는다.</p>

<h3>가설 5 — 선택 폭 N</h3>
<p>IDQL 기본값은 N=64(태스크별 32~128). 우리 실물 실험은 N=8이었다. 우리 critic에서 평균 이상 후보
비율이 18.7%이므로 N=8이면 평균 이상 후보가 기대 1.5개뿐이고, <b>16% 확률로 8개 전부가 합격선 아래</b>가
되어 implicit은 균등 추첨으로 퇴화한다. argmax 쪽은 반대로 N이 커질수록 과대평가를 더 잘 찾아낸다.</p>

<h3>무엇을 하면 판별되는가</h3>
<ol>
<li><b>앙상블을 10으로 올린 critic</b>을 학습해 같은 조향 스윕을 반복한다. 가설 2가 맞다면 QPILOTS의
붕괴가 완화되어야 한다. 선택(implicit)은 크게 변하지 않아야 한다 — 이미 순위만 쓰기 때문이다.</li>
<li><b>∇Q 유용성 직접 측정</b>: 데모 액션에서 ∇Q 방향으로 조금 이동했을 때 실제로 더 나은 액션이 되는지를
오프라인에서 잰다. FlowDPG·QAM·FlowDAgger가 전부 ∇Q에 의존하므로 셋의 기대치를 한 번에 정한다.</li>
<li><b>청크 차원 축소</b>: 30스텝 대신 5~10스텝 청크를 채점하는 critic으로 같은 비교를 한다. 가설 1이
맞다면 짧은 청크에서 argmax가 살아나야 하고, 이는 "짧게 실행할수록 좋다"는 BC 스윕 결과와도 맞물린다.</li>
<li><b>N=64 재실행</b>: 선택 규칙 비교를 논문 기본값에서 반복한다.</li>
</ol>

<p><b>한계.</b> 이 서베이는 <b>코드에서 읽은 사실</b>과 <b>가설</b>을 구분한다. 표의 값은 전부 파일:줄로
확인했지만, 가설 1~5의 인과는 아직 어느 것도 검증되지 않았다. QPILOTS 항목의 일부(앙상블·온라인 학습)는
공식 코드가 없어 논문 본문·부록에서 왔다. 우리 쪽 실물 근거는 조건당 10 에피소드다.</p>"""

EN = f"""<table class='num'><tr><th>item</th><th>content</th></tr>
<tr><th>who</th><td>Worker B, prompted by the user's question: why did BoN/DDPG-style extraction work in the baselines' papers?</td></tr>
<tr><th>where</th><td>Six local clones: IDQL, CFGRL, QAM, LPS, Diffusion-QL, FlowDAgger, plus the QPILOTS paper</td></tr>
<tr><th>what</th><td>Implementation-detail survey of the critics: ensemble, normalisation, action dimensionality, observation, evaluation environment, how the critic is trained</td></tr>
<tr><th>how</th><td>Read from the CODE at file:line, not from the papers' prose; our own numbers recomputed from the critic's config.json</td></tr>
<tr><th>why</th><td>With one critic, selection (implicit) worked and steering (QPILOTS) collapsed — what is different about our setting?</td></tr>
<tr><th>code</th><td><code>{branch}@{stamp}</code></td></tr></table>

<p><b>The question.</b> The baselines we ported mostly use IQL-family critics, and their papers report BoN
and DDPG-style extraction working. In <span class='xref' data-eid='serving-rollouts-yam'>our real-robot
rollouts</span> argmax (BoN) matched doing no selection at all, and QPILOTS steering cost from the moment it
was switched on. We are the ones suffering OOD overestimation. What differs?</p>

<h3>First hypothesis, rejected: "our critic is less regularised"</h3>
<p>LayerNorm was the obvious suspect (the standard remedy for OOD extrapolation in the RLPD line).
<b>Wrong.</b> Our critic has LayerNorm in every block (<code>patch_critic/critic.py:60,76,78,80</code>),
while <b>IDQL's critic has none</b> — <code>ddpm_iql_learner.py:170</code> builds the MLP without passing
<code>use_layer_norm</code>, which defaults to False. We are not the less-regularised side.</p>

{TABLE}

<h3>Hypothesis 1 — the action dimensionality differs by two orders of magnitude</h3>
<p>Their Q scores a <b>3–8 dimensional single action</b>. Ours scores a <b>{OURS_DIM}-dimensional chunk</b>
({cc["horizon"]} steps × {cc["action_dim"]} joints). Asking for generalisation over that space from 938k
frames is a different problem, and N=8 samples cover essentially none of it. Nothing constrains the critic's
extrapolation off the data manifold, so argmax finds not "the best chunk" but <b>the most over-valued
one</b> — which is exactly the polarisation observed (7 of 10 episodes stalled at stage 1, 2 reached 4).</p>

<h3>Hypothesis 2 — every method that uses ∇Q uses an ensemble of 10. We used 2</h3>
<p>The table splits cleanly. Methods that use only the <b>ranking</b> of Q (IDQL, CFGRL, LPS) are content
with 2. Methods that use <b>∇Q</b> (QAM <code>num_qs=10</code>; QPILOTS <code>J=10</code> with ρ=0.5
pessimism) all use 10. We ran both families on the same K=2 critic.</p>
<p>That maps exactly onto our result — <b>selection succeeded, steering failed.</b> Values and gradients are
different requirements, and gradients demand far more ensemble. Running both on K=2 was a condition unfair
to one of them.</p>

<h3>Hypothesis 3 — QPILOTS' critic is <b>online</b> (the one I had recorded and failed to connect)</h3>
<p>Noted in the paper digest and never joined up: the QPILOTS pi0.5-LIBERO experiment trains its critic
<b>online for 5e5 environment steps</b> with <b>SARSA targets on the buffer's stored next action</b>. Their
steering critic is continually corrected on the distribution its own steering produces. Ours is fully
offline and <b>frozen</b>: it is never corrected on the actions steering creates. Same equations, different
method — we reproduced their formulas, not their training loop.</p>

<h3>Hypothesis 4 — evaluated inside the simulator vs on a real robot</h3>
<p>D4RL and OGBench evaluate in <b>the very simulator that produced the data</b>: no embodiment gap, no
camera variation, no contact noise. We evaluate on hardware, and the critic's input is DINOv2 features of
real images. Under observation shift, Q's absolute values no longer mean what they meant in training.</p>

<h3>Hypothesis 5 — the width of the selection, N</h3>
<p>IDQL's default is N=64 (32–128 per task). Our robot runs used N=8. With only 18.7% of candidates above V,
N=8 yields ~1.5 above-V candidates and a <b>16% chance that all eight are below</b>, where implicit
degenerates to a uniform draw. argmax has the opposite exposure: larger N finds over-estimates better.</p>

<h3>What would settle it</h3>
<ol>
<li><b>Train a 10-ensemble critic</b> and repeat the steering sweep. If hypothesis 2 holds, QPILOTS' collapse
should soften, while selection should barely move (it only uses ranking).</li>
<li><b>Measure ∇Q's usefulness directly</b>: offline, step from a demo action along ∇Q and check whether the
action actually improves. FlowDPG, QAM and FlowDAgger all lean on ∇Q, so one diagnostic sets all three.</li>
<li><b>Shorten the chunk</b>: score 5–10 step chunks instead of 30. If hypothesis 1 holds, argmax should
recover — and that meshes with the BC sweep's finding that shorter execution is better.</li>
<li><b>Rerun selection at N=64</b>, the papers' default.</li>
</ol>

<p><b>Limits.</b> This survey separates <b>facts read from code</b> from <b>hypotheses</b>. Every table entry
was checked at file:line, but none of hypotheses 1–5 is verified. Parts of the QPILOTS row (ensemble, online
training) come from the paper's text and appendix since no official code exists. Our robot evidence is 10
episodes per condition.</p>"""

entry = {
    "eid": "critic-detail-survey",
    "date": "2026-09-01 23:10",
    "worker": "B",
    "title": "📚 [워커B] baseline은 왜 BoN이 됐고 우리는 안 됐나 — critic 구현 디테일 서베이 (코드 6종)",
    "summary": (
        f"가설 '우리 critic이 덜 정규화됐다'는 기각 — 우리는 전 층 LayerNorm, IDQL critic은 LayerNorm이 없다. "
        f"코드에서 읽은 실제 격차: ① Q가 채점하는 액션이 3~8차원(단일 액션) vs 우리 {OURS_DIM}차원(30스텝 청크), "
        "② ∇Q를 쓰는 방법(QAM·QPILOTS)은 전부 앙상블 10인데 우리는 2 — 순위만 쓰는 IDQL·CFGRL·LPS는 2로 충분하며 "
        "이는 '선택은 되고 조향은 안 된' 우리 실물 결과와 정확히 일치, ③ QPILOTS의 critic은 온라인 5e5 스텝 SARSA로 "
        "조향이 만든 분포 위에서 계속 교정되는데 우리는 동결, ④ 시뮬 내 평가 vs 실물, ⑤ N=64 vs 8. "
        "판별 실험 4개 제시."
    ),
    "tags": ["워커B", "서베이", "critic", "OOD", "구현디테일"],
    "status": "finding",
    "phase": "진단·방법",
    "links": ["serving-rollouts-yam", "extraction-suite-yam", "wa-emaq-bon", "conservatism"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/critic_survey_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "| our chunk dim:", OURS_DIM, "| stamp", branch, stamp)
