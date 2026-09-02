"""Hub entry: every LEGOPROG robot condition against its method-only-diff control.

Every number is recomputed by scripts/legoprog_stats.py from .scratch/legoprog_v3.xlsx on each run.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
S = json.loads((R / ".scratch/extraction/legoprog_stats.json").read_text())


def _git(*args):
    return subprocess.run(["git", "-C", str(R), *args], capture_output=True, text=True, check=False).stdout.strip()


stamp = _git("log", "-1", "--format=%h") + ("+dirty" if _git("status", "--porcelain", "-uno") else "")
branch = _git("rev-parse", "--abbrev-ref", "HEAD")
T = {r["name"]: r for r in S["table"]}
D = S["decomposition"]
ST = S["steering"]
CTRL, CM = S["control"], S["control_mean"]
DL = list(D)
n_raw = sum(r["p"] < 0.05 for r in S["table"])
n_holm = sum(r["holm"] < 0.05 for r in S["table"])
BLK = {"select": "선택 규칙", "adaptive": "적응 커밋", "steer": "QPILOTS 조향"}
BLK_EN = {"select": "selection", "adaptive": "adaptive commitment", "steer": "QPILOTS steering"}


def table(*, en=False):
    rows = sorted(S["table"], key=lambda r: -r["mean"])
    h = (
        "<tr><th>condition</th><th>block</th><th>mean</th><th>Δ vs control</th><th>95% CI</th><th>p</th><th>Holm</th></tr>"
        if en
        else "<tr><th>조건</th><th>블록</th><th>평균</th><th>Δ vs 통제군</th><th>95% CI</th><th>p</th><th>Holm</th></tr>"
    )
    b = "".join(
        f"<tr><td>{r['name']}</td><td>{(BLK_EN if en else BLK)[r['block']]}</td><td>{r['mean']:.2f}</td><td>{r['delta']:+.2f}</td>"
        f"<td>[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]</td><td>{'<b>' if r['p'] < 0.05 else ''}{r['p']:.3f}{'</b>' if r['p'] < 0.05 else ''}</td><td>{r['holm']:.3f}</td></tr>"
        for r in rows
    )
    return f"<table class='num'>{h}{b}</table>"


def dec_rows(*, en=False):
    labs = list(D)[:4]
    ko = [
        "체크포인트 h50→h30 (실행 10스텝 고정)",
        "실행 길이 10→30 (h30 고정)",
        "실행 길이 10→30 (h50 고정)",
        "둘 다 동시에 = 실제로 했던 비교",
    ]
    eng = [
        "checkpoint h50→h30 (execution fixed at 10)",
        "execution 10→30 (checkpoint fixed at h30)",
        "execution 10→30 (checkpoint fixed at h50)",
        "both at once = the comparison that was made",
    ]
    return "".join(
        f"<tr><td>{(eng if en else ko)[i]}</td><td>{D[k]['delta']:+.2f}</td><td>[{D[k]['ci'][0]:+.2f}, {D[k]['ci'][1]:+.2f}]</td><td>{D[k]['p']:.3f}</td></tr>"
        for i, k in enumerate(labs)
    )


FIG = "<figure><img src='figures/operating-point-30/fig_legoprog_vs_control.png' alt='all conditions vs the matched control'><figcaption>{cap}</figcaption></figure>"
b8, a8, q0, q1 = T["fixed_implicit_bon8"], T["fixed_argmax_bon8"], T["fixed_qpilot_0.000"], T["fixed_qpilot_0.1"]

KO = f"""<table class='num'><tr><th>항목</th><th>내용</th></tr>
<tr><th>who</th><td>워커B — 사용자가 "우리는 30스텝을 정한 거고, 그 안에서 얼마나 올리느냐를 보는 것"이라고 바로잡은 데서 출발</td></tr>
<tr><th>when</th><td>2026-09-02</td></tr>
<tr><th>where</th><td>YAM lego-taxi 실물 로봇, LEGOPROG 4개 블록 전부 (<code>.scratch/legoprog_v3.xlsx</code>, 조건당 n=10)</td></tr>
<tr><th>what</th><td>critic을 쓰는 15개 조건을 <b>method-only-diff 통제군</b>({CTRL} = {CM:.2f})과 비교. 결과: 유의한 차이는 {n_raw}/15, Holm 보정 후 {n_holm}/15 — 전부 <b>조향</b>이고 전부 <b>음수</b></td></tr>
<tr><th>how</th><td><code>scripts/legoprog_stats.py</code>: Welch t, 95% CI, Holm 보정. 잘못된 비교(vs bc50_ex_10)를 체크포인트 효과와 실행 길이 효과로 분해. 조향 스윕은 경향 검정</td></tr>
<tr><th>why</th><td>같은 데이터를 두고 오늘 하루 동안 "BoN 실패", "critic이 15/15 BC보다 나쁘다", "실행 길이 혼동" — 세 가지 다른 결론을 냈다. 전부 통제군을 잘못 잡은 탓이다. 이 글이 올바른 통제군으로 끝을 맺는다</td></tr>
<tr><th>코드</th><td><code>{branch}@{stamp}</code></td></tr></table>

<h3>운용점</h3>
<p>실행 길이 30스텝은 <b>선택한 운용점</b>이다. 질문은 "30이 최적인가"가 아니라 <b>"30에서 critic이 얼마나 올려주는가"</b>이다.
그러면 통제군은 하나로 정해진다: <b>같은 체크포인트(h30 BC), 같은 30스텝 실행, critic 없음</b> — 시트의 <code>{CTRL}</code>, 평균 <b>{CM:.2f}</b>.
critic 조건이 서빙하는 체크포인트가 h30이고(<code>serving.py</code> BC_CKPT), <code>bon</code>/arm 모드가 청크 전체를 실행하며
(<code>patch_critic_policy.py</code>: n_exec = min(30, 30)), <code>fixed</code> critic은 macro_group_size=30이라 adaptive 모드에서도
prefix가 하나뿐이므로, 이 통제군은 모든 critic 조건과 <b>critic 유무 하나만</b> 다르다.</p>

<h3>결과 — 15개 조건 전부, 올바른 통제군 대비</h3>
{table()}
{FIG.format(cap=f"음영이 통제군 {CTRL}의 95% CI. * raw p<0.05, ** Holm p<0.05. 점은 에피소드.")}
<p><b>critic이 아무 결정도 안 하는 두 조건은 통제군과 정확히 같다</b> — 조향 OFF {q0["mean"]:.2f} (Δ={q0["delta"]:+.2f}, p={q0["p"]:.3f}),
선택 OFF {T["fixed_implicit_bon1"]["mean"]:.2f} (p={T["fixed_implicit_bon1"]["p"]:.3f}). 서빙 스택은 베이스 정책을 그대로 재현한다.</p>
<p><b>선택·적응 커밋 10개 조건은 전부 판별 불가</b>(p 0.10~0.85). 가장 큰 효과는 여전히 expectile 추첨
{b8["mean"]:.2f} (Δ={b8["delta"]:+.2f}, p={b8["p"]:.3f})이지만 n=10으로는 유의하지 않고, argmax BoN은 Δ={a8["delta"]:+.2f}로 통제군 그대로다.</p>
<p><b>유의한 것은 조향뿐이고, 방향은 아래다.</b> α=0.1에서 {q1["mean"]:.2f} (Δ={q1["delta"]:+.2f}, Holm p={q1["holm"]:.3f}), α=0.025에서
Δ={T["fixed_qpilot_0.025"]["delta"]:+.2f} (raw p={T["fixed_qpilot_0.025"]["p"]:.3f}). 사용자가 "랜덤보다 나쁜 경우가 많았다"고 한 것은 정확히 이 블록이다.</p>

<h3>조향은 "용량 반응"이 아니다 — 평평하다가 무너진다</h3>
<p>α 여섯 수준 전체로는 경향이 강하다(Spearman p={ST["spearman_all_p"]:.0e}). 그러나 <b>α=0.1 한 점을 빼면 경향이 사라지고</b>
(p={ST["spearman_no01_p"]:.3f}), 중간 네 수준(0.005~0.05)은 서로 구분되지 않는다(Kruskal p={ST["kruskal_mid_p"]:.3f}).
모양은 "켜면 일정량 손해(α>0 풀링 n=50: Δ={ST["pooled_on_delta"]:+.2f} [{ST["pooled_on_ci"][0]:+.2f}, {ST["pooled_on_ci"][1]:+.2f}], p={ST["pooled_on_p"]:.3f}),
0.1에서 붕괴(stage 0이 7/10)"이다. 이는 "gradient가 조금씩 잘못된 방향을 가리킨다"보다 <b>"조향이 켜지는 순간 무언가가 깨지고, 0.1에서는 정책이
동작 자체를 못 한다"</b>에 가깝다. 어느 쪽인지는 조향 구현 쪽(정규화 박스 이탈, ODE 파손)을 따로 봐야 한다. 임계값은 <b>방향에 무관</b>하므로 이 데이터만으로는 "gradient가 틀렸다"와 "그 크기의 주입은 방향이 뭐든 액션을 망친다"를 가를 수 없고, 가르는 통제군은 <b>같은 조향 루프에서 g를 랜덤 단위벡터로 바꾼 것</b>이다(commit 7c6af78).</p>

<div style='border-left:4px solid #2a7;background:#f2fbf6;padding:10px 14px;margin:12px 0'>
<b>추가 (2026-09-02) — 그 통제군의 결과가 도착했다.</b> 워커B(ACRFT-WS)가 같은 로봇·같은 critic 계열에서
바로 그 실험을 돌렸다. 시연자 앵커 기준, 정규화 행동단위 3까지 이동했을 때:
<br><code>∇_a Q 방향 +32.8 ± 5.0 &nbsp; vs &nbsp; 무작위 단위방향 3개 −0.13 ± 0.49</code> — <b>200배, critic 9종 전부</b>.
그리고 ∇_a Q 에너지의 <b>73%가 BC draw 16개가 펼치는 공간 밖</b>이다(우연이면 96%).
<br>따라서 위 양자택일은 <b>해소된다</b>: 과대추정은 <b>방향-특이적</b>이며 "그 크기면 뭐든 망가진다"가 아니다.
조향이 무너지는 이유는 critic의 gradient가 가리키는 그 방향에서만 Q가 부풀기 때문이다.
<br>같은 인계에서 <b>축 분리</b>도 보고됐다 — BC 체크포인트를 100k/150k/200k로 바꾸면 선택 이득은
+1.08 → +0.68로 줄지만 ∇Q 과대추정은 32.6/32.9/32.8로 <b>전혀 움직이지 않는다</b>. 조향 문제는
정책을 바꿔서 고칠 수 없고 critic의 성질이라는 뜻이다.
<br><b>출처와 한계</b>: 이 수치는 워커B의 <code>scripts/probe_q_landscape.py</code>(브랜치
<code>probe/q-landscape</code>)에서 온 것으로, 그 브랜치가 origin에 없어 <b>이 세션에서 재현하지 않았다</b>.
인계 내용 그대로 인용하며, 이 리포트의 나머지 수치와 달리 독립 검증되지 않았다.
</div>

<h3>오늘 세 번 틀린 이유 — 한 표로</h3>
<p>오전의 "15/15가 BC보다 나쁘다"는 <code>bc50_ex_10</code>(3.10)과 비교한 것이다. 그 비교는 <b>두 가지를 동시에</b> 바꾼다:</p>
<table class='num'><tr><th>바꾼 것</th><th>Δ</th><th>95% CI</th><th>p</th></tr>{dec_rows()}</table>
<p>실제로 서빙하는 h30 체크포인트에서는 <b>실행 길이 10→30이 유의하지 않다</b>(Δ={D[DL[1]]["delta"]:+.2f}, p={D[DL[1]]["p"]:.3f}). 차이의 대부분은
<b>체크포인트 교체</b>(h50→h30, Δ={D[DL[0]]["delta"]:+.2f})였고, 게다가 bc50_ex_10은 자기 스윕 여섯 점 중 <b>정점을 고른 값</b>이다
(나머지 다섯 점 풀링 대비 p={D["peak_pick_p"]:.4f}). 즉 "실행 길이 혼동"이라는 오후의 진단도 4분의 1만 맞았다.
이 리포의 규칙 — <b>비교는 method-only-diff끼리만</b> — 을 통제군 선택에서 두 번 어긴 것이고, 통제군이 시트 안에 있었는데 못 봤다.</p>

<h3>확정과 잠정</h3>
<ul>
<li><b>확정.</b> 30스텝 운용점에서 서빙 스택은 베이스를 재현한다. 조향 α=0.1은 정책을 망가뜨린다(Holm p={q1["holm"]:.3f}).</li>
<li><b>잠정.</b> 선택·적응 커밋 어느 것도 n=10으로는 판정 불가. 추첨의 +{b8["delta"]:.2f}는 이 데이터셋의 최대 효과지만 CI가 0을 포함한다.
Δ=1.0을 80% 검정력으로 잡으려면 조건당 30~40 에피소드.</li>
<li><b>영향받지 않는 것.</b> <span class='xref' data-eid='argmax-width'>argmax-width</span>의 오프라인 측정(판별 0.566, 폭-편향 멱함수, 앙상블 4.7%)은 로봇 숫자와 무관하다.
다만 그 글이 실물 결과를 인용한 문장은 이 표를 기준으로 읽어야 한다.</li>
</ul>"""

EN = f"""<table class='num'><tr><th>field</th><th></th></tr>
<tr><th>who</th><td>worker B — prompted by the user's correction: "we chose 30 steps; the question is how much the critic adds within that"</td></tr>
<tr><th>when</th><td>2026-09-02</td></tr>
<tr><th>where</th><td>YAM lego-taxi real robot, all four LEGOPROG blocks (<code>.scratch/legoprog_v3.xlsx</code>, n=10 per condition)</td></tr>
<tr><th>what</th><td>All 15 critic-using conditions against the <b>method-only-diff control</b> ({CTRL} = {CM:.2f}). Result: {n_raw}/15 significant, {n_holm}/15 after Holm — all of them <b>steering</b>, all of them <b>negative</b></td></tr>
<tr><th>how</th><td><code>scripts/legoprog_stats.py</code>: Welch t, 95% CI, Holm. The wrong comparison (vs bc50_ex_10) decomposed into a checkpoint effect and an execution-length effect; trend tests on the steering sweep</td></tr>
<tr><th>why</th><td>Over one day the same data produced three different conclusions — "BoN failed", "15/15 worse than BC", "an execution-length confound" — each from a wrong control. This entry closes it with the right one</td></tr>
<tr><th>code</th><td><code>{branch}@{stamp}</code></td></tr></table>

<h3>The operating point</h3>
<p>Thirty executed steps is the <b>chosen operating point</b>. The question is not "is 30 optimal" but <b>"how much does the critic add at 30"</b>.
That fixes the control: <b>same checkpoint (h30 BC), same 30 executed steps, no critic</b> — the sheet's <code>{CTRL}</code>, mean <b>{CM:.2f}</b>.
The critic conditions serve the h30 checkpoint (<code>serving.py</code> BC_CKPT), <code>bon</code>/arm modes execute the whole chunk
(<code>patch_critic_policy.py</code>: n_exec = min(30, 30)), and the <code>fixed</code> critics have macro_group_size=30 so even adaptive mode has a single
prefix. This control therefore differs from every critic condition in <b>exactly one thing</b>: the critic.</p>

<h3>Result — all 15 conditions against the right control</h3>
{table(en=True)}
{FIG.format(cap=f"Shaded band: 95% CI of the control {CTRL}. * raw p<0.05, ** Holm p<0.05. Dots are episodes.")}
<p><b>The two conditions in which the critic makes no decision land exactly on the control</b> — steering off {q0["mean"]:.2f} (Δ={q0["delta"]:+.2f}, p={q0["p"]:.3f}),
selection off {T["fixed_implicit_bon1"]["mean"]:.2f} (p={T["fixed_implicit_bon1"]["p"]:.3f}). The serving stack reproduces the base policy.</p>
<p><b>All ten selection and adaptive-commitment conditions are undetermined</b> (p 0.10–0.85). The largest effect is still the expectile lottery at
{b8["mean"]:.2f} (Δ={b8["delta"]:+.2f}, p={b8["p"]:.3f}), not significant at n=10; argmax BoN is Δ={a8["delta"]:+.2f}, the control exactly.</p>
<p><b>Only steering is significant, and it points down.</b> α=0.1 gives {q1["mean"]:.2f} (Δ={q1["delta"]:+.2f}, Holm p={q1["holm"]:.3f}); α=0.025 gives
Δ={T["fixed_qpilot_0.025"]["delta"]:+.2f} (raw p={T["fixed_qpilot_0.025"]["p"]:.3f}). This block is what the user meant by "often worse than random".</p>

<h3>Steering is not a dose-response — it is flat, then it collapses</h3>
<p>Across all six α levels the trend is strong (Spearman p={ST["spearman_all_p"]:.0e}). But <b>remove the single α=0.1 point and the trend is gone</b>
(p={ST["spearman_no01_p"]:.3f}); the four intermediate levels (0.005–0.05) are indistinguishable from each other (Kruskal p={ST["kruskal_mid_p"]:.3f}).
The shape is "a fixed cost for turning it on (pooled α>0, n=50: Δ={ST["pooled_on_delta"]:+.2f} [{ST["pooled_on_ci"][0]:+.2f}, {ST["pooled_on_ci"][1]:+.2f}], p={ST["pooled_on_p"]:.3f}),
then collapse at 0.1 (7/10 episodes at stage 0)". That reads less like "the gradient points slightly wrong" and more like
<b>"something breaks the moment steering is on, and at 0.1 the policy cannot act at all"</b>. Which it is needs a look at the steering implementation
(leaving the normalized action box, breaking the ODE), not at the critic. A threshold is <b>direction-agnostic</b>, so this data alone cannot separate "the gradient points the wrong way" from "an injection that large damages the action whatever its direction"; the control that separates them is <b>the same steering loop with g replaced by a random unit vector</b> (commit 7c6af78).</p>

<div style='border-left:4px solid #2a7;background:#f2fbf6;padding:10px 14px;margin:12px 0'>
<b>Added 2026-09-02 — that control has now been run.</b> Worker B (ACRFT-WS) ran exactly it, on the same
robot and the same critic family. Against the demonstrator anchor, moving out to 3 normalized action units:
<br><code>along ∇_a Q: +32.8 ± 5.0 &nbsp; vs &nbsp; three random unit directions: −0.13 ± 0.49</code> — <b>200x, across all 9 critics</b>.
And <b>73% of the ∇_a Q energy lies outside the span of 16 BC draws</b> (96% would be chance).
<br>So the disjunction above <b>resolves</b>: the over-estimation is <b>direction-specific</b>, not "anything
that large breaks it". Steering collapses because Q inflates along the one direction the critic's gradient points.
<br>The same handoff reports an <b>axis separation</b>: sweeping the BC checkpoint 100k/150k/200k shrinks the
selection benefit (+1.08 → +0.68) while the ∇Q over-estimation does not move at all (32.6/32.9/32.8). Steering
cannot be fixed by changing the policy — it is a property of the critic.
<br><b>Provenance and limit</b>: these numbers come from worker B's <code>scripts/probe_q_landscape.py</code>
(branch <code>probe/q-landscape</code>), which is not on origin, so they were <b>not reproduced in this session</b>.
They are quoted as handed over and, unlike every other number in this entry, are not independently verified.
</div>

<h3>Why the same data gave three wrong answers — one table</h3>
<p>The morning's "15/15 worse than BC" compared against <code>bc50_ex_10</code> (3.10). That comparison changes <b>two things at once</b>:</p>
<table class='num'><tr><th>what changed</th><th>Δ</th><th>95% CI</th><th>p</th></tr>{dec_rows(en=True)}</table>
<p>On the h30 checkpoint actually served, <b>execution length 10→30 is not significant</b> (Δ={D[DL[1]]["delta"]:+.2f}, p={D[DL[1]]["p"]:.3f}). Most of the gap is the
<b>checkpoint swap</b> (h50→h30, Δ={D[DL[0]]["delta"]:+.2f}), and bc50_ex_10 is moreover the <b>peak of its own six-point sweep</b>
(vs the other five pooled, p={D["peak_pick_p"]:.4f}). So the afternoon's "execution-length confound" diagnosis was only a quarter right.
The repo's rule — <b>compare method-only-diff only</b> — was broken twice in the choice of control, and the right control was in the sheet the whole time.</p>

<h3>Settled and provisional</h3>
<ul>
<li><b>Settled.</b> At the 30-step operating point the serving stack reproduces the base policy. Steering at α=0.1 breaks it (Holm p={q1["holm"]:.3f}).</li>
<li><b>Provisional.</b> No selection or adaptive-commitment rule can be judged at n=10. The lottery's +{b8["delta"]:.2f} is the largest effect in the dataset and its CI includes zero.
Detecting Δ=1.0 at 80% power needs 30–40 episodes per condition.</li>
<li><b>Unaffected.</b> The offline measurements in <span class='xref' data-eid='argmax-width'>argmax-width</span> (0.566 discrimination, the width–bias power law, the 4.7% ensemble ratio)
never depended on the robot numbers. Its sentences that cite the robot should be read against this table.</li>
</ul>"""

entry = {
    "eid": "operating-point-30",
    "worker": "B",
    "date": "2026-09-02 13:40",
    "status": "finding",
    "title": f"🤖 [워커B] 30스텝 운용점에서 critic 15조건 vs 올바른 통제군 — 유의한 건 조향뿐이고, 아래로 ({n_raw}/15)",
    "summary": (
        f"사용자 지적('30은 우리가 정한 운용점')대로 통제군을 {CTRL}(같은 h30 체크포인트·같은 30스텝·critic 없음, {CM:.2f})로 잡으면: "
        f"critic이 결정을 안 하는 두 조건은 통제군과 정확히 같고(p=1.000, 0.824), 선택·적응 커밋 10조건은 전부 판별 불가, "
        f"유의한 것은 조향 α=0.1(Δ={q1['delta']:+.2f}, Holm p={q1['holm']:.3f})과 α=0.025뿐이며 둘 다 음수. 조향은 용량 반응이 아니라 "
        f"'켜면 일정 손해, 0.1에서 붕괴'(α=0.1 제거 시 경향 p={ST['spearman_no01_p']:.3f}). 오늘의 '15/15 BC보다 나쁘다'는 bc50_ex_10(3.10)과 비교한 것으로 "
        f"체크포인트(h50→h30, Δ={D[DL[0]]['delta']:+.2f})와 실행 길이(h30에서는 n.s.)를 동시에 바꾼 데다 스윕의 정점을 고른 값이었다."
    ),
    "tags": ["워커B", "실물", "LEGOPROG", "통제군", "정정", "QPILOTS"],
    "phase": "실물 평가",
    "links": [
        "serving-rollouts-yam",
        "argmax-width",
        "q-landscape-ood",
        "critic-detail-survey",
        "extraction-suite-yam",
    ],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/operating_point_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print(f"wrote {out}  | control {CTRL}={CM:.2f} | sig {n_raw}/15 raw, {n_holm}/15 Holm")
