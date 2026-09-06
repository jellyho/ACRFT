"""Hub entry: an alarm I raised about every robot result, and the evidence that settled it.

New eid. Numbers come from .scratch/extraction/image_contract.json, which fig_image_contract.py
regenerates from the rollout recording itself.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
D = json.loads((R / ".scratch/extraction/image_contract.json").read_text())


def _git(*a):
    return subprocess.run(["git", "-C", str(R), *a], capture_output=True, text=True, check=False).stdout.strip()


STAMP = f"{_git('rev-parse', '--abbrev-ref', 'HEAD')}@{_git('log', '-1', '--format=%h')}" + (
    "+dirty" if _git("status", "--porcelain", "-uno") else ""
)
BAR = D["bar_rows_if_padded"]
CAMS = D["cameras"]
TOPMAX = min(c["top_band_max"] for c in CAMS)
FIG = "<figure><img src='figures/image-contract/fig_image_contract.png' alt='row-brightness profile and a recorded frame beside the letterboxed version it was not'><figcaption>{cap}</figcaption></figure>"

rows = "".join(
    f"<tr><td>카메라 {i}</td><td>{c['top_band_max']:.0f}</td><td>{c['bottom_band_max']:.0f}</td>"
    f"<td>{c['top_band_mean']:.1f}</td><td>{c['middle_mean']:.1f}</td></tr>"
    for i, c in enumerate(CAMS)
)
TBL = (
    f"<table class='num'><tr><th></th><th>상단 {BAR}행 최대</th><th>하단 {BAR}행 최대</th>"
    f"<th>상단 평균</th><th>가운데 평균</th></tr>{rows}</table>"
)
TBL_EN = (
    TBL.replace("카메라", "camera")
    .replace(f"상단 {BAR}행 최대", f"max in top {BAR} rows")
    .replace(f"하단 {BAR}행 최대", f"max in bottom {BAR} rows")
    .replace("상단 평균", "top-band mean")
    .replace("가운데 평균", "middle mean")
)

KO = f"""<table class='num'><tr><th>항목</th><th>내용</th></tr>
<tr><th>who</th><td>워커B</td></tr><tr><th>when</th><td>2026-09-05</td></tr>
<tr><th>where</th><td>디스크에 남아 있던 롤아웃 기록 <code>pc_rollouts_yam/lego_taxi</code>
({D["episodes"]}에피소드 / {D["success"]}성공). CPU only</td></tr>
<tr><th>what</th><td>critic 감사가 찾은 "이미지 계약이 강제되지 않는다"는 결함이 <b>실제로 발현했는지</b> 판정</td></tr>
<tr><th>how</th><td><code>scripts/fig_image_contract.py</code> — 파이프라인을 실제로 통과한 프레임에서
letterbox 띠의 유무를 직접 확인</td></tr>
<tr><th>why</th><td>발현했다면 지금까지의 critic 로봇 결과 전부의 해석이 흔들린다</td></tr>
<tr><th>코드</th><td><code>{STAMP}</code></td></tr></table>

<p><b>배경.</b> critic의 feature 캐시는 480×640 원본을 224로 <b>찌부러뜨려(squash, cv2.INTER_AREA,
종횡비 미보존)</b> 만들었다. 그런데 openpi의 문서화된 클라이언트, <code>examples/droid/main.py</code>,
랩의 YAM 브리지는 모두 <b><code>resize_with_pad</code></b>로 전처리한다 — letterbox다. 그런 클라이언트는
<b>이미 224×224인</b> 프레임을 보내고, 서버의 resize는 no-op이 되며, critic은 학습에서 본 적 없는
레터박스 이미지를 채점한다. <b>예외도 안 나고, Q도 그럴듯하게 나온다.</b> 그리고 도착 형상을 남기는
로그가 어디에도 없었다.</p>

<p><b>측정된 대가</b>(18프레임/6에피소드): patch 토큰이 상대 L2 <b>0.636</b> 어긋나고 V가
<b>평균 222.5 / 최대 894</b> 움직이는데 <b>모든 프레임에서 부호가 양수</b> — critic이 상태를 일관되게
목표에 더 가깝다고 읽는다. V의 상태 간 퍼짐이 338이고 arg-max 선택 효과 전체가 +100.6인 것과 비교하면 크다.
(대조로, 학습 전처리로 재인코딩하면 토큰 0.032 / ΔV 1.4로 캐시를 재현한다.)</p>

<h3>서버 쪽에서는 답할 수 없었다</h3>
<p>형상 검사로는 원리적으로 못 잡는다. <b>224×224 도착은 두 관례 모두에서 통과</b>하고 서로 다른 것을
뜻한다. <code>spec.check</code>는 <code>num_cameras</code>와 <code>img_size</code>만 비교하는데 둘 다
<b>서버 자신의 상수</b>다. 그래서 <b>계약이 검사된 적이 없다</b>는 것은 사실이었지만,
<b>어긋났는지</b>는 서버 코드만으로는 알 수 없었다.</p>

<h3>기록에서는 답할 수 있었다</h3>
<p><code>pc_rollouts_yam/lego_taxi</code>에 <b>파이프라인을 실제로 통과한 이미지</b>가 그대로 남아 있다
(<code>images.dat [{D["frames_sampled"] and ""}162375, 3, 224, 224, 3]</code>). 그리고 판정이 기하학적으로
자명하다: 480×640을 224 안에 fit하면 224×168이 되어 <b>위아래 {BAR}행이 정확히 검정</b>이어야 한다.
squash면 224행이 전부 내용이다.</p>
{TBL}
<p>{D["frames_sampled"]}프레임 표본에서 <b>그 {BAR}행의 최대 픽셀값이 {TOPMAX:.0f}</b>이다. letterbox면 0이어야 한다.
<b>판정: squash — 캐시 관례와 일치.</b></p>
{FIG.format(cap="왼쪽: 세 카메라의 행별 평균 밝기. 음영이 letterbox 띠가 놓일 자리이고, 거기에 내용이 있다. 가운데: 실제 기록된 프레임. 오른쪽: 레터박스였다면 이랬을 프레임.")}

<h3>이게 진짜 로봇 롤아웃인지도 확인했다</h3>
<p>데이터셋 재렌더라면 아무것도 증명하지 못한다. 두 가지로 배제했다: 성공률이
<b>{D["success"]}/{D["episodes"]} = {D["success"] / D["episodes"] * 100:.0f}%</b>인데 데이터셋은 300/347 = 86%이고,
기록 프레임과 데이터셋 프레임의 최소 평균 절대차가 <b>3.5 / 20.9 / 25.8</b>(0–255 척도)로 재렌더의
0–2와 거리가 멀다.</p>

<h3>판정, 그리고 남는 것</h3>
<p><b>②는 이미 일어난 사고가 아니라 막아둔 위험이었다.</b> 내가 경보를 울린 근거는 <i>계약이 검사되지
않는다</i>였고 그건 참이었지만, <b>어긋났다는 증거는 없고 어긋나지 않았다는 증거가 나왔다.</b>
따라서 <span class='xref' data-eid='serving-rollouts-yam'>지금까지의 critic 로봇 결과 중 이 이유로 무효가
되는 것은 없다.</span></p>
<p><b>한계 둘.</b> ① 이 기록은 <b>2026-08-13</b>이고 LEGOPROG 평가는 8/26–9/1이다. 같은 스택이지만 그
사이 클라이언트가 안 바뀌었다는 보장은 없다. ② patch critic 롤아웃이 이 기록과 <b>같은 경로</b>였는지는
확정하지 못했다(디렉토리 이름 <code>pc_</code>가 patch critic을 가리키는 것으로 보이나 확인은 아니다).</p>
<p><b>그래도 계측은 남긴다.</b> 이번엔 사후에 73GB짜리 기록을 뒤져서 답했다. 앞으로는 서버가 도착 형상을
로그로 남기고, 정사각으로 도착하면 측정된 대가를 명시하며 경고한다 — <b>로그 한 줄로 끝난다.</b>
그리고 캐시가 <code>source_hw</code>/<code>resize_mode</code>를 기록하고 <code>input_spec</code>이
이를 전달하므로, 다음 캐시부터는 판정에 이 조사가 필요 없다.</p>
<p><b>여전히 보류인 것 하나.</b> 이미지와 무관하게, <span class='xref' data-eid='serving-rollouts-yam'>QPILOTS α
스윕</span>은 critic의 gradient를 깨끗하게 검정하지 못했다 — 조향이 미분한 값이
<code>mean − 0.5·std</code>인데 K=2에서 std는 <code>|q1−q2|/2</code>이고, 그것이 gradient 크기의
<b>34%</b>를 값 gradient와 <b>거의 직교한 방향</b>(cos −0.078)으로 차지했다. 랜덤 방향 통제군이 이를 가른다.</p>"""

EN = f"""<table class='num'><tr><th>field</th><th></th></tr>
<tr><th>who</th><td>worker B</td></tr><tr><th>when</th><td>2026-09-05</td></tr>
<tr><th>where</th><td>a rollout recording still on disk, <code>pc_rollouts_yam/lego_taxi</code>
({D["episodes"]} episodes / {D["success"]} success). CPU only</td></tr>
<tr><th>what</th><td>Whether the unenforceable image contract the critic audit found <b>actually fired</b></td></tr>
<tr><th>how</th><td><code>scripts/fig_image_contract.py</code> — look for letterbox bars in the frames that
actually went through the pipeline</td></tr>
<tr><th>why</th><td>If it fired, every critic robot result to date is reinterpretable</td></tr>
<tr><th>code</th><td><code>{STAMP}</code></td></tr></table>

<p><b>Background.</b> The critic's feature cache was built by <b>squashing</b> native 480×640 frames to 224
(cv2.INTER_AREA, aspect ratio not preserved). But openpi's documented client,
<code>examples/droid/main.py</code> and the lab's YAM bridge all pre-process with
<b><code>resize_with_pad</code></b>, which letterboxes. Such a client sends an <b>already-224×224</b> frame,
the server's resize becomes a no-op, and the critic scores an image it never saw in training.
<b>Nothing raises, and Q comes back plausible.</b> No log anywhere retained the arriving shape.</p>

<p><b>Measured cost</b> (18 frames / 6 episodes): patch tokens drift <b>0.636</b> relative L2 and V moves
<b>222.5 mean / 894 max, positive on every frame</b> — the critic reads the state as systematically closer
to the goal. Against a V spread of 338 and a whole arg-max selection effect of +100.6, that is large.
(Control: re-encoding through the training preprocessing reproduces the cache to 0.032 / ΔV 1.4.)</p>

<h3>The server side could not answer it</h3>
<p>A shape check cannot decide this in principle: <b>a 224×224 arrival passes under both conventions</b> and
means different things under each. <code>spec.check</code> compares only <code>num_cameras</code> and
<code>img_size</code>, both of which are the <b>server's own constants</b>. So "the contract was never
checked" was true, while "the contract was violated" was unanswerable from the server code.</p>

<h3>The recording could</h3>
<p><code>pc_rollouts_yam/lego_taxi</code> retains <b>the images that actually went through the pipeline</b>.
And the test is geometric: fitting 480×640 inside 224 yields 224×168, leaving <b>exactly {BAR} rows of black
at the top and bottom</b>. A squash fills all 224.</p>
{TBL_EN}
<p>Over {D["frames_sampled"]} sampled frames the <b>maximum pixel value in those {BAR} rows is {TOPMAX:.0f}</b>.
A letterbox bar would be 0. <b>Verdict: squash — matching the cache.</b></p>
{FIG.format(cap="Left: per-row mean brightness for the three cameras; the shaded bands are where a letterbox bar would sit, and there is content there. Middle: a frame as recorded. Right: the letterboxed frame it was not.")}

<h3>And these are genuine robot rollouts</h3>
<p>A dataset re-render would prove nothing. Two checks rule it out: the success rate is
<b>{D["success"]}/{D["episodes"]} = {D["success"] / D["episodes"] * 100:.0f}%</b> against the dataset's 300/347 = 86%,
and the minimum mean-absolute difference between recorded and dataset frames is <b>3.5 / 20.9 / 25.8</b> on a
0–255 scale, where a re-render would be 0–2.</p>

<h3>Verdict, and what remains</h3>
<p><b>This was a risk that was guarded, not an accident that happened.</b> The alarm rested on
<i>the contract is never checked</i>, which was true — but there is no evidence it was violated and there is
evidence it was not. <span class='xref' data-eid='serving-rollouts-yam'>No critic robot result is
invalidated for this reason.</span></p>
<p><b>Two limits.</b> (1) This recording is from <b>2026-08-13</b> while the LEGOPROG evaluations ran
Aug 26 – Sep 1; same stack, but nothing guarantees the client did not change in between. (2) I could not
confirm the patch-critic rollouts took <b>the same path</b> as this recording (the <code>pc_</code> prefix
suggests patch critic, which is not a confirmation).</p>
<p><b>The instrumentation stays.</b> This time the answer required digging through a 73 GB recording after
the fact. From now the server logs the arriving geometry and warns, naming the measured cost, when frames
arrive already square — <b>one log line settles it</b>. And the cache records
<code>source_hw</code>/<code>resize_mode</code> with <code>input_spec</code> carrying them forward, so from
the next cache on this investigation is unnecessary.</p>
<p><b>One thing still open.</b> Independent of the images, the
<span class='xref' data-eid='serving-rollouts-yam'>QPILOTS α sweep</span> did not cleanly test the critic's
gradient: what the steering differentiated was <code>mean − 0.5·std</code>, and at K=2 the std is
<code>|q1−q2|/2</code>, contributing <b>34%</b> of the gradient magnitude in a direction
<b>near-orthogonal</b> (cos −0.078) to the value gradient. A random-direction control arm separates them.</p>"""

entry = {
    "eid": "image-contract",
    "worker": "B",
    "date": "2026-09-05 06:20",
    "status": "finding",
    "title": "🔍 [워커B] critic 입력이 틀렸을 뻔한 이야기 — 그리고 롤아웃 기록이 준 답",
    "summary": (
        "critic 감사가 '이미지 squash-vs-pad 계약이 강제되지 않는다'를 찾았고, 어긋났다면 V가 프레임마다 "
        "+222.5(최대 894, 전 프레임 양수) 치우친다 — V의 상태 간 퍼짐 338, arg-max 선택 효과 전체가 +100.6인 "
        "것과 비교하면 크다. 형상 검사로는 원리적으로 못 잡는다(224 도착은 두 관례 모두 통과). 서버 코드로는 "
        f"답이 안 나와서 롤아웃 기록 {D['episodes']}에피소드를 뒤졌더니, letterbox 띠가 놓일 {BAR}행의 "
        f"최대 픽셀값이 {TOPMAX:.0f}(띠라면 0) — squash, 캐시와 일치. 진짜 로봇 롤아웃인 것도 성공률 "
        f"{D['success']}/{D['episodes']}와 데이터셋 대비 프레임 차이로 확인. **이 이유로 무효가 되는 로봇 결과는 "
        "없다.** 한계: 기록은 8/13이고 평가는 8/26–9/1. 계측은 남겨서 앞으로는 로그 한 줄로 끝난다."
    ),
    "tags": ["워커B", "critic", "서빙", "감사", "정정"],
    "phase": "진단·방법",
    "links": ["serving-rollouts-yam", "argmax-width", "critic-detail-survey", "q-landscape-ood"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/image_contract_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print(f"wrote {out} | verdict {D['verdict']} | bar {BAR} rows | max in band {TOPMAX:.0f} | stamp {STAMP}")
