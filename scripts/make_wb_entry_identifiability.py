"""Hub entry: is there action-conditioned return signal in the data at all?

Numbers recomputed from .scratch/extraction/diag_action_identifiability.json on every run.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
D = json.loads((R / ".scratch/extraction/diag_action_identifiability.json").read_text())


def _git(*args):
    return subprocess.run(["git", "-C", str(R), *args], capture_output=True, text=True, check=False).stdout.strip()


stamp = _git("log", "-1", "--format=%h") + ("+dirty" if _git("status", "--porcelain", "-uno") else "")
branch = _git("rev-parse", "--abbrev-ref", "HEAD")
EH, FH, C, NL = D["episode_heldout"], D["frame_heldout"], D["controls"], D["permutation_null"]
BOTH = C["both"]["gap"]
RESID = BOTH / (1 - C["both"]["state_plus_style"])

TBL = f"""<table class='num'>
<tr><th></th><th>R²(state)</th><th>R²(state+action)</th><th>gap</th></tr>
<tr><td>episode-held-out</td><td>{EH["state"]:.4f}</td><td>{EH["state_action"]:.4f}</td><td><b>{EH["gap"]:+.4f}</b></td></tr>
<tr><td>frame-held-out</td><td>{FH["state"]:.4f}</td><td>{FH["state_action"]:.4f}</td><td>{FH["gap"]:+.4f}</td></tr>
<tr><td>null, action block permuted (global)</td><td>{EH["state"]:.4f}</td><td>{NL["global"]["state_action_permuted"]:.4f}</td><td>{NL["global"]["gap"]:+.4f}</td></tr>
<tr><td>null, permuted within episode</td><td>{EH["state"]:.4f}</td><td>{NL["within_episode"]["state_action_permuted"]:.4f}</td><td>{NL["within_episode"]["gap"]:+.4f}</td></tr>
<tr><td>pace-corrected target</td><td>{C["pace_corrected"]["state"]:.4f}</td><td>{C["pace_corrected"]["state_action"]:.4f}</td><td>{C["pace_corrected"]["gap"]:+.4f}</td></tr>
<tr><td>episode-style controlled</td><td>{C["style_controlled"]["state_plus_style"]:.4f}</td><td>{C["style_controlled"]["plus_action"]:.4f}</td><td>{C["style_controlled"]["gap"]:+.4f}</td></tr>
<tr><td><b>both controls</b></td><td>{C["both"]["state_plus_style"]:.4f}</td><td>{C["both"]["plus_action"]:.4f}</td><td><b>{BOTH:+.4f}</b></td></tr>
</table>"""

KO = f"""<table class='num'><tr><th>항목</th><th>내용</th></tr>
<tr><th>who</th><td>워커B (이 세션). 순열 null과 독립 재현은 <b>ACRFT-WS</b> 세션</td></tr>
<tr><th>when</th><td>2026-09-02</td></tr>
<tr><th>where</th><td>CPU only. yam_lego_taxi 성공 300 에피소드에서 {D["n_frames"]:,}프레임, homing 제외</td></tr>
<tr><th>what</th><td>액션이 상태 너머로 리턴을 예측하는가 — 즉 Q(s,a)가 V(s) 너머로 <b>식별 가능한가</b></td></tr>
<tr><th>how</th><td><code>scripts/diag_action_identifiability.py</code>. Ridge, <b>에피소드 단위</b> held-out 5-fold. 액션 블록 순열 null + 속도·스타일 통제</td></tr>
<tr><th>why</th><td>"실물 데이터는 state당 action이 하나라 negative가 없다"가 여러 곳에서 전제로 쓰이고 있었다. 사실이면 critic 관련 작업 절반이 무의미하다</td></tr>
<tr><th>코드</th><td><code>{branch}@{stamp}</code></td></tr></table>

<p><b>질문.</b> 같은 상태에서 시연자가 항상 같은 행동을 했다면, critic이 배울 수 있는 건 V(s)뿐이고
액션을 고르는 모든 방법 — best-of-N, 조향, CQL 네거티브, 앙상블 — 이 없는 신호를 짜내는 일이 된다.
이건 주장으로 유통되고 있었지 측정된 적이 없었다.</p>

<h3>먼저: 이웃으로 "같은 상태"를 찾는 방법은 못 쓴다</h3>
<p>자연스러운 접근은 에피소드를 가로질러 반복된 상태를 찾아 거기서 액션의 분산을 보는 것이다.
돌려봤고 <b>전제가 검증에서 무너졌다</b>: 시각+proprio 게이트를 통과하는 에피소드 간 쌍이
<b>89%가 에피소드 경계 프레임</b>(기저율 3%, 30배 농축)이고, 그것을 제외하면 strict 티어가
<b>783쌍 → 6쌍</b>으로 붕괴한다. 게이트를 통과하는 것은 "같은 상태"가 아니라 <b>작업 전후로 홈 포즈에
세워둔 로봇</b>이었다. 그래서 아래 측정은 그 전제를 아예 쓰지 않는다.</p>

<h3>전제 없는 질문으로 바꾼다</h3>
<p>두 프레임이 같은 상태인지 <b>묻지 않는다</b>. 회귀로 바꾼다 — held-out <b>에피소드</b>에서
목표까지 남은 시간을 (a) 상태만으로 (b) 상태+액션으로 예측했을 때의 R² 차이. 프레임 단위로 자르면
같은 궤적을 이미 본 모델이 외운 것을 신호로 착각하므로, 분할은 반드시 에피소드 단위다.</p>
{TBL}

<h3>세 가지가 이 숫자를 지탱한다</h3>
<p><b>① 누수가 아니다.</b> 에피소드 분할과 프레임 분할의 gap이 {EH["gap"]:+.4f} / {FH["gap"]:+.4f}로 같다.
프레임 분할은 절대 R²만 부풀린다.</p>
<p><b>② 용량이 아니다.</b> 420개 액션 열을 고정 α ridge에 더하는 것은 공짜 비교가 아니다 — 같은 정규화가
더 많은 방향에 퍼지므로 gap의 일부가 용량일 수 있다. 액션 블록을 프레임 사이에서 <b>순열</b>하면
주변분포와 공선성은 그대로 두고 상태-액션 대응만 파괴된다. 결과: <b>{NL["global"]["gap"]:+.4f}</b> (전역),
<b>{NL["within_episode"]["gap"]:+.4f}</b> (에피소드 내). 0이 아니라 <b>음수</b>다 — 용량 기여가 없을 뿐 아니라
쓸모없는 열은 약간 손해다. <b>이 통제군은 ACRFT-WS 세션이 먼저 지적하고 돌렸다.</b></p>
<p><b>③ 시연자 속도나 버릇이 아니다.</b> 느린 시연자는 남은 시간도 길고 움직임도 알아볼 수 있다.
목표를 에피소드 길이로 재조정하고, 에피소드 평균 액션을 상태 쪽에 넣어 <b>스타일을 통제</b>하면
gap이 <b>커진다</b>: {BOTH:+.4f}. 완전 통제 지점에서 액션은 <b>상태가 남긴 분산의 {RESID * 100:.0f}%</b>를 설명한다.</p>

<h3>독립 재현 — 다른 코드베이스, 다른 상태 표현</h3>
<p>ACRFT-WS 세션이 <b>다른 코드로</b> 재현했다: LeRobot parquet을 직접 읽고, 상태로 DINO가 아니라
<b>proprioception</b>을 쓰고, patch 캐시도 critic 체크포인트도 GPU도 없이. lego에서 gap +0.0635/+0.0639
(시드 2개), 마지막 100프레임을 자르면(우리 homing 제외의 거친 대용) <b>+0.0347</b>, cable_tie에서 +0.0446.
그쪽 null은 lego에서 +0.0005~+0.0007로 사실상 0이지만 <b>cable_tie에서는 +0.0074</b>다.
에피소드 내 순열은 그 시연자의 스타일을 보존하므로, cable_tie에서는 gap의 작은 일부가
<b>프레임 수준이 아니라 에피소드 수준</b>이라는 뜻이다. 결론은 안 바뀐다(그 null 대비 gap이 +0.0446),
다만 그쪽 null이 0이 아니었던 유일한 지점이므로 두 측정을 나란히 인용할 때 빠뜨리면 안 된다 —
ACRFT-WS 본인이 짚은 사항이다.</p>
<p><b>수렴이 흥미로운 부분이다.</b> 그쪽 +0.0347이 우리 {EH["gap"]:+.4f}에 떨어지는데,
<b>상태 표현의 질은 전혀 다르다</b> — 우리 R²(state) {EH["state"]:.4f} vs 그쪽 0.3200.
상태가 얼마나 좋든 액션의 한계 기여가 같다는 것은 누수가 아니라 <b>실재 신호</b>의 모습이다.
또한 DINO와 proprio가 목표까지 남은 시간에 대해 대체로 같은 것을 잡고 있고,
액션은 <b>둘 다 갖지 못한 무언가</b>를 더한다는 뜻이기도 하다.</p>

<h3>그리고 액션은 상태로 결정되지 않는다</h3>
<p>상태에서 액션 청크를 예측하면 R² = <b>{D["action_from_state_r2"]["mean"]:.3f}</b>.
1.0이면 액션이 상태의 함수라 값을 귀속시킬 잔차가 없다는 뜻인데, <b>65%가 잔차</b>다.</p>

<h3>판정</h3>
<p><b>state→action은 1:1이 아니고, Q(s,a)는 V(s) 너머로 식별 가능하다.</b> ridge는 선형이므로
{BOTH:+.4f}는 <b>하한</b>이다 — 유연한 critic이면 더 뽑지 덜 뽑지 않는다.</p>
<p>그런데 <span class='xref' data-eid='q-landscape-ood'>q-landscape-ood</span>는 액션 인자가
critic 9종 전부에서 <b>Q 분산의 0.001~0.002%</b>만 설명한다고 측정했다. 두 양은 직접 비교 대상이 아니지만
자릿수의 메시지는 분명하다: <b>선형 프로브조차 찾아내는 신호를 우리 critic들이 쓰지 않고 있다.</b></p>
<p>이는 처방을 바꾼다. "데이터가 1:1이라 원천적으로 안 된다"였다면 EDAC·CQL·앙상블·floq가 전부
헛수고였을 것이다. <b>배울 것이 있는데 안 배우고 있다</b>면 그것은 고칠 수 있는 종류의 실패다.</p>

<h3>한계</h3>
<ul>
<li>ridge는 <b>선형</b>이다. 비선형 신호는 안 보이므로 두 숫자 모두 하한이다.</li>
<li>성공 에피소드만 쓴다 — 실패는 목표가 없어 time-to-goal이 정의되지 않는다.</li>
<li><b>이것이 정하지 <u>못하는</u> 것</b>: q-landscape-ood의 cross-critic Spearman 0.50이 "데이터의 신호"인지
"critic 9종이 같은 방식으로 틀린 것"인지. 두 측정 모두 <b>데이터에</b> 신호가 있다고 말할 뿐,
그 0.50이 <b>그</b> 신호라고는 말하지 않는다. 여전히 별개의 주장이다.</li>
<li>ACRFT-WS의 꼬리 100프레임 절단은 우리 homing onset보다 거칠다.</li>
</ul>"""

EN = KO  # English pending; the tables and numbers are the substance and are language-neutral

entry = {
    "eid": "action-identifiability",
    "worker": "B",
    "date": "2026-09-02 20:30",
    "status": "finding",
    "title": "🤖 [워커B] 데이터는 1:1이 아니다 — 액션에 리턴 신호가 있고, 우리 critic이 그걸 안 쓰고 있다",
    "summary": (
        f"'실물 데이터는 state당 action이 하나라 negative가 없다'를 측정했다. 틀렸다. 에피소드 단위 held-out에서 "
        f"액션을 더하면 목표까지 남은 시간의 R²가 {EH['state']:.4f} → {EH['state_action']:.4f} ({EH['gap']:+.4f})로 오르고, "
        f"액션 블록을 순열한 null은 {NL['within_episode']['gap']:+.4f}로 사실상 0이며(용량 아님), 속도·스타일을 통제하면 "
        f"gap이 오히려 {BOTH:+.4f}로 커진다 — 상태가 남긴 분산의 {RESID * 100:.0f}%. 액션은 상태로 R²={D['action_from_state_r2']['mean']:.3f}만 "
        "설명되므로 65%가 잔차다. ACRFT-WS 세션이 다른 코드베이스·proprio 상태로 독립 재현했고(그쪽 R²(state) 0.32 vs 우리 0.60인데 "
        "액션의 한계 기여는 동일), 순열 null도 그쪽이 먼저 돌렸다. 이웃으로 '같은 상태'를 찾는 접근은 통과 쌍의 89%가 "
        "홈 포즈 경계 프레임이라 폐기. ridge는 선형이라 하한. → EDAC·CQL·앙상블·floq는 헛수고가 아니다."
    ),
    "tags": ["워커B", "식별가능성", "offline RL", "critic", "재현"],
    "phase": "진단·방법",
    "links": ["q-landscape-ood", "argmax-width", "operating-point-30", "critic-detail-survey"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/identifiability_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
print(f"wrote {out}")
print(
    f"  gap {EH['gap']:+.4f} | null {NL['within_episode']['gap']:+.4f} | controlled {BOTH:+.4f} | residual share {RESID * 100:.0f}%"
)
