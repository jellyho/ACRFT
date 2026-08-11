"""Publish the floq + 'What does flow matching bring to TD learning?' paper review to the hub.

A reference/review entry (RULES.md: verbatim quotes beside claims, KO/EN, 5W1H, links, git stamp).
"""

import datetime
import json
import pathlib
import re
import subprocess
import tempfile

from huggingface_hub import CommitOperationAdd
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

SPACE = "jellyho/acrft-reports"

branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=False).stdout.strip()
sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
).stdout.strip()
dirty = bool(
    subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()
)
stamp = f"{branch}@{sha}" + (" (+uncommitted)" if dirty else "")
now = (
    datetime.datetime.now(datetime.UTC)
    .astimezone(datetime.timezone(datetime.timedelta(hours=9)))
    .strftime("%Y-%m-%d %H:%M")
)


def spec(rows):
    tr = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='tblwrap'><table class='spec w6'>{tr}</table></div>"


def quote(text, cite):
    return f"<blockquote style='border-left:3px solid #c7cad1;margin:.5em 0;padding:.2em .9em;color:#444'>“{text}” <span style='color:#888'>({cite})</span></blockquote>"


SPEC_KO = spec(
    [
        ("누가", "워커A (리뷰)"),
        ("언제", now + " KST"),
        ("어디서", "arXiv — floq (2509.06863) + 후속 (2603.04333)"),
        ("무엇을", "flow-matching으로 critic을 학습하는 계열 정독 — 방법·실험·'왜 되는가'"),
        ("어떻게", "본문 정독 + 원문 인용 병기, 우리 스택(coverage/반사실)과 대조"),
        ("왜", "사용자 요청: critic을 통한 generalization의 다른 접근 조사"),
        ("코드", f"<code>{stamp}</code>"),
    ]
)
SPEC_EN = spec(
    [
        ("Who", "worker-A (review)"),
        ("When", now + " KST"),
        ("Where", "arXiv — floq (2509.06863) + follow-up (2603.04333)"),
        ("What", "the flow-matching critic family — method, experiments, and *why* it works"),
        ("How", "close read with verbatim quotes, contrasted with our coverage/counterfactual stack"),
        ("Why", "user request: survey a different route to generalization through the critic"),
        ("Code", f"<code>{stamp}</code>"),
    ]
)

KO = f"""
{SPEC_KO}
<p class='sub'>배경: 사용자 요청 — "critic을 통한 generalization의 다른 방식"으로 floq와 그 후속을 조사.
둘 다 Agrawalla·Nauman·(Agarwal)·Kumar. floq(2025)는 방법 제안, 후속(2026)은 "왜 되는가"의 해부다.</p>

<p><b>① floq — critic을 velocity field로.</b> 스칼라 Q를 직접 회귀하는 대신, Q를 시간·(s,a) 조건부
<b>속도장</b> v_θ(t, z | s,a)로 두고 노이즈 z(0)~Unif[l,u]를 K스텝 적분해 값을 읽는다:
Q(s,a) = ψ_θ(1, z). 학습은 flow-matching 손실을 TD 타깃 y=r+γ·(EMA 타깃장 적분)에 맞추는 것.
핵심 동기는 "iterative computation + 매 스텝 감독"이 표준 critic에 없다는 점:</p>
{quote("ResNets ... lack one ingredient that makes iterative computation effective in transformers or diffusion models: supervision at every step", "floq §1")}
{quote("we train the velocity to match the evolving TD-targets", "floq §4.2")}
<p><b>설계 3요소</b>(안 하면 'flow collapse'로 monolithic으로 퇴화): ⓐ 노이즈 구간 폭 κ=(u−l)/range≈0.1,
ⓑ 보간 입력 z(t)를 <b>HL-Gauss 히스토그램으로 인코딩</b>(σ=16, bin 80% 활성 — 입력 비정상성 완화),
ⓒ 시간 t의 64-d Fourier 임베딩. 성능(OGBench 50태스크, 상태기반): hard 태스크에서 FQL 대비 <b>≈1.8×</b>
(평균 hard 21→38; antmaze-giant 27→51). 적분 스텝 K는 4~8이 sweet spot, 16이면 과적합으로 하락.</p>
{quote("floq improves performance by nearly 1.8× on hard tasks", "floq Abstract")}

<p><b>② 후속 — "flow matching이 TD에 가져오는 것은 분포 모델링이 아니다."</b> 통념(성공=distributional RL)을
정면 반박한다. 통제 실험에서 <b>distributional 변형이 오히려 더 나쁘다</b>(hmmaze-large 30% vs 기대값 floq 52%):</p>
{quote("The advantage of flow matching does not come from modeling return distributions", "follow-up §4")}
<p>대신 두 메커니즘: <b>(a) test-time recovery</b> — 적분이 초기 값오차를 이후 스텝에서 감쇠(c-conic 조건
∂v/∂z ≤ −c/(1−t)로 t→1에 수렴 강화, β_K=O(K^−c)). 실험: 초기 25–50% 적분을 <b>낡은(stale) 속도장</b>으로
바꿔도 성능 유지/향상(antsoccer 45→50), monolithic은 붕괴. <b>(b) plasticity</b> — 매 보간점에서 속도를
감독하니 비정상 TD 타깃에 <b>피처를 덮어쓰지 않고 재가중</b>으로 적응. 증거: TD 학습 시 flow critic의
penultimate 피처 norm이 낮아지고(monolithic은 증가; SARSA/MC엔 없음 = TD 특이적), 50% 지점에서 피처를
얼려도 flow는 계속 개선·monolithic은 급락.</p>
{quote("Flow-matching critics can learn fairly general features that can represent multiple TD targets over the course of learning ... monolithic critics need to modify features to track the current TD target", "follow-up §6.1")}
{quote("Supervising velocities is essential for these benefits; directly supervising absolute TD targets collapses flow matching to monolithic behavior", "follow-up §6.2")}
<p>응용: high-UTD(32–128) 온라인 RL(+오프라인 데이터)에서 <b>최종 성능 2×, 표본효율 5×</b> — 가소성 손실이
심한 바로 그 체제.</p>
{quote("a 2× performance gain and a 5× improvement in sample efficiency in high update-to-data online RL with offline data", "follow-up Introduction")}

<p><b>우리 스택과의 관계 — 축이 다르다(그래서 상보적).</b></p>
<ul>
<li><b>이건 critic의 용량·최적화 축이지, 커버리지 축이 아니다.</b> 우리 null(BoN 무익·V≈MC·후보 붕괴)은
"타깃에 후보를 가를 <b>신호가 없다</b>(반사실 부재)"는 데이터 문제다. floq는 "신호가 있을 때 critic이 그걸
<b>잘 맞추고 잃지 않는가</b>"를 푼다. demo-only에서 후보가 사실상 동일하면 floq를 써도 후보 판별은 그대로
안 된다 — 우리 병목엔 직접 약이 아니다.</li>
<li><b>온라인 단계엔 직접 유효하다.</b> 우리 로드맵의 WSRL warmup→online critic·high-UTD 국면은 정확히
후속 논문이 2×/5×를 보인 조건(가소성 손실)이다. AC-RFT 온라인 phase의 critic 파라미터화 1순위 후보.</li>
<li><b>지금 돌리는 분포형 probe에 대한 경고이자 구분선.</b> 후속은 "flow critic의 확률적 출력은 의미있는
return 분포가 아니다"라고 못박는다. 단, 그건 <b>flow critic 한정</b>이고 우리 것은 HL-Gauss <b>범주형</b>
critic(cross-entropy로 가우시안-평활 타깃에 회귀)이라 그 분포는 타깃 분포 그 자체다. 그래도 "성능을 위해
분포를 모델링할 필요는 없다"는 결론은, 우리 다봉성 시각화를 <b>제어 도구가 아니라 진단 도구</b>로만 쓰라는
경계로 받아들인다.</li>
<li><b>HL-Gauss의 두 얼굴.</b> 우리는 HL-Gauss를 <b>출력 value head</b>로(다봉 진단), floq는 <b>보간 입력
인코딩</b>으로 쓴다(비정상성 완화). 같은 도구의 다른 용처 — floq의 σ=16·80% bin 활성 트릭은 우리 head
설계에도 참고 가치.</li>
</ul>
<p class='sub'>연결된 리포트: <span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>관계 기하 판정(r49)</span> ·
<span class='xref' onclick='openReport(#T:BoN 사전등록)'>BoN null(r53)</span>. git: <code>{stamp}</code></p>
"""

EN = f"""
{SPEC_EN}
<p class='sub'>Context: user request — survey "a different route to generalization through the critic" via floq
and its follow-up. Both Agrawalla·Nauman·(Agarwal)·Kumar; floq (2025) proposes the method, the follow-up
(2026) dissects *why* it works.</p>

<p><b>① floq — the critic as a velocity field.</b> Instead of regressing a scalar Q, parameterize Q as a
time/(s,a)-conditioned <b>velocity field</b> v_θ(t, z | s,a) and read the value by integrating noise
z(0)~Unif[l,u] for K steps: Q(s,a)=ψ_θ(1,z). Train the flow-matching loss toward the TD target
y=r+γ·(integrate the EMA target field). The motivation: standard critics lack step-wise supervision:</p>
{quote("ResNets ... lack one ingredient that makes iterative computation effective in transformers or diffusion models: supervision at every step", "floq §1")}
{quote("we train the velocity to match the evolving TD-targets", "floq §4.2")}
<p><b>Three design choices</b> (else 'flow collapse' → monolithic): ⓐ noise width κ=(u−l)/range≈0.1,
ⓑ encode the interpolant z(t) as an <b>HL-Gauss histogram</b> (σ=16, ~80% bins active — tames input
non-stationarity), ⓒ 64-d Fourier embedding of t. Results (OGBench, 50 state-based tasks): <b>≈1.8×</b> over
FQL on hard tasks (hard avg 21→38; antmaze-giant 27→51). Integration steps K=4–8 are the sweet spot; K=16
overfits and drops.</p>
{quote("floq improves performance by nearly 1.8× on hard tasks", "floq Abstract")}

<p><b>② Follow-up — "what flow matching brings to TD is NOT distributional modeling."</b> It refutes the
common belief head-on: a distributional variant is actually <b>worse</b> (hmmaze-large 30% vs expected-value
floq 52%):</p>
{quote("The advantage of flow matching does not come from modeling return distributions", "follow-up §4")}
<p>Two mechanisms instead. <b>(a) Test-time recovery</b> — integration dampens early value errors over later
steps (c-conic condition ∂v/∂z ≤ −c/(1−t), so contraction strengthens as t→1; β_K=O(K^−c)). Evidence:
replacing the first 25–50% of integration with a <b>stale</b> velocity field keeps or improves performance
(antsoccer 45→50) while monolithic critics collapse. <b>(b) Plasticity</b> — supervising velocities at every
interpolant lets the net adapt to non-stationary TD targets by <b>reweighting features, not overwriting</b>
them. Evidence: under TD, flow critics' penultimate feature norm drops (monolithic's rises; absent for
SARSA/MC = TD-specific); freezing features at 50% of training, flow keeps improving while monolithic collapses.</p>
{quote("Flow-matching critics can learn fairly general features that can represent multiple TD targets over the course of learning ... monolithic critics need to modify features to track the current TD target", "follow-up §6.1")}
{quote("Supervising velocities is essential for these benefits; directly supervising absolute TD targets collapses flow matching to monolithic behavior", "follow-up §6.2")}
<p>Application: high-UTD (32–128) online RL (+offline data) gives <b>2× final performance, 5× sample
efficiency</b> — exactly the plasticity-loss regime.</p>
{quote("a 2× performance gain and a 5× improvement in sample efficiency in high update-to-data online RL with offline data", "follow-up Introduction")}

<p><b>Relation to our stack — a different axis (hence complementary).</b></p>
<ul>
<li><b>This is the critic's capacity/optimization axis, not the coverage axis.</b> Our nulls (BoN inert, V≈MC,
candidate collapse) are a <b>data</b> problem — no signal in the target to separate candidates (absent
counterfactuals). floq solves "when signal exists, does the critic fit it well and not lose it." With
near-identical demo-only candidates, floq would not make candidate discrimination appear — no direct cure for
our bottleneck.</li>
<li><b>Directly useful for the online phase.</b> Our roadmap's WSRL warmup→online critic / high-UTD stage is
precisely where the follow-up shows 2×/5× (plasticity loss). Top candidate for the AC-RFT online-phase critic
parameterization.</li>
<li><b>A caution and a boundary for the distributional probe running now.</b> The follow-up states a flow
critic's stochastic output is <b>not</b> a meaningful return distribution — but that is <b>flow-specific</b>;
ours is a HL-Gauss <b>categorical</b> critic (cross-entropy to Gaussian-smoothed targets), so its distribution
*is* the target distribution. Still, "you don't need to model the distribution for performance" tells us to
treat our multimodality video as a <b>diagnostic, not a control lever</b>.</li>
<li><b>HL-Gauss, two uses.</b> We use HL-Gauss as the <b>output value head</b> (multimodality diagnostic);
floq uses it to <b>encode the interpolant input</b> (non-stationarity). Same tool, different slot — floq's
σ=16 / 80%-active-bins trick is worth borrowing for our head.</li>
</ul>
<p class='sub'>Links: <span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>relational-geometry verdict (r49)</span> ·
<span class='xref' onclick='openReport(#T:BoN 사전등록)'>BoN null (r53)</span>. git: <code>{stamp}</code></p>
"""

ENTRY = {
    "date": now,
    "title": "📄 [워커A] 논문 리뷰 — floq & 'What does flow matching bring to TD?' (critic를 velocity field로)",
    "summary": (
        "critic을 스칼라 회귀 대신 velocity field로 두고 노이즈를 K스텝 적분해 값을 읽는 floq(OGBench hard "
        "1.8×) + 후속 해부. 후속 핵심: 이득은 distributional 모델링이 아니라 ① test-time recovery(적분이 "
        "초기오차 감쇠) ② plasticity(dense velocity 감독→비정상 TD 타깃에 피처 재가중)이며, high-UTD에서 "
        "2×/5×. 우리와의 관계: 커버리지 축이 아니라 용량·최적화 축 — 온라인 phase엔 직접 유효, demo-only "
        "후보붕괴엔 무효. 지금 돌리는 HL-Gauss 다봉 probe는 '진단 도구로만' 쓰라는 경계도 제공."
    ),
    "tags": ["워커A", "논문리뷰"],
    "status": "finding",
}


def resolve(reports, txt):
    def repl(m):
        key = m.group(1)
        for i, r in enumerate(reports):
            if key in r["title"]:
                return f"openReport({i})"
        return "goHome()"

    return re.sub(r"openReport\(#T:([^)]+)\)", repl, txt)


api = HfApi()
for attempt in range(6):
    head = api.repo_info(SPACE, repo_type="space").sha
    p = hf_hub_download(SPACE, "index.html", repo_type="space", revision=head, force_download=True)
    s = pathlib.Path(p).read_text()
    m = re.search(r"const REPORTS\s*=\s*(\[.*?\]);", s, re.DOTALL)
    reports = json.loads(m.group(1))
    if any(e.get("title") == ENTRY["title"] for e in reports):
        print("already present")
        break
    ko_l, en_l = resolve(reports, KO), resolve(reports, EN)
    idx = len(reports)
    reports.append(ENTRY)
    s = s[: m.start(1)] + json.dumps(reports, ensure_ascii=False) + s[m.end(1) :]
    section = (
        f'<section class="report" id="r{idx}" hidden>'
        f'<div class="wbx wbx-ko">{ko_l}</div><div class="wbx wbx-en">{en_l}</div></section>'
    )
    last = s.rindex("</section>") + len("</section>")
    s = s[:last] + "\n" + section + s[last:]
    assert len(s.encode()) < 9_500_000
    tmp = pathlib.Path(tempfile.mkstemp(suffix=".html")[1])
    tmp.write_text(s)
    try:
        api.create_commit(
            SPACE,
            [CommitOperationAdd("index.html", str(tmp))],
            repo_type="space",
            commit_message=f"worker-A r{idx}: floq + flow-matching-TD paper review [{stamp}]",
            parent_commit=head,
        )
        print(f"published r{idx}")
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:100]}")
else:
    raise SystemExit("could not publish")
