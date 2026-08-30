"""Hub entry for the policy-extraction suite (eid extraction-suite-yam).

Ten extraction arms implemented at pi0.5 scale against the one frozen serving critic, with
line-level provenance; wave-1 offline comparison (training-free arms) recomputed from the
eval_extraction.py JSONs. Trained-arm evals and robot rollouts follow in later entries.
"""

import json
import pathlib
import subprocess

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
D = R / ".scratch/extraction/eval"

base = json.loads((D / "bc_base.json").read_text())
bm = base["metrics"]
ORDER = ["bon_n8", "idql_n64", "qpilots_a01", "qpilots_a02", "qpilots_a03", "flowdagger"]
NAMES = {
    "flowdagger": "FlowDAgger",
    "bon_n8": "BoN N=8",
    "idql_n64": "IDQL N=64",
    "qpilots_a01": "QPILOTS-U α=0.1",
    "qpilots_a02": "QPILOTS-U α=0.2",
    "qpilots_a03": "QPILOTS-U α=0.3",
}


def paired(label, key):
    ps = json.loads((D / f"{label}.json").read_text())["per_state"]
    d = np.array(ps[key]) - np.array(base["per_state"][key])
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(d.size)


rows = ""
vals = {}
for a in ORDER:
    dq, dqci = paired(a, "q_mean")
    dm, _ = paired(a, "demo_mse")
    dj, _ = paired(a, "jerk")
    vals[a] = (dq, dqci, dm, dj)
    rows += f"<tr><td>{NAMES[a]}</td><td>{dq:+.2f} ± {dqci:.2f}</td><td>{dm:+.4f}</td><td>{dj:+.4f}</td></tr>"

n_states = base["n_states"]
q3 = vals["qpilots_a03"][0]

stamp = subprocess.run(
    ["git", "-C", str(R), "log", "-1", "--format=fql-one-step-actor@%h"], capture_output=True, text=True, check=False
).stdout.strip()
if subprocess.run(
    ["git", "-C", str(R), "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False
).stdout.strip():
    stamp += "+dirty"

PROV = (
    "<div class='tblwrap'><table class='num'><tr><th>arm</th><th>근거 (코드/논문)</th><th>π0.5 이식 요점</th><th>상태</th></tr>"
    "<tr><td>AWR</td><td>xbpeng/awr awr_agent.py:403,407</td><td>A z-score→exp→clip20 가중 flow-BC</td><td>학습 중</td></tr>"
    "<tr><td>CFGRL</td><td>kvfrans/cfgrl iql_diffusion.py:157,170-179,213</td><td>O=1{A&gt;0}, cond+0.1·uncond, CFG 샘플링(w)</td><td>학습 중</td></tr>"
    "<tr><td>FlowDPG</td><td>논문 2606.22303 Eq.4-9 (코드 없음)</td><td>Tweedie â, twin-min ∇Q, λ-warmup</td><td>학습 중</td></tr>"
    "<tr><td>QAM</td><td>ColinQiyangLi/qam qam.py:49-145</td><td>slow/fast 필드, SDE 롤아웃, VJP adjoint</td><td>학습 대기</td></tr>"
    "<tr><td>LPS / LPSD</td><td>저자 코드 lps.py:185-224</td><td>α-Flow 200k one-step 위 latent actor</td><td>학습 중</td></tr>"
    "<tr><td>FlowDAgger</td><td>microsoft/FlowDAgger</td><td>perstep_fp seed 역산 + DCT-K10 + steering head; corrector=critic ascent(오프라인 치환)</td><td><b>완료</b></td></tr>"
    "<tr><td>DQL</td><td>Zhendong-Wang ql_diffusion.py:140-148</td><td>BC + η(−Q_i/sg|Q_j|), 샘플러 전체 BPTT</td><td>학습 중</td></tr>"
    "<tr><td>QPILOTS-U</td><td>논문 2606.14801 (코드 없음)</td><td>테스트타임 스티어링(Eq.14/15/17), 학습 0</td><td><b>평가 완료</b></td></tr>"
    "<tr><td>IDQL / BoN</td><td>philippe-eecs/IDQL ddpm_iql_learner.py:360-403</td><td>N-샘플 argmax min-Q(=BoN arm-0) + expectile 변형</td><td><b>평가 완료</b></td></tr>"
    "</table></div>"
)

LADDER = (
    "<table class='num'><tr><th>라운드</th><th>탈락</th><th>원인</th><th>고침</th></tr>"
    "<tr><td>1 (10 arms)</td><td>cfgrl·lps·lpsd·qam·flowdagger</td><td>import 경로 / numpy-closure tracer</td><td>Pi0Config 경로, frozen params를 jit 인자로</td></tr>"
    "<tr><td>2</td><td>lps·lpsd·qam·flowdagger</td><td>static ts→tracer / msgpack tuple / XLA constant-folding OOM</td><td>python tuple 시간격자, [w,b] 리스트 직렬화</td></tr>"
    "<tr><td>3-5 (qam)</td><td>qam</td><td>full-state grad 13GB + fp32 파라미터 2벌 28GB</td><td>expert-only grad + slow/fast 백본 공유</td></tr>"
    "<tr><td>6</td><td>—</td><td colspan='2'><b>10/10 전 arm 통과</b></td></tr></table>"
)

head = (
    f"<table class='num'><tr><th>항목</th><th>내용</th></tr>"
    f"<tr><th>who</th><td>워커B — 사용자 지시(공식 코드/논문 라인 단위 provenance 필수)의 밤샘 산출물</td></tr>"
    f"<tr><th>where</th><td>openpi-alphaflow · YAM lego-taxi 347ep · critic g5_tau9_min 고정 · L40S jobs 37026-37050</td></tr>"
    f"<tr><th>what</th><td>VLA policy-extraction 10 arms 구현 + 스모크 검증 + wave-1 오프라인 비교(무학습 arms, n={n_states})</td></tr>"
    f"<tr><th>how</th><td>단일 프로토콜(eval_extraction.py): 고정 strided 상태셋, arm 간 동일 노이즈(paired), critic-Q/demo-MSE/jerk</td></tr>"
    f"<tr><th>why</th><td>어떤 extraction 기제가 π0.5 스케일에서 실제로 Q를 올리고 무엇을 대가로 치르는지 — 롤아웃 전 오프라인 예비 판정</td></tr>"
    f"<tr><th>코드</th><td><code>{stamp}</code></td></tr></table>"
)

KO = f"""{head}
<p><b>왜.</b> 지금까지의 서빙은 BoN·adaptive-K(<span class='xref' data-eid='iql-pair-yam'>critic pair</span>)
한 계열이었다. 문헌에는 같은 frozen critic에서 정책을 뽑아내는 서로 다른 기제 — 가중 회귀(AWR),
조건화(CFGRL), 증류(DQL·FlowDPG·QAM·LPS), 시드 조향(FlowDAgger), 테스트타임 조향(QPILOTS), 선택(IDQL/BoN) —
가 있고, 이를 <b>method-only-diff</b>로 붙는 링을 만들었다: 같은 BC h30 init, 같은 critic, action expert만 학습.</p>

<p><b>무엇을 구현했나.</b> 각 파일 상단에 공식 코드/논문의 파일·줄 단위 근거를 남겼다(사용자 규칙).</p>
{PROV}

<p><b>스모크 사다리.</b> 순차 스모크 6라운드로 전 arm 통과 — 탈락 원인은 전부 π0.5 스케일 특유의
메모리/트레이싱이었다(방법론 자체 아님):</p>
{LADDER}

<p><b>Wave-1 오프라인 비교 (무학습 arms, paired, n={n_states}).</b> BC 대비 페어드 Δ — 같은 상태·같은 노이즈.</p>
<table class='num'><tr><th>arm</th><th>ΔQ (95% CI)</th><th>Δ demo-MSE</th><th>Δ jerk</th></tr>{rows}</table>
<figure><img src='figures/extraction-suite-yam/fig_extraction_wave1.png' alt='wave-1 paired comparison'>
<figcaption>왼쪽: BC 대비 페어드 critic-Q 상승. 오른쪽: 그 대가(분포 이탈·저크).</figcaption></figure>

<p><b>잠정 판정 1 — 조향은 세고, 비싸다.</b> QPILOTS-U는 α에 단조로 ΔQ {q3:+.1f}까지 올리지만 demo-MSE
{vals["qpilots_a03"][2]:+.3f}(기저 {bm["demo_mse"]["mean"]:.4f}의 ~5배), jerk +28%를 치른다 — critic이 좋아하는
방향이 데모 분포 밖이라는 뜻이고, K=2 앙상블 pessimism으로는 critic-exploitation을 못 막을 수 있다.
<b>ΔQ는 자기 심판 지표다</b>(조향이 그 critic을 직접 올림) — 진짜 판정은 로봇 롤아웃.</p>

<p><b>잠정 판정 2 — 선택은 정직하고 겸손하다.</b> IDQL N=64 ΔQ +2.7, BoN N=8 +1.55 — 분포 안에 머물러
(ΔMSE ≈ +0.003) 대가가 거의 없다. N을 8→64로 8배 올려도 +1.55→+2.7뿐: <b>BC 샘플 분포의 Q-스프레드가
좁다</b>는 뜻으로, EMaQ 계열의 'N 증가는 금방 포화' 예측과 부합.</p>

<p><b>잠정 판정 3 — FlowDAgger(오프라인 치환)는 무효.</b> 조향 헤드가 시드 공간 MSE 0.32에서 수렴했지만
ΔQ −0.02 — 보정 시드의 정보가 헤드 용량/표현(pooled-DINO)으로는 전달이 안 되거나, recon 게이트(0.001)를
통과한 보정 자체가 작았다. 원 논문의 corrector는 실제 전문가 개입이라 이 오프라인 치환의 실패가
방법 자체의 반증은 아니다.</p>

<p><b>다음.</b> ① 학습 arms(awr·cfgrl·flowdpg·dql·lps·lpsd·qam) 체크포인트 도착 시 같은 프로토콜로 wave-2.
② 사용자 로봇 롤아웃용 arm 서빙 준비(체크포인트는 순차 HF 업로드). ③ ΔQ-vs-이탈 프런티어에서 α·N 선정.</p>

<p><b>한계.</b> 전 지표가 오프라인 프록시 — 특히 ΔQ는 critic-ascending arms에 자기 편향. demo-MSE는
"데모와 같으면 좋다"는 가정(BC 최적화와 동어반복 위험). n={n_states} 상태는 in-distribution strided 셋.</p>"""

EN = f"""{head}
<p><b>Why.</b> Our serving so far has been one family (BoN / adaptive-K over the
<span class='xref' data-eid='iql-pair-yam'>critic pair</span>). The literature offers distinct mechanisms for
extracting a policy from the same frozen critic — weighted regression (AWR), conditioning (CFGRL), distillation
(DQL, FlowDPG, QAM, LPS), seed steering (FlowDAgger), test-time steering (QPILOTS), selection (IDQL/BoN) — and
we built a <b>method-only-diff</b> ring: same BC h30 init, same critic, action expert only.</p>

<p><b>What was implemented.</b> Every file carries file/line-level provenance from the official code or the
paper (user rule).</p>
{PROV}

<p><b>The smoke ladder.</b> Six sequential rounds to 10/10 — every failure was pi0.5-scale memory/tracing,
none methodological:</p>
{LADDER}

<p><b>Wave-1 offline comparison (training-free arms, paired, n={n_states}).</b> Paired deltas vs BC —
same states, same noise.</p>
<table class='num'><tr><th>arm</th><th>ΔQ (95% CI)</th><th>Δ demo-MSE</th><th>Δ jerk</th></tr>{rows}</table>
<figure><img src='figures/extraction-suite-yam/fig_extraction_wave1.png' alt='wave-1 paired comparison'>
<figcaption>Left: paired critic-Q uplift vs BC. Right: the price (distribution shift, jerk).</figcaption></figure>

<p><b>Provisional verdict 1 — steering is strong and expensive.</b> QPILOTS-U climbs monotonically to
ΔQ {q3:+.1f} at α=0.3 but pays demo-MSE {vals["qpilots_a03"][2]:+.3f} (~5× the base {bm["demo_mse"]["mean"]:.4f})
and +28% jerk — the critic's preferred direction leaves the demo distribution, and K=2 ensemble pessimism may
not stop critic exploitation. <b>ΔQ is a self-refereed metric here</b> (steering ascends that very critic) —
the real referee is the robot rollout.</p>

<p><b>Provisional verdict 2 — selection is honest and modest.</b> IDQL N=64 ΔQ +2.7, BoN N=8 +1.55, staying
in-distribution (ΔMSE ≈ +0.003). An 8× increase in N buys only +1.55→+2.7: <b>the BC sample distribution's
Q-spread is narrow</b>, matching the EMaQ-line prediction that N saturates fast.</p>

<p><b>Provisional verdict 3 — FlowDAgger (offline substitution) is null.</b> The steering head converged
(seed-space MSE 0.32) yet ΔQ −0.02 — either the corrected-seed signal doesn't survive the head's
capacity/representation (pooled DINO), or the corrections passing the 0.001 recon gate were tiny. The original
corrector is a live expert, so this offline substitution's failure does not refute the method itself.</p>

<p><b>Next.</b> (1) Wave-2 under the same protocol as trained-arm checkpoints land (awr, cfgrl, flowdpg, dql,
lps, lpsd, qam). (2) Serving prep for the user's robot rollouts (checkpoints uploaded to HF as they arrive).
(3) Pick α/N on the ΔQ-vs-shift frontier.</p>

<p><b>Limitations.</b> All metrics are offline proxies — ΔQ is self-biased for critic-ascending arms;
demo-MSE assumes demo-likeness is good (near-tautological for BC); n={n_states} in-distribution strided states.</p>"""

entry = {
    "eid": "extraction-suite-yam",
    "date": "2026-08-30 10:55",  # KST (machine clock is UTC)
    "worker": "B",
    "title": "🧪 [워커B] policy-extraction 10 arms 링 — 구현·검증 완료, wave-1: 조향은 세지만 비싸고, 선택은 정직하다",
    "summary": f"AWR·CFGRL·FlowDPG·QAM·LPS/LPSD·FlowDAgger·DQL·QPILOTS-U·IDQL/BoN을 π0.5 스케일에 라인 단위 provenance로 구현, 스모크 6라운드에 10/10 통과. 무학습 arms 오프라인 비교(paired n={n_states}): QPILOTS-U ΔQ {q3:+.1f}(α=.3)로 압도하나 demo-MSE ~5배·jerk +28%의 분포 이탈을 치르고, IDQL/BoN은 +1.5~2.7로 겸손하되 무비용, FlowDAgger 오프라인 치환은 무효. ΔQ는 자기 심판 — 판정은 로봇 롤아웃으로.",
    "tags": ["워커B", "extraction", "YAM", "비교링"],
    "status": "ongoing",
    "phase": "방법 비교",
    "links": ["iql-pair-yam", "alphaflow-1step-gate", "jvp-offdiag"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/extraction_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "stamp", stamp)
