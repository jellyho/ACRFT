"""Split worker-A's overnight mega-report into per-experiment hub entries (worker-B format).

User feedback (2026-08-08): "실험 세팅과 그 분석이 너무 빈약해. 아직도 나는 니가 무슨 실험을
돌린지 알 수가 없네" — every entry now carries a full spec (누가/언제/무엇을/어떻게: job IDs,
commands, data, seeds, checkpoints) and its own analysis, one experiment per entry.

Outputs /tmp/wa_entries.json: [{entry: {...REPORTS fields...}, body: "<html>"}].
Figures are reused from the overnight report (extracted to /tmp/wa_fig{0..3}.b64).
"""

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = [pathlib.Path(f"/tmp/wa_fig{i}.b64").read_text() for i in range(4)]
GIT = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
).stdout.strip()

STYLE = """<style>
.wa-rpt{overflow-x:hidden}
.wa-rpt p,.wa-rpt li,.wa-rpt div,.wa-rpt code{overflow-wrap:anywhere}
.wa-rpt table{width:100% !important;max-width:100% !important;table-layout:fixed;border-collapse:collapse;font-size:.88em}
.wa-rpt th,.wa-rpt td{overflow-wrap:anywhere;word-break:break-word;white-space:normal !important;padding:5px 8px;vertical-align:top}
.wa-rpt .spec th{width:92px}
.wa-rpt .num th:first-child,.wa-rpt .num td:first-child{width:38%}
.wa-rpt .num td:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}
.wa-rpt img{max-width:100%;height:auto}
</style>"""


def wrap(body):
    return f"{STYLE}<div class='wa-rpt'>{body}<p style='color:#888;font-size:.85em'>git: fix/probe-eval-jit @ {GIT}</p></div>"


def img(i):
    return f"<img src='data:image/jpeg;base64,{FIG[i]}'>"


E = []

# ---------------- entry 1: the 7+4-arm paired rollout campaign ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 11:30",
            "title": "🎰 [워커A] PC 롤아웃 대작전 — critic 권한 7-arm + bounded-authority 4-arm, 재현까지",
            "summary": "유일한 유의 결과는 부정: 오프라인 전 조건 통과한 φ Cal-QL+swap critic이 전권 모드에서 .300 (vla .700, McNemar p=.004). 첫 30-trial의 모든 양성(+.10 prefix, +.067 σ-커밋)은 seed-set 재현에서 소거 — 60 paired에서 정확히 동률. 커밋 분포: 최고 arm은 이봉형, 붕괴 arm은 2스텝 스래싱 59%.",
            "tags": ["워커A", "RoboCasa", "critic", "rollout", "MB-AC"],
            "status": "finding",
        },
        "body": f"""
<h2>무엇을, 어떻게</h2>
<table class='spec'>
<tr><th>언제</th><td>2026-08-08 01:00–11:30 (밤샘, 순차 5개 slurm job)</td></tr>
<tr><th>VLA</th><td><code>pi05_robocasa_PrepareCoffee_rlt</code> / <code>PrepareCoffee_rlt5_pardec_noprop/70000</code> (로컬에 없어 HF <code>pi05-robocasa-prepcoffee-rlt-pardec-noprop-70k</code>에서 복원). 오버라이드 <code>rlt_decoder_mode=parallel, rlt_include_proprio=false</code></td></tr>
<tr><th>환경</th><td>PrepareCoffee, max 1000 steps/trial, N=16 candidates/replan, flow 10 steps, camera 256, MUJOCO_GL=egl</td></tr>
<tr><th>critic A</th><td><code>critic_vlaz_ref</code>: raw RLT 토큰 2048+proprio16, plain IQL(τ.7) — 오프라인 action-blind(sens .001)</td></tr>
<tr><th>critic B</th><td><code>critic_rltphi_calswap</code>: HILP φ readout 128+16, IQL+Cal-QL(α1)+swap negatives — 오프라인 전 조건 통과(sens .524, binding .996). 롤아웃에는 <code>--phi rlt_hilp_readout/phi.pt</code> 어댑터(2048→128 numpy gelu MLP, torch 대비 1e-6 일치 검증)</td></tr>
<tr><th>σ-커밋</th><td><code>--dyn phi_dyn_v1_h1/ensemble_v1.pt</code> (φ-공간 DynV1 5-ensemble, hist1) — 슬롯별 disagreement가 러닝 미디언의 tau배를 넘으면 컷. mbacv=커밋만, mbac=critic 선택+커밋, mbacf=σ-veto 절반 후 critic argmax(+커밋)</td></tr>
<tr><th>시드 규약</th><td>set A = seed 0 (trial i는 scene seed 0+i, 30 trials) 전 arm 공유 → 완전 paired. set B = seed 30 (재현, 새 장면 30개, 자체 vla 기준선 포함)</td></tr>
<tr><th>job</th><td>34632(control 4-arm) → 34633(φ 3-arm) → 34652(mbacv/mbac/mbacf, tau2.0) → 34656(mbacv tau1.3) → 34661(재현: vla/mbacv/mbacf/prefix, tau1.3)</td></tr>
<tr><th>명령</th><td><code>uv run --group eval python examples/robocasa/eval_critic.py --config … --checkpoint … --critic … [--phi …] [--dyn … --dyn-tau …] --task PrepareCoffee --num-trials 30 --seed 0|30 --modes …</code></td></tr>
</table>

<h2>왜</h2>
<p>오프라인 진단(binding·action-sensitivity)을 전부 통과한 critic이 실제 배포에서 이득을 주는가 —
그리고 critic에게 주는 <b>권한의 폭</b>(선택만 / 커밋만 / 전권)이 결과를 바꾸는가. worker-B의 23-config
피해 테이블(raw 토큰)과 독립 스택(φ 표현, Cal-QL+swap)에서 교차 검증하는 목적도 있다.</p>

<h2>결과</h2>
{img(0)}
<p>30 paired trials 기준, 채워진 점만 유의(McNemar p&lt;.05). <b>φ critic 전권 모드가 .300으로 붕괴한 것이
밤 전체의 유일한 유의 결과다</b>(+2/−14, p=.004). action-blind critic은 무의미(bon .600≈critic .567 — 동전던지기,
오프라인 sens .001의 롤아웃 재현). φ bon은 vla와 정확히 동률(+5/−5) — 선택-만 권한은 안전하지만 무익.</p>
{img(1)}
<p>커밋 길이 분포가 세 체제를 갈라 보여준다: 야간 최고 성적(.800, 유의 미달)의 control prefix는 <b>이봉형</b>
(30%는 2스텝 컷, 60%는 14–16 풀커밋 — 소박한 적응 커밋), 붕괴한 φ 전권 critic은 <b>59%를 2스텝에 몰아넣는
스래싱</b>(mean 4.5), σ-rule(tau 2.0)은 88% 풀커밋으로 사실상 무개입(E1 배터리가 예측한 그대로).</p>
<table class='num'><tr><th>arm</th><th>set A</th><th>set B(재현)</th><th>pooled 60</th><th>vs vla</th></tr>
<tr><td>vla</td><td>.700</td><td>.633</td><td>.667</td><td>—</td></tr>
<tr><td>prefix (action-blind)</td><td>.800</td><td>.500</td><td>.650</td><td>+10/−11, p=1.0</td></tr>
<tr><td>mbacv σ-커밋만 (tau1.3)</td><td>.767</td><td>.567</td><td>.667</td><td>+8/−8, p=1.0</td></tr>
<tr><td>mbacf σ-veto BoN (tau2.0/1.3)</td><td>.700</td><td>.500</td><td>—(설정 불일치)</td><td>n.s.</td></tr>
<tr><td>φ critic 전권</td><td><b>.300</b></td><td>—</td><td>—</td><td><b>+2/−14, p=.004</b></td></tr>
</table>

<h2>해석</h2>
<p>① <b>권한이 변수다</b>: 같은 critic이 선택-만에선 무해, 전권(16후보×8prefix=128 옵션 argmax)에선 파국.
아는 것(binding)을 고쳐도 argmax가 Q의 꼬리 노이즈를 매 replan 같은 방향으로 착취하면 오차가 복리로 쌓인다
(랜덤 선택은 편향이 없어 vla와 같고, argmax는 편향을 증폭해 랜덤보다 나빠질 수 있다).
② <b>±.1은 잡음</b>: 30-trial arm의 모든 양성이 새 장면 30개에서 부호가 뒤집혔다. worker-B의 n≥8 규칙과 합치 —
이후 모든 판정은 3 seed-set × 50 trials(150 paired)로 사전 등록.
③ 이 태스크·이 체크포인트에서 적응 청킹의 지렛대가 작은 구조적 이유: 실패가 청크 사이가 아니라 청크 안에서
결정되고(잘못된 파지 등), 같은 flow의 N=16 후보가 선택이 갈릴 만큼 다르지 않다.</p>
<p>원시 데이터: <code>.scratch/rollout_{{control,rltphi,mbac,mbacv_tau13,rep2_seed30}}.json</code> (per-trial trace 포함)</p>
""",
    }
)

# ---------------- entry 2: phi-space dynamics ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 06:00",
            "title": "🌀 [워커A] φ-공간 dynamics — 같은 모델, 표현만 바꿔 전 구간 압승",
            "summary": "DynV1(macro-stride4 causal transformer 5-ensemble, 40k steps)을 DINO 공간 대신 HILP φ(128d)에서 학습: R² .29/.51/.61/.66 @4/8/12/16 (DINO .06/.30/.43/.52), AUSE .07, OOD action-swap 1.41→1.54. hist=1이 hist=3와 동급이라 배포에 토큰 히스토리 불필요.",
            "tags": ["워커A", "RoboCasa", "dynamics", "표현"],
            "status": "done",
        },
        "body": f"""
<h2>무엇을, 어떻게</h2>
<table class='spec'>
<tr><th>언제</th><td>2026-08-08 05:00–07:30 (job 34635 hist3, 34641 hist1)</td></tr>
<tr><th>모델</th><td>DynV1: block-causal transformer d288 depth4 heads8, macro-stride s=4(모델 1스텝=제어 4스텝, 16-청크=4 macro-슬롯), Gaussian NLL(softplus-bounded logvar), teacher-forcing+2-step rollout loss 1:1, 완전 독립 5-member 앙상블. 40k steps, batch 256, AdamW 3e-4 cosine</td></tr>
<tr><th>데이터</th><td><code>annot_noprop</code> 279,534 frames의 (φ_t 히스토리, 데모 청크 16×12, φ_{{t+4/8/12/16}}), φ = <code>rlt_hilp_readout/z.npy</code>(128d, 표준화), 에피소드 15% held-out</td></tr>
<tr><th>명령</th><td><code>uv run python scripts/train_cheapz_dynamics_v1.py --z-dir .scratch/rlt_hilp_readout --annot .scratch/annot_noprop --stride 4 --hist 3|1 --members 5 --steps 40000</code></td></tr>
<tr><th>비교 대상</th><td>동일 아키텍처·스텝의 DINO v4b 공간(256d) run (<code>cheapz_dyn_v1</code>, 전날)</td></tr>
</table>

<h2>왜</h2>
<p>MB-AC의 모든 것(σ-커밋, veto, 상상 기반 가치)이 dynamics 품질 위에 선다. 예측이 쉬운 공간이 어디인지 —
외형 잡음이 남은 DINO 공간인가, TD readout이 도달가능성 기하만 남긴 φ 공간인가.</p>

<h2>결과</h2>
{img(2)}
<table class='num'><tr><th>지표</th><th>DINO(256d)</th><th>φ hist3</th><th>φ hist1</th></tr>
<tr><td>R² vs copy-forward @16</td><td>.519</td><td>.659</td><td>.665</td></tr>
<tr><td>AUSE(캘리브레이션, 낮을수록↑)</td><td>.089–.114</td><td>.072–.080</td><td>.072–.088</td></tr>
<tr><td>OOD action-swap disagreement 비</td><td>1.09</td><td>1.41</td><td>1.54</td></tr>
</table>

<h2>해석</h2>
<p>TD readout이 외형 잡음을 미리 버려서, 남은 기하가 정확히 행동이 움직이는 것이 됐다 — 같은 용량으로 DINO 공간
대비 전 구간 R² 우위, 그리고 <b>틀린 행동을 넣으면 앙상블이 1.5배 더 불안해진다</b>(swap 1.54). 단 R²(+4)=.29로
4-스텝 게이트는 미달: 모델은 첫 macro-step에 대해선 여전히 copy-forward 수준이므로, 커밋 floor는 4스텝이 맞다.
hist=1 동급 판정으로 롤아웃 배포에서 토큰 히스토리 링버퍼가 불필요해졌다(mbacv/mbac/mbacf가 이 hist1 판을 사용).</p>
""",
    }
)

# ---------------- entry 3: MB-AC offline battery ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 06:40",
            "title": "🧭 [워커A] MB-AC 오프라인 배터리 E1–E3 — σ는 binding하고, 모델-가치는 역전한다",
            "summary": "held-out 8000 anchors: 앙상블 disagreement 단독으로 demo vs 타상태-demo 청크 binding .817(CQL 없이, 첫 macro-step에서 이미 최고 .837) — 그러나 모델이 예측한 종말상태의 goal 거리로 랭킹하면 .371로 우연 이하(함정). σ-적응 컷은 fixed-k 프론티어를 근소하게 밑돈다.",
            "tags": ["워커A", "RoboCasa", "dynamics", "MB-AC"],
            "status": "finding",
        },
        "body": f"""
<h2>무엇을, 어떻게</h2>
<table class='spec'>
<tr><th>언제</th><td>2026-08-08 06:10–06:40 (job 34644 DINO판, 34645 φ판 — torch weights_only 버그 수정 후 재제출분)</td></tr>
<tr><th>평가</th><td><code>scripts/eval_mbac_offline.py</code>: 학습 split을 시드로 재현한 held-out 에피소드에서 n=8000 anchors. E1: 데모 자기 청크를 따라 σ_k·오류_k 측정, quantile-τ 컷 vs fixed-k의 '평균 커밋당 종말 오류' 프론티어. E2: 데모 청크 vs jnp.roll 타상태-demo 청크(--cql-swap과 같은 negative)를 ①모델 예측 종말상태의 에피소드-goal 거리 ②총 disagreement로 랭킹. E3: E2를 macro-step별로.</td></tr>
<tr><th>모델</th><td>phi_dyn_v1 (hist3, 5-ensemble)</td></tr>
</table>

<h2>왜</h2>
<p>모델을 <b>가치평가</b>에 쓸 수 있는가(선택 신호), 아니면 <b>신뢰영역</b>으로만 쓸 수 있는가(커밋·거부권 신호).
MB-AC의 권한 배분 설계가 이 답에 달려 있다.</p>

<h2>결과</h2>
{img(3)}
<table class='num'><tr><th>지표</th><th>φ 공간</th><th>DINO 공간</th></tr>
<tr><td>binding by disagreement</td><td><b>.817</b> (σ비 1.61)</td><td>.780 (1.10)</td></tr>
<tr><td>binding by 예측-goal거리</td><td><b>.371</b> (우연 이하!)</td><td>.315</td></tr>
<tr><td>E3: disagreement binding @4/8/12/16</td><td>.837/.815/.807/.798</td><td>—</td></tr>
<tr><td>E1: σ-적응 q0.3 (mean_k 14.2)</td><td>오류 4.13 vs fixed 보간 4.19</td><td>σ가 거의 안 끊음(97% 풀커밋)</td></tr>
</table>

<h2>해석</h2>
<p>① disagreement는 CQL 없이 행동을 상태에 binding한다(.817) — "데이터가 이런 식으로 이어진 적 없다"는 알람으로서의
모델. 가시성이 첫 macro-step에서 이미 최고라 veto는 깊은 롤아웃이 필요 없다(→ mbacf가 1-슬롯 σ만 쓰는 근거).
② 모델의 예측값을 goal 거리 휴리스틱으로 읽으면 <b>우연보다 나쁘다</b> — 틀린 행동의 예측이 progress를 흉내내며
표류한다. 이것이 이후 MVE critic 설계에서 '모델 출력은 반드시 학습된 V에 넣고, 랭킹 휴리스틱으로 쓰지 않는다'는
원칙이 됐다. ③ σ-컷 지점은 오류가 실제로 빨리 자라는 anchor에 몰리고(컷-조기 그룹 오류 기울기 3.01 vs 2.00)
태스크 후반부(파지·배치 국면)에 집중된다.</p>
""",
    }
)

# ---------------- entry 4: MVE critic (ongoing) ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 16:00",
            "title": "⚗️ [워커A] MVE critic (in-sample MAC) — policy 없이 해상도·비과대평가 잡기, 4-변형 대기중",
            "summary": "MAC(2512.08108)·IQL-TD-MPC(2306.00867) 정독 후 설계: V는 φ-단독 in-sample IQL, Q는 저장된 VLA 후보 16개 전부에 대해 y=r̂+γ^16·min₅V(φ̂)로 증류 — CQL·λσ 패널티·live policy 전부 없음. 변형 4개(min/mean 집계 × r̂ 유무 × τ.7/.9) GPU 대기.",
            "tags": ["워커A", "RoboCasa", "critic", "MB-AC"],
            "status": "ongoing",
        },
        "body": """
<h2>무엇을, 어떻게</h2>
<table class='spec'>
<tr><th>설계 근거</th><td>MAC(Park·Park·Lee·Levine): 착취 방지는 λσ 패널티가 아니라 'BC flow 샘플만 질의'. IQL-TD-MPC: 계획은 정책 샘플 위에서만(n_r=0). 우리 제약: live policy 없음(VLA는 이미지 필요·비쌈) → <b>annotation의 프레임당 VLA 후보 16개 = frozen policy 아카이브</b>로 대체</td></tr>
<tr><th>알고리즘</th><td>①V: 청크 단위 IQL(expectile τ, φ-단독 — 상상 상태에서 평가 가능해야 하므로 proprio 제외). sparse reward 닫힌형: mc_t=γ^(T−t)이므로 성공이 청크 안이면 r_n=mc_t. ②전 후보 사전계산: y_i = clamp(r̂(φ,a_i) + γ^16·min_m V(φ̂_m(φ,a_i)), 0, 1) — min-앙상블이 유일한 보수화 장치. ③최종 Q(φ⊕proprio,a): 배치 절반 데모(실타겟)+절반 후보(y). 배포는 bon 그대로 → calswap과 Q만 다른 A/B</td></tr>
<tr><th>변형</th><td>A=min+r̂+τ.7(34691) / B=mean 집계(34694) / C=r̂ 제거(34695) / D=τ.9(34696)</td></tr>
<tr><th>게이트</th><td>action sens ≥.03, binding ≥.9, <b>Q_demo−mc ≤ .05</b>, support 초과≈0, Spearman ≥.75 (기준선: calswap .524/.996/+.029)</td></tr>
<tr><th>상태</th><td>4 job 모두 L40S 큐 대기 (GPU 기근). 코드 <code>scripts/train_mve_critic.py</code></td></tr>
</table>
<h2>왜</h2>
<p>두 워커의 유의한 부정 결과 두 개 — 전권 argmax의 파국(워커A, p=.004)과 CQL 과잉억압(워커B calql_v14, p=.001) —
을 동시에 회피하는 critic: 비관은 in-sample V와 min-앙상블에서, 해상도는 모델의 반사실 착지점에서, OOD 질의는 구조적으로 0.</p>
""",
    }
)

# ---------------- entry 5: GP protocol (ongoing) ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 15:00",
            "title": "🥞 [워커A] GarnishPancake 파이프라인 개시 — 100k 완주, 3×50 CI 프로토콜 사전등록",
            "summary": "GP mae0.5 학습 100k 완주. PC 교훈을 프로토콜로: 모든 arm은 seed set 3개(0/100/200)×50 trials=150 paired, VLA 기준선·체크포인트 스윕(50k–100k)부터. annotation(mae0.5@100k, N=16)→raw calswap critic→φ-체인 critic 의존성 사다리 제출. 전부 GPU 대기중.",
            "tags": ["워커A", "RoboCasa", "GP", "프로토콜"],
            "status": "ongoing",
        },
        "body": """
<h2>무엇을, 어떻게</h2>
<table class='spec'>
<tr><th>학습 완료</th><td><code>GarnishPancake_rlt7_pardec_noprop_mae0.5</code> 100k steps (job 34197, B200 30h). 체크포인트 10k–100k 전부 HF <code>pi05_robocasa_garnishpancake_rlt7_mae05</code>에 검증-아카이브</td></tr>
<tr><th>평가 프로토콜</th><td>arm당 seed set 3개(0–49/100–149/200–249) × 50 trials — PC의 '30-trial ±.1 잡음' 교훈 반영, 세트별 paired McNemar + 세트간 run-level CI 병기. 모든 후속 arm이 같은 150 장면 재사용</td></tr>
<tr><th>제출된 사다리</th><td>VLA 기준선: 34674-76(100k, 세트별) + 34677-81(50k–90k 스윕, job당 3세트 순회). annotation: 34671(mae0.5@100k, N=16, batch16, XLA_MEM_FRACTION=.85 — 이전 mae0.5 OOM의 restore-시점 원인 대응). critic: 34672(raw+Cal-QL+swap) / 34673(토큰캐시→HILP φ readout(NaN 수정 이식판)→φ+calswap) 자동 연쇄</td></tr>
<tr><th>상태</th><td>L40S 16장 타점유로 전체 대기. YAM 병행: s200 200k 연장 ~171k, 청크60(h60 config 신설, 전용 norm stats) ~29k/100k</td></tr>
</table>
<h2>왜</h2>
<p>PC에서 확립한 전 파이프라인(annotation→critic 매트릭스→paired 롤아웃)을 두 번째 태스크에서 반복해 경향성을
확인한다 — 이번엔 검정력 문제를 프로토콜 수준에서 선결한 채로.</p>
""",
    }
)

out = [{"entry": e["entry"], "body": wrap(e["body"])} for e in E]
pathlib.Path("/tmp/wa_entries.json").write_text(json.dumps(out, ensure_ascii=False))
print("entries:", len(out), "| total body KB:", sum(len(o["body"]) for o in out) // 1024)
