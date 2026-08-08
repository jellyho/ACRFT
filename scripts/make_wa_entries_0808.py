"""Worker-A hub entries, v2: tableless layout + every term explained in place.

User feedback round 2 (2026-08-08): (1) "각 metric이 무슨 의미이고 … swap negatives 이러는데
내가 뭔지 알 도리가 없다" — every metric/term is now defined in plain language at first use,
plus a standalone glossary entry. (2) tables overlapped following text — NO <table> elements
remain; specs and results are flex key-value rows (normal block flow cannot overlap).

Outputs /tmp/wa_entries.json: [{entry, body}]. Figures reused from /tmp/wa_fig{0..3}.b64.
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
.wa-rpt img{max-width:100%;height:auto}
.wa-kv{border:1px solid #e2e2e2;border-radius:10px;margin:10px 0;overflow:hidden}
.wa-kv>div{display:flex;gap:12px;padding:8px 14px;border-bottom:1px solid #ececec;align-items:baseline}
.wa-kv>div:last-child{border-bottom:none}
.wa-kv .k{flex:0 0 118px;font-weight:700;color:#555;font-size:.88em}
.wa-kv .v{flex:1;min-width:0;font-size:.92em}
.wa-row{display:flex;gap:12px;padding:6px 14px;border-bottom:1px solid #f0f0f0;align-items:baseline;font-size:.92em}
.wa-row .k{flex:0 0 210px;min-width:0;font-weight:600}
.wa-row .v{flex:1;min-width:0;font-variant-numeric:tabular-nums}
.wa-res{border:1px solid #e2e2e2;border-radius:10px;margin:10px 0;overflow:hidden}
.wa-res .hd{background:#f6f5f4;padding:6px 14px;font-weight:700;font-size:.88em;color:#444}
.wa-term{background:#f4f6fb;border-left:3px solid #6b7fc7;border-radius:0 8px 8px 0;padding:8px 13px;margin:8px 0;font-size:.9em}
</style>"""


def wrap(body):
    return (
        f"{STYLE}<div class='wa-rpt'>{body}"
        f"<p style='color:#888;font-size:.85em'>git: fix/probe-eval-jit @ {GIT}</p></div>"
    )


def img(i):
    return f"<img src='data:image/jpeg;base64,{FIG[i]}'>"


def kv(*pairs):
    rows = "".join(f"<div><span class='k'>{k}</span><span class='v'>{v}</span></div>" for k, v in pairs)
    return f"<div class='wa-kv'>{rows}</div>"


def res(title, *pairs):
    rows = "".join(f"<div class='wa-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>" for k, v in pairs)
    return f"<div class='wa-res'><div class='hd'>{title}</div>{rows}</div>"


def term(name, expl):
    return f"<div class='wa-term'><b>{name}</b> — {expl}</div>"


E = []

# ---------------- entry 0: glossary ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 17:30",
            "title": "📖 [워커A] 용어·지표 사전 — 워커A 엔트리를 읽기 위한 모든 개념",
            "summary": "RLT 토큰·φ readout·IQL·Cal-QL·swap negatives·binding·action sensitivity·σ-disagreement·AUSE·McNemar 등 워커A 실험에 등장하는 모든 용어를 정의. 각 지표의 계산법, 좋은 값/나쁜 값, 왜 재는지까지.",
            "tags": ["워커A", "용어", "참고"],
            "status": "living",
        },
        "body": f"""
<h2>표현(입력) 쪽</h2>
{term("RLT 토큰", "VLA(π0.5) 내부에 병목으로 끼워 넣은 <b>2048차원 벡터 1개</b>. 매 순간의 관측(카메라 3대)을 이 한 벡터로 요약해야 행동 디코더가 작동하도록 학습됐다. critic이 이미지를 다시 보지 않고 이 벡터만 보게 하는 것이 RLT의 목적.")}
{term("annotation(어노테이션)", "학습된 VLA를 데이터셋 전체(PC 기준 279,534 프레임)에 한 번 돌려 프레임마다 (RLT 토큰, 데모가 실제 실행한 행동 청크 16스텝, <b>VLA가 그 상태에서 샘플한 후보 청크 16개</b>, proprio, 남은 성공까지의 할인 보상 mc_return)을 저장한 것. 이후 모든 critic/dynamics 학습이 VLA를 다시 부르지 않고 이 캐시만 쓴다.")}
{term("HILP φ readout", "얼린 RLT 토큰 위에 2-층 MLP(2048→256→128)만 학습한 것. 학습 신호는 단 하나: −‖φ(s)−φ(g)‖가 '상태 s에서 목표 g까지 몇 스텝 남았나'의 가치함수가 되도록 하는 TD 손실(HILP 논문의 1단계). 결과적으로 φ 공간에서는 <b>거리 = 도달 가능성</b>이 된다. '어느 에피소드였나' 같은 정체성 정보는 지워진다(도달성과 무관하므로).")}
{term("mc_return", "Monte-Carlo return. 성공 프레임에서 1을 받고 γ=0.99로 할인해 거꾸로 편 값 — 프레임 t의 mc = γ^(성공까지 남은 스텝 수). '이 상태의 진짜 가치'의 관측치로, critic의 Q가 이것과 얼마나 맞는지가 기본 검증이 된다.")}

<h2>critic 학습 쪽</h2>
{term("IQL / expectile", "오프라인 RL에서 '데이터에 없는 행동을 Q에 넣어보는 것'(OOD 질의)이 과대평가의 근원이다. IQL은 V(s)를 학습할 때 <b>데이터셋이 실제 실행한 행동의 Q만</b> 보고, expectile 회귀(τ=0.7이면 위쪽 오차를 더 무겁게 벌하는 비대칭 제곱손실)로 'in-sample 행동 중 좋은 편의 가치'를 추정한다. OOD 질의가 없어 애초에 과대평가할 통로가 없다.")}
{term("Cal-QL", "CQL 계열 보수화: 데모의 Q는 올리고 <b>후보(실행 안 된) 청크들의 Q는 눌러서</b> argmax가 이상한 걸 고르지 못하게 한다. Cal-QL은 누르는 하한을 mc_return으로 잡아(칼리브레이션) 과하게 눌러 죽이는 것을 막는다. 워커B는 실패-다수 데이터에서 이 '누르기'가 과해져 유의한 해(p=.001)가 됨을 발견했다.")}
{term("swap negatives", "배치 안에서 <b>다른 상태의 데모 청크</b>를 끌어와(배열을 한 칸 돌려서) 지금 상태의 가짜 후보로 쓰는 것. '좋은 행동이긴 한데 <i>이 상태의</i> 행동은 아님'이라는, 진짜 후보보다 훨씬 어려운 음성 예제다. 이걸 Cal-QL의 누르기 대상에 추가하자 binding(아래)이 우연 수준(.53)에서 .996으로 뛰었다.")}
{term("MVE critic (in-sample MAC)", "이번에 설계한 critic. V는 위의 IQL로 in-sample 학습하고, Q는 저장된 후보 16개 각각에 대해 dynamics 모델이 예측한 '그 청크를 실행했을 때의 도착 상태 φ̂'를 만들어 y = r̂ + γ¹⁶·min(앙상블 5개의 V(φ̂))를 타겟으로 증류한다. 후보마다 <b>모델이 시뮬레이션한 근거 있는 값</b>이 생기므로, 일괄적으로 눌러버리는 CQL 없이 해상도를 얻는 것이 목표.")}

<h2>critic 진단 지표 (held-out 에피소드에서 측정)</h2>
{term("action sensitivity", "같은 상태의 후보 16개에 매긴 Q의 표준편차(상태별 계산 후 평균). <b>0이면 critic이 행동을 아예 안 본다</b>(action-blind) — best-of-N이 무의미. 성공 데모로만 학습하면 '어느 행동이든 결국 성공했으니'가 정답이 되어 이렇게 된다. 참고: plain IQL .001(blind), Cal-QL+swap .524.")}
{term("binding", "지금 상태의 진짜 데모 청크와, <b>다른 상태에서 가져온</b> 데모 청크 중 어느 쪽 Q가 높은가의 정답률. 0.5면 동전던지기 — critic이 행동을 상태에 묶지(bind) 못하고 '데모스러운 행동인가'만 본다는 뜻. 게이트 ≥0.9.")}
{term("Q−mc (과대평가 검사)", "데모 청크의 Q에서 그 상태의 실측 mc_return을 뺀 평균. 양수로 크면 critic이 실제보다 낙관하고 있고, argmax가 그 낙관을 착취해 롤아웃이 망가진다. 게이트 ≤ +0.05.")}
{term("Spearman(Q, mc)", "Q의 순위와 mc_return 순위의 상관(1이면 완벽). '어느 상태가 더 좋은 상태인지' 서열을 맞추는지 본다. 게이트 ≥ 0.75.")}

<h2>dynamics(세계 모델) 쪽</h2>
{term("DynV1 / macro-stride", "φ 공간에서 (현재 φ, 행동 청크) → 미래 φ를 예측하는 causal transformer. 제어 4스텝을 모델 1스텝으로 묶어(macro-stride 4) 16-청크가 예측 4번이 된다 — 예측 횟수가 줄면 오차 누적도 준다(MAC 논문의 action-chunk model과 같은 원리).")}
{term("앙상블 σ-disagreement", "같은 데이터로 초기화만 다르게 학습한 모델 5개의 예측이 서로 벌어진 정도(분산). <b>데이터에 없던 (상태,행동) 조합일수록 다섯이 서로 다른 소리를 한다</b> — '이렇게 이어진 적 없음' 알람. 틀린 행동을 넣으면 1.5배 커지는 것을 측정으로 확인했다.")}
{term("R² vs copy-forward", "'미래 φ = 현재 φ 그대로'라는 가장 게으른 예측(copy-forward) 대비 얼마나 나은가. 0이면 모델이 무의미, 1이면 완벽. 4스텝 앞은 로봇이 별로 안 움직여 copy-forward가 이미 세서 R²가 낮게 나온다(.29) — 모델이 나빠서가 아니라 기준선이 세서.")}
{term("AUSE", "불확실성 캘리브레이션 지표. '모델이 스스로 불확실하다고 한 순서'로 샘플을 지워가며 남은 평균 오류를 재고, '실제 오류가 큰 순서'로 지운 이상적 곡선과의 면적 차. 0이면 σ가 자기 오류를 완벽히 순위매김. 우리 값 .07은 σ를 컷/거부권 신호로 믿어도 된다는 뜻.")}
{term("OOD action-swap 비", "행동 청크를 다른 상태의 것과 바꿔치기했을 때 σ가 몇 배로 커지는가. 1.0이면 모델이 행동을 안 본다는 뜻. φ-공간 모델은 1.54.")}

<h2>롤아웃 평가 쪽</h2>
{term("arm / mode", "같은 VLA 위에서 선택·커밋 규칙만 바꾼 실험군. <b>vla</b>=첫 샘플을 16스텝 그대로 실행(기준선) · <b>bon</b>=critic이 후보 16개 중 하나 선택(best-of-N) · <b>prefix</b>=critic이 '몇 스텝 실행할지'(2/4/…/16)를 선택 · <b>전권(critic)</b>=후보×길이 128개 조합에서 argmax · <b>mbacv</b>=σ-disagreement가 커밋 길이만 결정 · <b>mbacf</b>=σ가 후보 절반을 거부(veto)한 뒤 critic이 남은 것 중 선택.")}
{term("paired / McNemar", "모든 arm이 <b>정확히 같은 장면 시드</b>에서 평가되므로 trial 단위 1:1 비교가 된다. McNemar 검정은 '한쪽만 성공한 trial'의 수(+w/−l)만 보고 차이가 우연인지 판정 — 장면 난이도 분산이 통째로 소거돼 같은 n에서 검정력이 훨씬 세다.")}
{term("seed-set 재현 / n≥8 규칙", "30-trial 하나에서의 ±.10은 잡음이었다(재현에서 전부 소거). 이후 규약: seed set 3개(0/100/200)×50 trials로 사전 등록하고, 부호가 세트를 가로질러 유지될 때만 신호로 취급. 워커B의 16-run CI 경험과 합치한다.")}
""",
    }
)

# ---------------- entry 1: rollout campaign ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 11:30",
            "title": "🎰 [워커A] PC 롤아웃 대작전 — critic 권한 7-arm + bounded-authority 4-arm, 재현까지",
            "summary": "유일한 유의 결과는 부정: 오프라인 전 조건 통과한 φ Cal-QL+swap critic이 전권 모드에서 .300 (vla .700, McNemar p=.004). 첫 30-trial의 모든 양성은 재현에서 소거(60 paired 동률). 용어는 📖 사전 엔트리 참조.",
            "tags": ["워커A", "RoboCasa", "critic", "rollout", "MB-AC"],
            "status": "finding",
        },
        "body": f"""
<h2>무엇을 돌렸나</h2>
<p>같은 VLA·같은 장면 위에서 <b>critic에게 주는 권한만 바꾼</b> arm들을 1:1 비교했다. 권한이란: 후보 16개 중
무엇을 실행할지(<b>선택</b>), 그 청크를 몇 스텝 실행하고 다시 계획할지(<b>커밋</b>), 혹은 둘 다(<b>전권</b>).
각 arm의 정확한 규칙은 📖 사전의 'arm / mode' 항목에 있다.</p>
{kv(("언제", "2026-08-08 01:00–11:30, slurm job 5개 순차 (34632→34633→34652→34656→34661)"),
    ("VLA", "<code>PrepareCoffee_rlt5_pardec_noprop/70000</code> — π0.5+RLT, PrepareCoffee 태스크 70k스텝 학습. 로컬에서 아카이브돼 있어 HF에서 복원 후 사용"),
    ("환경", "RoboCasa PrepareCoffee(머그를 커피머신에 놓고 버튼 누르기), trial당 최대 1000 제어스텝, replan마다 후보 16개 샘플"),
    ("critic A", "raw RLT 토큰(2048) + plain IQL — 오프라인에서 action-blind(sensitivity .001) 판정을 받은 <b>음성 대조군</b>. '행동을 못 보는 critic은 롤아웃에서도 무의미해야 한다'의 검증"),
    ("critic B", "HILP φ(128) + Cal-QL + swap negatives — 오프라인 전 게이트 통과(sensitivity .524, binding .996, Q−mc +.029). '오프라인 합격이 롤아웃 이득이 되는가'의 본 실험"),
    ("σ-커밋", "φ-공간 dynamics 앙상블의 disagreement가 최근 중앙값의 tau배를 넘는 지점에서 청크를 끊고 재계획. tau 2.0과 1.3 두 설정"),
    ("시드 규약", "set A: 장면 시드 0~29를 모든 arm이 공유(완전 paired). set B: 시드 30~59로 유망 arm 전부 재현(자체 vla 기준선 포함)"))}

<h2>왜</h2>
<p>오프라인 진단(binding·action sensitivity)을 전부 통과한 critic이 실제 배포에서 이득을 주는지, 그리고
권한의 폭이 결과를 바꾸는지. 워커B의 raw-토큰 23-config 피해 테이블과 독립 스택(φ 표현)에서 교차 검증하는
목적도 있었다.</p>

<h2>결과</h2>
{img(0)}
<p>그림: 각 arm 성공률에서 vla 기준선(.700)을 뺀 값. 30 paired trials, <b>채워진 점만 유의</b>(McNemar p&lt;.05).</p>
{res("set A (시드 0–29, 30 trials, 모든 arm 같은 장면)",
     ("vla (기준선)", ".700"),
     ("prefix — critic A가 커밋 길이만", ".800 · +6/−3, 유의 미달"),
     ("bon — critic B가 후보만", ".700 · +5/−5, vla와 정확히 동률"),
     ("mbacv — σ가 커밋만 (tau1.3)", ".767 · +5/−3, 유의 미달"),
     ("전권 — critic B가 후보×길이 argmax", "<b>.300 · +2/−14, p=.004 — 유일한 유의 결과</b>"))}
{res("set B 재현 (시드 30–59, 새 장면) → 60 paired 합산",
     ("vla", ".633 — 새 장면이 더 어려움"),
     ("prefix", ".500 → 합산 .650 vs vla .667 (+10/−11, p=1.0) — <b>+.10은 잡음이었다</b>"),
     ("mbacv tau1.3", ".567 → 합산 .667 vs .667 (+8/−8) — 정확히 동률"))}
{img(1)}
<p>그림: replan당 실제 실행한 스텝 수의 분포. 최고 성적 prefix는 <b>이봉형</b>(30%는 2스텝에서 끊고 60%는 거의
풀커밋 — 소박한 적응 커밋을 이미 하고 있었다). 붕괴한 전권 critic은 <b>59%를 2스텝에 몰아넣는 스래싱</b> —
계획을 2스텝마다 갈아치워 일관된 파지 동작을 한 번도 완주하지 못한다. σ-rule(tau 2.0)은 88% 풀커밋으로 사실상
무개입(오프라인 배터리가 예측한 그대로 — tau가 너무 관대했다).</p>

<h2>해석</h2>
<p>① <b>권한이 변수다.</b> 같은 critic이 선택-만에선 무해(동률), 전권에선 파국. 이유: 랜덤 선택은 후보 풀의
평균을 얻지만(편향 없음 → vla와 같음), argmax는 'Q가 가장 과대평가한 옵션'을 매 replan <b>확실하게</b> 고른다.
그 과대평가가 하필 나쁜 옵션(최단 커밋, 데이터가 이어가 본 적 없는 청크)에 몰려 있으면 오차가 복리로 쌓인다.
② <b>±.1은 잡음.</b> 30-trial의 모든 양성이 새 장면에서 부호가 뒤집혔다 → 이후 판정은 3 seed-set×50으로
사전 등록. ③ 이 태스크·체크포인트에서 선택/커밋 개입의 지렛대가 작은 구조적 이유: 실패는 청크 <b>안</b>에서
결정되고(잘못된 파지는 끊어도 소용없음), 같은 flow의 16 후보는 서로 너무 비슷하다.</p>
""",
    }
)

# ---------------- entry 2: dynamics ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 06:00",
            "title": "🌀 [워커A] φ-공간 dynamics — 같은 모델, 표현만 바꿔 전 구간 압승",
            "summary": "미래 상태 예측 모델을 DINO 공간 대신 HILP φ(128d)에서 학습: R² .66@16스텝(DINO .52), 캘리브레이션 AUSE .07, 틀린 행동에 대한 앙상블 불안 1.54배(DINO 1.09). hist=1 동급 → 배포 간단.",
            "tags": ["워커A", "RoboCasa", "dynamics", "표현"],
            "status": "done",
        },
        "body": f"""
<h2>무엇을 돌렸나</h2>
<p>(현재 상태 φ, 행동 청크) → 4/8/12/16스텝 뒤의 φ를 예측하는 <b>세계 모델</b>을 학습했다. 완전히 같은
아키텍처·스텝 수로 표현 공간만 바꿔(전날의 DINO 공간 vs 이번의 φ 공간) 어느 공간이 예측하기 쉬운지 비교.</p>
{kv(("언제/job", "08-08 05:00–07:30, job 34635(과거 3프레임 입력), 34641(현재 1프레임만 입력)"),
    ("모델", "causal transformer(d288, 4층) 5개를 독립 학습한 앙상블. 제어 4스텝=모델 1스텝(macro-stride), 40k 학습스텝"),
    ("데이터", "PC annotation 279,534 프레임의 (φ, 데모 청크, 미래 φ) 쌍. φ = HILP readout 출력 128차원. 에피소드 15%는 학습 제외(held-out), 평가는 거기서만"),
    ("명령", "<code>train_cheapz_dynamics_v1.py --z-dir .scratch/rlt_hilp_readout --stride 4 --hist 3|1 --members 5 --steps 40000</code>"))}

<h2>결과</h2>
{img(2)}
<p>왼쪽 <b>Accuracy</b>: copy-forward('미래=현재'라는 게으른 예측) 대비 개선율 R². 오른쪽 <b>Calibration</b>:
모델이 스스로 매긴 불확실성이 실제 오류를 얼마나 잘 순위매기는지(AUSE, 낮을수록 좋음). 지표 정의는 📖 사전.</p>
{res("핵심 수치 (16스텝 예측, held-out)",
     ("R² vs copy-forward", "φ .66 · DINO .52 — 전 구간(4/8/12/16)에서 φ 우위"),
     ("AUSE", "φ .07 · DINO .09~.11 — σ를 신호로 믿을 수 있는 수준"),
     ("OOD action-swap 비", "행동을 딴 상태 것으로 바꿔치면 앙상블 불안이 φ 1.54배 · DINO 1.09배 — φ 모델은 행동을 진짜로 본다"),
     ("hist=1 vs hist=3", "R² .665 vs .659 — 동급. 배포 시 과거 토큰 링버퍼가 필요 없어짐"))}

<h2>해석</h2>
<p>φ의 TD 학습이 외형 잡음(색·질감·장면 정체성)을 미리 버려줘서, 모델이 예측할 것이 정확히 '행동이 움직이는
기하'만 남았다. 단, 4스텝 앞 R²는 .29로 낮다 — 로봇이 4스텝에 별로 안 움직여 copy-forward가 이미 세기 때문
(모델 결함이 아니라 기준선 강세). 그래서 커밋 최소 단위는 4스텝이 맞고, 이 앙상블의 σ가 이후 mbacv/mbacf의
커밋·거부권 신호와 MVE critic의 상상 타겟을 공급한다.</p>
""",
    }
)

# ---------------- entry 3: battery ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 06:40",
            "title": "🧭 [워커A] MB-AC 오프라인 배터리 — σ는 binding하고, 모델-가치는 역전한다",
            "summary": "held-out 8000 anchors: 앙상블 disagreement만으로 '이 상태의 진짜 행동 vs 딴 상태 행동' 구분 정확도 .817 (보수화 손실 없이). 반면 모델이 예측한 도착지의 goal 거리로 랭킹하면 .371 — 우연(.5)보다 나쁨. 모델은 신뢰영역이지 가치평가자가 아니다.",
            "tags": ["워커A", "RoboCasa", "dynamics", "MB-AC"],
            "status": "finding",
        },
        "body": f"""
<h2>무엇을 돌렸나</h2>
<p>학습된 dynamics 앙상블을 <b>어디에 써도 되는지</b> 판정하는 3종 검사. <b>E1</b>: σ로 청크를 언제 끊을지
정하는 적응 커밋이 고정 길이보다 나은가. <b>E2</b>: '이 상태의 진짜 데모 청크'와 '<b>다른 상태에서 가져온</b>
데모 청크'(swap negative — 좋은 행동이지만 이 상태의 행동은 아닌 것)를 모델만으로 구분할 수 있는가 —
①모델이 예측한 도착 상태가 에피소드 목표에 얼마나 가까운지로, ②앙상블 disagreement(σ)로. <b>E3</b>: E2를
예측 깊이(4/8/12/16스텝)별로 나눠, 어느 깊이부터 구분이 가능한지.</p>
{kv(("언제/job", "08-08 06:10–06:40, job 34644(DINO 공간 판)·34645(φ 공간 판)"),
    ("데이터", "학습에서 제외된 held-out 에피소드의 anchor 8000개 — 모델이 처음 보는 상태들"),
    ("명령", "<code>eval_mbac_offline.py --ensemble phi_dyn_v1/ensemble_v1.pt</code>"))}

<h2>결과</h2>
{img(3)}
<p>왼쪽: 평균 커밋 길이당 '커밋 끝 시점의 예측 오류' — σ-적응 컷(보라)이 고정 길이 곡선(회색)을 근소하게 밑돈다
(같은 평균 길이에서 더 낮은 오류 = 프론티어 아래). 오른쪽: 초록(σ 기준 구분율)은 모든 깊이에서 .8 위,
빨강(예측-goal거리 기준)은 우연(점선 .5) <b>아래</b>.</p>
{res("핵심 수치 (φ-공간 모델)",
     ("binding by σ", ".817 — 틀린 행동엔 앙상블이 1.61배 더 불안해진다. CQL 같은 보수화 손실 없이 얻은 구분력"),
     ("binding by 예측-goal거리", ".371 — <b>우연 이하.</b> 틀린 행동의 예측 궤적이 progress를 흉내내며 목표 쪽으로 표류(함정)"),
     ("깊이별 σ-binding", "4스텝 .837 / 8 .815 / 12 .807 / 16 .798 — <b>첫 4스텝에서 이미 최고</b>, 거부권엔 깊은 예측이 필요 없다"),
     ("σ-적응 컷 (q0.3)", "평균 14.2스텝 커밋에서 오류 4.13 vs 고정-k 보간 4.19 — 얇지만 실재하는 우위"))}

<h2>해석</h2>
<p>모델의 <b>불확실성</b>은 진짜 신호다('이렇게 이어진 적 없다' 알람) — 그러나 모델의 <b>예측값을 가치로 읽으면</b>
우연보다 나쁘다. 이 비대칭이 이후 설계 전부를 결정했다: σ는 거부권(mbacf)과 커밋(mbacv)에 쓰고, 모델 예측은
반드시 학습된 가치함수 V에 넣어 읽으며(MVE critic), 예측-거리 휴리스틱은 금지. σ-컷이 실제로 오류가 빨리
자라는 지점(조기-컷 그룹의 오류 성장 3.01 vs 2.00)과 태스크 후반 접촉 국면에 몰리는 것도 확인했다.</p>
""",
    }
)

# ---------------- entry 4: MVE critic ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 16:00",
            "title": "⚗️ [워커A] MVE critic — policy 없이 '해상도 있고 과대평가 없는' critic, 4-변형 대기중",
            "summary": "MAC·IQL-TD-MPC 정독 후 설계. V는 in-sample IQL(OOD 질의 원천 차단), Q는 저장된 후보 16개마다 dynamics가 예측한 도착지의 V(5-앙상블 min)로 타겟 제조 — CQL·불확실성 패널티·live policy 전부 불필요. 변형 4개 GPU 대기.",
            "tags": ["워커A", "RoboCasa", "critic", "MB-AC"],
            "status": "ongoing",
        },
        "body": f"""
<h2>설계 — 왜 이렇게</h2>
<p>요구는 세 가지: <b>policy 샘플링 없이</b>(VLA는 이미지가 필요해 상상 속에서 못 돌리고, 돌리면 비싸다),
<b>해상도 있게</b>(후보 16개를 실제로 구분), <b>과대평가 없이</b>. 정독한 두 논문이 방향을 줬다 — MAC: 모델
착취 방지는 패널티가 아니라 'BC flow가 낸 샘플만 질의'로 한다. IQL-TD-MPC: 계획은 정책 샘플 위에서만(무작위
행동 시퀀스 0개). 우리에겐 live policy가 없지만 <b>annotation에 프레임마다 VLA 후보 16개가 얼려져 있다</b> —
이것이 frozen policy 아카이브 역할을 한다.</p>
{kv(("V (1단계)", "청크 단위 IQL(expectile τ=0.7). φ-단독 입력 — 다음 단계에서 <b>상상된 상태</b>에 V를 매겨야 하는데 dynamics는 proprio를 예측하지 않으므로. 데이터가 실행한 행동만 봐서 과대평가 통로가 구조적으로 없다"),
    ("타겟 제조 (2단계)", "저장된 후보 16개 전부에 대해(280k 프레임×16, VLA 호출 0회): dynamics 앙상블 5개가 '그 청크를 끝까지 실행한 도착지 φ̂'를 각자 예측 → y = r̂ + γ¹⁶ · <b>min₅</b> V(φ̂ₘ), [0,1] 클램프. min이 유일한 보수화 장치 — 다섯 예측 중 가장 비관적인 도착지만 믿는다"),
    ("Q (3단계)", "최종 critic Q(φ⊕proprio, 청크)를 배치 절반 데모(실제 타겟)+절반 후보(y)로 증류. 배포는 bon 그대로 — 기존 Cal-QL+swap과 <b>Q만 다른 깨끗한 A/B</b>"),
    ("변형 4개", "A: min집계+r̂+τ.7 (job 34691) · B: mean집계 (34694 — min의 보수성 대가 측정) · C: r̂ 제거 (34695 — 보상 헤드 기여 분리) · D: τ.9 (34696 — 날카로운 V의 과대평가 위험 측정)"),
    ("통과 게이트", "action sensitivity ≥.03 / binding ≥.9 / <b>Q−mc ≤ +.05</b> / support 초과 0 / Spearman ≥.75 — 정의는 📖 사전"),
    ("상태", "4 job 전부 L40S 큐 대기(GPU 기근). 코드 <code>scripts/train_mve_critic.py</code>"))}
<p>맥락: 두 워커가 각자 유의한 <b>부정</b> 결과를 하나씩 갖고 있다 — 전권 argmax의 파국(워커A, p=.004)과 CQL
과잉억압(워커B, p=.001). 이 설계는 그 두 실패 기전을 동시에 회피하도록 짰다.</p>
""",
    }
)

# ---------------- entry 5: GP ----------------
E.append(
    {
        "entry": {
            "date": "2026-08-08 15:00",
            "title": "🥞 [워커A] GarnishPancake 개시 — 100k 완주, 3×50 CI 프로토콜 사전등록",
            "summary": "GP mae0.5 VLA 100k 완주(체크포인트 전부 HF 아카이브). 평가는 seed set 3개×50 trials=150 paired로 사전등록(PC의 ±.1 잡음 교훈). VLA 기준선+체크포인트 스윕(50k–100k) → annotation → critic 2종 의존성 사다리 제출, GPU 대기중.",
            "tags": ["워커A", "RoboCasa", "GP", "프로토콜"],
            "status": "ongoing",
        },
        "body": f"""
<h2>무엇을</h2>
{kv(("학습 완료", "<code>GarnishPancake_rlt7_pardec_noprop_mae0.5</code> 100k 스텝(job 34197, B200 30시간). mae0.5 = 학습 중 관측 토큰의 50%를 무작위로 가리는 마스킹 목적함수 변형. 체크포인트 10k~100k 전부 HF에 검증-업로드 후 optimizer state 정리(디스크 391G→152G)"),
    ("평가 프로토콜", "arm당 seed set 3개(장면 0–49 / 100–149 / 200–249) × 50 trials = <b>150 paired</b>. PC에서 30-trial의 ±.1이 전부 잡음이었던 교훈을 프로토콜로 선결. 모든 후속 arm이 같은 150개 장면을 재사용"),
    ("VLA 기준선", "job 34674–76(100k 체크포인트, 세트별 병렬) + 34677–81(50k~90k 스윕 — 체크포인트별 학습 곡선을 성공률로)"),
    ("annotation", "job 34671: mae0.5@100k로 GP 데이터셋에 (토큰, 후보16, mc_return) 캐시 생성. 이전 mae0.5 annotation OOM은 체크포인트 복원 시점 메모리 문제로 진단 — batch 16 + MEM_FRACTION .85로 대응"),
    ("critic", "job 34672(raw 토큰+Cal-QL+swap) / 34673(φ readout → φ+Cal-QL+swap; 워커B의 NaN 수정 이식판 사용) — annotation 완료 시 자동 연쇄"),
    ("상태", "L40S 16장 타 사용자 점유로 전체 대기. 병행: YAM s200 200k 연장 ~173k, 청크길이 60 신규 학습 ~30k/100k(2초 청크 ablation, 전용 norm stats)"))}
<h2>왜</h2>
<p>PC에서 확립한 파이프라인(annotation → critic 매트릭스 → paired 롤아웃)을 두 번째 태스크에서 반복해 결론의
태스크-일반성을 확인한다. 이번에는 검정력 문제를 실험 설계 단계에서 해결한 채로 들어간다.</p>
""",
    }
)

out = [{"entry": e["entry"], "body": wrap(e["body"])} for e in E]
pathlib.Path("/tmp/wa_entries.json").write_text(json.dumps(out, ensure_ascii=False))
print("entries:", len(out), "| total body KB:", sum(len(o["body"]) for o in out) // 1024)
