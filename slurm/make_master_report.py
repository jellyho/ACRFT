"""Build master_report.html — every experiment of the RLT-critic project, one page, one tab each.

The user picks an experiment tab and gets that experiment start-to-finish: provenance spec
(checkpoints, data, scenes, pairing), the question, expected vs actual, figures, and the
interpretation in prose. Figures stay clean; all reading guidance lives in the text.

Quantitative sections recompute from raw artefacts where they exist on disk (v11/v12 run-level
stats, autopsy taxonomy, band widths); historical eras cite the numbers recorded in their
original reports.

    uv run --no-sync python slurm/make_master_report.py   # -> $CACHE_DIR/master_report.html
"""

import base64
import glob
import html as _html
import json
import os
import pathlib
from collections import Counter

import numpy as np

C = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
OUT = C / "master_report.html"
TCRIT = {8: 2.36, 10: 2.26, 15: 2.14, 16: 2.13}


def img(path, alt=""):
    p = pathlib.Path(path)
    if not p.exists():
        return f"<p class='missing'>figure not yet available: {p.name}</p>"
    b = base64.b64encode(p.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{b}' alt='{_html.escape(alt)}'/>"


def spec(rows):
    body = "".join(f"<tr><th>{_html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table class='spec'>{body}</table>"


def run_level(root, mode, prefixes=("std", "old", "nseed", "ev")):
    ds = []
    for pre in prefixes:
        for f in sorted(glob.glob(str(C / f"critic_runs/{root}/rollout/{pre}_s*.json"))):
            j = json.loads(pathlib.Path(f).read_text())
            if mode not in j:
                continue
            a = np.mean([t["success"] for t in j[mode]["trials"]])
            v = np.mean([t["success"] for t in j["vla"]["trials"]])
            ds.append(a - v)
    return np.asarray(ds)


def ci_row(name, ds):
    if len(ds) < 2:
        return f"<tr><td>{name}</td><td colspan=4>n={len(ds)} — 수집 중</td></tr>"
    n = len(ds)
    m = ds.mean()
    se = ds.std(ddof=1) / np.sqrt(n)
    t = TCRIT.get(n, 2.1)
    verdict = (
        "<b style='color:#b91c1c'>확실한 해악</b>"
        if m + t * se < 0
        else ("<b style='color:#15803d'>vla를 이김</b>" if m - t * se > 0 else "효과 없음 (CI가 0 포함)")
    )
    return (
        f"<tr><td>{name}</td><td>{n}</td><td>{m:+.3f}</td>"
        f"<td>[{m - t * se:+.3f}, {m + t * se:+.3f}]</td><td>{verdict}</td></tr>"
    )


TABS = []


def tab(tid, title, body):
    TABS.append((tid, title, body))


# ------------------------------------------------------------------ 0 overview
tab(
    "overview",
    "0 · 개요",
    f"""
<h2>프로젝트: RLT 임베딩 위의 critic으로 VLA를 이기기</h2>
<p><b>설정.</b> RoboCasa PrepareCoffee. 고정된 3B VLA(π₀-RLT, <code>pardec_noprop/70000</code>)가 매 재계획마다
후보 action chunk 16개를 샘플링하고, critic Q(z, a, h)가 (후보, 커밋 길이 h)를 골라 실행한다.
z는 VLA 백본의 RL 토큰(2048d) + proprio(16d). 보상은 sparse(성공 프레임에 1), 참값 V = γ^(성공까지 스텝).</p>
<p><b>목표.</b> critic 버전이 순정 vla(후보 0번, 풀 커밋)를 <i>성공률로</i> 이기는 방법 하나를 찾는 것.</p>
<h3>서사 요약 (탭 순서 = 시간 순서)</h3>
<ol>
<li><b>TD 시대</b>: critic이 vla보다 크게 나빴다(−0.26~−0.32). 원인 규명: 부트스트랩 max의 낙관 + h-바이어스 + winner's curse.</li>
<li><b>Fit 프로브</b>: 에피소드 전 구간 value fit 도구를 만들어 — IQL이 캘리브레이션을 완치함을 확인. 대신 액션 축이 붕괴.</li>
<li><b>동전던지기 실험</b>: critic의 선택이 무작위 선택보다 나쁨(winner's curse의 직접 증거).</li>
<li><b>부검</b>: 실패의 2/3가 엔드게임. 파지 실패 0. 버튼 상태가 관측 불가(앨리어싱)라는 태스크 결함 발견.</li>
<li><b>장면 풀 효과</b>: "critic이 나쁘다/좋다"가 주방 세트에 따라 뒤집힘 → 평가 방법론을 런-레벨 CI로 격상.</li>
<li><b>v11 공정 비교</b>: method만 다른 4 체크포인트 × 16시드 — TD만 확실히 해롭고 나머지는 무효과로 확정.</li>
<li><b>v12 혼합 데이터</b> (진행 중): 실패 롤아웃 249개를 어노테이션해 혼합 학습 → 후보 밴드가 10~30배 개방. 성공률 CI 수집 중.</li>
</ol>
<p class="meta">이 문서는 <code>slurm/make_master_report.py</code>가 생성하며, 정량 표는 디스크의 원시 JSON에서 재계산된다.
새 실험이 끝나면 재실행으로 탭이 갱신된다.</p>
""",
)

# ------------------------------------------------------------------ 1 td era
tab(
    "td-era",
    "1 · TD 시대 진단",
    f"""
<h2>TD critic은 왜 vla보다 크게 나빴나</h2>
{
        spec(
            [
                ("시기", "프로젝트 초반 (v3_fixedmask / v4_hlgfloor / fix_main 패밀리)"),
                ("학습", "TD: y_h = r + γ^h·max_(후보,h') min-앙상블 Q_target — 후보 16개에 대한 max 부트스트랩"),
                ("평가", "시드 0–300 풀, 모드별 30장면 페어드 × 다수 런"),
            ]
        )
    }
<p><b>관측.</b> critic−vla 런 평균 −0.26~−0.32, 14/14런 음수(부호검정 p≈10⁻⁴). </p>
<p><b>규명된 메커니즘 세 겹.</b>
① <b>후보 신호 부재</b>: 데모 데이터에는 "어느 후보가 더 나은가"의 정보가 없어 Q의 후보 간 차이는 고정 함수 노이즈
(분산 분해: 후보 주효과 81%가 상태와 무관, 순위 품질은 우연 수준).
② <b>winner's curse</b>: 그 노이즈에 argmax를 걸면 가장 과대평가된 후보가 체계적으로 선택됨 — best-of-N이 무작위 선택보다 나빠짐.
③ <b>h-바이어스</b>: TD 타깃이 거리 구조로 기울어(원거리 5.07× 팽창) 커밋 길이 선택이 최단으로 붕괴(61%가 h=2).</p>
<p><b>Takeaway.</b> 이 시대의 적자는 재현 가능한 진짜 효과였고, 원인은 전부 TD 부트스트랩과 무신호 argmax의 상호작용이었다.
이후 v11 공정 비교(탭 6)에서 TD의 해악은 조건을 완전히 통제해도 16/16런에서 재확정된다.</p>
""",
)

# ------------------------------------------------------------------ 2 fit probes
tab(
    "fit",
    "2 · Value-fit 프로브",
    f"""
<h2>에피소드 전 구간 value fit — 캘리브레이션의 완치와 액션 축의 붕괴</h2>
{
        spec(
            [
                (
                    "도구",
                    "critic을 기록된 에피소드의 모든 프레임에 쿼리: demo chunk Q(prefix별), 후보 16개 Q 분포 밴드(q01–q99), V(z), 참값 γ^(K−t)",
                ),
                (
                    "지표",
                    "mean|err| = 프레임 평균 |Q−참값| (0=완벽) · last20 = terminal 직전 20프레임 (terminal 처리 검증)",
                ),
                ("대상", "단일 에피소드 학습본 → 1/4/16/64 사다리 → full-data 실전 critic 5종"),
            ]
        )
    }
{img(C / "plots/12_fullrun_fit.png", "full-data critics fit")}
<p><b>그림 읽기.</b> 행=critic, 열=에피소드(둘 다 학습 데이터 안). 검정 점선이 참값, viridis가 demo chunk의 prefix별 Q,
회색 밴드가 후보 16개의 Q 분포, 빨강이 V(z).</p>
<p><b>결과와 해석.</b>
TD(1–2행)는 전체 데이터 + 200k 스텝에도 자기 학습 궤적 위에서 +0.03~0.08 낙관 오프셋이 남는다 — 부트스트랩 낙관은 데이터로 안 사라진다.
IQL 계열(3–5행)은 오차 0.002~0.031로 사실상 완벽: <b>캘리브레이션 문제는 IQL로 종결</b>.
그러나 회색 밴드가 모든 행에서 demo 곡선에 붙어 있다 — 후보 16개에 같은 값을 준다는 뜻이고,
이것이 "잘 맞추는데 고를 줄 모르는" critic의 초상이다. γ=0.999 사다리에서는 TD가 소예산에서 support 상단으로
폭주하는 것도 확인했다(수축률 0.999^h≈0.98의 자기증폭).</p>
<p><b>Takeaway.</b> 좋아 보이는 value 곡선은 필요조건일 뿐이다. 배포 성능은 곡선의 높이가 아니라
<i>후보 간 차이</i>에 있고, 그건 데이터에 실패가 있어야 생긴다(탭 7).</p>
""",
)

# ------------------------------------------------------------------ 3 collapse
band_rows = ""
bw = C / "probes/band_width.json"
if bw.exists():
    data = json.loads(bw.read_text())
    for lab in dict.fromkeys(r["label"] for r in data):
        b = np.mean([r["band"] for r in data if r["label"] == lab])
        band_rows += f"<tr><td>{lab}</td><td>{b:.4f}</td></tr>"
tab(
    "collapse",
    "3 · 액션 축 붕괴 정량",
    f"""
<h2>후보를 구분하는가 — 밴드폭(q99−q01)으로 잰 액션 민감도</h2>
{
        spec(
            [
                ("측정", "학습 에피소드 2개의 모든 프레임에서 후보 16개 Q의 q99−q01 평균 (클수록 후보를 구분)"),
                ("비교", "데모-only 시대 critic 5종 vs 혼합-데이터 v12 2종 — 체크포인트 외 동일 조건"),
            ]
        )
    }
{img(C / "plots/18_band_open.png", "band opening")}
<table class='num'><tr><th>critic</th><th>band q99−q01</th></tr>{band_rows}</table>
<p><b>해석.</b> 데모-only에서는 어떤 objective/γ를 골라도 밴드가 0.002~0.023 — 후보 간 차이가 측정 노이즈 수준이라
argmax가 읽을 정보 자체가 없었다(TD의 0.02조차 순위로는 무의미함을 별도 검증).
실패 데이터를 섞은 v12(초록)는 0.065~0.107로 <b>10~30배 개방</b> — 실패 궤적이 "이 상태로 가는 액션은 낮다"는
대조를 처음 제공한 결과다. 단, 밴드 개방은 "구분한다"이지 "옳게 구분한다"가 아니다 — 순위의 옳음은 탭 7의 성공률 CI가 심판한다.</p>
""",
)

# ------------------------------------------------------------------ 4 coinflip
tab(
    "coinflip",
    "4 · 동전던지기 실험 (randh)",
    f"""
<h2>critic의 선택은 무작위보다 나은가</h2>
{
        spec(
            [
                ("체크포인트", "v6_iql/iql_e70 (IQL e70, 200k, 데모-only)"),
                (
                    "모드",
                    "critic(joint argmax) · rand(후보만 무작위, 풀커밋) · randh(후보+h 모두 무작위) · vla — 4모드 같은 장면 페어드",
                ),
                ("장면", "시드 0–300, 120쌍"),
            ]
        )
    }
<table class='num'><tr><th>모드</th><th>성공</th></tr>
<tr><td>rand</td><td>82/120 (0.683)</td></tr><tr><td>randh</td><td>78/120 (0.650)</td></tr>
<tr><td>vla</td><td>73/120 (0.608)</td></tr><tr><td><b>critic</b></td><td><b>65/120 (0.542)</b></td></tr></table>
<p><b>해석.</b> critic vs rand 페어드 McNemar p=0.019 — <b>능동적으로 잃는다</b>는 첫 유의 증거였다.
vla와 rand의 차이는 이론상 0이어야 하고(후보는 교환가능한 iid 샘플임을 코드로 확인) 실제로 배치마다 부호가 뒤집혀 노이즈로 판정.
randh≈rand는 커밋 길이 축이 이 태스크에서 둔감함을 시사.
단, 이후 장면 풀 실험(탭 5)에서 이 능동 손실이 <i>이 특정 주방 세트</i>에 묶인 효과임이 드러난다 — 결론의 일반화가 데이터에 의해 제한된 사례.</p>
""",
)

# ------------------------------------------------------------------ 5 autopsy
tax_rows = ""
for name, path, mode in (
    ("vla", C / "critic_runs/v6_iql/iql_e70/rollout/autopsy_s*.json", "vla"),
    ("critic(iql_e70)", C / "critic_runs/v6_iql/iql_e70/rollout/cr_autopsy_s*.json", "critic"),
):
    trials = []
    for f in sorted(glob.glob(str(path))):
        trials += json.loads(pathlib.Path(f).read_text())[mode]["trials"]

    def _cls(t):
        sa = t.get("stage_at", {}) or {}
        if t["success"]:
            return "성공"
        if "grasped" not in sa:
            return "파지실패"
        if "placed" not in sa:
            return "운반/배치실패"
        if "machine_on" not in sa:
            return "버튼실패"
        return "press후실패"

    c = Counter(_cls(t) for t in trials)
    tax_rows += (
        f"<tr><td>{name}</td><td>{c['성공']}</td><td>{c['파지실패']}</td>"
        f"<td>{c['운반/배치실패']}</td><td>{c['버튼실패']}</td><td>{c['press후실패']}</td></tr>"
    )
tab(
    "autopsy",
    "5 · 실패 부검",
    f"""
<h2>어느 단계에서 죽는가 — env 술어 기반 단계 로그</h2>
{
        spec(
            [
                (
                    "단계 정의",
                    "grasped(머그 파지) → placed(디스펜서 아래 4cm) → machine_on(버튼 접촉) → 성공(+ 그리퍼 25cm/15cm 후퇴 동시조건)",
                ),
                (
                    "측정",
                    "매 스텝 env 내부 술어 판정, 트라이얼당 각 단계 최초 도달 스텝 기록 (육안 분류 금지 원칙의 출발점)",
                ),
                ("장면", "코호트별 같은 장면 페어드 (아래 그림의 행 = 코호트, 행 간 비교 금지)"),
            ]
        )
    }
{img(C / "plots/15_autopsy.png", "failure taxonomy")}
<table class='num'><tr><th>모드 (시드 2000–2300)</th><th>성공</th><th>파지</th><th>운반/배치</th><th>버튼</th><th>press후</th></tr>{
        tax_rows
    }</table>
<p><b>그림 읽기.</b> 행=코호트(주방 세트×체크포인트가 같아 페어드인 집합). 패널1 = 단계별 생존율(높을수록 멀리 감),
패널2 = 결과 구성(초록=성공), 패널3 = 단계 도달 시점 분포.</p>
<p><b>해석.</b> 파지는 전 모드에서 사실상 100% — 병목이 아니다. 실패의 약 2/3가 배치 이후(엔드게임)에 몰리며 모드 간 차이는 작다.
press후 실패의 하위 모드를 end_state로 가르면 critic 쪽은 전부 '머그 이탈'(관측 가능한 물리 실수 — 학습 가능한 범주),
vla는 이탈:후퇴 ≈ 4:3 혼합. 후퇴형은 <b>버튼을 눌러도 관측이 안 변하는 태스크 결함</b>(액체 site 알파만 변함, 버튼 자체는 무변화)과 연결 —
정책 스스로 '이미 켜졌음'을 못 보고 머신 앞을 배회한다. 앨리어싱 귀속 실패는 전체의 ~5%로 유계라서 태스크 잔류를 결정,
대신 실패 데이터로 엔드게임 반례를 주입하는 노선을 택했다(탭 7).</p>
""",
)

# ------------------------------------------------------------------ 6 pools
tab(
    "pools",
    "6 · 장면 풀 효과와 방법론 교정",
    f"""
<h2>같은 critic, 다른 주방 세트 — 결론의 부호가 뒤집히다</h2>
{
        spec(
            [
                ("장면 풀", "평가 트라이얼의 장면은 (seed+trial)로 결정 — 시드 세트가 곧 '주방들의 집합'"),
                ("사건", "iql_e70 critic: 풀 0–300에서 Δ−0.066/−0.067(재방문 재현) vs 풀 2000–2300에서 +0.017"),
                ("교정", "이후 모든 판정을 런(시드)-레벨 평균±95% t-CI, 다중 풀로 격상"),
            ]
        )
    }
<p><b>해석.</b> 30장면 페어드 비교는 모드 간 차이(±0.05)보다 큰 풀 간 변동(±0.1)에 노출된다.
재방문 실험이 구풀 적자를 소수점까지 재현(−0.067)해 "노이즈가 아니라 풀 특이적 진짜 효과"임을 보였고 —
그 적자의 대부분이 특정 시드(s100) 주방들에서 critic이 '배치 후 버튼' 단계를 부수는 것으로 특정됐다(h-붕괴로 인한 2스텝 커밋 디더링).
같은 데이터가 "옛 결과가 왜 그랬는지"와 "왜 일반화하면 안 됐는지"를 동시에 설명한다.
이 사건이 v11 공정 비교의 설계(체크포인트 동일 조건 + 16시드 CI)를 만들었다.</p>
""",
)

# ------------------------------------------------------------------ 7 v11
v11_rows = "".join(
    ci_row(n, run_level(f"v11_std/{m}", md, ("std", "old", "nseed")))
    for n, m, md in (("TD", "td", "critic"), ("IQL", "iql", "critic"), ("QC", "qc", "critic"), ("AQC", "aqc", "aqc"))
)
tab(
    "v11",
    "7 · v11 공정 비교 (16시드 CI)",
    f"""
<h2>method만 다른 4 체크포인트 — 데모-only의 완결 판정</h2>
{
        spec(
            [
                (
                    "체크포인트",
                    "v11_std/{{td,iql,qc,aqc}} — 공통: annot/noprop(데모-only), γ0.99, 100k, batch256, lr3e-4, HLG51, mc_floor, seed0, 현행 코드(액션 z-score). 차이는 method 플래그뿐",
                ),
                (
                    "method 축",
                    "td↔iql=objective 분리 · iql↔qc=chunk 구조 분리 · iql↔aqc=선택 규칙 분리 — 각 쌍이 요인 하나만 다름",
                ),
                ("배포 규칙", "td/iql/qc: 자기 Q의 joint argmax · aqc: z_ε(Q_h−b_h) argmax — 진단 모드 없음"),
                ("평가", "시드 16개(풀 4개) × 30장면, 잡내 vla 페어드. 판정 = 런-레벨 Δ̄의 95% t-CI"),
            ]
        )
    }
{img(C / "plots/16_run_level.png", "run level CI")}
<table class='num'><tr><th>방법</th><th>n런</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{v11_rows}</table>
<p><b>그림 읽기.</b> 점 = 시드런 하나의 (방법−vla) 성공률 차, 색 = 장면 풀, 검은 막대 = 평균±95% CI.
오른쪽 패널은 같은 규칙·같은 장면에서 체크포인트 세대만 바꾼 대조(구세대 −0.067 → v11 +0.100 — 풀 하나 기준).</p>
<p><b>해석.</b> TD는 16/16런 음수, CI [−0.214, −0.119] — 조건을 완전히 통제해도 <b>방법 자체가 해롭다</b>는 인과 결론.
실패 해부에서도 TD의 파손은 placed_no_press에 집중(120쌍 중 28건) — h-붕괴 디더링의 서명.
IQL/QC/AQC는 CI가 0을 포갠다: 데모-only 데이터에서는 <b>어떤 objective/구조/선택 규칙도 vla를 넘을 정보를 얻지 못한다</b>.
이로써 "남은 지렛대는 데이터뿐"이 실험적으로 완결됐다.</p>
""",
)

# ------------------------------------------------------------------ 8 v12
v12_rows = "".join(
    ci_row(n, run_level(f"v12_mixed/{m}", md, ("ev",)))
    for n, m, md in (("IQL(혼합)", "iql", "critic"), ("AQC(혼합)", "aqc", "aqc"))
)
tab(
    "v12",
    "8 · v12 혼합 데이터 (진행 중)",
    f"""
<h2>실패 롤아웃을 학습에 — 밴드는 열렸고, 성공률은 심판 중</h2>
{
        spec(
            [
                (
                    "데이터",
                    "annot/mixed = 데모 279,534 + 롤아웃 528,100 프레임(720궤적, 성공471/실패249; vla·rand 행동정책, 시드 1000–1900). 실패는 mc_return=0",
                ),
                (
                    "어노테이션",
                    "궤적 npz(3캠 jpeg+state+action) → VLA 재통과로 RL토큰+후보16 재계산, 데모와 동일 memmap 포맷",
                ),
                ("학습", "v11과 동일 레시피(100k·batch256·γ0.99·seed0), method=iql/aqc"),
                ("평가", "v11과 동일 프로토콜 — 시드 16개 × 잡내 vla 페어드 (수집 중)"),
            ]
        )
    }
<table class='num'><tr><th>방법</th><th>n런</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{v12_rows}</table>
<p><b>지금까지의 해석.</b> 기전 지표는 예측대로 움직였다: 후보 밴드 10~30배 개방(탭 3), 학습 중 q_mean이 0.5→0.14로 하강
(데이터의 35%가 실패임을 가치 지형이 반영). 남은 질문은 하나 — 열린 밴드의 <i>순위가 옳은가</i>.
16런 CI가 0 위로 뜨면 프로젝트 목표 달성, 0을 포개면 "구분하되 잘못 구분" → margin 순위 손실로 교정하는 다음 단계가 명확해진다.
실패 궤적 위 V-fit 프로브와 HUD 비디오도 생성 중이며 완료 시 이 탭에 추가된다.</p>
""",
)

# ------------------------------------------------------------------ assemble
nav = "".join(
    f"<button class='tabbtn' onclick=\"showTab('{tid}')\" id='btn-{tid}'>{title}</button>" for tid, title, _ in TABS
)
sections = "".join(f"<section class='tabpane' id='pane-{tid}'>{body}</section>" for tid, _, body in TABS)
css = """
body{font-family:system-ui,sans-serif;max-width:1150px;margin:20px auto;padding:0 16px;color:#111;line-height:1.6}
h2{margin-top:0.4em} img{max-width:100%;border:1px solid #e5e7eb;border-radius:6px;margin:8px 0}
.tabbar{display:flex;flex-wrap:wrap;gap:6px;position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:2px solid #e5e7eb;z-index:5}
.tabbtn{border:1px solid #d1d5db;background:#f9fafb;border-radius:8px;padding:7px 12px;font-size:.9em;cursor:pointer}
.tabbtn.active{background:#1d4ed8;color:#fff;border-color:#1d4ed8}
.tabpane{display:none;padding:18px 0}.tabpane.show{display:block}
table{border-collapse:collapse;margin:10px 0;font-size:.92em}td,th{border:1px solid #d1d5db;padding:5px 10px;text-align:left}
th{background:#f3f4f6}.spec th{width:130px;background:#eef2ff}.num td:nth-child(n+2){text-align:right}
.meta{color:#666;font-size:.9em}.missing{background:#fef9c3;padding:6px 10px;border-radius:6px}
code{background:#f3f4f6;padding:1px 5px;border-radius:4px}
"""
js = """
function showTab(id){
  document.querySelectorAll('.tabpane').forEach(e=>e.classList.remove('show'));
  document.querySelectorAll('.tabbtn').forEach(e=>e.classList.remove('active'));
  document.getElementById('pane-'+id).classList.add('show');
  document.getElementById('btn-'+id).classList.add('active');
  location.hash=id;
}
window.addEventListener('load',()=>showTab(location.hash?location.hash.slice(1):'overview'));
"""
OUT.write_text(
    f"<!doctype html><meta charset='utf-8'><title>RLT critic — master report</title>"
    f"<style>{css}</style><h1>RLT critic — 실험 마스터 리포트</h1>"
    f"<div class='tabbar'>{nav}</div>{sections}<script>{js}</script>"
)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(TABS)} tabs)")
