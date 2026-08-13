"""Build the project's master experiment repo
<p><b>⑨ BoN pre-registration closed (worker A r53, 08-10 23:00) — test-time BoN confirmed futile.</b> The
seed-0 positive signal (.800 vs .700) held over from last cycle was <b>noise</b>: seeds 30 and 60 tied exactly,
pooled bon .711 vs vla .678 (+11/−8, McNemar <b>p=0.65</b>) — below the pre-registered p&lt;.05, null. With the
full-authority catastrophe (.133), <b>"full authority harms, selection-only is futile" is the ceiling of a
demo-only critic</b> — meeting our FINAL 14-arm null again from an independent stack. Test-time selection (BoN)
is closed across two stacks and three datasets; the remaining doors are training-time intervention (the
vector-SF critic) and on-policy counterfactuals.</p>
<p><b>08-11 — floq (flow-matching critic) cross-review, and its contact with TD-SF-ARQ.</b> Worker A r56
reviewed floq ("critic as a velocity field, integrate K steps to read the value," OGBench hard 1.8×) and its
dissection: the gain is not distributional modeling but ① test-time recovery (integration damps initial error)
and ② <b>plasticity</b> (dense velocity supervision re-weights features under non-stationary TD targets), 2×/5×
at high UTD. Worker A's read: a capacity/optimization axis, not a coverage axis — useful online, useless for
demo-only candidate collapse (consistent with us). <b>Implications for TD-SF-ARQ:</b> ① our vector-SF target
(128 dims per transition) already <b>bakes in floq's plasticity mechanism</b> — the same cure for scalar TD's
one-dimensional starvation; ② floq's velocity-field critic is the same family as our actor-critic ladder
(∂Q/∂a flow steering), so add it as a phase-2 on-policy critic-form candidate; ③ floq still cannot manufacture
coverage — reaffirming our ordering (complementary to on-policy counterfactuals).</p>
<p><b>⑪ floq implemented & tested → split into its own report:</b> see <span class="xref" data-eid="floq-impl">floq — flow-matching critic (implementation, test, visualization)</span> (08-12).</p>rt — every experiment, grouped by DATE, then by
experiment name within the date. Two-tier navigation: pick a day, pick the experiment.

Output:
  $CACHE_DIR/master_report.html           standalone page
  $CACHE_DIR/master_report_artifact.html  same content, no doctype/head (Artifact-compatible)

Figures are embedded as downscaled JPEGs (max 1400px wide) so the page stays shippable;
originals stay in $CACHE_DIR/plots. Quantitative tables recompute from raw JSONs on rerun.

    uv run --no-sync python slurm/make_master_report.py
"""

import base64
import glob
import html as _html
import io
import json
import os
import pathlib

import numpy as np
from PIL import Image

C = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
TCRIT = {2: 12.7, 3: 4.30, 4: 3.18, 8: 2.36, 10: 2.26, 15: 2.14, 16: 2.13}

# JSON 기반 figure는 리포트 생성 때마다 원본 데이터에서 재생성 — 그림과 데이터가 어긋날 수 없다.
try:
    import make_figures

    make_figures.main()
except Exception as _e:
    print(f"[make_figures 건너뜀: {_e}]")


def img(path, alt=""):
    p = pathlib.Path(path)
    if not p.exists():
        return f"<p class='missing'>figure not yet available: {p.name}</p>"
    im = Image.open(p).convert("RGB")
    if im.width > 1400:
        im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=86)
    b = base64.b64encode(buf.getvalue()).decode()
    return f"<img src='data:image/jpeg;base64,{b}' alt='{_html.escape(alt)}'/>"


def spec(rows):
    return "<table class='spec'>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def run_level(root, mode, prefixes):
    ds = []
    for pre in prefixes:
        for f in sorted(glob.glob(str(C / f"critic_runs/{root}/rollout/{pre}_s*.json"))):
            j = json.loads(pathlib.Path(f).read_text())
            if mode in j:
                ds.append(
                    np.mean([t["success"] for t in j[mode]["trials"]])
                    - np.mean([t["success"] for t in j["vla"]["trials"]])
                )
    return np.asarray(ds)


def ci_row(name, ds):
    if len(ds) < 2:
        reason = (
            "학습 진행 중 (segfault 해결 후 재가동, td-segv 참조)" if len(ds) == 0 else f"n={len(ds)} — 시드 도착 중"
        )
        return f"<tr><td>{name}</td><td colspan=4 class='pending'>{reason}</td></tr>"
    n, m = len(ds), ds.mean()
    se = ds.std(ddof=1) / np.sqrt(n)
    t = TCRIT.get(n, 2.1)
    verdict = (
        "<b class='bad'>확실한 해악</b>"
        if m + t * se < 0
        else "<b class='good'>vla를 이김</b>"
        if m - t * se > 0
        else "효과 없음 (CI가 0 포함)"
    )
    return f"<tr><td>{name}</td><td>{n}</td><td>{m:+.3f}</td><td>[{m - t * se:+.3f}, {m + t * se:+.3f}]</td><td>{verdict}</td></tr>"


ENTRIES = []  # (date, eid, title, status, body)
EN_BODIES = {}  # eid -> English body (KO/EN toggle on the hub; translated progressively)
EN_TITLES = {}  # eid -> English title


def en(eid, title, body):
    EN_TITLES[eid] = title
    EN_BODIES[eid] = body


def entry(date, eid, title, status, body):
    ENTRIES.append((date, eid, title, status, body))


P = C / "plots"

FLOW = [
    (
        "~07-28",
        "파이프라인 구축",
        "어노테이션(RL토큰+후보16) → critic 학습 → 페어드 롤아웃 평가의 3단 파이프라인 완성.",
        "",
    ),
    (
        "~08-01",
        "첫 판정: TD critic의 일관된 적자",
        "critic−vla 14/14런 음수(−0.2~−0.3, p≈10⁻⁴). '왜 지는가'가 프로젝트의 질문이 됨.",
        "→ 원인 추적 시작",
    ),
    (
        "~08-01",
        "h축 오염 발견: b(d) 거리-바이어스",
        "TD 타깃이 원거리에서 5.07× 팽창 — 커밋 길이 선택이 최단(h=2, 61%)으로 붕괴하는 원인.",
        "→ 타깃을 고치자: IQL",
    ),
    (
        "08-02",
        "후보축 사망 선고: winner's curse",
        "후보 간 Q차이의 81%가 상태 무관 고정 노이즈(ICC 0.878), 순위는 우연 수준. argmax = 가장 과대평가된 후보 집기.",
        "→ 추론 기교로 구제 시도",
    ),
    (
        "08-03",
        "IQL 전환의 성과와 한계",
        "가치 캘리브레이션 완치(오차 0.01 수준), prefix 모드 vla 회복. 그러나 후보축은 여전히 무신호 — 액션축이 아예 붕괴(밴드 0.002~0.007).",
        "",
    ),
    (
        "08-05",
        "사다리 실험: 귀속 종결",
        "같은 데이터·예산에서 IQL은 정확, TD γ0.999는 학습궤적조차 실패(자기증폭 인플레) — 실패의 범인은 데이터가 아니라 부트스트랩.",
        "→ '잘 맞추는데 못 고르는' critic의 초상 완성",
    ),
    (
        "08-05",
        "추론 기교 전멸 (고검정력)",
        "softcand·softmax·제로크로싱 재현 모두 무효과 — 무신호 순위는 샘플링으로도 못 살린다.",
        "→ 학습 신호 주입만 남음",
    ),
    (
        "08-06",
        "능동 손실의 정량: 동전던지기 실험",
        "critic(0.542) < rand(0.683), p=0.019 — critic의 선택이 무작위보다 나쁨을 유의하게 확인.",
        "",
    ),
    (
        "08-06",
        "AQC: 손실의 무해화",
        "per-h 베이스라인 + z-score로 critic을 동전던지기 밴드(0.658)로 복원, h 분포 정상화(평균 h=11). 이기지는 못함 — 분자에 내용이 없어서.",
        "",
    ),
    (
        "08-06",
        "부검과 태스크 결함 발견",
        "실패의 2/3가 엔드게임, 파지 실패 0. 버튼이 눌러져도 관측이 거의 안 변하는 앨리어싱 발견(귀속 ~5%로 유계 → 태스크 잔류).",
        "",
    ),
    (
        "08-06",
        "방법론 위기와 격상: 장면 풀 효과",
        "같은 critic이 주방 세트에 따라 −0.067 ↔ +0.017 — 30장면 비교는 풀 노이즈에 지배됨. 이후 모든 판정을 시드-레벨 95% CI × 다중 풀로.",
        "→ v11 공정 비교 설계",
    ),
    (
        "08-07",
        "v11 완결: 데모-only의 최종 판정",
        "method만 다른 4 체크포인트 × 16시드: TD만 확실한 해악(16/16 음수, CI [−0.21,−0.12]), IQL/QC/AQC는 무효과. 데모 데이터에서는 이길 정보가 없다.",
        "→ 남은 지렛대 = 데이터",
    ),
    (
        "08-07",
        "v12 혼합 데이터: 밴드가 열리다",
        "실패 롤아웃 249궤적 혼합 학습 → 후보 밴드 10~30배 개방, 실패 궤적 가치 인식. 성공률 16시드 CI 심판 진행 중.",
        "→ CI>0이면 목표 달성 / CI∋0이면 margin 순위 손실",
    ),
]
_tl_items = "".join(
    f"<div class='node'><div class='when'>{d}</div><div class='card'><h3>{t}</h3><p>{b}</p>"
    + (f"<div class='next'>{n}</div>" if n else "")
    + "</div></div>"
    for d, t, b, n in FLOW
)
entry(
    "타임라인",
    "flow",
    "전체 흐름 · Takeaways",
    "살아있음",
    f"""
<div class='sub'>처음부터 지금까지의 발견과 흐름만. 카드 = 국면의 takeaway, 파란 줄 = 다음 국면으로 이어진 논리. 상세는 날짜 탭에.</div>
<div class='tl'>{_tl_items}</div>
<div class='now'><b>현재 위치 (2026-08-08).</b> 데모-only(v11)와 mixed(v12) 모두 성공률 판정 null로 완결 —
실패 데이터가 후보 구분(밴드 개방)은 만들었으나 성공률 전환에는 실패. 현재 세 갈래가 병행 중:
① FINAL 전 요인 스윕(IQL 계열 null 확정, TD 계열은 XLA segfault 근본 해결 후 학습 재가동),
② 장면 정체성 지름길을 차단한 K-per-scene 데이터(v14, 60.5만 프레임)로 재학습,
③ CalQL(후보축 학습 신호) 검증. 다음 벤치마크로 GR1 tabletop 시뮬레이션 스택 가동 완료.</div>
""",
)

# =========================================================== 07-28 ~ 08-01
entry(
    "~08-01",
    "genesis",
    "TD critic 초기 세대와 첫 적자",
    "완결",
    f"""
{
        spec(
            [
                (
                    "체크포인트",
                    "v3_fixedmask / v4_hlgfloor / fix_main 패밀리 — TD 부트스트랩(후보 16 max), ARQ, γ0.99, HLG51",
                ),
                ("평가", "시드 0–300 풀, 모드별 30장면 페어드, 다수 런"),
                ("상태", "완결 — 결론은 이후 v11에서 인과로 재확정"),
            ]
        )
    }
<p>파이프라인(어노테이션→critic 학습→롤아웃)을 세운 시기. 첫 대규모 결과는 명확한 적자였다:
critic−vla 런 평균 −0.20~−0.32, <b>14/14런 음수(부호검정 p≈10⁻⁴)</b>.</p>
{img(P / "5_per_run_harm.png", "per-run harm")}
<p><b>해석.</b> 위 그림의 각 점이 한 평가 런의 critic−vla 차이다 — 전부 0 아래로 몰려 있어 "설정 노이즈"로는 설명이 안 됐다.
이 적자의 원인 추적이 이후 2주의 서사를 만든다: 타깃 바이어스(h축) + 무신호 argmax(후보축)의 합작이라는 것이
단계적으로 규명된다(다음 날짜들의 탭).</p>
""",
)

entry(
    "~08-01",
    "vbias",
    "TD 타깃의 거리-구조 바이어스 (b(d) 프로브)",
    "완결",
    f"""
{
        spec(
            [
                ("도구", "vbias 프로브: 상태의 '성공까지 거리 d'별로 Q−γ^d 오프셋 b(d)를 측정 — 정확하면 b(d)=0"),
                ("대상", "TD 패밀리 critic들, 데모 데이터"),
            ]
        )
    }
<p class="sub">아래 그림은 08-01 원본(v3 스윕)이며 원시 데이터가 남아있지 않아 스타일 재생성 대상에서 제외 — 아카이브 원본 유지.</p>{
        img(P / "2_value_bias.png", "value bias by distance")
    }
<p><b>해석.</b> TD critic의 b(d)는 원거리에서 크게 양수(최대 5.07× 팽창) — 타깃이 거리 구조로 기울어 있었다.
이 기울기는 커밋 길이 선택(h축)을 오염시켜, 이후 측정에서 joint argmax의 61%가 최단 h=2로 붕괴하는 원인이 된다.
γ를 0.999로 올리면 캘리브레이션은 좋아지지만(±0.01) 롤아웃은 나아지지 않는 것도 이때 확인 —
"캘리브레이션 ≠ 선택 능력"이라는 프로젝트의 반복 주제가 처음 등장했다.</p>
""",
)

# =========================================================== 08-02 ~ 08-03
entry(
    "08-02~03",
    "families",
    "방법 패밀리별 롤아웃 총결산 (TD→IQL 전환 근거)",
    "완결",
    f"""
{
        spec(
            [
                ("비교", "패밀리(v3/v4/fix_main=TD, v6=IQL, v8=IQL+γ↑/dueling)별 critic−vla, prefix−vla 런 델타"),
                ("주의", "세대 간 체크포인트 조건이 달라 방향성 참고용 — 엄밀 비교는 08-07의 v11"),
            ]
        )
    }
{img(P / "1_success_by_mode.png", "success by mode/family")}
<p><b>해석.</b> TD 패밀리는 critic−vla −0.26~−0.32로 일관 열세, IQL 도입(v6)으로 −0.06까지 회복,
prefix-only 모드는 유일하게 0 위(+0.02). 이 그림이 "IQL로 가치 축을 고치고, 후보 축은 별도 신호가 필요하다"는
프로젝트 노선을 결정했다. v8(γ0.999·dueling)이 오히려 −0.16으로 후퇴한 것은 나중에
"γ↑가 h-대비를 죽이고 dueling의 zero-mean A가 prefix 신호를 지운다"로 설명된다.</p>
""",
)

entry(
    "08-02~03",
    "wcurse",
    "Winner's curse 해부 — 분산 분해와 두 개의 argmax",
    "완결",
    f"""
<p class='sub'>용어: <b>winner's curse</b>=잡음 낀 추정치들의 max를 고르면 체계적으로 과대추정된 것을 집는 현상, <b>분산 분해</b>=Q 오차를 상태축(모든 후보 공유)과 후보축(후보별 고유)으로 가르는 것. 상세는 conservatism 탭.</p>
{
        spec(
            [
                ("도구", "상태 내 Q[후보,prefix] 행렬의 분산 분해 + 결합/행 argmax 일치율"),
                ("핵심 수치", "후보 주효과 81.1% · 후보 내 오차 상관 ICC 0.878 · 순위 품질 ≈ 우연"),
            ]
        )
    }
<p><b>해석.</b> 후보 간 Q 차이의 대부분이 상태와 무관한 <i>고정 함수 노이즈</i>였다 — critic은 후보의 정체성에
점수를 매길 뿐 결과를 예측하지 않는다. 그 위의 argmax는 가장 과대평가된 후보를 체계적으로 집는 winner's curse가 되고,
실제로 best-of-N이 무작위 선택보다 나빠진다(08-06 randh에서 p=0.019로 유의 확인).
또한 "학습 시 argmax(TD 타깃)"와 "배포 시 argmax(선택)"가 별개 문제임을 분리 — IQL은 전자만 제거한다는 점이
이후 실험 설계의 기준이 됐다.</p>
""",
)

entry(
    "08-02~03",
    "duel",
    "Dueling 게이지 실패 두 번과 zero-mean 해법",
    "완결",
    """
<p>Q = V + A 분해는 게이지 자유도((V+c, A−c))가 있어 두 번 실패했다: ① detach-V 판은 |V|~200으로 표류,
② sigmoid 바운드 판은 V가 0에서 죽음. 해법은 <b>prefix 축 zero-mean advantage</b>(A에서 mean_h A를 빼 게이지를 V에 고정).
이 기계는 이후 AQC의 per-h 베이스라인과 개념적으로 연결된다(Q_h − b_h가 곧 advantage).
부산물 교훈: dueling의 A가 액션을 완전히 무시하도록 붕괴하면(측정: 밴드 0.0002) prefix 선택이 동전던지기가 된다 —
v8 패밀리 열세의 한 원인.</p>
""",
)

# =========================================================== 08-05
entry(
    "08-05",
    "singlefit",
    "단일 궤적 fit — terminal 처리 검증",
    "완결",
    f"""
{
        spec(
            [
                ("체크포인트", "v7_single/ep1 (에피소드 1개만 학습, TD γ0.99, 20k)"),
                ("검증", "자기 에피소드 전 프레임에서 Q vs 참값 γ^(K−t) — terminal 경계 누수 여부"),
            ]
        )
    }
{img(P / "7_single_traj_fit.png", "single trajectory fit")}
<p><b>해석.</b> 전 prefix 헤드가 참값 곡선을 따라가고 terminal 직전 20프레임 오차 0.002 — 타깃/terminal 계산이
정확함을 사용자 검수용으로 확정한 실험. 원거리의 ~0.06 바닥은 HL-Gauss 하한 아티팩트로, 이후 반복 확인된다.</p>
""",
)

entry(
    "08-05",
    "ladders",
    "데이터 사다리 (1→64 에피소드) × objective × γ",
    "완결",
    f"""
{
        spec(
            [
                (
                    "설계",
                    "1/4/16/64 에피소드로 학습한 critic을 학습궤적(ep0)과 held-out(ep65)에 fit — TD γ0.99 / TD γ0.999 / IQL γ0.999 3세트, 동일 20k 예산",
                ),
                (
                    "γ=0.999의 이유",
                    "0.99는 원거리가 평지(0.99^300≈0.05)라 '못 배움'과 '평지'가 구분 불가 — 0.999는 전 구간 기울기(시작값 0.48)",
                ),
            ]
        )
    }
{img(P / "11_ladder_traj_fit.png", "IQL ladder")}
<p><b>IQL γ0.999 (위).</b> held-out 오차 0.157→0.056 단조 개선. 액션을 안 보는 V(빨강)가 Q보다 모든 rung에서
잘 일반화(0.036 vs 0.056) — <b>관측 표현은 전이되는데 액션 평가가 병목</b>이라는 직접 증거.</p>
{img(P / "10_ladder_traj_fit.png", "TD gamma 0.999 ladder")}
<p><b>TD γ0.999 (위).</b> 파국: 같은 예산에서 학습 궤적조차 0.18~0.20 오차로 support 상단(~0.86)에 들러붙는다.
γ=0.999의 백업 수축률(0.999^h≈0.98)이 max-부트스트랩의 과대평가를 거의 감쇠 없이 자기증폭시키는 구조.
같은 데이터·예산에서 IQL은 0.013~0.027로 정확 — <b>사다리 실패의 귀속이 '데이터 부족'이 아니라 '부트스트랩'으로 종결</b>.</p>
""",
)

entry(
    "08-05",
    "fullfit",
    "Full-data 실전 critic 검수",
    "완결",
    f"""
{
        spec(
            [
                (
                    "대상",
                    "실제 롤아웃에 썼던 5종: TD γ0.99 / TD γ0.999 / IQL e70 γ0.99 / IQL e70 γ0.999 / IQL duel — 각 200k, 데모 전체",
                ),
                ("로딩", "배포 경로(load_trained) 그대로 — dueling 재합성·정규화 포함"),
            ]
        )
    }
{img(P / "12_fullrun_fit.png", "full-data fit")}
<p><b>해석.</b> IQL 계열은 자기 학습 궤적을 0.002~0.031로 맞춘다(캘리브레이션 완치).
TD는 full-data 200k에도 +0.03~0.08 낙관이 남는다. 그러나 다섯 모두 후보 밴드(회색)가 demo 곡선에 붙어 있다 —
후보 구분 없음. 부수 발견: 에피소드 시작 ~10프레임의 가치 스파이크(전 critic 공통, 리셋 프레임 토큰 오염 의심).</p>
""",
)

entry(
    "08-05",
    "highpower",
    "고검정력 롤아웃 판정 (softcand / e70 재현 / softmax)",
    "완결",
    f"""
{spec([("체크포인트", "v6_iql/iql_e70 (200k, 데모-only)"), ("장면", "시드 0–300, 페어드 McNemar")])}
<table class='num'><tr><th>실험</th><th>결과</th><th>판정</th></tr>
<tr><td>softcand 4시드 (후보 샘플링+행내 argmax)</td><td>74/120 vs vla 81/120, p=0.37</td><td>무효과</td></tr>
<tr><td>iql_e70 제로크로싱(+0.033) 재현 3시드</td><td>critic 48/90 vs vla 53/90</td><td><b>재현 실패</b></td></tr>
<tr><td>softmax T=0.02</td><td>21/30 vs 20/30</td><td>무효과</td></tr></table>
<p><b>해석.</b> "argmax 대신 샘플링"류의 추론 기교로는 무신호 순위를 구제할 수 없음을 검정력 있게 확정.
한 시드의 +0.033은 노이즈였다. 이 판정이 "학습 신호 주입(실패 데이터)"으로의 전환점.</p>
""",
)

# =========================================================== 08-06
entry(
    "08-06",
    "randh",
    "동전던지기 실험 — critic의 능동 손실",
    "완결",
    f"""
<p class='sub'>용어: <b>rand</b>=16후보 중 무작위 실행(구조상 VLA와 동일 분포 — 기준선), <b>randh</b>=커밋 길이까지 무작위 (critic의 전체 결정공간에 대한 정직한 무작위 대조). critic이 rand보다 나빠지면 '능동적 해악'.</p>
{
        spec(
            [
                (
                    "모드",
                    "critic(joint argmax) / rand(후보 랜덤·풀커밋) / randh(후보+h 랜덤) / vla — 같은 장면 4모드 페어드",
                ),
                ("체크포인트", "v6_iql/iql_e70"),
                ("장면", "시드 0–300, 120쌍"),
            ]
        )
    }
<table class='num'><tr><th>모드</th><th>성공</th></tr><tr><td>rand</td><td>0.683</td></tr><tr><td>randh</td><td>0.650</td></tr>
<tr><td>vla</td><td>0.608</td></tr><tr><td><b>critic</b></td><td><b>0.542</b></td></tr></table>
<p><b>해석.</b> critic이 <b>무작위 후보 선택보다 유의하게 나쁨</b>(vs rand p=0.019) — winner's curse의 행동적 확정.
vla↔rand는 이론상 동일(후보는 iid 교환가능 — 코드 검증)하고 실제로 배치마다 부호가 뒤집혀 노이즈 판정.
randh≈rand로 커밋 길이 축의 둔감함도 확인. 단, 이 능동 손실 자체가 장면 풀 특이적임이 같은 날 뒤에 드러난다(pools 탭).</p>
""",
)

entry(
    "08-06",
    "aqc",
    "AQC 구현과 데모-only 판정",
    "완결",
    f"""
{
        spec(
            [
                (
                    "구현",
                    "ValueNet에 per-prefix 베이스라인 헤드 b_h 추가(κ_b=0.9, 자기 짝 Q_h에 expectile) — 부트스트랩 스칼라 V는 불변",
                ),
                ("배포 규칙", "score = z_ε(Q_h − b_h), ε=10⁻³ (γ^h 나눗셈은 z-score와 상쇄됨을 확인 후 제거)"),
                ("판정", "aqc vs vla 4시드 120쌍 페어드"),
            ]
        )
    }
<p><b>결과.</b> aqc 79/120 vs vla 83/120 (p=0.64) — 못 이겼지만, <b>critic의 능동 손실(0.542)을 동전던지기 밴드(0.658)로
복원</b>했고, h 선택 분포가 61% 최단-붕괴에서 평균 h=11의 전 구간 스펙트럼으로 정상화됐다.
AQC 논문 Prop 5.1("신호가 없으면 편향된 오답을 무편향 랜덤으로")의 재현. 남은 갭의 원인은 advantage 분자에
내용이 없는 것 — 실패 데이터로 이어진다.</p>
""",
)

entry(
    "08-06",
    "autopsy",
    "실패 부검 — env 술어 단계 로그",
    "완결",
    f"""
<p class='sub'>용어: <b>stage_flags</b>=환경 술어(파지·거치·머신on)를 매 스텝 프로그램적으로 기록한 것 — 실패 유형 분류는 전부 이 술어 기반(육안 분류 없음).</p>
{
        spec(
            [
                (
                    "단계",
                    "grasped → placed(4cm) → machine_on → 성공(+그리퍼 25cm/15cm 후퇴 동시조건) — 매 스텝 env 내부 술어로 판정",
                ),
                ("원칙", "육안 분류 금지(육안 예비판독이 프로그램 집계와 정반대였던 사고 이후 확립)"),
            ]
        )
    }
{img(P / "15_autopsy.png", "autopsy")}
<p><b>읽는 법.</b> 행=코호트(같은 주방 세트×같은 체크포인트 — 행 안에서만 비교). 패널1=단계 생존율, 패널2=결과 구성(초록=성공),
패널3=단계 도달 시점.</p>
<p><b>해석.</b> 파지 실패 0. 실패의 ~2/3가 배치 이후 엔드게임. 모드 간 차이보다 태스크 병목이 지배적.
press후 실패의 하위모드: critic은 전부 '머그 이탈'(관측 가능→학습 가능), vla는 이탈:후퇴≈4:3.
후퇴형은 같은 날 발견된 <b>버튼 앨리어싱</b>(누름의 유일한 시각 변화가 가는 액체 줄기의 알파뿐, 버튼 자체 무변화)과 연결 —
정책이 '켜졌음'을 못 봐 머신 앞을 배회한다. 앨리어싱 귀속은 전체의 ~5%로 유계 → 태스크 잔류 결정.</p>
""",
)

entry(
    "08-06",
    "pools",
    "장면 풀 효과 — 평가 방법론의 교정",
    "완결",
    f"""
{
        spec(
            [
                ("사건", "같은 critic(iql_e70)·같은 규칙이 풀 0–300에선 Δ−0.066, 풀 2000–2300에선 +0.017"),
                ("재방문", "구풀을 오늘 다시 돌리면 −0.067 — 소수점까지 재현되는 풀 특이적 진짜 효과"),
            ]
        )
    }
<p><b>해석.</b> 30장면 비교는 풀 간 변동(±0.1)이 모드 간 차이(±0.05)를 압도한다.
구풀 적자의 해부: 특정 시드(s100) 주방들에서 critic이 '배치 후 버튼' 단계를 부수는 것(placed_no_press 10 vs vla 4) —
h-붕괴 디더링이 그 주방 구성에서 특히 치명적이었던 것. 교훈이 방법론이 됐다:
이후 모든 판정은 <b>런(시드)-레벨 평균 ± 95% t-CI, 다중 풀, method-only-diff 체크포인트</b>로 격상(v11).</p>
""",
)

entry(
    "08-06",
    "failpipe",
    "실패 데이터 파이프라인 + in-distribution 장면 재현",
    "완결",
    f"""
{
        spec(
            [
                (
                    "수집",
                    "--dump-traj: 롤아웃 매 스텝 (3캠 jpeg, state, action, prompt, 성공) npz — 시드 1000–1900, 720궤적(성공471/실패249)",
                ),
                ("어노테이션", "annotate_rollouts.py: VLA 재통과로 RL토큰+후보16, 데모와 동일 memmap — 8샤드 병렬"),
                (
                    "장면 재현",
                    "원본 HF repo의 extras(ep_meta/model.xml/states)로 학습 데모의 주방을 정확 재구성하는 --scenes-from-extras 구현",
                ),
                ("사고", "1차 수집분 240궤적은 image_right 누락으로 폐기 후 재수집 — 3캠 필수"),
            ]
        )
    }
<p><b>해석.</b> AQC 논문이 후보 선택에 성공한 조건(혼합 롤아웃 학습, per-step −1 + 성공창 보상)을 우리 스택에
이식하기 위한 기반 공사. 보상 규약은 우리 관례(성공 프레임 1, 실패 전부 0 — 실패 궤적의 mc_return=0이
V를 성공확률-가중으로 끌어내림)를 유지했다.</p>
""",
)

# =========================================================== 08-07
v11_rows = "".join(
    ci_row(n, run_level(f"v11_std/{m}", md, ("std", "old", "nseed")))
    for n, m, md in (("TD", "td", "critic"), ("IQL", "iql", "critic"), ("QC", "qc", "critic"), ("AQC", "aqc", "aqc"))
)
entry(
    "08-07",
    "v11",
    "v11 공정 비교 — 16시드 CI 완결",
    "완결",
    f"""
<p class='sub'>용어: <b>BoN</b>=후보 N개 중 critic 최고점 실행, <b>expectile τ</b>=상위쪽 치우친 회귀(τ0.9≈상위 10% 지향), <b>run-level CI</b>=시드(잡) 단위 평균차의 신뢰구간. 상세 해설은 FINAL 탭의 용어 박스.</p>
{
        spec(
            [
                (
                    "체크포인트",
                    "v11_std/{{td,iql,qc,aqc}} — method 플래그 외 전 조건 동일(데모-only·γ0.99·100k·batch256·seed0·현행 코드)",
                ),
                ("요인 분리", "td↔iql=objective · iql↔qc=chunk 구조 · iql↔aqc=선택 규칙 — 쌍마다 한 요인"),
                ("평가", "시드 16개(풀 4개)×30장면, 잡내 vla 페어드, 판정=런-레벨 95% t-CI"),
            ]
        )
    }
{img(P / "16_run_level.png", "run-level CI")}
<table class='num'><tr><th>방법</th><th>n</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{v11_rows}</table>
<p><b>해석.</b> TD는 16/16런 음수, CI 전체가 0 아래 — 조건을 완전히 통제한 <b>인과 수준의 해악 판정</b>
(파손은 placed_no_press에 집중: h-붕괴 디더링의 서명). IQL/QC/AQC는 CI가 0을 포갬 —
<b>데모-only 데이터에서는 어떤 objective/구조/선택 규칙도 vla를 넘지 못한다</b>가 이 프로젝트의 확정 결론.
오른쪽 패널: 같은 체크포인트(v11 iql)를 풀별로 분해한 것 — 한 풀은 +0.10으로 읽히지만 16런 종합은 0(효과 없음). 풀 하나에서의 숫자는 성능 주장이 될 수 없다는 것이 이 패널의 유일한 메시지다. (이전 판의 v6↔v11 세대 비교 패널은 정규화·스텝/배치·코드가 동시에 달라 confound였고 "+10%"로 오독되어 제거함.)</p>
""",
)

band_rows = ""
bw = C / "probes/band_width.json"
if bw.exists():
    data = json.loads(bw.read_text())
    for lab in dict.fromkeys(r["label"] for r in data):
        band_rows += f"<tr><td>{lab}</td><td>{np.mean([r['band'] for r in data if r['label'] == lab]):.4f}</td></tr>"
v12_rows = "".join(
    ci_row(n, run_level(f"v12_mixed/{m}", md, ("ev",)))
    for n, m, md in (("IQL(혼합)", "iql", "critic"), ("AQC(혼합)", "aqc", "aqc"))
)
entry(
    "08-07",
    "v12",
    "v12 혼합 데이터 — 밴드 개방과 성공률 심판",
    "진행 중",
    f"""
<p class='sub'>용어: <b>밴드(q99−q01)</b>=한 상태에서 후보 16개 Q값의 폭 — critic이 후보를 구분하는 정도의 정적 측정, <b>held-out</b>=학습에 안 쓴 데이터. 상세 해설은 FINAL 탭의 용어 박스.</p>
{
        spec(
            [
                ("데이터", "annot/mixed = 데모 279,534 + 롤아웃 528,100 프레임(실패 249궤적 포함)"),
                (
                    "VLA",
                    "<b>동결</b> — 실패 데이터는 critic(~10M)만 학습. 후보 분포 불변, 이득 귀속은 100% 가치-기반 선택",
                ),
                ("학습", "v11과 동일 레시피, method=iql/aqc — a6000(데이터 17GB 상주)"),
                ("평가", "v11 동일 프로토콜, 시드 16개 완결 (iql/aqc 각 n=16)"),
                (
                    "사고록",
                    "3090 메모리 초과 → a6000 이전 · make_diag 통짜 jit가 스텝5000에서 무로그 사망 → 256슬라이스 샤딩 픽스 커밋",
                ),
            ]
        )
    }
{img(P / "18_band_open.png", "band opening")}
<table class='num'><tr><th>critic</th><th>band q99−q01</th></tr>{band_rows}</table>
{img(P / "17_rollout_traj_fit.png", "mixed critic on rollout trajectories")}
<table class='num'><tr><th>방법</th><th>n</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{v12_rows}</table>
<p><b>해석.</b> 기전 지표 두 개가 예측 방향으로 움직였다. ① 후보 밴드 10~30배 개방(초록) — 후보를 처음으로 구분한다.
② 실패 롤아웃 궤적 fit(위 그림): 데모-only critic이 실패 궤적에서 보이던 '상승 망상'이 혼합 critic에서 완화되는지의 검증 —
성공 궤적 추적은 유지하면서 실패 구간 가치가 낮아지는 정도를 본다.
성공률 판정(16런): <b>효과 없음</b> — iql Δ̄=−0.017 CI[−0.068,+0.035], aqc Δ̄=−0.029(14런). 밴드는 열렸지만 성공률로 전환되지 않았다. 해석: on-policy Q^π에서 같은 상태의 iid 후보 간 참 가치 차이는 "한 chunk 뒤 같은 정책"이라 원래 작고, 열린 밴드의 상당 부분은 상태 진행도 차이를 반영했을 가능성. 다음 갈래(사전 등록): ① in-distribution 평가(학습 주방 정확 재현)로 일반화 요인 분리 — 제출됨, ② AQC 논문식 보상 성형(−1 스텝 벌점 + 성공 창 확장)으로 후보 간 가치 격차를 증폭하는 v13.</p>
{img(P / "19_rollout_traj_fit.png", "held-out trajectories")}
<p><b>held-out memorization 판정 (시드 9100 — 학습 외 주방, aqc n=16 갱신: Δ̄=−0.019 CI[−0.068,+0.030]).</b>
학습 실패 궤적의 평평한 V=0은 자기-학습-데이터 평가의 아티팩트였다. held-out 실패 t05/t07/t09에서는
<b>진행에 따라 V가 0.6~1.0까지 상승했다가 실수 시점에 붕괴 후 0 유지</b> — 기대했던 V^π 형태가 일반화 데이터에서 처음 관측됐다.
즉 혼합 critic은 진행·실패-시점 감지를 실제로 배웠다. 남은 한계: held-out 성공 t01을 거의 끝까지 저평가(err 0.147) —
낯선 주방의 성공을 못 믿는 보수 편향. K-per-scene 데이터(수집 완료 → v14, 'K-per-scene' 리포트 참조)가 정확히 이 지점을 겨냥한다.</p>
""",
)


# =========================================================== 08-07 FINAL
FINAL_ARMS = [
    ("td_max", "critic"),
    ("td_soft", "critic"),
    ("td_aqcmax", "critic"),
    ("iql", "critic"),
    ("qc", "critic"),
    ("td_max_a101", "critic"),
    ("td_max_a201", "critic"),
    ("iql_a101", "critic"),
    ("iql_a201", "critic"),
    ("td_max_online", "critic"),
    ("iql_online", "critic"),
    ("td_max_demo", "critic"),
    ("iql_demo", "critic"),
    ("qc_demo", "critic"),
]
_frows = "".join(ci_row(a, run_level(f"final/{a}", m, ("f",))) for a, m in FINAL_ARMS)
_abs_rows = ""
for a, m in FINAL_ARMS:
    import glob as _g

    S = V = N = 0
    for f in sorted(_g.glob(str(C / f"critic_runs/final/{a}/rollout/f_s*.json"))):
        j = json.loads(pathlib.Path(f).read_text())
        if m in j:
            S += sum(t["success"] for t in j[m]["trials"])
            V += sum(t["success"] for t in j["vla"]["trials"])
            N += len(j[m]["trials"])
    if N:
        _abs_rows += f"<tr><td>{a}</td><td>{S}/{N} ({S / N:.3f})</td><td>{V}/{N} ({V / N:.3f})</td></tr>"
    else:
        _abs_rows += (
            f"<tr><td>{a}</td><td colspan=2 class='pending'>평가 대기 (학습 재가동 — td-segv 리포트 참조)</td></tr>"
        )
_mc_rows = ""
from math import comb as _comb  # noqa: E402 - local helper for the block below

for a, m in FINAL_ARMS:
    b = c = n = 0
    for f in sorted(glob.glob(str(C / f"critic_runs/final/{a}/rollout/f_s*.json"))):
        j = json.loads(pathlib.Path(f).read_text())
        if m in j and "vla" in j:
            for tc, tv in zip(j[m]["trials"], j["vla"]["trials"], strict=True):
                n += 1
                b += int(tc["success"] and not tv["success"])
                c += int(tv["success"] and not tc["success"])
    if n == 0:
        continue
    mtot = b + c
    pv = min(1.0, sum(_comb(mtot, k) for k in range(min(b, c) + 1)) * 2 / 2**mtot) if mtot else 1.0
    mark = (
        "<b class='bad'>유의한 해악</b>"
        if (pv < 0.05 and c > b)
        else ("<b class='good'>유의한 이득</b>" if (pv < 0.05 and b > c) else "무차이")
    )
    _mc_rows += f"<tr><td>{a}</td><td>+{b}</td><td>−{c}</td><td>{n}</td><td>{pv:.3f}</td><td>{mark}</td></tr>"

entry(
    "08-07",
    "final",
    "FINAL 캠페인 — 전 요인 사전등록 스윕",
    "완결",
    f"""
{
        spec(
            [
                (
                    "공통",
                    "γ0.995 · 100k · b256 · seed0 · mc_floor · z-score · 타깃τ0.005 · IQL τ0.9 · 배포=공통 joint argmax",
                ),
                (
                    "요인",
                    "A 방법×부트스트랩(max/softmax/aqcmax) · B atoms(51/101/201) · C 타깃넷(EMA/online) · D 데이터(mixed/demo)",
                ),
                (
                    "평가",
                    "arm당 시드 4개(5000–5300)×50장면 잡내 페어드 — 전 arm 동일 장면이라 arm 간도 페어드 · arm당 HUD 비디오 6장면",
                ),
                ("판정", "사전 등록: 95% t-CI(n=4)가 0을 벗어나는가 · 절대 성공률 병기"),
            ]
        )
    }
<h3>상대 성적 (arm − 잡내 vla)</h3>
{img(P / "20_final_forest.png", "FINAL forest")}
<table class='num'><tr><th>arm</th><th>n런</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{_frows}</table>
<h3>절대 성공률</h3>
<table class='num'><tr><th>arm</th><th>arm 성공</th><th>잡내 vla 성공</th></tr>{
        _abs_rows if _abs_rows else "<tr><td colspan=3>평가 도착 대기</td></tr>"
    }</table>
<h3>용어 해설 — 이 탭을 읽는 데 필요한 것들</h3>
<p><b>BoN (Best-of-N).</b> VLA에서 행동 후보 N개(우리는 16)를 샘플링하고 critic 점수가 가장 높은 것을 실행하는 배포
방식. VLA 자체는 그중 아무거나 하나를 실행하는 것과 같으므로, BoN의 이득은 "후보들 사이에 가치 차이가 있고 critic이
그 순서를 맞힐 때"만 생긴다.</p>
<p><b>expectile τ (IQL).</b> 평균(τ=0.5)이 아니라 상위쪽으로 치우친 값을 회귀하는 손실. τ=0.9면 "좋았던 결과 쪽"을
좇는 V가 되어, OOD 행동을 평가하지 않고도 max에 가까운 값을 얻는다. τ가 보증하는 범위는 대략 상위 1/(1−τ)개 샘플의
max까지(τ=0.9 ≈ 샘플 10개의 max).</p>
<p><b>HL-Gauss (atoms).</b> 가치를 스칼라 하나로 회귀하지 않고 [0,1]을 atoms개 구간으로 나눈 히스토그램 분류로 학습하는
기법 — 회귀보다 안정적. atoms(51/101/201)는 그 해상도.</p>
<p><b>ensemble-min.</b> 쌍둥이 critic 2개의 min을 쓰는 것 — 둘 중 하나라도 낮게 보면 낮게 평가하는 가장 단순한 보수화.</p>
<p><b>in-job 페어드.</b> critic 모드와 vla 모드를 같은 잡·같은 장면 세트에서 돌려 짝 비교 — 장면 난이도 운을 소거한다.</p>
<h3>최종 판정 (2026-08-08, 14/14팔 완결)</h3>
<p><b>어떤 요인 조합도 VLA를 이기지 못했다.</b> run-level 95% t-CI(n=4)로는 14팔 전부 null이고,
점추정 Δ̄는 −0.190 ~ +0.040 사이(14팔 평균 −0.051). 방법(TD/IQL/QC), 부트스트랩 연산자(max/softmax/aqcmax),
atoms(51/101/201), 타깃넷(EMA/online), 데이터(demo/mixed) 어느 축도 판정을 바꾸지 못했다.
같은 레시피·같은 장면·잡내 페어드라는 통제 하에서의 결론이므로, "튜닝이 부족했다"보다는
<b>이 스택의 구조적 한계</b>로 읽는 것이 타당하다.</p>
<h3>트라이얼 페어드 McNemar (워커A 관례 도입 — 같은 장면 200쌍, +: critic만 성공 / −: vla만 성공)</h3>
<table class='num'><tr><th>arm</th><th>+</th><th>−</th><th>n쌍</th><th>p</th><th>판정</th></tr>{_mc_rows}</table>
<p><b>검정력을 올리자 드러난 것.</b> run-level CI(n=4, 보수적)로는 안 보이던 신호가 트라이얼 페어드에서 확정된다:
<b>td_max_demo(+21/−59, p&lt;0.001)·td_aqcmax(+28/−53, p=0.007)·td_max_online(+30/−51, p=0.026)은 유의한 해악.</b>
즉 "TD 계열은 demo-only·aqcmax 부트스트랩·타깃넷 제거에서 능동적으로 해롭고", 나머지는 무차이, <b>유의한 이득은 0팔</b>.
워커A의 독립 스택(HILP φ + Cal-QL+swap, McNemar) 판정과 서로 재현 관계다 — full-authority 해악(p=.004)<i>(08-10 갱신: φ 정규화 버그로 원수치 .300/p=.004는 무효였으나, 수정 후 동일 시드 재실행에서 파국이 더 강하게 재현 — .133, +2/−19, p=.0002. 정정된 것은 숫자이지 결론이 아니다)</i>, BoN 무익.</p>
<p><b>구조적 이유 — 왜 아무것도 안 바뀌는가.</b> ① rand(16후보 중 무작위)는 VLA와 구조적으로 동일 분포이고
실측도 무차이(n=71, Δ̄=−0.020 CI[−0.054,+0.015]); randh(길이 무작위)도 null(n=4). ② 모든 critic의 argmax도 null —
max가 mean을 못 이긴다는 것은 <b>같은 상태에서 뽑힌 16후보의 참 가치 스프레드가 선택 이득을 만들기에 너무 작다</b>는 뜻
(한 chunk 뒤엔 같은 정책이 이어받는 on-policy Q^π의 구조). ③ wcurse 분해의 후보축 분산 ≪ 상태축 분산,
demo-only 밴드 0.002–0.023이 같은 사실의 정적 측정이다. 반면 K-수집에서 주방의 45%는 정책시드에 따라
성공/실패가 갈렸다 — 궤적 수준 확률성은 크지만 chunk 단위에서 순위화 가능한 형태가 아니다.</p>
<p><b>부수 발견.</b> v11의 확정 해악이던 TD(−0.167, 16시드)가 mixed에서 −0.050 null로 완화 —
실패 데이터가 TD의 파괴적 선택(placed_no_press형)을 억제한다. td_max_demo −0.190은 이 시그니처의 재확인.
td_aqcmax는 −0.125로 TD 계열 중 나쁜 축이나 CI는 0을 포함.</p>
<p><b>다음 단계 (우선순위 순).</b>
① <b>후보 다양화</b> — 스프레드 부족이 근본 원인이라면 공격 지점은 critic이 아니라 후보 생성이다:
VLA 샘플링 온도/노이즈 스케일을 키워 16후보를 의도적으로 흩뿌리고(나쁜 꼬리 포함) critic이 걸러내게 한다.
BoN의 전제(선택할 거리가 있음)를 복원하는 실험 — 소규모 스모크(온도 스윕 × 후보 스프레드 측정)부터.
② <b>과제 이전(GR1 tabletop)</b> — 행동 선택이 결과를 크게 가르는 과제군에서 같은 스택 재검증 (데이터 다운로드 중).
③ iql_vcand(후보 위 V-expectile)·τ=0.94(N=16 정합) 등 잔여 아이디어는 ①의 결과가 긍정일 때만 의미가 있다.</p>
<p><b>잠정 해석 (아카이브 — 도착분 기준 기록).</b>
IQL 계열 5팔(iql/iql_a101/iql_a201/iql_online/iql_demo)과 qc_demo가 먼저 도착했고 <b>전부 null</b> —
CI가 모두 0을 포함하며, 점추정은 −0.065~+0.040 사이. v11(demo-only 16시드)과 v12(mixed 16시드)의 null 판정과 정합적이다.
atoms 51→201 증가도, 타깃넷 EMA→online도 IQL에서는 판정을 바꾸지 않았다.
<b>td_max_demo 완결(n=4):</b> Δ̄=−0.190 CI[−0.407,+0.027] — 사전등록 기준으로는 null(CI가 0을 간신히 포함)이지만
점추정이 v11의 확정 해악(TD −0.167, n=16)과 방향·크기 모두 일치. n=4의 검정력 한계로 읽는 것이 맞다.
<b>qc(mixed) 완결(n=4): Δ̄=−0.020 CI[−0.174,+0.134] — null.</b> segfault 해결 후 TD 계열 첫 완주 팔.
나머지 TD 계열 6팔 + calql은 학습 진행 중(step 13k–27k/100k) — '침묵사 규명' 탭 참조.
06:00 무결 감사: v11·v12의 공표 수치를 원본 JSON에서 재계산해 일치 확인(TD −0.167 CI[−0.214,−0.119] 재현).</p>
""",
)

# ============================================== 08-08 v14 + CalQL 판정
_v14_rows = "".join(ci_row(a, run_level(f"v14/{a}", "critic", ("f",))) for a in ["iql_v14", "td_max_v14", "calql_v14"])
_cq_rows = "".join(ci_row(a, run_level(f"v13_calql/{a}", "critic", ("f",))) for a in ["calql_noprop", "calql_mixed"])
entry(
    "08-07",
    "v14",
    "v14 — 장면 지름길 제거 데이터 판정",
    "진행 중",
    f"""
{
        spec(
            [
                (
                    "데이터",
                    "annot/mixed_v14 = 데모 514 에피소드 + K-per-scene 롤아웃 450 (605,684 프레임) — 주방당 정책시드 3롤아웃, 45% 주방이 혼합 결과",
                ),
                (
                    "질문",
                    "critic이 장면 정체성으로 결과를 외우는 지름길을 차단하면, v12에서 열린 후보 밴드가 성공률로 전환되는가",
                ),
                ("학습", "FINAL 공통 레시피 그대로 (γ0.995 · 100k · b256 · τ0.9 · mc_floor · z-score)"),
                ("평가", "시드 4개(5000–5300) × 50장면, 잡내 vla 페어드 — FINAL과 동일 장면"),
                ("판정", "run-level 95% t-CI가 0을 벗어나는가"),
            ]
        )
    }
<table class='num'><tr><th>arm</th><th>n런</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{_v14_rows}</table>
<p><b>결과.</b> iql_v14(n=4): Δ̄=−0.035 CI[−0.083,+0.013] — <b>null</b>, CI가 좁아 "큰 효과가 숨어있다"고 볼 여지도 작다.
td_max_v14(n=4): Δ̄=−0.110 CI[−0.378,+0.158] — null이나 시드 3/4가 음수로 TD의 음(−) 경향 유지.
<b>해석:</b> 장면 정체성 지름길 제거는 held-out 프로브가 보인 보수 편향(낯선 성공 저평가)의 원인 후보였지만,
그것만으로는 성공률이 움직이지 않는다. v12의 결론(밴드는 열리나 전환 실패)이 지름길 없는 데이터에서도 유지 —
즉 병목은 데이터의 암기 가능성이 아니라 <b>후보 간 참 가치 격차 자체가 작은 것</b>(on-policy Q^π의 구조적 한계)일 가능성이 커졌다.
calql_v14는 학습 중(74k).</p>
""",
)
entry(
    "08-07",
    "calql",
    "CalQL(CO-RFT) — 후보축 학습 신호의 성공률 판정",
    "진행 중",
    f"""
{
        spec(
            [
                (
                    "동기",
                    "지금까지의 모든 방법은 후보 축을 학습에서 쓰지 않았다(실행 행동만 회귀). CalQL의 CQL 항은 16후보를 데모 대비 밀어내리는 최초의 학습-시간 후보축 신호",
                ),
                ("구성", "TD 타깃 + mc_floor(CalQL 보정) + α·(T·logsumexp Q(z,cand)/T − Q(z,demo)), α=1.0, T=0.1"),
                ("평가", "demo-only(noprop)는 시드 8개로 확장(5000–5700), mixed는 학습 중"),
                ("판정", "run-level 95% t-CI"),
            ]
        )
    }
<table class='num'><tr><th>arm</th><th>n런</th><th>Δ̄</th><th>95% CI</th><th>판정</th></tr>{_cq_rows}</table>
<p><b>결과 (완결).</b> calql_noprop(n=8): Δ̄=−0.018 CI[−0.103,+0.068] — null (초기 n=4의 쌍봉 ±0.1 분산은 시드가
쌓이며 씻김). <b>calql_mixed(n=4): Δ̄=+0.015 CI[−0.069,+0.099], McNemar +37/−34 p=0.813 — null.</b>
주목할 점 하나: 전 mixed 팔 중 유일하게 점추정이 양수이고 시드 3/4이 양수 — 학습-시간 후보축 신호(CQL 항)가
방향은 옳게 미는 정황이나 크기가 잡음 이하다. conservatism 프레임으로 읽으면: 후보가 전부 in-support인 우리
배포에서는 억압할 OOD가 등장하지 않아 다이얼이 공회전한다 — 'conservatism 스펙트럼' 리포트 참조.</p>
<p><b>calql_v14 (지름길 제거 데이터, n=4): Δ̄=−0.140, McNemar +19/−47 p=0.001 — 유의한 해악.</b>
v12 데이터에서 유일한 양(+) 팔이던 CalQL이 v14에서는 뒤집혔다. 가설: v14의 실패 비중(K-수집 32% 실패)에서
CQL 항이 좋은 후보까지 과잉 억압 — 실패 상태의 후보 push-down이 성공 경로의 행동과 겹치는 상태-행동에서
일반화됐을 수 있다. calql의 α·T 민감도(현재 α=1.0 고정)와 데이터 구성의 교호작용이 확인 대상.</p>
""",
)

# ============================================== 08-08 논문 리뷰
entry(
    "08-07",
    "papers-value-steering",
    "논문 리뷰 — Robo-ValueRL + 가치 기반 VLA 조향 3편",
    "완결",
    """
<h3>⓪ Robo-ValueRL (gewu-lab, arXiv:2607.09866) — 요청 논문</h3>
<p><b>구도.</b> 오프라인→온라인 로봇 RL에서 가치함수를 "데이터 활용 인터페이스"로 쓴다. 3단: ① 히스토리 조건부
가치 추정, ② 품질 조건부 정책 사전학습, ③ 온라인 잔차(residual) 적응. 결과: chip insertion BC 대비 +26%,
block disassembly +34%, 온라인 3회 반복으로 46%→86%.</p>
<p><b>우리 스택과의 평행선 (수렴 진화 수준).</b> 가치 헤드가 <b>동결 VLM 백본 위 경량 Transformer + HL-Gauss
분포 헤드(K=256 bins)</b> — 우리 ARQ critic(동결 π0.5 토큰 + HL-Gauss 51–201 atoms)과 같은 설계 계열.
타깃도 정규화 진행도 + <b>실패 궤적 페널티</b> — 우리 mc_floor·실패 데이터 혼합과 같은 문제 의식.</p>
<p><b>결정적 차이 — 가치를 어디에 쓰는가.</b> 이들은 <b>BoN 재순위를 아예 하지 않는다.</b> 가치는
(a) 학습-시간 신호: ΔV를 3단계 품질 라벨(Low/Med/High)로 이산화해 <b>"Quality: High" 텍스트 프롬프트로 VLA에
주입</b>(품질 조건부 사전학습), (b) 온라인 롤아웃을 고품질 세그먼트로 필터링해 <b>동결 베이스 + 경량 잔차 어댑터</b>만
학습. 즉 우리 FINAL이 도달한 결론("테스트타임 이산 선택은 이 세팅에서 못 이긴다")을 이들은 설계로 우회했다.</p>
<p><b>우리가 가져올 것 세 가지.</b> ① <b>히스토리 조건화</b>: 5프레임 히스토리(SigLIP+Perceiver 압축)가 가치
신뢰도와 성공률 모두 최고(30프레임보다 나음) — 우리 critic은 단일 프레임이라 가림·반복 동작에서 모호하고,
held-out 보수 편향의 원인 후보. 토큰에 5프레임 히스토리를 붙이는 critic 변형은 저비용 실험. ② <b>실패 페널티
타깃</b>: v13 보상 성형 아이디어와 동일 계열 — 문헌 근거 확보. ③ <b>잔차 어댑터 경로</b>: 베이스 동결을 유지한 채
critic-필터링 데이터로 작은 어댑터만 학습 — "critic이 VLA를 이긴다"를 선택이 아니라 학습된 델타로 달성하는
경로다. 단, 어댑터는 critic 외 추가 학습 파라미터이므로 <b>"이득의 100% 가치-기반 선택 귀속" 원칙과의 관계는
사용자 결정 필요</b>.</p>
<hr>
<p class='sub'>이하는 같은 주제(동결 VLA 가치 조향)의 인접 3편 — FINAL null 판정의 해석에 직접 걸린다.</p>
<h3>① V-GPS — Steering Your Generalists (arXiv:2410.13816)</h3>
<p>우리와 같은 구도(동결 정책 + 오프라인 가치함수 + 테스트타임 재순위)의 원조. Cal-QL로 가치함수를 학습하고
정책에서 <b>K=50 후보</b>를 뽑아 재순위. OpenVLA·Octo 등 5개 정책 × 12과제에서 일관된 개선을 보고.
<b>우리와의 차이가 곧 교훈:</b> (a) K=50 vs 우리 16 — 꼬리 샘플 확보량이 3배, (b) 그들의 베이스는 혼합 품질
대규모 데이터로 학습된 제너럴리스트라 정책 자체가 차선(=선택 여지가 큼); 우리 π0.5는 과제 데모로 강하게
미세조정되어 후보가 균질하다. BoN의 이득은 "베이스가 데이터 대비 차선일 때"에 몰린다는 방증.</p>
<h3>② Q-VGM — Q-Guided Value-Gradient Matching (arXiv:2606.08015)</h3>
<p>롤아웃 데이터로 Cal-QL critic을 학습(우리 calql 팔과 동일 구성)한 뒤, <b>BoN 대신 critic의 행동-그래디언트를
flow-matching 디노이징의 속도장에 주입</b>해 생성 자체를 고가치 방향으로 민다. 논문 자신의 진단이 우리 FINAL
결론과 일치: "재순위의 이산적 선택은 이미 생성된 후보를 다듬을 수 없다" — 16개 이산 후보의 스프레드가 작으면
선택으로는 이득이 없고, 연속 그래디언트 조향은 그 한계를 우회한다. <b>우리 스택에 이식 가능성 높음:</b>
π0.5도 flow-matching이고 CalQL critic·후보 z-score 인프라가 이미 있다 — critic의 ∂Q/∂a를 디노이징 스텝에
더하는 velocity correction 실험이 가능.</p>
<h3>③ Frozen VLA 가치 프로빙 (arXiv:2605.28527)</h3>
<p>동결 VLA 표현 위 선형 프로브로 가치류 신호가 <b>이미 디코딩 가능</b>함을 보임(π0.5 비전 인코더 R²=0.551,
matched-pair 순서 정확도 94.2% vs 셔플 50.1%). 단, 행동 개선은 <b>headroom 있는 과제에서만</b>
(push-plate +17.7pp, 반면 near-ceiling 과제는 0). 우리 관측과 정합: critic은 진행·실패 감지를 배웠지만
(held-out 상승→붕괴 V), 후보가 균질한 과제에서 성공률은 안 움직인다.</p>
<h3>종합 Takeaway — 우리 로드맵 반영</h3>
<table class='num'><tr><th>교훈</th><th>근거</th><th>우리 액션</th></tr>
<tr><td>BoN은 후보 스프레드가 전제</td><td>V-GPS K=50·차선 베이스에서만 이득, 우리 null</td><td>후보 다양화 프로브(21번, 실행중) + N 증대 검토</td></tr>
<tr><td>이산 선택 → 연속 조향</td><td>Q-VGM: 재순위는 후보를 다듬지 못함</td><td><b>∂Q/∂a velocity correction 실험 설계</b> (기존 CalQL critic 재사용)</td></tr>
<tr><td>가치의 용처는 학습-시간</td><td>Robo-ValueRL: BoN 없이 품질 조건화+잔차 어댑터로 +26~34%</td><td>히스토리 조건 critic·실패 페널티 타깃 도입, 어댑터 경로는 사용자 결정</td></tr>
<tr><td>가치 신호 자체는 존재</td><td>프로빙 R²=0.55, 우리 held-out rise-collapse</td><td>critic 학습은 성공 — 병목은 배포 방식이라는 확신 강화</td></tr></table>
""",
)

# ============================================== 08-10 TD-JEPA 리뷰
entry(
    "08-10",
    "papers-tdjepa",
    "논문 리뷰 — TD-JEPA, HILP의 상위호환인가",
    "완결",
    """
<p class='sub'>사용자 요청 리뷰: "HILP 말고 TD-JEPA 같은 건 어때?" (arXiv:2510.00739, Bagatella·Pirotta·Touati·
Lazaric·Tirinzoni, FAIR·Meta, 2025-10-02)</p>

<h3>① 문제 설정 — zero-shot RL이란</h3>
<p>보상 없는 전이 데이터 D={(s,a,s′)}만으로 사전학습해 두고, <b>테스트 시점에 임의의 보상 r이 주어지면
추가 학습 없이</b> 그 보상을 최대화하는 정책을 내놓는 것이 목표다. 이걸 가능하게 하는 고전적 열쇠가
<b>successor measure</b> M^π(X|s,a) = Σ_t γ^t Pr(s_{t+1}∈X|s,a,π) — "이 상태·행동에서 정책 π로 출발하면
미래에 어디를 얼마나 방문하나"의 <b>할인 방문 장부</b>다. 어떤 보상이든 Q^π_r(s,a) = ∫ M^π(ds⁺|s,a)·r(s⁺),
즉 <b>Q = 방문 장부 × 보상</b>으로 즉석 조립된다. 문제는 M이 상태공간 크기의 거대한 객체라는 것 —
그래서 저차원 인수분해가 필요하고, 그 인수분해를 어떻게 배우느냐가 이 계열(FB, HILP, TD-JEPA)의 승부처다.</p>

<h3>② 방법 — 손실의 유도 과정 (MC → TD)</h3>
<p><b>1단계 (이상형, MC-JEPA).</b> 정책 패밀리 {π_z}에 대해, 예측기가 "π_z로 미래에 방문할 상태들의 임베딩"을
직접 맞히게 한다: 손실 E‖T(φ(s),a,z) − φ(s⁺)‖², s⁺ ~ M^{π_z}(·|s,a). 이 손실의 최적 예측기는 정확히
<b>φ의 successor feature</b> F^{π_z}_φ(s,a)다 (Prop.1). 그러나 s⁺를 뽑으려면 각 π_z를 실제로 굴려야 해서
(on-policy) 오프라인 데이터로는 불가.</p>
<p><b>2단계 (핵심 기여, TD-JEPA).</b> successor feature가 벨만 방정식 F(s,a) = E[φ(s′) + γF(s′,a′)]을
만족한다는 사실로 MC 손실을 TD로 변환:</p>
<p style='text-align:center'><b>L = E‖T(φ(s),a,z) − φ̄(s′) − γ·T̄(φ̄(s′),a′,z)‖²</b>, a′~π_z(s′), 바(¯)=타깃망(stop-grad)</p>
<p>이제 <b>한 스텝 전이만 있으면</b> 되므로 오프라인·off-policy 데이터로 학습 가능. a′는 데이터가 아니라
현재 정책망 π_z에서 샘플링한다 — 우리 IQL/TD 실험과 같은 부트스트랩 구조다.</p>
<p><b>3단계 (비대칭 이중 인코더).</b> 상태 인코더 φ(제어용 저수준 정보)와 태스크 인코더 ψ(보상 정의용 고수준
정보)를 분리하고, 예측기도 두 벌(T_φ: φ→ψ 방향, T_ψ: ψ→φ 방향)을 서로 예측하게 학습. 로봇 비유:
φ는 "관절·속도", ψ는 "건물 토폴로지" — 하나의 임베딩에 둘 다 담으면 어느 쪽이든 손해라는 논리.</p>
<p><b>4단계 (정책 학습과 제로샷 추론).</b> 잠재 정책 π(φ(s), z)들을 "ψ-선형 보상 r_z(s)=⟨ψ(s),z⟩에 대해
최적"이 되도록 actor-critic으로 동시 학습(z~Z 무작위 샘플). 테스트에 보상 r이 오면 <b>선형 회귀 한 번</b>:
ω_r = (ψᵀψ)⁻¹ψᵀr → 정책 π(·, z=ω_r) 실행. 이게 전부다 — 재학습 없음.</p>

<h3>③ 이론 — 왜 collapse하지 않고, 왜 근거가 있나</h3>
<table class='num'><tr><th>정리</th><th>내용</th><th>의미</th></tr>
<tr><td>Th.1/3</td><td>MC/TD-JEPA의 최적 예측기·그래디언트가 successor measure 인수분해 손실 ‖φT_zψᵀ − M^{π_z}‖²의 것과 일치(TD 쪽은 사선 투영)</td><td>"잠재 예측"이 눈속임이 아니라 실제로 M의 <b>저랭크 인수분해</b>를 배우고 있음</td></tr>
<tr><td>Th.2</td><td>예측기를 표현보다 빠르게 학습하면 φᵀφ, ψᵀψ 공분산이 시간 불변</td><td>단위 공분산으로 초기화하면 <b>collapse(전부 0으로 수렴) 방지</b> — 직교 정규화·타깃망과 함께 실무 안정장치</td></tr>
<tr><td>Th.4</td><td>모든 단위노름 보상에 대한 정책평가 오차 ≤ 2·L_SM ≤ c·L_TD</td><td>TD 손실을 줄이면 <b>제로샷 평가 오차의 상계</b>가 줄어든다 — 손실이 곧 보증</td></tr></table>
<p class='sub'>한계(저자 명시): 보증은 P^π 대칭 가정에 기댄다 — 실제 로봇 dynamics는 비대칭이라 이론은
방향 제시용이고, 실전은 실험이 말한다.</p>

<h3>④ 실험 — 어디서 이기고 어디서 지나 (13 데이터셋 65태스크, 정직 판독)</h3>
<table class='num'><tr><th>벤치</th><th>HILP</th><th>FB</th><th>TD-JEPA</th><th>판독</th></tr>
<tr><td>DMC 픽셀 (return, avg)</td><td>391.2±23.8</td><td>456.2±8.6</td><td><b>628.8±5.5</b></td><td>픽셀에서 압도 — 전 도메인 1위(walker 738.9)</td></tr>
<tr><td>DMC proprio (avg)</td><td>620.1±8.4</td><td>648.2±4.1</td><td><b>661.2±6.3</b></td><td>동급 상위 — 저차원 입력에선 이점 축소</td></tr>
<tr><td>OGBench 픽셀 (success, avg)</td><td>32.6±0.9</td><td>39.9±0.5</td><td><b>41.3±0.5</b></td><td>BYOL-γ(41.6)와 공동 선두</td></tr>
<tr><td>OGBench proprio (avg)</td><td>38.0±1.1</td><td><b>39.0±0.7</b></td><td>38.0±0.8</td><td>동률 — 그리고 <b>cube-single에서 HILP 74.2 vs TD-JEPA 34.2로 대패</b></td></tr></table>
<p><b>주목할 정직 포인트:</b> proprio manipulation(cube 계열)에서는 HILP가 크게 이기는 도메인이 있다 —
"TD-JEPA가 HILP의 전면 상위호환"은 과장이고, 정확히는 <b>"픽셀 입력 + 넓은 커버리지에서 상위호환,
proprio manipulation에선 도메인 의존"</b>이다. 우리 GR1은 ego 픽셀 중심이라 유리한 쪽이긴 하다.
ablation 요지(§5.2, 두 축): ① <b>몇 스텝을 내다보나</b> — one-step(BYOL*)→multi-step(BYOL-γ*)에서 DMC픽셀
513.8→582.4, ② <b>누구의 미래를 예측하나</b> — behavior policy→정책조건부(TD-JEPA)에서 582.4→628.8.
"급락"이 아니라 <b>평균적 이득의 누적</b>이며, 원문은 expert-like 데이터에선 behavior 근사도 효과적일 수
있다고 단서를 단다(우리 GR1 데모가 그 경우다)
② 공유 인코더(φ=ψ)보다 분리가 낫다 ③ 사전학습된 φ를 <b>동결한 채로도</b> TD3 미세조정이 from-scratch보다
훨씬 빠르다(fast adaptation) — 표현이 실제 정보를 담고 있다는 방증.</p><blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"While BYOL* and BYOL-γ* approximate one-step and multi-step transitions of the <i>behavioral policy</i>, respectively, TD-JEPA models multi-step transitions of the <i>zero-shot policies</i>. While approximating the behavioral dynamics can be effective for expert-like data (i.e., in OGBench), we observe a general pattern suggesting that directly modeling policy-conditional successor measures is <b>on average beneficial</b>." (§5.2)</blockquote>

<h3>⑤ HILP와의 구조 비교 — 우리가 겪은 한계와 1:1 대응</h3>
<table class='num'><tr><th></th><th>HILP (우리 사용)</th><th>TD-JEPA</th></tr>
<tr><td>배우는 것</td><td>도달가능성 <i>거리</i> 임베딩 (상태만)</td><td>임베딩 + 행동조건 장기예측 + 정책</td></tr>
<tr><td>행동 정보</td><td>없음 → f(z,a)를 따로 지도학습으로 부착</td><td>T(φ(s),<b>a</b>,z)에 내장, TD로 장기 전파</td></tr>
<tr><td>태스크 공간</td><td>거리 기반 goal-reaching에 특화</td><td>ψ-span의 <b>모든 선형 보상</b> — goal 넘어 일반 보상</td></tr>
<tr><td>우리 실측 약점</td><td>"거리만 남고 디테일 소실"(디코더 프로브: proprio R² raw .760→φ .546), 합성게이트 action-blind(.487)</td><td>SF 근사가 가치 추정용 정보 보존을 목적식으로 강제 + frozen-adaptation 실험이 정보 보존을 실증</td></tr>
<tr><td>구성</td><td>3단 합성: HILP φ + 지도 f + IQL V — 각 접합부가 오차원</td><td><b>단일 TD 손실</b> — 접합부 소멸</td></tr>
<tr><td>주의점</td><td>proprio manipulation에선 여전히 강함(cube-single 74.2)</td><td>하이퍼 민감성·대칭성 가정, z-정책 다양성 필요</td></tr></table>

<h3>⑥ 우리 판정과의 교차 — 어디에 꽂히는가</h3>
<p>두 워커가 수렴한 벽("demo-only 데이터엔 동일-상태 반사실이 없다")은 TD-JEPA도 공짜로 못 넘는다:
z-조건 예측이 의미를 가지려면 <b>데이터에 서로 다른 행동 모드가 실재</b>해야 하는데, 단일 텔레옵 데모에선
z 공간이 사실상 1개 정책으로 붕괴한다(논문의 학습 데이터는 ExoRL 탐사 데이터·OGBench 다양 커버리지 —
정책 다양성이 전제돼 있다). 그러나 뒤집으면 z-조건 구조는 "같은 상태, 다른 정책 → 다른 미래"라는 반사실을
<b>정책축에서 제조하는 장치</b>다. 우리가 phase-2에 계획한 on-policy K-per-scene 수집은 vla·rand·noise-scale
변형이라는 <b>실재하는 정책 패밀리</b>를 공급한다 — z의 값이 실제로 갈리는 데이터가 생기는 순간,
이 프레임은 우리 문제(후보 chunk의 가치 비교)에 정확히 맞물린다: 후보 chunk를 "그 chunk를 실행하는
단기 정책"으로 보고 T(φ(s), chunk, z)의 착지 SF로 가치를 매기는 것이 우리 model-based 시도
Q=γ^h·V(f(z,a))의 원리적 완성형이다.</p>
<p><b>→ 설계 제안은 독립 리포트로 분리:</b> <span class='xref' data-eid='tdsf-arq'>TD-SF-ARQ 설계</span> — 단일태스크 증류(⑦)와 actor-critic 확장 사다리(⑧)를 그쪽에서 완결적으로 다룬다.</p>
""",
)

# ============================================== 08-10 TD-SF-ARQ 설계
entry(
    "08-10",
    "tdsf-arq",
    "TD-SF-ARQ 설계 — 벡터 SF 타깃의 단일태스크 critic (사전등록)",
    "살아있음",
    """
<p class='sub'>설계 제안 (사전등록 문서). 배경: <span class='xref' data-eid='papers-tdjepa'>TD-JEPA 리뷰</span>에서
사용자 질문 두 개("z가 local하게만 의미 있는 단일 태스크에서 JEPA 이점을 살릴 수 있나", "actor-critic도 되나")로
발전한 우리 자체 방법 제안. 아래는 그 완결 정리다.</p>
<h3>동기 — 우리 진단의 요약</h3>
<p>이틀간의 판정으로 확정된 사실: ① 후보 chunk 간 가치 차이는 demo-only 데이터에서 극도로 희미하다
(σ_signal≈0, 압축 좌표에서만 행동 정보 +7.3%). ② 그 희미한 신호마저 스칼라 TD로는 전이당 1차원의
그래디언트만 받아 굶주린다. ③ 3단 합성(HILP φ + 지도 f + IQL V)은 접합부마다 오차가 쌓여 action-blind로
판정났다. 처방은 "행동조건 예측을 목적식에 내장한 단일 손실" — TD-JEPA의 핵심 구조다. 단, 우리는
behavior foundation model이 아니라 단일 태스크 critic이 목표이므로 그 프레임을 증류해야 한다.</p>
<h3>설계 — 무엇을 버리고 무엇을 살리나</h3>
<p><b>사용자 논점.</b> 우리는 behavior foundation model이 목표가 아니다 — 태스크 하나, z는 local하게만
의미 있는 데이터셋. 그래도 JEPA류 표현의 이점을 살리면서 우리 adaptive transformer critic을 유지하려면?</p>
<p><b>답: 버릴 것과 살릴 것을 나눈다.</b> 태스크 인코더 ψ·전역 z·제로샷 추론(ω_r 회귀)·잠재 정책 학습은
전부 "보상이 여러 개일 때"의 장치 — 버린다. 살릴 것은 단 하나, <b>critic 출력을 스칼라 Q에서 벡터
successor feature로 바꾸고 TD 타깃도 벡터로 주는 것</b>:</p>
<p style='text-align:center'><b>F(s, chunk) ≈ φ̄(s′) + γ·F̄(s′, chunk′)</b> (chunk-단위 MDP, s′=chunk 실행 후 상태),
&nbsp; BoN 점수 Q = ⟨F, w⟩ (w = progress/성공 라벨의 φ 릿지 회귀, 닫힌형)</p>
<p><b>왜 이것이 우리 진단의 처방인가.</b> 측정으로 확정된 병목은 "행동 정보는 존재하나 희미하다"(압축 좌표
+7.3%)였다. 스칼라 TD는 그 희미한 행동 의존 경로에 <b>전이당 1차원</b>의 그래디언트만 흘린다 — 약한 신호가
굶주리는 구조(GR1 그리퍼 정규화 사고의 학습신호 버전). 벡터 SF 타깃은 <b>전이당 128차원의 밀집 감독</b>을
행동 조건 경로에 직접 붓는다. TD-JEPA의 §5.2 ablation이 같은 방향을 가리킨다: one-step·behavior-policy
타깃(BYOL*)에서 정책조건 multi-step(TD-JEPA)으로 갈수록 DMC픽셀 513.8→582.4→628.8 — "directly modeling
policy-conditional successor measures is on average beneficial"(원문). 급락이 아니라 누적적 이득이고,
expert-like 데이터에선 격차가 줄 수 있다는 단서도 원문에 있다 — 그래서 우리도 이 설계를 보장이 아닌
가설로 사전등록한다. 또한 model-based 시도 Q=γ^h·V(f(z,a))가 하려던 "착지점의 가치"를 f·V 접합부 없이 단일
TD 손실로 해낸다. <b>ARQ transformer는 아키텍처 그대로, 출력 헤드만 교체</b>(HL-Gauss는 보조 헤드로 유지 가능).</p>
<table class='num'><tr><th>단계</th><th>내용</th><th>통제</th></tr>
<tr><td>A</td><td>φ = <b>고정</b> PCA-128, ARQ 출력만 F로 교체, chunk-단위 TD + w 회귀</td><td>collapse 원천 차단, 변인 하나. 판정 = 오프라인 게이트(demo_winrate·band) vs IQL critic 나란히</td></tr>
<tr><td>B</td><td>φ 공동 학습으로 unfreeze</td><td>collapse 방지 3종 이식: 단위 공분산 초기화·직교 정규화·예측기 lr &gt; 표현 lr(Th.2)</td></tr>
<tr><td>C</td><td>on-policy K-per-scene 후 z=정책변형(vla/rand/noise) local 조건화</td><td>반사실이 z축에 실리는 시점 — 그때만 z 도입</td></tr></table>
<p><b>솔직한 기대치:</b> 동일-상태 반사실 부재의 벽은 A·B에선 그대로다 — 이것은 "약한 신호를 최대로 뽑는
더 나은 수도관"이지 보장이 아니다. 다만 우리가 잰 +7.3%가 정확히 압축 좌표의 예측 신호였으므로, 그 신호를
정면으로 먹는 목적식이라는 점에서 시도 가치가 가장 높은 단일 변경이다. (Task#8, GR1 phase-1 게이트 통과 시
A단계 사전등록.)</p>
<h3>Actor-critic 확장 — 개입 강도 사다리</h3>
<p><b>가능하고, 프레임에 내장돼 있다</b> — TD-JEPA 원본부터 잠재 정책을 Q에 대한 DPG로 학습하는
actor-critic이다. 우리 버전에서 Q=⟨F(s,chunk),w⟩는 chunk에 미분 가능하고, 그 그래디언트의 해석이 깨끗하다:
∂Q/∂chunk = wᵀ·∂F/∂chunk = <b>"chunk를 어느 방향으로 틀면 예측 착지 임베딩이 φ-공간의 가치 상승 방향으로
움직이는가"</b>. BoN(16개 중 고르기)의 다음 단계인 연속 다듬기다.</p>
<table class='num'><tr><th>형태 (약→강)</th><th>무엇</th><th>학습</th><th>위험</th></tr>
<tr><td>∂Q/∂a flow 조향</td><td>VLA flow 디노이징에 velocity correction 주입</td><td>불필요(테스트타임)</td><td>낮음 — 스텝 크기로 통제, critic 생기면 공짜 팔</td></tr>
<tr><td>AWR 잔차 어댑터</td><td>exp(A/β) 가중 BC로 소형 어댑터 학습 (A=Q−V)</td><td>가벼움</td><td>낮음 — 가중치가 데이터 chunk에만 붙어 in-support 보장 (Robo-ValueRL 어댑터 계열)</td></tr>
<tr><td>DPG 잔차 액터</td><td>δ(φ(s),chunk)를 ∂Q/∂a로 직접 상승</td><td>있음</td><td><b>높음</b> — 아래</td></tr></table>
<p><b>단, 우리 conservatism 2축 진단이 정확히 여기서 돌아온다.</b> critic-argmax(TD 3팔)조차 추정 오차를
골라 밟아 유의한 해악을 냈다(McNemar p&lt;0.01). ∂Q/∂a 상승은 과대평가 방향으로 <b>연속적으로 파고들어</b>
축2(winner's curse)를 이산 선택보다 세게 착취한다 — TD-JEPA 벤치에서 되는 이유는 탐사 데이터로 Q가 넓게
교정돼 있기 때문이고, demo-only에선 몇 스텝만 밀어도 OOD다. <b>개입 강도 사다리: BoN 게이트(신호 확인) →
flow 조향·AWR 어댑터(BC-앵커 내장) → DPG 액터는 on-policy 수집으로 Q가 자기 분포에서 교정된 후(C단계 이후)에만.</b></p>
<h3>사전등록 — A단계 판정 기준</h3>
<p>GR1 phase-1 게이트(헤드룸·rand-vs-vla) 통과 시 A단계를 다음 기준으로 사전등록한다:
동일 주석 데이터에서 IQL critic과 TD-SF-ARQ를 나란히 학습, 오프라인 게이트(held-out demo_winrate·band)에서
① demo_winrate가 0.5에서 유의하게 벗어나고 ② IQL 대비 개선이 확인되며 ③ <b>시간해상도가 γ-천장
(ΔQ≈V·|lnγ|·Δt, 워커A 08-10 보정)의 30% 이상</b>일 때만 롤아웃 팔로 승격 — 천장 초과 sensitivity는
인공 마진으로 간주해 기각한다.
둘 다 아니면 null로 기록하고 C단계(on-policy) 전제 조건으로 이동 — 트릭 추가 없음.</p>
""",
)

# ============================================== 08-11 BYOL-gamma 리뷰
entry(
    "08-11",
    "papers-byolg",
    "논문 리뷰 — BYOL-γ, TD-JEPA의 선조 (그리고 우리 A단계의 MC 팔)",
    "완결",
    """
<p class='sub'>사용자 요청 리뷰: "TD-JEPA의 선조격 같은데 자세히" (arXiv:2506.10137, Lawson·Hugessen·Cloutier·
Berseth·Khetarpal, Mila·DeepMind, ICLR 2026)</p>
<h3>문제 — stitching(조합 일반화)</h3>
<p>goal-조건 BC는 학습에서 본 (상태,목표) 조합은 잘하지만 데이터에 완결 궤적이 없는 새 조합엔 약하다.
궤적 s₀→s_h와 s_b→s_f가 w에서 교차하는 데이터로 학습, s₀→s_f(조각 잇기 필수)로 평가하는 설정.</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"while BC methods can
perform well on tasks directly observed in the dataset, they often fail to perform zero-shot transfer to tasks
requiring novel combinations of in-distribution behavior, known as combinatorial generalization." (§1)</blockquote>
<p>BC엔 MDP 귀납 편향이 없고, TD는 있지만 "challenging to scale due to the instability of bootstrapping in
TD learning when combined with fully offline training"(§1) — 이 딜레마를 푸는 것이 목표.</p>
<h3>해법 — 기하분포 오프셋의 잠재 자기예측 (TD 없는 successor 근사)</h3>
<p>successor measure(γ-할인 미래 방문 분포)를 TD 부트스트랩 대신 <b>예측 시점 k를 기하분포
k~Geom(1−γ)로 샘플링하는 MC</b>로 근사한다: 전방 예측기 ψ_f(φ(s_t), a_t) → sg[φ(s_{t+k})], 여기에
후방 예측기 ψ_b(φ(s_{t+k})) → φ(s_t)를 함께(FB backward의 친척). BC 보조 손실로 사용하며,</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"With ϕ affected by
both terms, the BC loss ensures that the representation is sufficient for action prediction, preventing
collapse." (§4.2)</blockquote>
<p>— target-net 곡예 없이 안정한 이유. 이론: 유한 단일정책 MDP에서 SR을 근사. 혼합정책 데이터에선
TD-SR이 "혼합정책의 SM"을, MC 계열은 "SR들의 혼합"을 잡는데, 대조학습(TRA)과의 결정적 차이:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"CL as used in TRA
leads to pessimism in the relationship between states sampled by different policies. … this pessimism is not
encountered with BYOL-γ, which does not utilize negative examples." (§4.2)</blockquote>
<p>negative가 없어 <b>서로 다른 궤적의 상태를 낙관적으로 잇는(bridging)</b> 표현 — 워커A의 08-10 측정
"φ는 관계 기하로 bridging한다"와 같은 주제의 목적식 측 대응물이다.</p>
<h3>결과 — 정직 판독</h3>
<p>OGBench stitch에서 BYOL-γ+GCBC가 GCBC·TRA·오프라인 RL(IQL/QRL/CRL)을 평균 상회. 단
<b>TD-SR이 작은 상태공간(antmaze)에선 더 좋고</b> BYOL-γ는 큰 상태공간(humanoidmaze·visual)에서 우세 —
"BYOL-γ's simpler training procedure is beneficial in environments with larger state spaces"(§5.2).
비용: BYOL-γ O(B) vs CL O(B²) negatives vs TD-SR O(B²) 부트스트랩.</p>
<h3>계보와 우리 함의 — A단계에 MC 팔</h3>
<p><b>계보:</b> BYOL(1-step) → BYOL-γ(기하 오프셋 MC, behavior 정책) → TD-JEPA(TD off-policy화 +
정책조건 z + 태스크 인코더 = 임의 정책 SF). TD-JEPA 표에서 BYOL-γ*는 픽셀 최강 베이스라인(DMC-RGB
582.4 vs 628.8)이었다.</p>
<p><b>함의:</b> BYOL-γ는 (a) 단일 behavior 혼합 데이터 (b) TD 불필요 (c) 정책조건 불필요 — <b>demo-only인
우리 설정엔 TD-JEPA보다 더 가까운 조상</b>이다. 액션: <span class='xref' data-eid='tdsf-arq'>TD-SF-ARQ</span>
A단계에 <b>MC-기하 변형 팔</b>(T(φ(s),chunk)가 φ(s_{t+k}), k~Geom(1−γ)을 맞힘 — 부트스트랩 제거, O(B))을
TD 팔과 나란한 한 변인 비교로 추가한다. "BC 손실이 collapse를 막는다"는 관찰은 B단계 안정장치 후보,
TRA-비관 분석은 대조학습 계열 제외의 근거.</p>
""",
)


# ============================================== 08-11 임베딩 구조 비교
entry(
    "08-11",
    "embed-compare",
    "임베딩 구조 비교 — raw/PCA/phi/BYOL-gamma/TD-JEPA (무엇이 궤적을 가로질러 붙이나)",
    "완결",
    """
<p class='sub'>사용자 요청: "HILP·cheap-z 비교처럼 TD-JEPA·BYOL-gamma도 시각화" + "한 궤적의 임베딩 하나 뽑아
다른 궤적의 최근접 이웃을 이미지로 보여달라(내가 판단하겠다)" + BC probing 통합. 데이터: PrepareCoffee mixed
어노테이션(807,634 토큰-프레임·1,234 에피소드, 토큰/에피소드 중앙값 582 — 결코 짧지 않음).</p>
<p class='missing'><b>핵심 결과(TL;DR) — DiT 닫힌-루프 probe.</b> 오프라인 지표(BC R² 0.64~0.71·stage-purity
0.76~0.83)는 전부 평평해 이 붕괴를 못 예측했다 — <b>닫힌-루프만 진실을 드러냈다</b>. PCA-128과 φ-128은 같은
128차원인데 14배 차이 → <b>차원이 아니라 학습 readout의 기하가 제어 정보를 파괴</b>.</p>
<table class='num'><tr><th>임베딩</th><th>DiT 닫힌-루프 성공률 (25트라이얼) ↑</th></tr>
<tr><td>raw 2048</td><td><b>0.60</b></td></tr>
<tr><td>PCA-128</td><td><b>0.40</b></td></tr>
<tr><td>φ (HILP)</td><td>0.04</td></tr>
<tr><td>TD-JEPA</td><td>0.04</td></tr>
<tr><td>BYOL-γ</td><td>0.00</td></tr></table>
<h3>설계</h3>
<p>TD-JEPA·BYOL-gamma를 φ(HILP)가 쓴 <b>같은 frozen RLT 토큰</b> 위에 readout으로 학습(모두 같은 MLP 구조 —
목적식만 다름). φ는 goal-조건 expectile TD(cross-episode goal 샘플링), BYOL-gamma는 기하 오프셋 자기예측
(EMA 타깃), TD-JEPA는 action-free TD successor. 같은 배터리로 채점: kNN 에피소드 purity(↓=에피소드 정체성 적음),
cross-ep phase 오차(↓), progress R²(↑).</p>
<h3>정량 결과 (배터리)</h3>
<table class='num'><tr><th>임베딩</th><th>purity ↓</th><th>phase_err ↓</th><th>prog R² ↑</th><th>해석</th></tr>
<tr><td>raw 2048</td><td>0.589</td><td>0.114</td><td>0.690</td><td>기준</td></tr>
<tr><td>PCA-128</td><td>0.511</td><td>0.111</td><td>0.694</td><td>차원만 축소</td></tr>
<tr><td><b>phi (HILP)</b></td><td><b>0.382</b></td><td><b>0.097</b></td><td><b>0.715</b></td><td><b>유일하게 에피소드 정체성 제거</b></td></tr>
<tr><td>BYOL-gamma</td><td>0.917</td><td>0.175</td><td>0.180</td><td>정체성 <b>증폭</b> (γ=0.9; γ=0.98이면 R²=−3.7로 붕괴)</td></tr>
<tr><td>TD-JEPA (SR)</td><td>0.610</td><td>0.126</td><td>0.650</td><td>≈ raw (제거 실패)</td></tr></table>
<p><img src="videos/embed/24_embed_compare.png" alt="embedding battery + PCA-2 projection"></p>
<p class='sub'>위 산점도는 <b>PCA-2</b>(선형) 투영이다. 아래는 같은 임베딩의 <b>t-SNE</b>(비선형) 투영 — 색은 task progress:</p>
<p><img src="videos/embed/26_tsne_embed.png" alt="t-SNE of 5 embeddings colored by progress"></p>
<p><b>t-SNE 판독:</b> φ는 progress가 <b>하나의 일관된 매니폴드</b>로 흐른다(궤적을 가로질러 과제 단계 정렬).
<b>BYOL-gamma는 ~60개 작은 섬으로 파편화</b> — 각 섬이 자기 안에 progress 그라디언트를 가짐 = 에피소드별
독립 타임라인(purity .92의 육안 증거). raw/PCA는 그라디언트는 있으나 덜 정돈, TD-JEPA는 중간.</p>
<h3>핵심 통찰 — TD vs MC가 아니라 "cross-episode 대조가 있느냐"</h3>
<p>BYOL-gamma(미래 자기예측)도 TD-JEPA readout(TD successor, action-free)도 <b>궤적 내부 미래만</b> 예측한다 →
궤적을 가로질러 붙일 압력이 없음. 우리 RLT 토큰은 이미 에피소드 정체성이 강해서(raw purity .59),
순수 자기예측은 <b>가장 예측하기 쉬운 방향 = 에피소드 내내 천천히 변하는 정체성 특징</b>에 lock-on 하고 오히려
증폭한다(BYOL-gamma purity .92). 오직 φ(HILP)만 goal을 <b>cross-episode로 샘플링</b>해 다른 궤적 간 대조를
강제하므로 정체성을 벗긴다(.38). <b>DBC도 같은 패턴</b> — |r_i−r_j|+W₂가 permuted-batch의 임의 상태 쌍 대조라
궤적을 가로지른다. <b>결론: 원하는 cross-trajectory invariance엔 목적식에 궤적을 가로지르는 대조/goal 샘플링이
반드시 필요하다.</b> 순수 self-prediction(BYOL-gamma·TD-SR)엔 그게 없다.</p>
<p class='sub'><b>정정:</b> 첫 BYOL-gamma 붕괴를 "짧은 에피소드+end-clamp"로 오진했으나, 확인 결과 에피소드는
중앙값 582 토큰으로 길다. γ를 0.98→0.9로 낮춰도 purity가 .92로 여전히 높아(붕괴 완화일 뿐), 원인은 horizon이
아니라 위의 <b>self-prediction 정체성 증폭</b>으로 확정됐다.</p>
<h3>정성 이웃 행렬 (사용자 설계) — 세로줄 = 같은 궤적</h3>
<p>레이아웃: <b>행 = query 에피소드 7개</b>(각각 랜덤 프레임 하나), <b>열 = 고정된 다른 에피소드 7개</b>(모든 행에서 동일),
셀[i,j] = 열-에피소드 j에서 query i와 코사인 최근접 프레임. 각 셀은 <b>위 agentview / 아래 wrist</b>, 숫자는 코사인 유사도.
세로줄이 항상 같은 궤적이라 "각 임베딩이 다른 궤적들에서 query와 무엇을 같다고 보는가"를 직접 비교할 수 있다.</p>
<p><b>φ (HILP)</b><br><img src="videos/embed/25_xneighbor_phi128.png" alt="phi neighbor matrix"></p>
<p><b>raw 2048</b><br><img src="videos/embed/25_xneighbor_raw2048.png" alt="raw neighbor matrix"></p>
<p><b>BYOL-gamma</b><br><img src="videos/embed/25_xneighbor_byolg128.png" alt="byolg neighbor matrix"></p>
<p><b>PCA-128</b><br><img src="videos/embed/25_xneighbor_pca128.png" alt="pca neighbor matrix"></p>
<p><b>TD-JEPA</b><br><img src="videos/embed/25_xneighbor_tdjepa128.png" alt="tdjepa neighbor matrix"></p>
<p>φ의 이웃은 다른 궤적에서 같은 과제 순간(같은 팔-물체 배치, wrist 뷰의 컵/머그)으로 일관되는 경향. <b>정직한 한계 둘:</b>
① PrepareCoffee는 주방 장면이 반복돼 <b>모든</b> 임베딩(raw 포함)이 시각적으로 그럴듯한 cross-궤적 이웃을 찾는다 —
패널만으로 방법을 강하게 가르긴 어렵다(판별은 정량 purity·t-SNE가 한다). ② 이웃들이 "의미 같고 시각 다름"보다 거의
<b>near-duplicate</b>라, 원하는 invariance(시각 변이 불변)를 이 데이터로는 완전히 stress-test 하지 못한다 —
<b>시각 다양성이 큰 데이터(GR1 신규물체·OGBench 시각변형)</b>가 그 시험대다.</p>
<h3>정렬 축 분석 (사용자 가설 검증, 08-11) — 무엇으로 정렬되나</h3>
<p>사용자 관찰 둘: (a) φ는 progress(goal-거리)로 정렬해 같은 모션을 속도로 갈라놓는다, (b) raw는 최근접조차
유사도가 낮아 궤적 간 그룹화가 안 된다. 궤적 가로지르는 20만 쌍에서 임베딩 유사도가 <b>행동(모션) vs progress</b>
중 무엇과 상관되는지, 그리고 최근접 vs 랜덤 유사도 gap(판별력)을 쟀다:</p>
<table class='num'><tr><th>임베딩</th><th>corr(sim,action)</th><th>corr(sim,progress)</th><th>최근접 sim</th><th>랜덤 sim</th><th>판별 gap</th></tr>
<tr><td>raw 2048</td><td>0.178</td><td>0.193</td><td>0.529</td><td>0.188</td><td><b>0.341</b>(최저)</td></tr>
<tr><td>PCA-128</td><td>0.186</td><td>0.209</td><td>0.591</td><td>0.001</td><td><b>0.590</b>(최고)</td></tr>
<tr><td>φ (HILP)</td><td>0.152</td><td><b>0.019</b></td><td><b>0.829</b></td><td><b>0.281</b></td><td>0.549</td></tr>
<tr><td>BYOL-γ</td><td>0.044</td><td>0.035</td><td>0.738</td><td>0.154</td><td>0.584</td></tr>
<tr><td>TD-JEPA</td><td>0.110</td><td>0.073</td><td>0.635</td><td>0.060</td><td>0.576</td></tr></table>
<p><b>판독:</b> ① (b) 확증 — raw 판별 gap 0.341 최저(최근접이 랜덤보다 별로 안 가까움) = "같은 상태다"를 못 집음.
② (a) <b>반증</b> — φ는 progress 상관 0.019(최저)이고 오히려 action(0.152)과 상관. φ는 순수 progress가 아니라
reachability(장면+행동)를 인코딩. 같은-모션 내 progress 의존도도 φ(.249)=raw(.25)로 동일. ③ φ는 압축돼
랜덤 쌍도 0.281로 가까움(비판별 성분), PCA는 랜덤 0.001로 가장 깔끔히 판별. <b>모든 상관이 약함(≤0.21) =
어느 임베딩도 행동/progress 축으로 깔끔히 조직되지 않음, 주로 장면 외형이 지배.</b></p>
<h3>Stage 정렬 (옳은 타깃 — 속도·배경 불변) — 거친 O, 정밀 X</h3>
<p>progress는 속도 혼입 타깃이라 틀렸다. 옳은 타깃은 <b>하위과제 stage</b>(kroll 플래그 grasped/placed/machine_on의
조합) — 속도·배경 무관한 task-relevant 라벨. "cross-궤적 최근접이 같은 stage인가"(우연 0.207):</p>
<table class='num'><tr><th>임베딩</th><th>최근접 same-stage</th><th>chance 대비 lift</th></tr>
<tr><td>raw / PCA</td><td>0.825 / 0.828</td><td>+0.62</td></tr>
<tr><td>φ (HILP)</td><td>0.815</td><td>+0.61</td></tr>
<tr><td>TD-JEPA</td><td>0.807</td><td>+0.60</td></tr>
<tr><td>BYOL-γ</td><td>0.762</td><td>+0.55</td></tr></table>
<p><b>판정 — 비관론의 정정:</b> 전 임베딩이 다른 궤적 최근접을 <b>~80%로 같은 하위과제</b>에서 찾는다(우연 21% 대비
+0.6). 즉 "cross-궤적에서 아무것도 같은 상태로 안 본다"는 <b>거친 하위과제 수준에선 틀리다</b> — 배경 nuisance를 안고도
하위과제 정렬은 이미 된다. <b>정밀 종합(거친 O / 정밀 X):</b> 거친 stage 정렬 ~0.80이지만 정밀 관계 기하(정확한 컵
pose·그리퍼 배치, 정렬-축 corr .15)는 약하다. 사용자 명제 "배경 무관·관계상태만 중요"는 <b>정밀 수준에서 참</b> —
부족한 건 <b>같은 하위과제 안에서의 정밀 판별</b>이고, 그건 더 나은 invariance가 아니라 <b>반사실(커버리지)</b>이
푸는 문제다. 임베딩 라인과 커버리지 라인이 같은 지점에서 만난다.</p>
<h3>DiT 닫힌-루프 probe (본 판정, 08-11) — 오프라인 지표를 뒤집다</h3>
<p>오프라인 BC MSE는 compounding error를 못 본다. 그래서 임베딩별 <b>DiT 정책 헤드</b>(action chunk를 H토큰으로
시간축 self-attention + AdaLN-Zero, rectified-flow)를 학습해 PrepareCoffee 시뮬에서 <b>닫힌-루프 성공률</b>로 잰다
(25트라이얼, 동일 VLA 백본 토큰):</p>
<table class='num'><tr><th>임베딩</th><th>DiT 닫힌-루프 성공률 ↑</th><th>(참고) BC R²</th><th>(참고) stage-purity</th></tr>
<tr><td>raw 2048</td><td><b>0.60</b></td><td>0.708</td><td>0.825</td></tr>
<tr><td>PCA-128</td><td><b>0.40</b></td><td>0.697</td><td>0.828</td></tr>
<tr><td>φ (HILP)</td><td>0.04</td><td>0.682</td><td>0.815</td></tr>
<tr><td>TD-JEPA</td><td>0.04</td><td>0.688</td><td>0.807</td></tr>
<tr><td>BYOL-γ</td><td>0.00</td><td>0.640</td><td>0.762</td></tr></table>
<p><b>판정 — 세 가지 결정타:</b> ① <b>차원이 아니라 기하가 범인.</b> PCA-128(0.40)과 φ-128(0.04)은 같은 128차원인데
14배 차이 — 분산 보존 선형투영은 제어를 살리고, <b>학습된 readout(φ·TD-JEPA·BYOL-γ)의 reachability/self-predictive
기하가 정밀 제어 정보를 파괴</b>한다. ② <b>오프라인 지표 전멸.</b> BC R²(0.64~0.71 평평)도 stage-purity(0.76~0.83
평평)도 이 붕괴를 예측 못 함 — <b>닫힌-루프만 진실을 드러낸다</b>(compounding error). ③ <b>"φ는 표현으로 충분"은
확정 기각</b> — φ는 단일-스텝 BC·거친 stage는 되지만 닫힌-루프 제어엔 불충분(디코더 프로브의 proprio 손실 .760→.546이
실은 제어 결정적이었음). <b>TD-SF-ARQ 함의:</b> critic 임베딩은 φ가 아니라 <b>PCA-128/raw</b>로 — 학습 readout은
후보 판별에 필요한 정밀 제어 신호를 버린다. (사용자가 닫힌-루프+DiT를 고집한 판단이 정확히 옳았다.)</p>
<h3>BC probing 통합 (5개 임베딩) — BC-충분성과 invariance는 직교</h3>
<p>임베딩→데모 action chunk 재현 BC를 5개 전부에 대해 동일 프로토콜로(kroll, held-out):</p>
<table class='num'><tr><th>임베딩</th><th>BC action R² ↑</th><th>(참고) 구조 purity ↓</th></tr>
<tr><td>raw 2048</td><td>0.708</td><td>0.589</td></tr>
<tr><td>PCA-128</td><td>0.697</td><td>0.511</td></tr>
<tr><td>phi (HILP)</td><td>0.682</td><td><b>0.382</b></td></tr>
<tr><td>TD-JEPA</td><td>0.688</td><td>0.610</td></tr>
<tr><td>BYOL-gamma</td><td>0.640</td><td>0.917</td></tr></table>
<p><b>판독:</b> BC R²가 <b>전 임베딩에서 거의 평평(0.64~0.71)</b> — 에피소드 정체성에 붕괴한 BYOL-gamma조차
행동 정보는 대부분 보존(0.640). 즉 <b>행동 예측(BC)과 cross-trajectory invariance는 직교</b>한다: BYOL-gamma는
BC는 되지만(0.64) bridging은 최악(purity 0.92)이고, φ는 둘 다(0.68 / 0.38). 이는 "BC를 잘한다 ≠ 좋은 임베딩"을
정량화한다 — 우리 목표(같은 상태 후보 판별)엔 BC-충분성이 아니라 invariance가 필요하고, 그 invariance는
cross-episode 대조 목적식(φ)만 준다. 종합: φ는 (a) 행동정보 보존(BC .68) (b) 정체성 제거(purity .38)
(c) 관계 bridging(워커A act-cos .661) — <b>표현으로 충분</b>. φ-critic의 BoN 실패는 표현이 아니라
<b>반사실 부재(데이터)</b>. 디코더·BC·구조 세 독립 프로브가 "표현 충분, 데이터 부족"으로 수렴.</p>
""",
)

# ============================================== 08-11 DBC/bisimulation 리뷰
entry(
    "08-11",
    "papers-dbc",
    "논문 리뷰 — DBC(bisimulation): 우리가 원하는 invariance의 정면 정의",
    "완결",
    """
<p class='sub'>사용자 질문 발: "우리가 원하는 invariance는 semantic하게 비슷한 장면(컵 손잡이로 다가간다)이
visual하게 달라도 비슷한 임베딩이 되는 것. BYOL-γ도 도움이 되나, 다른 연구가 있나?" → 그 invariance를 <b>정면으로
정의</b>하는 계보가 bisimulation이다. (arXiv:2006.10742, Zhang·McAllister·Calandra·Gal·Levine, ICLR 2021 oral)</p>
<h3>무엇을 정의하나 — behavioral 등가</h3>
<p>bisimulation은 두 상태를 "픽셀이 비슷한가"가 아니라 <b>"행동적으로 구분 불가능한가"</b>로 같다고 본다:
같은 보상을 주고, 어떤 행동에도 (다음 상태의) bisimulation-등가 분포로 전이하면 등가. DBC는 이 거리를 임베딩
거리로 <b>직접</b> 심는다:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"Our method trains
encoders such that distances in latent space equal bisimulation distances in state space." (Abstract)</blockquote>
<p style='text-align:center'><b>J(φ) = ( ‖z_i−z_j‖₁ − |r_i−r_j| − γ·W₂(P̂(·|z̄_i,a_i), P̂(·|z̄_j,a_j)) )²</b></p>
<p>즉 두 임베딩 거리가 <b>보상 차이 + 다음-상태 분포의 Wasserstein 거리</b>와 같아지게 회귀한다(z̄=stop-grad,
동역학 모델 P̂ 동시 학습). 재구성이 전혀 없어서 과제-무관 시각 요소는 애초에 표현되지 않는다:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"state elements are
relevant not only if they influence the current reward, but also if they influence state elements in the future
that in turn influence future rewards … an ideal representation is one that is predictive of reward, and also
predictive of itself in the future." (§1)</blockquote>
<p>실험도 정확히 우리 관심사: MuJoCo 배경을 움직이는 distractor·자연 영상으로 바꿔도, 운전 태스크에서 구름·날씨·
시간대가 바뀌어도 invariant.</p>
<h3>SR/BYOL-γ와의 관계 — 보상축의 유무</h3>
<table class='num'><tr><th></th><th>SR/BYOL-γ (successor)</th><th>bisimulation/DBC</th></tr>
<tr><td>같다고 보는 기준</td><td>미래 <b>방문 분포</b>가 같으면</td><td>미래 <b>보상 결과</b>가 같으면(보상+전이)</td></tr>
<tr><td>보상축</td><td>없음 (dynamics-only)</td><td>있음 — |r_i−r_j| 항</td></tr>
<tr><td>우리 want와의 거리</td><td>proxy (semantics≡미래결과일 때 tight)</td><td><b>정의 그 자체</b></td></tr>
<tr><td>실무 위험</td><td>collapse (BYOL-γ는 BC앵커로 방지)</td><td>W₂·동역학 공동학습 불안정성(논문도 언급)</td></tr></table>
<p>즉 BYOL-γ는 "미래 방문이 같으면 붙인다"는 <b>보상 없는 사촌</b>, DBC는 "보상 결과가 같으면 붙인다"는 정본.
우리가 원하는 "손잡이 접근은 배경 달라도 같게"는 — 손잡이 접근이 같은 미래 보상(잡기 성공)으로 이어지는 한 —
둘 다 잡지만, DBC가 더 직접적이다.</p>
<h3>인접 계보 (같은 목표, 다른 손잡이)</h3>
<table class='num'><tr><th>계열</th><th>대표</th><th>거리의 의미</th></tr>
<tr><td>Bisimulation</td><td>DBC(2021), DeepMDP(2019)</td><td>보상+전이 등가</td></tr>
<tr><td>Value-implicit</td><td>VIP(2023)</td><td>목표 진행도</td></tr>
<tr><td>Quasimetric</td><td>QRL(2023)</td><td>최적 도달 스텝(비대칭)</td></tr>
<tr><td>Temporal-contrastive</td><td>TRA(2025)·CRL·TCN</td><td>시간 근접/같은 목표</td></tr>
<tr><td>Object-centric</td><td>slot attention 계열</td><td>관계 배치 직접(배경 무관)</td></tr></table>
<h3>우리 결론과의 연결 — invariance = 커버리지 해금</h3>
<p>핵심 통찰: 더 나은 semantic invariance는 <b>궤적 간 반사실 공유</b> 장치다. φ가 궤적 A의 "손잡이 접근"과 B의
"손잡이 접근"을 같은 임베딩으로 붙이면, A(a₁→성공)와 B(a₂→실패)가 <b>같은-임베딩 상태의 반사실 쌍</b>이 된다 —
원 궤적은 달라도. 즉 invariance는 오프라인 데이터에 숨은 반사실을 풀어내 <b>유효 커버리지를 늘린다</b>(OGBench
stitch가 되는 이유). <b>한계 둘:</b> ① 데이터에 잠재 반사실이 있어야 풀 게 있다(단일 데모 모드면 0). ② 우리
downstream(후보 판별)엔 반사실이 어디든 존재해야 한다. 그래서 invariance(잠재 해금)와 on-policy(신규 제조)는
상보. <b>우리 증거:</b> φ(successor 계열)가 이미 관계 기하로 궤적 간 bridging(워커A act-cos .661 vs .334) —
이 계보가 우리 스택에서 실제로 작동함을 확인했다. <b>다음:</b> TD-JEPA·BYOL-γ readout을 φ·raw와 같은 배터리로
비교(embed-compare 엔트리), bisimulation(보상축 추가)은 GR1에서 progress 보상으로 시도 가능.</p>
""",
)


# ============================================== 08-11 커버리지 역방향 ablation 설계
entry(
    "08-11",
    "horizon-probe",
    "설계 — 커버리지가 범인인가: OGBench에서 역방향 ablation (반사실 다이얼)",
    "보류",
    """
<p class='missing'><b>보류 (08-11, 사용자 판단):</b> OGBench 재확인은 "커버리지가 OGBench 장점"이라는
결론이 이미 우리 2축 프레임으로 설명되므로 새로 얻을 게 적다 — 실험은 접고, 결론(커버리지=인과 변인,
처방=on-policy 커버리지 제조)만 로드맵에 반영한다. 본류는 GR1 phase-1 + phase-2 on-policy.</p>
<p class='sub'>사용자 제안(08-11) + 결정적 사실 확정: "TD-BoN(우리 초기 버전)은 OGBench에서 굉장히 잘 된다."
그리고 OGBench의 장점은 <b>diverse/play 데이터의 반사실 풍부함(커버리지)</b>이다. 이 엔트리는 그 관찰을
<b>범인 특정 실험</b>으로 구체화한다.</p>
<h3>재구성 — 진단이 뒤집혔다</h3>
<p>TD-BoN이 OGBench에서 잘 된다는 사실은 "우리 파이프라인이 신호가 있을 때 잡는가?"에 <b>이미 답</b>한다: 잡는다.
따라서 PrepareCoffee·YAM의 null은 <b>임베딩/critic 결함이 아니라 그 데이터에 신호(반사실)가 없어서</b>다 — 우리
2축 결론(축2 = 같은-상태 반사실 부재)을 반대편에서 확증하며 <b>우리 방법을 면죄</b>한다. 그러므로 질문이 바뀐다:</p>
<p style='text-align:center'><b>OGBench엔 있고 VLA-manipulation엔 없는, TD-BoN을 죽이는 그 속성 하나는?</b></p>
<p>사용자 확정: <b>커버리지(반사실 밀도)</b>가 1순위 용의자이자 OGBench의 핵심 장점. "임베딩이 흩어져서"가 아니라
"같은 상태를 다른 행동으로 지나간 데이터가 없어서" 후보를 못 가른다는 가설.</p>
<h3>결정적 실험 — 같은 환경에서 커버리지만 다이얼 (한 변인)</h3>
<p>새 환경을 만들 필요가 없다. <b>OGBench 그 자체에서 커버리지만 얇게 만든다</b>: env·horizon·관측·알고리즘 전부
고정, 데이터의 <b>반사실 밀도</b>만 낮춘다(같은 상태를 지나는 서로 다른 궤적/행동 수를 프로그램적으로 감축 →
demo-only 레짐으로 수렴). 반사실 밀도는 측정 가능 — 상태 빈당 데이터에 등장한 서로 다른 next-action 수.</p>
<table class='num'><tr><th>관측</th><th>결론</th></tr>
<tr><td>커버리지↓에 따라 TD-BoN sensitivity·성공이 <b>단조 붕괴</b>, demo-only 레짐에서 우리 VLA null 재현</td><td><b>커버리지가 인과 변인으로 확정</b> — 2년치 null의 뿌리를 같은 환경에서 통제 증명. 처방은 명확: VLA엔 반사실을 <b>만들어야</b> 한다(on-policy)</td></tr>
<tr><td>커버리지를 얇게 해도 TD-BoN 유지</td><td>커버리지는 범인이 아님 → horizon·관측차원 등 2순위 용의자로 다이얼 이동</td></tr>
<tr><td>같은 커버리지에서 raw는 되고 φ만 붕괴</td><td>표현이 부차 변인 — φ가 버린 정보가 이 레짐에서 결정적(디코더 프로브 .546과 대조)</td></tr></table>
<h3>전략적 따름정리 — 이게 맞으면 우리 로드맵이 확정된다</h3>
<p>커버리지가 인과 변인으로 확정되면: ① <b>오프라인 전용 임베딩 라인(φ, TD-SF-ARQ A·B단계)은 천장이 있다</b> —
반사실이 없는 데이터에선 목적식을 아무리 바꿔도 상한이 낮다. ② 진짜 레버는 <b>커버리지 제조 = on-policy
K-per-scene 수집</b>(vla/rand/noise 변형으로 같은 장면을 다르게 지나가기) — GR1 phase-2 C단계로 이미 계획됨.
OGBench는 그것의 존재 증명이다: "반사실을 주면 TD-BoN이 된다". TD-SF-ARQ는 그 반사실을 <b>최대 효율로 흡수</b>하는
목적식(전이당 128차원 SF 감독)이라는 위치로 정리된다.</p>
<p><b>순서:</b> GR1 phase-1과 병렬(다른 자원). 우리 초기 TD-BoN 코드 + OGBench 로더 재사용이라 착수 빠름.
반사실-밀도 다이얼 3~4점 스윕이 첫 산출. (Task#10)</p>
""",
)


# ============================================== 08-09 아침 종합
entry(
    "08-07",
    "morning-0809",
    "아침 종합 (08-09) — 수렴의 밤: 모든 갈래가 닫히고 남은 것",
    "살아있음",
    """
<p><b>한 줄 요약.</b> 밤새 열려 있던 모든 갈래가 정직하게 닫혔다 — 트릭 계열(v17b), 임베딩 계열(φ·φ+proprio),
히스토리, model-based 오프라인 합성까지. 이틀의 수렴 결론은 하나: <b>이 과제·이 VLA에는 후보 선택으로 캘 가치
차이가 없다.</b> 남은 본질적 행보는 GR1 이전이며, 두 가지 사용자 결정을 기다린다.</p>
<h3>밤새 판정 (전부 사전등록 기준·페어드)</h3>
<table class='num'><tr><th>실험</th><th>판정</th><th>수치</th></tr>
<tr><td>v17b 다양화+σ-veto (n=16 확정)</td><td>null — 트릭 계열 종결</td><td>Δ̄=+0.019 CI[−0.021,+0.059], McN +136/−121 p=0.383</td></tr>
<tr><td>φ-128 사다리 (n=8 확정)</td><td>null</td><td>Δ̄=−0.010, n=4 신호는 잡음으로 판명</td></tr>
<tr><td>히스토리 critic (iql·td)</td><td>둘 다 null — 축 마감</td><td>−0.025 / +0.020</td></tr>
<tr><td>model-based 합성 게이트 (4좌표)</td><td>오프라인 기각 — 롤아웃 0회 소모</td><td>demo_winrate .479/.481/.487/.485 (0.5=blind)</td></tr>
<tr><td>φ+proprio (사용자 질문)</td><td>proprio 보존↑(R² .546→.617)이나 게이트 동일 기각</td><td>winrate .485</td></tr></table>
<h3>이틀의 수렴 — 왜 이것이 성과인가</h3>
<p>부정 결과 여덟 갈래가 서로 독립인 방법으로 같은 구조적 사실을 가리켰고(선택·임베딩·히스토리·학습신호·모델 합성),
각 갈래는 사전등록 기준과 페어드 통계로 닫혔다. 워커A의 독립 스택 판정(전권 파국 p=.004<i>(08-10 갱신: φ 정규화 버그로 원수치 .300/p=.004는 무효였으나, 수정 후 동일 시드 재실행에서 파국이 더 강하게 재현 — .133, +2/−19, p=.0002. 정정된 것은 숫자이지 결론이 아니다)</i>, BoN 동률)과도 상호 재현.
"무엇이 안 되는가"의 지도가 완성됐고, 그 지도가 다음 무대(GR1)의 실험 설계 — 파일럿에서 headroom과 스프레드부터
측정 — 를 결정한다.</p>
<h3>사용자 결정 대기</h3>
<table class='num'><tr><th>#</th><th>결정</th><th>선택지</th></tr>
<tr><td>1</td><td>GR1 학습 자원</td><td>A. node200/B200(빠름, 워커A 머신 승인 필요) / B. 이 클러스터 PRO6000(96GB)</td></tr>
<tr><td>2</td><td>PR#4 머지</td><td>iql-followups(110+커밋) → master 후 작업단위 브랜치 체제</td></tr></table>
<h3>오늘 계획</h3>
<p>① 자원 결정 시 GR1 config 구현→파일럿 미세조정 ② 용어 해설 패스(진행 중) ③ 워커A MVE 결과 교차 리뷰(게시되면)
④ 리포트 무결 유지·매 사이클 게시.</p>
""",
)

# ============================================== 08-09 GR1 이식 계획
entry(
    "08-07",
    "gr1-port",
    "GR1 tabletop 이식 계획 — 질문이 유효한 무대로",
    "진행 중",
    """
<p><b>왜 GR1인가.</b> 이틀간 모든 경로가 "PrepareCoffee + 과제특화 미세조정 VLA에서는 후보 간 캘 가치 차이가 없다"로
수렴했다(model-based 리포트 참조). AQC 논문의 세팅이기도 한 GR1 tabletop은 ① 고정 탁상(장면 일반화 아님)
② 양팔 휴머노이드의 정밀 조작이라 행동 선택이 결과를 가르고 ③ 베이스 성공률에 headroom이 있다 —
가치 기반 선택이라는 질문 자체가 유효한 무대다.</p>
<h3>준비 완료</h3>
<table class='num'><tr><th>항목</th><th>상태</th></tr>
<tr><td>시뮬레이터 (robosuite-gr1 fork + robocasa-gr1-tabletop)</td><td>08-07 스모크 통과 (.venv-gr1, gym 등록 확인)</td></tr>
<tr><td>데이터 (Teleop-Sim, 5태스크)</td><td>12G 다운로드·검증 완료 — 태스크당 1,000 에피소드, LeRobot 포맷, ego_view 256², state/action 44d, 20fps</td></tr></table>
<h3>필요한 변경 (config 초안 조사 결과)</h3>
<table class='num'><tr><th>변경</th><th>내용</th><th>난이도</th></tr>
<tr><td>action_dim 44</td><td>π0.5 기본 action_dim=32 &lt; GR1 44 → in/out 프로젝션 재초기화 필요(KeepMissing 로더가 처리, 새 프로젝션은 BC 미세조정으로 학습)</td><td>중</td></tr>
<tr><td>데이터 config</td><td>단일 ego_view 카메라(RoboCasa는 3캠) 매핑 + state 44d repack + norm stats 재계산</td><td>중</td></tr>
<tr><td>RLT 파이프라인</td><td>주석·critic은 과제 무관 — 그대로</td><td>하</td></tr>
<tr><td><b>평가 하네스 (10:10 조사 갱신)</b></td><td>메인 venv(mujoco 3.3.1)에서 fork env는 버전 assert 3종 완화 후에도
<b>렌더러 API 비호환</b>(MjRenderContextOffscreen, 3.2.6 전용)으로 불가. 물리 스택 추가 패치는 하지 않는다(원칙).
<b>확정 아키텍처: 정책 서버 분리</b> — 메인 venv에서 openpi serve_policy(VLA+critic), .venv-gr1에서 롤아웃
클라이언트(env)가 websocket 질의. <b>설계 확정(11:40): 프로토콜 확장 불필요</b> — 표준 infer(obs)→actions 인터페이스를
그대로 쓰고, ① BoN 로직(후보 16→critic→argmax)은 서버측 Policy 어댑터(BoNServePolicy) 내부에, ② 커밋 길이는
반환 chunk를 n_exec 길이로 잘라 보내는 것으로 표현(클라이언트는 받은 만큼 실행하는 표준 루프), ③ vla 기준선 =
같은 서버의 mode=vla. 클라이언트는 rollout.py의 시드·페어링·stage 로깅 프로토콜을 이식</td><td>중</td></tr></table>
<h3>학습 자원 — 사용자 결정 필요</h3>
<table class='num'><tr><th>옵션</th><th>내용</th><th>비고</th></tr>
<tr><td>A. node200 (B200, 다른 머신)</td><td>기존 VLA 미세조정이 돌던 인프라(train_rlt.slurm, /data5) — 가장 빠름</td><td>워커A 머신 자원 — 승인 필요, torch/B200 이슈 이력 참고</td></tr>
<tr><td>B. 이 클러스터 PRO6000/A6000</td><td>3B 미세조정을 bf16 + batch 축소로 — PRO6000(96GB)이면 무난, A6000(48GB)은 빠듯</td><td>큐 경쟁, 검증 안 된 경로</td></tr></table>
<p><b>발차 (08-09 ㄱㄱㄱ, 자원 B: PRO6000).</b> config 등록(pi05_gr1_rlt: action_dim 48, 파일럿 30k) →
데이터 v2.0→v3.0 변환 → norm stats <b>통과(15:00)</b> → 파일럿 미세조정 자동 개시. <b>이식 사고록(5건 — 나머지
4태스크 변환의 체크리스트):</b> ① LeRobot v2.0 포맷(BackwardCompatibilityError) → v2.1 태그+공식 변환기,
② 변환기 --root는 부모 디렉토리 시맨틱, ③ episodes_stats.jsonl은 변환기가 생성이 아니라 로드 — 자작 생성기
(수치는 parquet 실계산, 이미지는 중립 placeholder — openpi는 이미지 스탯 미사용), ④ state/action dtype이
'object'로 기록 → float32 교정, ⑤ frame_index 컬럼 부재 → AddProgress에 전역 index 폴백 구현(커밋),
⑥ torchcodec이 노드의 FFmpeg 공유 라이브러리 부재로 실패 → pyav 백엔드 env 배선(LEROBOT_VIDEO_BACKEND, 커밋).</p>
<p><b>하네스 완성(17:00):</b> 서버(serve_bon_policy.py — eval_critic 재사용, 표준 infer 계약) + 클라이언트
(rollout_client.py — .venv-gr1, 페어드 시드) 양단 커밋. 5태스크 전부 v3 변환 완료. 파일럿 자원 사고 2건 추가: ⑦ base 체크포인트의 action 프로젝션이 (32,·)로 존재해 shape 충돌 → 로더에
불일치-드롭+fresh 유지 구현(커밋), ⑧ A6000 48GB는 batch16도 OOM(기각), ⑨ PRO6000 96GB도 batch32 OOM →
batch16 채택. <b>학습 진입 성공(21:15, node57 PRO6000): 1.7s/it, 30k ETA ~13.7시간(08-10 오전 완주 예상).</b>
9건의 사고 전부 원인-수정 짝으로 기록 — 이식 완료.</p>
<p><b>하네스 계약 확정 (08-09 밤, GPU 노드 스모크 2건).</b> 로그인 노드는 GPU가 없어 EGL 오프스크린 렌더가
불가하므로 스모크를 3090 노드 슬럼 잡으로 돌렸다. 확정된 계약: ① obs 이미지 키
<code>video.ego_view_pad_res256_freq20</code>(256²×3 uint8), ② env action은 flat 벡터가 아니라
<b>부위별 Dict</b>(left/right arm 7 + left/right hand 6 + waist 3 = 29ch) — 데이터셋 44d 레이아웃
(modality.json: arm7·hand6·<i>leg6</i>·<i>neck3</i>·arm7·hand6·<i>leg6</i>·waist3)에서 legs·neck은 데모 전 구간
정확히 0(stats 확인)이라 state는 zero-fill 44d 조립, action은 flat 44 → 5-슬라이스 Dict 분해, ③ 학습 프롬프트는
tasks 테이블 그대로 <b>"PnPCanToDrawerClose"</b>(자연어 지시문 아님 — 평가도 동일 문자열 사용), ④ 성공 판정은
<code>info["success"]</code>(프로그램적). rollout_client에 반영·커밋, openpi-client는 .venv-gr1에 설치 완료.</p>
<p><b>사고 ⑩ — 선제 차단 (23:35).</b> 파일럿 체크포인트 목적지가 기본값 <code>/home</code>(200G 중 잔여 1.4G,
100% 풀)이었고 save_interval=10k라 첫 저장(step 10k, ~01:58)에 ~30GB 쓰기 → ENOSPC로 13.7시간 학습이
중반에 죽을 운명이었다. 저장 전 안전 창에서 체크포인트 디렉토리를 /scratch(12T 여유)로 심링크 스왑해 무중단
우회 — 10k 저장 성공 여부는 워처가 감시하고, 저장되는 즉시 그 중간 체크포인트로 서버+클라이언트 E2E 스모크를
돌려 30k 완주 전에 평가 루프 전체를 검증한다.</p>
<h3>E2E 하네스 부팅 사다리 (08-10 새벽, 10k 중간 체크포인트)</h3>
<p>10k 저장은 /scratch 심링크를 통해 성공(사고 ⑩ 우회 검증). 이어 서버+클라이언트 전 루프 스모크를 부팅하며
사고 4건을 사다리식으로 규명·수정했다 — 전부 phase-1 본 평가 템플릿에 선반영:</p>
<table class='num'><tr><th>#</th><th>증상</th><th>원인</th><th>수정</th></tr>
<tr><td>⑪</td><td>서버 즉사: HF 404 (RepositoryNotFoundError)</td><td>잡에 HF_LEROBOT_HOME 부재 → norm stats 로딩이 로컬 데이터셋 대신 HF 원격 조회</td><td>학습 잡과 동일 env 2종 주입</td></tr>
<tr><td>⑫</td><td>_METADATA not found (…/params/params)</td><td>--checkpoint는 step 디렉토리를 받는데 /params까지 붙여 이중 경로</td><td>step 디렉토리로 통일</td></tr>
<tr><td>⑬</td><td>서버 기동됐는데 readiness 타임아웃</td><td>print 마커가 stdout 버퍼에 갇힘</td><td>python -u + websockets 로그 라인을 마커로</td></tr>
<tr><td>⑭</td><td>클라이언트 keepalive 1011 사망</td><td>첫 추론의 JIT 컴파일(수 분)이 ws 이벤트 루프를 독점 → ping 응답 불가</td><td>서버 기동 전 웜업 추론(커밋)</td></tr></table>
<p>부수 확인: 3090 풀에서 cuInit CUDA_ERROR_UNKNOWN 불량 노드 2대 조우(1대는 bad_nodes.txt 기존 등재 —
참조 누락 반성, 제외 목록 상시 적용으로 전환). 3090 24GB는 이 서빙 구성(π0.5 + 16후보 flow)에 구조적으로
빠듯함이 확인됐다: MEM_FRACTION 0.80→0.92에서 OOM 지점이 2.4GB→267MB 상수로 이동(거의 맞지만 부족) →
<b>A6000 48GB로 전환 + PREALLOCATE=false</b>.</p>
<p><b>✅ E2E 스모크 통과 (05:45, node25 A6000).</b> 서버 웜업(JIT) → 클라이언트 2트라이얼 완주(각 720스텝,
페어드 시드 5000/5001) → JSON 기록까지 전 루프 검증. 두 트라이얼 모두 실패는 10k(1/3 학습) 중간 체크포인트로선
예상 범위 — 성공률 판정은 30k phase-1에서. phase-1 템플릿은 A6000·24h·시드 분할(25×2잡/팔, 팔 간 동일 시드
페어링 유지)로 확정.</p>
<h3>20k 진단 평가 — 0/25와 판별 체계 (08-10 오전)</h3>
<p>30k를 기다리는 동안 20k 중간 체크포인트로 vla 25트라이얼 진단을 돌렸다(학습 곡선 + 성공 감지 배선 검증 목적).
그 과정에서 사고 2건 추가: <b>⑮</b> phase-1 스크립트 재작성 때 <code>unset LD_LIBRARY_PATH</code> 누락 →
miniconda libcrypto 오염으로 서버 즉사. <b>⑯</b>(교훈적) readiness 마커 "serving"이 서버 <i>트레이스백</i>의
"openpi.<b>serving</b>" 문자열에 오탐 매치 → SERVER_UP 오판, 클라이언트 1시간 헛대기. 성공 마커는 실패 출력과
절대 겹치지 않는 문자열이어야 한다(<code>policy on :</code>으로 교체).</p>
<p><b>결과: 0/25, 전 트라이얼 720스텝 최대치.</b> 해석 두 갈래 — (a) 20k 학습 부족, (b) 하네스 행동 실행 의미론
불일치. 조사: env 컨트롤러는 <code>control_delta=False</code>(절대 관절각), action_space 선언은 명목 [-1,1]인데
데모 액션 실측 범위는 ±3.0 rad — 데모가 같은 env에서 수집됐으므로 raw 라디안 통과가 정상일 공산이 크나 확정 필요.
판별 잡 2종 제출: <b>개루프 프로브</b>(데모 프레임 24개 → 예측 chunk vs 데모 액션 MSE, hold-still 기준선 대조 —
서빙 스택 건전성 확정) + <b>2트라이얼 비디오</b>(클라이언트에 --video-dir 추가·커밋, 실패 양상 기록용).
phase-1 제출은 프로브 판독 후 — 하네스 결함 상태로 4잡×25트라이얼을 태우지 않는다.</p>
<h3>근본 원인 규명 — 그리퍼 차원의 quantile 붕괴 (08-10 오후, 판별 사다리 완결)</h3>
<p>30k 완주 후에도 phase-1을 보류하고 판별을 끝까지 밀었다. 사다리: ① 개루프 프로브 — 데모 프레임에서
예측 MSE 1.82 > hold-still 0.88, 예측 절대최대 10.8(데모 ±3.0) → 어딘가 결함. ② 로더 프로브(학습 파이프라인
그대로) — 정규화 공간 MSE 1.5 vs zero-기준선 65,315로 "완벽해 보임". ③ 그러나 이 대비는 착시:
기준선 65,315는 <b>정규화가 ±1198로 폭발한 준상수 차원들</b>이 지배하는 수치였고, 그 쉬운 차원을 제외한
실 관절 차원의 오차는 개루프와 같은 수준. ④ A/B 프로브(같은 element에서 서빙 후보 경로 vs 직접
sample_actions) — 두 경로 일치(MSE 1.3–2.0), 즉 서빙 무죄·<b>모델 자체가 유효 차원을 못 배움</b>.</p>
<p><b>범인: 양손 마지막 관절(그리퍼 개폐, flat-44 dim 12/34).</b> 데모의 >99% 프레임에서 3.0에 파킹 →
q01=q99=2.9994(스팬 0)인데 실제 grasp 때 0↔3을 오간다. quantile 정규화 y=2(x−q01)/(스팬+1e-6)−1이
grasp 순간을 <b>수백만 배로 폭발</b>시켜 flow 손실을 지배 — 과제에서 가장 결정적인 차원(잡기)이 가장
병리적이었고, 정책은 손 개폐를 전혀 못 배웠다. 0/25가 완전히 설명된다.</p>
<p><b>수리·재발차단:</b> <code>slurm/repair_gr1_norm_stats.py</code>(커밋) — 스팬&lt;0.1인 차원의 q01/q99를
데이터셋 실측 min/max로 확장(그리퍼 0..3, waist-yaw ±0.1; 상수 0인 legs/neck은 무해하므로 불변).
부수: GR1Outputs가 2D를 가정해 3D 후보 배열에서 truncate가 무연산이 되는 버그도 수정(커밋).
<b>pilot-2 재학습 발차</b>(gigabyte_pro6000, 수리된 stats, 이번엔 체크포인트를 /scratch로 직접 지정 —
사고 ⑩ 재발 원천 차단). 다른 4개 태스크 데이터셋도 같은 수리를 적용할 것.</p>
<p><b>다음 순서:</b> pilot-2 학습 → 20k 중간 진단(개루프 MSE + 25트라이얼) → 30k phase-1 4잡
(vla/rand × 시드분할) → 유효 시 주석·critic·페어드. PR#4는 사용자 머지 대기.</p>
<h3>학습 자원 결정 — 이 클러스터 접고 node200/B200으로 (08-11)</h3>
<p><b>왜.</b> pi05(3B) GR1 파일럿을 이 클러스터 PRO6000(96GB)에서 반복 시도했으나, 노드들이 GPU를
<b>오버서브스크립션</b>(gres/gpu:1인데 물리 카드를 다른 잡과 공유)해서 batch16 워킹셋(~22GB)이 이미 74GB+
점유된 카드에 안 들어가 <b>공유-카드 OOM이 반복</b>됐다(node55/57 등). batch8로 낮추면 진입은 가능하나
언더트레이닝·불안정 우려. <b>사용자 판정(08-11): 학습은 여기서 하지 않는다</b> — 원래 옵션 A였던 node200/B200
(train_rlt.slurm·/data5, 기존 VLA 미세조정 인프라)으로 이관.</p>
<p><b>핸드오프 준비(전부 완료·리포 커밋).</b> ① config <code>pi05_gr1_rlt</code>(action_dim 48, 30k) 등록됨,
② <b>수리된 norm stats</b>(<code>assets/pi05_gr1_rlt/…/norm_stats.json</code> — 그리퍼 dim 12/34의 붕괴된
quantile을 데이터 실측 min/max로 확장, 사고록 참조) 커밋됨, ③ GR1 정책 변환(gr1_policy·weight_loaders shape-drop)
커밋됨, ④ 데이터 v3(PnPCanToDrawerClose 2.9G 등 5태스크) — node200 접근 스토리지로 복사 필요. ⑤ 체크포인트는
반드시 /scratch 등 여유 디스크로(사고 ⑩: /home 100% 풀 ENOSPC). <b>다음:</b> node200에서 학습 개시 → 완주 시
평가 하네스(이 리포에 커밋된 serve_bon_policy + rollout_client, A6000·PREALLOCATE=false E2E 검증 완료)로 phase-1.</p>
""",
)


# ============================================== 08-12 floq 구현·테스트·시각화
entry(
    "08-12",
    "floq",
    "floq — flow-matching critic 실증·시각화 (원본 충실 구현)",
    "완결",
    """
<p class='sub'>사용자 지시로 floq(arXiv:2509.06863, "Training Critics via Flow-Matching")를 <b>우리 critic에 직접
구현</b>하고 테스트·시각화했다. 배경: 워커A r56의 floq 리뷰(<span class='xref' data-eid='xworker-0808'>교차 워커 리뷰</span>)에서
파생 — "이득은 distributional이 아니라 capacity·plasticity"라는 해석을 우리 데이터·독립 구현으로 검증한다.</p>
<p><b>⑪ floq 직접 구현·테스트 (08-12, 사용자 지시 — 원본 충실).</b> 토이를 폐기하고 <b>우리 ARQ critic 아키텍처를
그대로 두고 bootstrap+head만 floq로 교체</b>: 같은 causal-transformer 트렁크, head를 velocity field로
(critic.py <code>flow_head</code>, non-breaking 커밋), 손실은 원본 Eq 4.2 <code>‖v_θ(t,z(t)|s,a)−(y−z)‖²</code>,
z(t)=(1−t)z+t·y, TD 타깃 y=r+γ^H·V_next는 <b>target velocity field의 K스텝 적분</b>으로 부트스트랩. collapse
방지 2종 이식: 초기 노이즈 <b>Uniform[−0.5,1]</b>(u=Q_max), interpolant z <b>categorical</b>+시간 t <b>Fourier</b>.
held-out(에피소드 분할):</p>
<table class='num'><tr><th>지표</th><th>스칼라 ARQ (expectile TD)</th><th>floq ARQ</th></tr>
<tr><td>Spearman(Q, mc) — value fit</td><td>0.395</td><td><b>0.521</b> (+32%)</td></tr>
<tr><td>action-sensitivity</td><td>0.024</td><td>0.071</td></tr>
<tr><td>demo_winrate (demo vs VLA후보, 낙관)</td><td>0.279</td><td>0.781</td></tr></table>
<p><b>판정 — capacity O, coverage X (강하게).</b> ① floq의 <b>용량 이득 확증</b>: 우리 실제 ARQ critic에서 value-fit
+0.13(+32% 상대), 토이보다 큼. ② 그러나 <b>배포 판별은 여전히 부재</b>: sensitivity 0.071은 γ-천장(γ=0.99·V~0.11·
Δt~16 → ΔQ상한≈<b>0.018</b>)을 넘어 <b>행동 스타일/분포 판별</b>(인공 마진)이지 참 가치 판별이 아니며, demo_winrate
0.78도 demo를 포함한 낙관치(배포 BoN엔 demo 없음). <b>결론: floq는 critic 용량·value-fit 업그레이드로 진짜
(TD-SF-ARQ head 채택 가치), 그러나 binding constraint인 coverage는 못 푼다 — 워커A의 floq 해석을 독립 구현에서
상호 재현.</b> (단순화: 단일-commit TD로 floq-vs-scalar만 격리; ARQ prefix/앙상블은 floq와 직교라 생략.)</p>
<p><b>flow 시각화.</b> Q를 어떻게 읽는지 — q(t)가 t=0의 노이즈(균등 [−0.5,1])에서 t=1의 Q값으로 velocity field를
따라 적분되는 <b>깔때기(funnel)</b>. 색 = 실제 mc_return. 서로 다른 노이즈에서 출발해도 같은 상태는 같은 Q로 수렴
(=velocity의 z-의존성, floq 용량의 물리적 원천). 대부분(낮은 mc)은 0 근처, 소수 고-mc는 0.75~0.85로 갈린다.
<a href="videos/floq/27_floq_flow.mp4" target="_blank">▶ 애니메이션</a></p>
<p><img src="videos/floq/27_floq_flow.png" alt="floq flow: q(t) integration funnel"></p>
<p class='sub'>곡률 0.0135로 작다 — rectified-flow는 궤적당 거의 직선이라(제 곡률 지표가 궤적 자기 직선 대비 이탈이라
사소) 극적 bending은 없고, 용량은 <b>노이즈 간 갈림(z-조건화)</b>에서 온다.</p>
<p><b>궤적 HUD 영상 (다봉 실험 스타일).</b> 성공한 held-out kroll 궤적을 따라 <b>왼쪽 로봇 ego 영상</b> +
<b>오른쪽 floq return 분포</b>(노이즈 256개 적분 샘플) + floq 평균과 스칼라 Q(점선) 대조.
<a href="videos/floq/28_floq_traj.mp4" target="_blank">▶ HUD 영상</a></p>
<p><img src="videos/floq/28_floq_traj.png" alt="floq trajectory HUD: robot + return distribution"></p>
<p class='sub'>판독: 분포는 대체로 <b>단봉·뾰족</b>(희소 이진 보상 → return이 거의 축퇴, 극적 다봉은 결과가
진짜 불확실할 때만). 프레임에 따라 floq 평균과 스칼라 Q가 크게 어긋나기도 한다(단순화 세팅의 스칼라 ARQ가
덜 캘리브레이션됨). floq은 점이 아니라 <b>분포</b>를 준다는 게 스칼라와의 질적 차이.</p>""",
)

# ============================================== 08-12 critic head 3종 + closed-loop BoN
entry(
    "08-12",
    "critic-heads",
    "critic head 3종 (scalar/HL-Gauss/floq) — 오프라인 랭킹 vs closed-loop BoN 판정",
    "완결",
    """
<p class='missing'><b>정정(2026-08-13):</b> 이 포스트의 "critic BoN이 VLA·랜덤보다 <b>유의하게 나쁘다</b>"는 판정은
<b>n=25 단일시드 노이즈</b>였다. 고통계력(6시드) 재검에서 critic은 VLA와 <b>동률</b>이다(못 이기고 안 해침).
<span class='xref' data-eid='deas'>DEAS 재현·정정</span> 참조.</p>
<p class='sub'>사용자 지시: floq을 우리 critic에 넣은 뒤 "이득이 <b>flow 메커니즘</b> 덕이냐 <b>categorical 표현</b>
덕이냐"를 가르고, 나아가 "<b>실제 evaluation으로 이 critic이 VLA를 향상시키는지</b>"를 closed-loop로 판정한다.
<span class='xref' data-eid='floq'>floq 구현 포스트</span>의 직접 후속.</p>

<h3>설계 — 같은 트렁크, head/loss만 3종 (method-only-diff)</h3>
<p>우리 표준 <b>AQC critic</b>(ARQCritic: [obs 토큰, action-macro 토큰] 위 causal transformer, macro_group_size=2,
full-chunk prefix로 스코어)의 <b>트렁크를 한 줄도 안 바꾸고</b> 출력 head와 학습 손실만 셋으로 바꿔 나란히 학습한다:</p>
<table class='num'><tr><th>head</th><th>표현</th><th>학습 손실</th><th>Q 읽기</th></tr>
<tr><td><b>scalar</b></td><td>스칼라 1개</td><td>expectile-TD(0.9) 회귀</td><td>그 스칼라</td></tr>
<tr><td><b>HL-Gauss</b></td><td>51-atom 로짓</td><td>Gaussian-smoothed 타깃에 cross-entropy(분류)</td><td>Σ softmax·centers</td></tr>
<tr><td><b>floq</b></td><td>속도장 v(t,z|s,a)</td><td>flow-matching(Eq 4.2)</td><td>노이즈→K스텝 적분</td></tr></table>
<p class='sub'>이 세 개가 <b>회귀 → 분류 → flow</b>의 사다리다. HL-Gauss가 결정적 중간항: floq의 붕괴방지도 z를
categorical로 인코딩하니, "같은 51-atom 분류지만 flow 없음"인 HL-Gauss를 끼우면 <b>이득의 출처가 categorical이냐
flow냐</b>가 갈린다. 공통 설정: γ=0.997(유효 지평 확대), 부트스트랩은 <b>단일 샘플 적분</b>(원본 floq 충실 — 평균은
분포를 뭉갠다), PrepareCoffee mixed annotation, held-out은 에피소드 분할. 각 head 30k step.</p>

<h3>① 오프라인 — head가 액션을 랭킹하는가</h3>
<table class='num'><tr><th>지표</th><th>scalar (회귀)</th><th>HL-Gauss (분류)</th><th>floq (flow)</th></tr>
<tr><td>Spearman(Q, mc) — value fit</td><td>0.398</td><td><b>0.594</b></td><td>0.518</td></tr>
<tr><td>action-sensitivity</td><td>0.018</td><td>0.011</td><td><b>0.281</b></td></tr>
<tr><td>demo_winrate (데모 vs VLA후보, 낙관)</td><td>0.226</td><td><b>0.905</b></td><td>0.851</td></tr></table>
<p><b>오프라인 판독.</b> ① <b>이득의 대부분은 categorical 표현</b>이다: HL-Gauss가 winrate 0.23→0.91, Spearman
0.40→0.59로 <b>flow 없이도 floq만큼(오히려 더) 잘 랭킹</b>한다. floq이 scalar를 이긴 건 flow 메커니즘이 아니라
categorical head 덕이 컸다. ② floq의 sensitivity 0.281은 <b>γ-천장</b>(γ=0.99·V~0.11·Δt~16 → 참 값차 상한
ΔQ≈<b>0.018</b>, 워커A 교정)을 <b>14배 초과</b> — 단일-샘플 부트스트랩이 액션 민감도를 키웠지만 이건 참 가치가 아니라
<b>행동 스타일 판별 아티팩트</b>다. winrate 0.85~0.91도 데모를 후보 풀에 포함한 낙관치(배포 BoN엔 데모 없음).</p>

<h3>② closed-loop — BoN이 실제로 VLA를 이기는가 (판정: 아니오)</h3>
<p>저장한 세 critic을 <b>실제 BoN 배포</b>에 꽂는다: PrepareCoffee 25장면, 매 replan마다 VLA가 <b>N=8 후보</b>를
샘플→critic이 스코어→argmax 후보 실행. 대조군 둘 — <b>VLA baseline</b>(후보 0 그대로 실행) + <b>rand null</b>
(랜덤 후보 실행). rand가 핵심 대조다: BoN이 VLA는 이겨도 rand를 못 이기면 "다시 뽑은 게 도움"일 뿐 critic이
고른 게 아니다. run_trials가 장면을 (seed,trial)로 고정하므로 <b>모든 모드가 동일 25장면</b>을 본다(scene-paired).</p>
<table class='num'><tr><th>모드</th><th>성공</th><th>성공률</th><th>Wilson 95% CI</th><th>vs VLA (paired McNemar: 승/패)</th></tr>
<tr><td><b>VLA (baseline)</b></td><td>20/25</td><td><b>0.80</b></td><td>[0.61, 0.91]</td><td>—</td></tr>
<tr><td>rand (null)</td><td>17/25</td><td>0.68</td><td>[0.48, 0.83]</td><td>3승 / 6패</td></tr>
<tr><td>scalar BoN</td><td>10/25</td><td>0.40</td><td>[0.23, 0.59]</td><td><b>0승 / 10패</b></td></tr>
<tr><td>HL-Gauss BoN</td><td>16/25</td><td>0.64</td><td>[0.45, 0.80]</td><td>3승 / 7패</td></tr>
<tr><td>floq BoN</td><td>11/25</td><td>0.44</td><td>[0.27, 0.63]</td><td><b>1승 / 10패</b></td></tr></table>
<p><img src="videos/critic-heads/29_bon_compare.png" alt="offline demo-winrate vs closed-loop BoN success"></p>
<p class='sub'>왼쪽: 오프라인 winrate(HL-Gauss·floq 0.91·0.85 압승). 오른쪽: closed-loop 성공률 — 검은 점선이 VLA
기준. 색: 회색=scalar, 초록=HL-Gauss, 빨강=floq, 검정=VLA, 회색(rand)=null. 오차막대는 Wilson 95%.</p>

<h3>판정</h3>
<p><b>어느 critic head도 BoN으로 VLA를 향상시키지 못한다.</b></p>
<p>① <b>scalar·floq은 랜덤 null보다도 나쁘다</b>(0.40·0.44 &lt; rand 0.68). paired McNemar로 scalar는 VLA 대비
<b>0승 10패</b>, floq은 <b>1승 10패</b> — 통계적으로 유의하게 유해하다. critic의 argmax가 랜덤보다 <b>더 나쁜</b>
후보를 고른다는 뜻: 후보 N개 중 critic이 <b>가장 과대평가</b>한 것을 뽑는데, 그 과대평가는 참 가치가 아니라
추정오차·off-distribution과 상관한다(<b>승자의 저주</b>=estimation-error exploitation, <span class='xref'
data-eid='conservatism'>2축 보수화</span>의 정확히 그 실패). 후보들이 다 웬만해서 rand는 0.68에 그치는데,
argmax는 그중 최악을 짚어 더 떨어진다.</p>
<p>② <b>오프라인 winrate가 closed-loop를 완전히 잘못 예측했다.</b> HL-Gauss winrate 0.905인데 BoN 0.64. 이것은
<span class='xref' data-eid='embed-compare'>임베딩-DiT 뒤집힘</span>(오프라인 BC/디코더는 "표현 충분"이라 했으나
closed-loop이 뒤집음)에 이은 <b>두 번째 "오프라인→closed-loop 불일치"</b>다. 이유: winrate는 <b>in-dist 데모 상태</b>에서
데모 vs 후보를 재지만, 배포는 <b>VLA가 방문한 상태에서 VLA 자기 후보</b>를 argmax한다 — 분포가 다르고, argmax가
오차를 증폭한다. 오프라인 랭킹 지표는 배포 성공의 신뢰할 대리가 아니다.</p>
<p>③ <b>head 층위 결론</b>: categorical(HL-Gauss)이 flow보다 싸고 오프라인·closed-loop 모두 낫다(BoN 0.64로 가장 덜
나쁨, 유일하게 rand와 비슷). floq의 flow는 sensitivity만 γ-천장 14배로 폭증시켜 <b>스타일 아티팩트로 후보를 갈라
오히려 유해</b>했다. → <span class='xref' data-eid='tdsf-arq'>TD-SF-ARQ</span> head는 <b>HL-Gauss</b>가 옳다(floq flow 아님).</p>
<p>④ <b>근본 재확인</b>: 문제는 head가 아니라 <b>배포 방식과 데이터</b>다. demo-only critic을 자기 후보 argmax(BoN)로
쓰는 건 보수화 없이는 승자의 저주에 진다. 어떤 head도 <b>coverage(반사실 후보)</b> 없이는 못 구한다 —
<span class='xref' data-eid='model-based'>후보축 학습신호 부재</span>·<span class='xref' data-eid='conservatism'>보수화 2축</span>과
같은 결론에 독립 경로로 재도달했다.</p>

<h3>한계 (정직)</h3>
<p>n=25 단일 시드라 <b>잠정</b>이다. VLA가 이미 0.80으로 강해 BoN <b>상승 여지가 작고 하락 여지는 큰</b> 천장 효과가
있다(더 약한 baseline 과제에서 재검 필요). full-chunk commit만 썼다(ARQ prefix 선택 미사용). HL-Gauss CI[0.45,0.80]는
VLA CI[0.61,0.91]와 겹쳐 "<b>HL-Gauss &lt; VLA</b>"는 미확정(McNemar p≈0.34) — <b>확정된 것은 scalar·floq의 유해성</b>이다.</p>

<h3>참고 — flow 시각화 (γ=0.997, 단일-샘플 부트스트랩)</h3>
<p>같은 재학습의 floq flow. γ를 0.99→0.997로 올리자 flow <b>곡률이 0.0135→0.0317</b>로 커졌다(유효 지평이 늘어 velocity
경로가 더 휨). 궤적 HUD: 왼쪽 로봇 ego + 오른쪽 floq return 분포(노이즈 256 적분). 단일-샘플 부트스트랩이라 결과가
불확실한 상태에서 분포가 더 벌어진다.
<a href="videos/critic-heads/28_floq_traj.mp4" target="_blank">▶ HUD 영상</a> ·
<a href="videos/critic-heads/27_floq_flow.mp4" target="_blank">▶ funnel 영상</a></p>
<p><img src="videos/critic-heads/28_floq_traj.png" alt="floq HUD gamma=0.997 single-sample bootstrap"></p>

<p class='sub'>재현: 오프라인 3-way <code>probes/floq_critic.py</code>(HL-Gauss head 추가), closed-loop
<code>probes/eval_bon.py</code>(critic 저장→VLA BoN 롤아웃), 그림 <code>probes/plot_bon.py</code>(JSON→figure).
결과 JSON: <code>floq_critic.json</code>·<code>bon_critic_compare.json</code>. 모두 커밋.</p>""",
)

# ============================================== 08-12 per-prefix td-max 재검
entry(
    "08-12",
    "critic-pfx",
    "부트스트랩 교정 — per-prefix TD-max + joint argmax로도 critic은 VLA를 못 이긴다",
    "완결",
    """
<p class='missing'><b>정정(2026-08-13):</b> 이 포스트의 "critic이 VLA·랜덤보다 <b>유의하게 나쁘다</b>(floq 2승10패 등)"는
<b>n=25 단일시드 노이즈</b>였다. 고통계력(6시드)에선 critic이 VLA와 <b>동률</b>이고 td-max ≈ DEAS다.
<span class='xref' data-eid='deas'>DEAS 재현·정정</span> 참조.</p>
<p class='sub'>바로 앞 <span class='xref' data-eid='critic-heads'>critic head 3종 비교</span>의 결함을 교정한 재검.
사용자 지적: "TD를 할 거면 데이터셋 액션을 샘플해 <b>max</b> 취해 부트스트랩해야 하고, critic은 per-prefix로
Q를 내야 한다." 앞 실험의 부트스트랩은 <b>데모의 다음 액션</b>으로 값을 이었는데(SARSA식), 그건 데모 정책의
값이지 optimal이 아니다. 이번엔 그 둘을 고쳐 결론이 바뀌는지 본다.</p>

<h3>무엇을 고쳤나 (프로덕션 트레이너 targets() 충실 재현)</h3>
<ul>
<li><b>부트스트랩 = TD-max over 후보</b>: 착지 상태에서 저장된 VLA 후보(base_action 16개 중 8개)에 대해
<b>V(s′)=max_j Q(s′, cand_j)</b>. 오프라인에서 max_a Q를 세우는 유일한 방법 — 상태당 데모가 하나뿐이라
"여러 액션"이 필요하고, 그 재료가 후보 풀이다.</li>
<li><b>per-prefix native</b>: prefix p(커밋 길이)마다 <b>y_p = Σ_{i&lt;p}γ^i r + γ^p·(1−ended_p)·V_next(착지_p)</b>,
mc 하한. floq은 prefix마다 적분해 Q를 읽는다.</li>
<li><b>배포 = (후보 × prefix) joint argmax</b>: 후보와 커밋 길이 n_exec를 동시에 고름(실제 AQC 배포 규칙).
null 대조에 <b>randh</b>(랜덤 후보 + 랜덤 prefix — joint argmax의 정직한 null) 추가.</li>
</ul>
<p class='sub'>세 head(scalar/HL-Gauss/floq)를 <b>같은 td-max 타깃</b>에 학습(표현만 차이). γ=0.997, 단일 critic(앙상블
없음, 명시적 단순화). PrepareCoffee, N=8, 25장면 scene-paired, Wilson 95% + paired McNemar.</p>

<h3>결과 — 여전히 아무 critic도 VLA를 못 이긴다</h3>
<table class='num'><tr><th>모드</th><th>성공</th><th>성공률</th><th>Wilson 95%</th><th>vs VLA (승/패)</th></tr>
<tr><td><b>VLA (baseline)</b></td><td>14/25</td><td><b>0.56</b></td><td>[0.37, 0.73]</td><td>—</td></tr>
<tr><td>rand (null)</td><td>17/25</td><td><b>0.68</b></td><td>[0.48, 0.83]</td><td>6승 / 3패 (+0.12)</td></tr>
<tr><td>randh (joint null)</td><td>13/25</td><td>0.52</td><td>[0.33, 0.70]</td><td>4승 / 5패</td></tr>
<tr><td>scalar BoN</td><td>9/25</td><td>0.36</td><td>[0.20, 0.55]</td><td>3승 / 8패 (−0.20)</td></tr>
<tr><td>HL-Gauss BoN</td><td>11/25</td><td>0.44</td><td>[0.27, 0.63]</td><td>3승 / 6패 (−0.12)</td></tr>
<tr><td>floq BoN</td><td>6/25</td><td><b>0.24</b></td><td>[0.11, 0.43]</td><td>2승 / <b>10패</b> (−0.32)</td></tr></table>
<p><img src="videos/critic-pfx/30_pfx_bon.png" alt="per-prefix td-max joint-argmax BoN success"></p>

<h3>판정</h3>
<p>① <b>부트스트랩·배포 규칙을 다 고쳐도 결론 불변.</b> scalar·HL-Gauss·floq 셋 다 VLA 아래, floq 최악(0.24).
<b>rand(0.68)이 critic들보다 높다</b> — "다시 뽑기"는 돕는데 "critic argmax"는 해친다. td-max·per-prefix로도
<b>승자의 저주</b>가 그대로다.</p>
<p>② <b>왜 안 바뀌나 — binding constraint는 head도 배포규칙도 아닌 coverage.</b> td-max가 max를 취해도, 후보 16개가
모두 <b>같은 VLA가 뽑은 near-demo</b>라 데모에서 크게 벗어난 <b>반사실 행동이 안 만들어진다.</b> 그래서 max는
"비슷비슷한 후보 중 critic이 가장 과대평가한 것"을 고르는 것으로 귀결 — 이건 정확히
<span class='xref' data-eid='model-based'>후보축 학습신호 부재</span>·<span class='xref' data-eid='conservatism'>보수화 2축</span>이
말한 실패다. <b>두 독립 배포 설계(데모-부트스트랩·full-chunk / td-max·per-prefix joint)에서 같은 음성 결론에 수렴</b>한다.</p>

<h3>정직한 경고 — n=25 단일시드는 과소검정</h3>
<p>VLA baseline이 앞 실험 <b>0.80 → 이번 0.56</b>으로 크게 흔들린다(같은 seed·N인데). 긴 지평 sim + bf16 VLA 추론의
노드 간 수치차가 borderline 장면을 뒤집는 것으로 보인다. 즉 <b>절대 성공률은 n=25에서 불안정</b>하고, 신뢰할
신호는 <b>런 내부 paired 방향</b>(critic argmax &lt; 재샘플 &lt; VLA, McNemar)이다. 확정 판정은 <b>multi-seed
run-level CI</b>로 넘긴다(후속). single critic(앙상블 없음)도 단순화 — 앙상블-min의 비관은 승자의 저주를
줄이는 직교 축이라 다음 후보다.</p>

<h3>그래서 다음</h3>
<p>head(HL-Gauss)도, 배포(joint argmax)도, 부트스트랩(td-max)도 coverage를 못 만든다. 남은 정면 승부는
<b>training-time에 후보축을 누르는 CalQL식 보수화</b>(반사실 후보를 데모 아래로)와 <b>on-policy 반사실 제조</b>다 —
<span class='xref' data-eid='conservatism'>보수화</span>·<span class='xref' data-eid='calql'>CalQL</span> 경로로 재수렴.</p>
<p class='sub'>재현: <code>probes/eval_bon_pfx.py</code>(per-prefix td-max 학습→VLA joint-argmax 롤아웃),
그림 <code>probes/plot_pfx.py</code>. 결과 <code>bon_pfx_compare.json</code> 커밋.</p>""",
)

# ============================================== 08-13 DEAS 방법론 — critic 부트스트랩 정정
entry(
    "08-13",
    "deas",
    "DEAS 재현·정정 — critic은 VLA와 동률(못 이기고 안 해침), 지난 'critic 유해' 판정은 n=25 노이즈였다",
    "완결",
    """
<p class='sub'><b>워커A에게 알림 + 우리 앞선 판정의 정정.</b> <span class='xref' data-eid='critic-heads'>critic-heads</span>·
<span class='xref' data-eid='critic-pfx'>critic-pfx</span>에서 "어느 critic도 VLA를 BoN으로 못 이긴다 → binding constraint는
coverage"라고 판정했다. 사용자가 이 결론을 의심했다("cand[0]도 VLA 샘플인데 argmax가 그보다 나쁠 리 없다").
문헌을 뒤진 결과 — <b>우리 결론은 부분적으로 틀렸을 수 있다. 원인은 coverage가 아니라 우리가 고른 부트스트랩
연산자(td-max)였다.</b></p>

<h3>DEAS (arXiv:2510.07730) — 우리와 같은 도메인이 이 문제를 정확히 지적</h3>
<p>DEAS("DEtached value learning with Action Sequence", Changyeon Kim·Younggyo Seo·Kimin Lee·Yuke Zhu)는
<b>VLA + RoboCasa Kitchen + action sequence(청크) + distributional value + BoN</b> — 우리 스택과 판박이다. 원문:</p>
<blockquote class='sub'>"directly adopting such sequences in actor-critic algorithms introduces <b>excessive value
overestimation</b>, which we address through <b>detached value learning that steers value estimates toward
in-distribution actions</b> that achieve high return in the offline dataset."</blockquote>
<p><b>우리 critic-pfx의 부트스트랩 `V_next = max_j Q(s′, cand_j)`(td-max)가 정확히 이 "excessive value
overestimation"이다.</b> 근사-데모 후보 8~16개 중 critic이 <b>가장 과대평가한</b> 것을 부트스트랩에 넣으니 값이
부풀고, 배포 argmax가 그 부푼 후보를 골라 승자의 저주가 심해진다. DEAS는 <b>max를 버리고</b> expectile 상태가치 V로
부트스트랩해 이를 억제한다. 그리고 <b>DEAS는 RoboCasa에서 GR00T baseline을 이긴다</b>(예: PnPCounterToMicrowave
~45%→~65%) — critic BoN이 VLA를 이길 수 있음을 같은 도메인에서 보인다.</p>

<h3>DEAS 방법론 (원본 코드 <code>deas_critic.py</code> 실측)</h3>
<p><b>① V 손실 = expectile + HL-Gauss(분류) 병용</b> — expectile은 손실에 곱하는 스칼라 가중이라 categorical CE와 공존한다:</p>
<table class='num'><tr><td><code>q_demo = min(Q1_tgt, Q2_tgt)(s, a_demo)</code>  # in-distribution 데모 액션, detached</td></tr>
<tr><td><code>g = where(q_demo &gt;= V, τ, 1−τ)</code>  ;  <code>L_V = mean( g · CE(V_logits, HLGauss(q_demo)) )</code></td></tr></table>
<p><b>② Q 손실 = V로 부트스트랩(후보 max 아님)</b> — dual discount γ1(청크 내 보상 합)·γ2(청크 간):</p>
<table class='num'><tr><td><code>target = Σ_i γ1^i·r_i + γ2^(nH)·(1−done)·V(s′)</code>  ;  <code>L_Q = (HLGauss_CE(Q1,target)+HLGauss_CE(Q2,target))/2</code></td></tr></table>
<p>double critic min + EMA target. 배포 BoN: <code>score=min(Q1,Q2)(z,cand)</code>, argmax.</p>

<h3>결과 (1) — 방법론만 이식, 백본은 우리 것 (DEAS GR00T 값 그대로)</h3>
<p>원본 Isaac-GR00T 스택 대신 <b>우리 pi05 백본·mixed 주석·AQC 트렁크에 DEAS 방법론만</b> 이식(<code>probes/eval_deas.py</code>):
V=HL-Gauss+expectile, Q는 V로 부트스트랩(td-max 폐기), double-min, <b>dual-discount γ1=0.9/γ2=0.99, negative-reward
(−1/스텝, support [−100,0]), τ=0.7</b> — DEAS의 RoboCasa 재현 명령 값 그대로. <b>PrepareCoffee 단일태스크</b>, N=10 argmax.</p>
<p><b>축1 — DEAS 고정, head 스윕 (n=25 잠정):</b> scalar 0.52 / <b>HL-Gauss 0.64 = VLA 0.64 (동률)</b> / floq 0.36.
HL-Gauss가 최선. floq은 support [−100,0]에서 flow 속도 폭발로 미수렴(q_loss 58.8)이었는데, <b>값을 [−1,0]으로 정규화하니
완전 수렴(q_loss 0.008)</b> — 스케일 문제였을 뿐(사용자 지적). 정규화 후에도 n=25 BoN 0.40으로 랭킹은 HL-Gauss만 못하나 n=25라 불확정.</p>

<h3>결과 (2) — 우리 td-max vs DEAS, 같은 장면 고통계력 판정</h3>
<p>n=25 단일시드가 런마다 0.52~0.64로 흔들려(같은 critic·다른 노드) 판정 불가였다. 그래서 <b>HL-Gauss 고정, 부트스트랩만
바꿔</b>(td-max = 착지 후보 max / DEAS = expectile-V) <b>6시드 × 25 = 150 trial/arm</b>, 같은 장면 paired, run-level t-CI:</p>
<table class='num'><tr><th>arm</th><th>부트스트랩</th><th>run-level 성공률</th><th>±95% t-CI</th></tr>
<tr><td>VLA (baseline)</td><td>—</td><td>0.640</td><td>±0.084</td></tr>
<tr><td>rand (null)</td><td>—</td><td>0.553</td><td>±0.062</td></tr>
<tr><td><b>td-max (우리)</b></td><td>max_j Q(s′,cand_j)</td><td><b>0.660</b></td><td>±0.115</td></tr>
<tr><td><b>DEAS</b></td><td>expectile-V</td><td><b>0.633</b></td><td>±0.097</td></tr></table>
<table class='num'><tr><th>paired Δ̄ (시드별 차)</th><th>값</th><th>95% t-CI</th><th>유의?</th></tr>
<tr><td>td-max − VLA</td><td>+0.02</td><td>±0.10</td><td>아니오 (0 포함)</td></tr>
<tr><td>DEAS − VLA</td><td>−0.01</td><td>±0.13</td><td>아니오</td></tr>
<tr><td>DEAS − td-max</td><td>−0.03</td><td>±0.14</td><td>아니오</td></tr></table>
<p><img src="videos/deas/31_runlevel_cmp.png" alt="run-level 6-seed comparison vla/rand/tdmax/deas"></p>
<p class='sub'>막대=run-level 평균±95% t-CI, 점=시드별 성공률(그 넓은 산포가 n=25 단일시드 판정을 못 믿게 한 이유).</p>

<h3>판정 — 셋 다 VLA와 통계적으로 구별 안 됨 (지난 판정 정정 포함)</h3>
<p>① <b>critic(td-max·DEAS)은 VLA와 동률 — 해치지 않는다.</b> ② <b>이기지도 못한다</b>(Δ̄≈0, CI가 0 포함).
③ <b>td-max ≈ DEAS</b> — 부트스트랩 연산자(max vs expectile-V)가 여기선 차이 없다. 내 "td-max가 THE 문제"도,
"DEAS가 고친다"도 <b>둘 다 지지 안 됨</b>. ④ <b>rand(0.55)만 살짝 아래</b> — critic은 랜덤보단 잘 고르나(0.66>0.55)
VLA 자기 top 샘플을 못 넘는다.</p>
<p><b>중요 정정.</b> <span class='xref' data-eid='critic-heads'>critic-heads</span>·<span class='xref' data-eid='critic-pfx'>critic-pfx</span>의
"critic이 VLA·랜덤보다 <b>유의하게 나쁘다</b>(McNemar 0/10 등)"는 <b>n=25 단일시드 노이즈 아티팩트</b>였다. 고통계력에선
좋은 head(HL-Gauss)+double-min이면 VLA 동률이다. 두 포스트에 정정 배너를 달았다.</p>
<p>⑤ <b>coverage 재확인.</b> 단일태스크 PrepareCoffee(VLA 이미 0.64)에선 BoN 상승 여지가 작다. DEAS가 GR00T에서
이긴 건 <b>24태스크 다양성 = 넓은 coverage</b> 덕이고, 우리 <b>단일태스크 near-demo</b>와 다르다 — 우리가 계속 말한
binding constraint=coverage와 정합.</p>
<p class='sub'><b>메타 교훈.</b> closed-loop 판정을 n=25 단일시드로 다섯 번 돌려 다 노이즈였다. 앞으로 판정은 처음부터
<b>run-level 다중시드</b>로. 재현: <code>probes/eval_deas.py</code>(DEAS 방법론)·<code>eval_compare.py</code>(td-max vs DEAS 고통계력)·
<code>plot_cmp.py</code>. 결과 JSON 커밋. DEAS 코드 <code>github.com/csmile-1006/DEAS-Isaac-GR00T</code> 정독.</p>""",
)

# ============================================== 08-13 실험 보드 (living)
entry(
    "08-13",
    "exp-board",
    "🧭 실험 보드 — 계획 / 진행중 / 완료 (담당·wandb·리포트)",
    "살아있음",
    """
<p class='sub'>실험이 흩어져 까먹거나(뭐 하려다 말고), 돌려놓고 정리가 안 되는 걸 막는 <b>living 보드</b>.
계획 → 진행중 → 완료로 흐른다. <b>갱신 규칙</b>: 실험을 제출/완료할 때 같은 사이클에 이 보드를 갱신한다
(<code>space_add_entry.py</code>로 같은 eid replace). 리포트 링크는 칩을 누르면 이동.</p>

<h3>🟡 진행중 (Running)</h3>
<table class='num'><tr><th>실험</th><th>담당</th><th>메모</th><th>wandb</th><th>리포트</th></tr>
<tr><td colspan="5" class="pending">현재 진행중인 잡 없음 (직전 배치 전부 완료). 아래 계획에서 다음 착수.</td></tr></table>

<h3>🔵 계획 (Planned) — 다음 후보</h3>
<table class='num'><tr><th>실험</th><th>담당</th><th>메모</th><th>리포트</th></tr>
<tr><td><b>MVE 게이트</b> — 수정구슬이 후보를 가르나</td><td>B</td><td>워커A <code>train_mve_critic</code> 게이트: 후보 판별 + disagreement 보수화 확인 → 통과 시 run-level 롤아웃</td><td><span class='xref' data-eid='mb-arq'>mb-arq</span></td></tr>
<tr><td><b>N-스윕 + λ-weighted min/max</b></td><td>B</td><td>EMaQ 지침: 작은 N(5) 재검, λ로 min쪽 보수화 (우린 8~10 고정이었음)</td><td><span class='xref' data-eid='deas'>deas</span></td></tr>
<tr><td><b>TD-SF-ARQ 설계</b> (phase-2 사전등록)</td><td>B</td><td>head=HL-Gauss+expectile-V 확정, 벽=coverage — 사전등록 문서화</td><td><span class='xref' data-eid='tdsf-arq'>tdsf-arq</span></td></tr>
<tr><td><b>on-policy 반사실 제조</b></td><td>B</td><td>근본 coverage 해결 — sim 롤아웃으로 counterfactual 데이터 생성</td><td><span class='xref' data-eid='conservatism'>conservatism</span></td></tr>
<tr><td><b>GR1 이전</b> (pi05 학습)</td><td>B→node200</td><td>node200/B200 외부 실행 대기, handoff·norm-stats·config 커밋 완료</td><td><span class='xref' data-eid='gr1-port'>gr1-port</span></td></tr></table>

<h3>🟢 완료 (Done) — 워커B</h3>
<table class='num'><tr><th>실험</th><th>상태</th><th>핵심 결과</th><th>wandb</th><th>리포트</th></tr>
<tr><td>DEAS 재현 + 고통계력</td><td>완결</td><td>critic=VLA <b>동률</b>(못 이기고 안 해침), n=25 판정은 노이즈였음</td><td>offline</td><td><span class='xref' data-eid='deas'>deas</span></td></tr>
<tr><td>critic head 3종 (scalar/HLG/floq)</td><td>완결(정정)</td><td>categorical이 이득 대부분 — 단 closed-loop은 n=25 노이즈</td><td>offline</td><td><span class='xref' data-eid='critic-heads'>critic-heads</span></td></tr>
<tr><td>per-prefix td-max + joint argmax</td><td>완결(정정)</td><td>연산자 바꿔도 동일 — 노이즈</td><td>offline</td><td><span class='xref' data-eid='critic-pfx'>critic-pfx</span></td></tr>
<tr><td>floq (flow-matching critic)</td><td>완결</td><td>용량 O·커버리지 X; 값 [−1,0] 정규화로 수렴 복구</td><td>offline</td><td><span class='xref' data-eid='floq'>floq</span></td></tr>
<tr><td>임베딩 비교 + DiT probe policy</td><td>완결</td><td>오프라인 지표가 closed-loop를 못 예측(뒤집힘)</td><td>offline</td><td><span class='xref' data-eid='embed-compare'>embed-compare</span></td></tr>
<tr><td>model-based 게이트</td><td>진행중(shelved)</td><td>후보축 행동정보 +7.3% — 약함</td><td>offline</td><td><span class='xref' data-eid='model-based'>model-based</span></td></tr>
<tr><td>horizon-decisive 진단</td><td>완결(shelved)</td><td>임베딩 결함 vs 신호부재 판별 환경</td><td>offline</td><td><span class='xref' data-eid='horizon-probe'>horizon-probe</span></td></tr>
<tr><td>보수화 2축 종합</td><td>살아있음</td><td>분포이동 vs 추정오차 exploit — EMaQ와 정합</td><td>—</td><td><span class='xref' data-eid='conservatism'>conservatism</span></td></tr></table>
<p class='sub'>더 이른 사다리(v11/v12/final/aqc/families/…)는 리포트 목록·마인드맵 참조.</p>

<h3>🟣 워커A 실험 (참고 — 워커A 게시·갱신)</h3>
<table class='num'><tr><th>실험</th><th>담당</th><th>메모</th><th>wandb</th><th>리포트</th></tr>
<tr><td>patch-critic (동결 DINOv2 + 분포형 ARQ IQL)</td><td>A</td><td>cost-to-goal reward, adaptive-K/BoN 배포</td><td>acrft / patch-critic</td><td><span class='xref' data-eid='wa-patchcritic-method'>patch-critic</span></td></tr>
<tr><td>MVE critic / cheap-z dynamics</td><td>A</td><td>model-based value expansion + 5-멤버 앙상블 동역학</td><td>—</td><td>코드 <code>train_mve_critic.py</code></td></tr>
<tr><td>EMaQ 논문 리뷰</td><td>A</td><td>BoN=Bellman연산자, 큰 N이 critic오차 exploit</td><td>—</td><td><span class='xref' data-eid='wa-emaq-bon'>EMaQ</span></td></tr>
<tr><td>policy server / value-guided serving</td><td>A</td><td>server-side BoN, HUD provenance</td><td>—</td><td>코드 <code>serve_policy.py</code></td></tr></table>

<p class='sub'><b>진짜 "실험 탭"</b>(템플릿 레벨의 별도 탭)은 워커A 고정 <code>index.html</code> 구조라 <b>워커A와 조율 필요</b> —
제안: 공유 <code>experiments.json</code> + 탭 렌더. 이 보드는 그 전까지의 living 대체물이며, 두 워커가 각자 행을 갱신한다.</p>""",
)

# ============================================== 08-13 model-based ARQ 쉬운 설명
entry(
    "08-13",
    "mb-arq",
    "쉽게 풀어쓴 model-based critic — '상상하는 심판'은 VLA를 이길 수 있나",
    "살아있음",
    """
<p class='sub'>이 글은 수식·용어 없이 <b>model-based critic(=model-based ARQ, MVE)</b> 하나만 처음부터 끝까지
비유로 설명한다. 우리가 왜 이걸 시도하는지, 무엇이고, 될 것 같은지·안 될 것 같은지까지.</p>

<h3>0. 무대 — 로봇, 후보, 심판</h3>
<p>우리 로봇 정책(<b>VLA</b>)은 매 순간 "다음에 이렇게 움직일게" 하는 <b>동작 후보를 여러 개</b> 제안한다(예: 8~16개).
같은 상황을 조금씩 다르게 실행하는 변주들이다. 우리는 그중 <b>제일 좋은 걸 골라줄 심판(critic)</b>을 원한다 —
이 "여러 개 뽑아 best 고르기"를 <b>Best-of-N(BoN)</b>이라 부른다. 심판이 잘 고르면 로봇이 더 잘하게 된다.</p>

<h3>1. 오늘 밤 우리가 확인한 것 — 심판이 로봇을 못 이긴다</h3>
<p>여러 방식의 심판을 만들어 붙여봤는데(<span class='xref' data-eid='deas'>DEAS 재현·정정</span>), <b>제대로 재보니
심판을 붙인 성공률이 그냥 로봇(첫 후보 실행)과 똑같았다.</b> 왜 못 이길까? 이유가 둘이다:</p>
<ul>
<li><b>후보가 다 비슷하다</b>: 같은 로봇이 뽑은 변주라 서로 엇비슷하다. 그중 "미세하게 나은 것"을 가려야 하는데 재료가 빈약하다.</li>
<li><b>심판이 대안을 본 적이 없다</b>: 학습 데이터(사람 시연)엔 매 상황에 <b>실제로 한 동작</b>만 있고, "다른 동작을 했으면
어땠을까"는 없다. 그래서 심판은 안 해본 후보에 대해 <b>사실상 찍는다</b>. 게다가 찍을 때 하필 <b>자기가 과대평가한
엉뚱한 후보</b>를 고르는 경향(=<b>승자의 저주</b>)까지 있어 오히려 손해를 본다.</li>
</ul>
<p class='sub'>핵심 병목을 우리는 <b>coverage(커버리지)</b>라 불러왔다 — "안 해본 행동의 결과를 모른다"는 데이터의 구멍.</p>

<h3>2. 새 아이디어 — 심판에게 '수정구슬'을 준다 (이게 model-based)</h3>
<p>심판이 후보를 직접 감으로 매기지 말고, <b>세상 돌아가는 법을 흉내내는 모델(수정구슬)</b>을 하나 줘서
<b>각 후보를 실행하면 무슨 일이 벌어질지 "상상"</b>하게 한 뒤, <b>그 상상된 결과가 얼마나 좋은지</b>를 평가하자.</p>
<p>체스 엔진을 떠올리면 된다: "여기 두면 판이 이렇게 되고 → 그 판은 유리해." 우리도 "이 후보로 팔을 움직이면 →
로봇이 이 상태에 도달하고 → 그 상태는 성공에 가깝다"로 점수를 준다. 이게 <b>model-based critic</b>이다.
(수식으로는 <code>후보 점수 = 상상한 즉시보상 + γ·V(상상한 도착지)</code> 한 줄. V는 "이 상태가 얼마나 좋은가"를 재는 기존 심판.)</p>

<h3>3. 과대평가 방지 — 수정구슬 5개로 '모르면 비관'</h3>
<p>상상은 틀릴 수 있다. 그래서 <b>서로 다르게 학습한 수정구슬 5개</b>를 두고, 한 후보에 대해 <b>5개가 크게 엇갈리면(불확실)
가장 나쁜 값으로 깎는다</b>(min). 자신 있게 예측하는(=엇갈림 적은) 후보만 높은 점수를 받는다. 이렇게 하면
<b>모르는 후보를 과대평가하는 승자의 저주</b>를 구조적으로 막는다. 이게 오늘 밤 우리에게 없던 안전장치다.
(이 방식을 <b>MVE, model-based value expansion</b>라 한다. 워커A가 이미 <code>train_mve_critic.py</code>로 구현해뒀다.)</p>

<h3>4. 그런데 — <u>수정구슬은 어떻게 만드나?</u> (여기가 진짜 관건)</h3>
<p>수정구슬(모델)은 <b>로봇이 실제로 한 것</b>으로 배운다: "이 상황에서 이 동작을 했더니 → 여기 도달했다"를 잔뜩 모아
(상황, 동작) → (도착지)를 맞히게 학습한다. 문제가 여기서 똑같이 터진다:</p>
<ul>
<li>데이터엔 <b>한 상황에 한 동작</b>만 있다. 그래서 모델은 "이 상황이면 대개 여기로 간다"는 잘 배우지만,
<b>"동작을 바꾸면 도착지가 어떻게 달라지나"는 거의 못 배운다.</b> (우리가 옛날에 쟀을 때 동작이 예측에 더하는 정보가
겨우 +7.3%였다.) → <b>같은 coverage 벽</b>이 모델 학습에도 그대로 있다.</li>
<li>특히 <b>"이 후보가 머그를 진짜 잡나?"</b> 같은 <b>물체 상호작용 결과</b>는 안 해본 동작에 대해 <b>원리적으로 못 배운다</b> —
그 데이터가 없으니까.</li>
</ul>

<h3>5. 그래도 되는 이유 — 물리(팔의 움직임)는 싸게 배운다</h3>
<p>수정구슬이 <b>확실히 잘 배우는 것</b>이 하나 있다: <b>팔의 물리</b>. "오른쪽으로 가라고 명령하면 팔이 오른쪽으로 간다" —
이건 결정적이고 시연에도 항상 보여서 잘 배운다. 그래서 모델은 <b>각 후보에 대해 '팔이 어디로 향하나'는 선명하게 상상</b>할 수 있다.
만약 좋은 후보와 나쁜 후보를 가르는 게 <b>"팔이 머그 쪽으로 제대로 가느냐"</b>라면, 물체 결과를 못 맞혀도 <b>궤적 품질</b>만으로
후보를 가를 수 있다. <b>단, 조건</b>: 모델이 도는 표현 공간이 그 <b>팔·위치 디테일을 보존</b>해야 한다. 우리
<span class='xref' data-eid='embed-compare'>임베딩 비교</span>에서 봤듯 "얼마나 다 됐나"만 남기는 진행도-표현(φ)은
<b>제어 정보를 파괴</b>하므로, 그런 공간에서 모델을 배우면 후보를 못 가른다. <b>표현 선택이 성패를 가른다.</b></p>

<h3>6. 정직한 결론 & 다음 한 수</h3>
<p><b>model-based critic은</b> "감으로 랭킹"을 "상상해서 랭킹 + 모르면 비관"으로 바꿔주는 원리적으로 옳은 방향이다.
<b>하지만</b> 수정구슬은 데모 데이터의 구멍(coverage)을 그대로 물려받는다: <b>팔 궤적은 상상하되, 안 해본 동작의 물체 결과는 못
상상한다.</b> 그러니:</p>
<ul>
<li><b>될 수 있는 경우</b>: 후보 판별이 대체로 "팔이 올바로 향하나"로 결정되면 — 궤적 채널 + 앙상블 비관으로 로봇을 이길 여지.</li>
<li><b>안 되는 경우</b>: 물체 결과 예측이 꼭 필요하면 — 결국 <b>후보를 실제로 시뮬에서 해봐서</b> 데이터를 만드는 수밖에 없다(on-policy).
우리가 계속 도달하는 그 결론.</li>
</ul>
<p><b>그래서 다음 실험은</b> "수정구슬이 정말 후보를 가르나"를 먼저 재는 <b>게이트</b>다: 학습한 모델이 (a) 아는 후보는 값이
갈리고, (b) 모르는 후보는 5개가 엇갈려 비관으로 빠지는지. 이게 통과해야 실제 롤아웃(그것도 <b>여러 시드 평균</b>으로 —
오늘 밤 배운 대로 n=25 한 번은 노이즈다)로 넘어간다. <span class='xref' data-eid='model-based'>이전 model-based 실험</span>과
이어진다.</p>""",
)

# ============================================== 08-09 model-based 본질 회귀
entry(
    "08-07",
    "model-based",
    "Model-based로 본질 회귀 — Q(z,a)=γ^h·V(f(z,a))",
    "진행 중",
    """
<p><b>방향 정리 (2026-08-09 00:40, 사용자 지시).</b> 지난 이틀 실험 축이 너무 벌어졌다(다양화·veto·φ결합·히스토리…).
트릭을 덧대는 대신 <b>간단한 방식 하나</b>로 집중한다: 측정으로 확정된 근본 문제는 "후보축에 학습 신호가 없다"이고,
이를 정면으로 푸는 가장 단순한 구성이 model-based다:</p>
<p style='text-align:center'><b>Q(z, a) = γ<sup>h</sup> · V( f(z, a) )</b></p>
<p>f = 잠재 dynamics("이 chunk를 실행하면 토큰이 어디로 가나" — 주석의 (z, chunk, z′) 쌍으로 <b>순수 지도학습</b>,
프레임당 2048차원 감독), V = 기존 IQL 상태가치망 재사용, 배포 = 표준 16후보에서 V(착지점) argmax.
CQL·veto·노이즈 풀 없음. 워커A가 repo에 커밋해 둔 train_latent_dynamics.py를 그대로 사용(재사용 원칙).</p>
<p><b>가지치기:</b> 신규 트릭 팔 제출 중단. 기제출된 v17b n=16·hist 평가·디코더 프로브는 결론만 기록하고 후속 없음.</p>
<h3>게이트 — 토큰 쌍에 행동 정보가 있는가</h3>
<p>같은 배치에서 세 기준선을 잰다: identity(z′=z, "h스텝에 뭐가 변하나") / no-action f(z)("상태만으로 정해지는 것") /
action-conditioned f(z,a). <b>마지막 둘의 격차가 곧 행동 정보</b> — 0이면 이 오프라인 경로는 구조적으로 닫혀 있다.</p>
<table class='num'><tr><th>좌표계</th><th>identity</th><th>no-action</th><th>action-cond.</th><th>행동 정보</th></tr>
<tr><td>raw 토큰 2048</td><td>0.197</td><td>0.128</td><td>0.125</td><td><b>+2.6%</b> — 희미함</td></tr>
<tr><td>PCA-128</td><td>1.225</td><td>0.649</td><td>0.607</td><td>+6.5%</td></tr>
<tr><td>φ-128</td><td>1.082</td><td>0.579</td><td>0.537</td><td><b>+7.3%</b></td></tr></table>
<p><b>판독 (01:20 갱신).</b> ① 행동 정보는 <b>압축 좌표에 산다</b>: 128차원으로 줄이면 상대량이 약 3배(2.6→6.5~7.3%) —
원시 토큰에서는 행동의 흔적이 외형 차원에 묻힌다. ② φ와 PCA의 차이는 작아(7.3 vs 6.5) 좌표 효과의 대부분은 압축 자체.
③ +7.3%는 존재 증명이지 지배 신호는 아니었다.</p>
<p class='sub'>용어 — <b>합성 게이트</b>: 조립된 Q=V(f(z,a))를 롤아웃에 태우기 전, held-out 프레임의 "실행된 데모
chunk + 후보 16개"에 Q를 계산해 (a) 데모가 후보를 이기는 비율(demo_winrate; 0.5=행동을 못 봄), (b) 후보 간 Q 폭(band)을
재는 오프라인 검문소. 정답이 알려진 과거 경기로 심판을 시험하는 것 — 통과 못 하면 실전(롤아웃)에 세우지 않는다.</p>
<h3>합성 게이트 — V(f(z,a))는 후보를 가르는가 (판정: 아니오)</h3>
<p>held-out kroll 2,000프레임(f·V 학습에 안 쓰인 데이터)에서 데모 chunk와 16후보 각각에 y=V(f(z,a))를 계산.
demo_winrate = 데모가 이기는 후보 비율(0.5면 action-blind), band = 후보 간 y의 q99−q01.</p>
<table class='num'><tr><th>좌표</th><th>demo_winrate</th><th>band</th></tr>
<tr><td>raw 2048</td><td>0.479</td><td>0.025</td></tr>
<tr><td>PCA-128</td><td>0.481</td><td>0.031</td></tr>
<tr><td>φ-128</td><td>0.487</td><td>0.023</td></tr>
<tr><td>φ-128+proprio</td><td>0.485</td><td>0.023</td></tr></table>
<p><b>in-dist 대조 (11:20, 사용자 질문).</b> 같은 게이트를 f·V가 학습에 사용한 mixed 프레임 2,000개에서 반복:
winrate .482/.489/.486/.489, band는 held-out보다 오히려 좁음(0.017–0.023). <b>학습 분포 안에서도 동전 —
일반화 격차가 아니라 근본 실패다.</b> 합성에 행동 신호가 실린 적이 없으며, 병목은 f가 행동 의존성을 배울 감독의
크기(행동 정보 +2.6~7.6%) 자체다. 데이터 증량으로 풀리는 종류가 아니라는 뜻.</p>
<p><b>판정.</b> 네 좌표 모두 winrate≈0.5 — <b>합성 model-based Q도 action-blind다.</b>
φ+proprio(사용자 질문의 마지막 갈래)는 proprio 정보를 실제로 더 보존하지만(R² .546→.617) 합성 게이트는
똑같이 닫혀 있다(winrate .485) — 보존된 proprio가 후보 구별로 이어지지 않는다. 행동 정보가 3배인 좌표에서도
f의 착지점 차이가 V의 해상도 아래로 사라진다. 롤아웃 없이 오프라인 게이트에서 기각 — 단순 구성 v1(f, V 재사용,
트릭 0)은 이 과제의 오프라인 데이터에서 닫혀 있다. <b>사고록:</b> ① 스크립트가 no-action 변형을 저장하던 버그를
게이트 전에 발견·수정(하마터면 action-blind 모델로 합성 판정할 뻔), ② node23 CUDA 런타임 붕괴 → bad-node 등록.</p>
<p><b>종합 — 이틀간의 수렴.</b> 선택 트릭(다양화·veto: n=16 null), 임베딩(φ: null + 디코더가 팔 자세 소실 확인),
히스토리(null), CalQL(null/해악), model-based 합성(오프라인 기각)까지 — <b>모든 경로가 같은 구조적 사실로 수렴한다:
과제 특화로 강하게 미세조정된 VLA의 PrepareCoffee에서는 후보 선택으로 캘 수 있는 가치 차이가 없다.</b>
남은 본질적 방향은 과제를 바꾸는 것 — 행동 선택이 결과를 크게 가르는 GR1 tabletop으로의 이전이며,
이는 트릭이 아니라 질문이 유효한 무대를 찾는 일이다. <b>준비 상태(03:50):</b> 5태스크 × 1,000 에피소드(12G,
LeRobot v2: ego_view·state·action, 20fps) 검증 완료, 시뮬레이터 스모크는 08-07에 통과. 남은 관문:
π0.5-RLT 미세조정 config + 학습 자원(node200/B200 — 사용자 결정) → 미세조정 → 주석 → critic 파이프라인 재실행.
φ+proprio 합성 게이트(마지막 오프라인 갈래)가 병행 실행 중이며 그 판정과 함께 아침 보고에 종합한다.</p>
""",
)

# ============================================== 08-08 phi 임베딩 사다리
entry(
    "08-07",
    "phi-ladder",
    "임베딩 사다리 — 차원인가 기하인가 (HILP φ 재현)",
    "진행 중",
    """
<p><b>동기.</b> RLT 토큰의 episode-정체성 병리(장면 암기 지름길의 표현 측 원인)를 워커A는 HILP식 TD readout(φ)으로
공격했다. 사용자 질문 두 개가 설계를 정했다: "φ의 이점이 그냥 128차원이라 작아서는 아닌가?"(차원 교란) →
PCA-128 통제군 추가, "도달성 거리만 남기면 디테일이 사라지는 것 아닌가?" → 디코더·상태변수 프로브 예정.
<b>실험은 엄밀하게, 한 번에 한 컴포넌트씩</b>: critic 레시피(FINAL iql) 완전 고정, 임베딩만 교체.</p>
<h3>φ 재현 (in-repo, train_hilp_readout.py)</h3>
<p>frozen 토큰 위 goal-조건 expectile TD로 V(s,g)=−‖φ(s)−φ(g)‖ 학습, 교차-에피소드 goal 샘플링(p_future=0.7).
<b>사고록:</b> ① 무-proprio 런이 loss=NaN 발산 — grad clip으로도 재발 → 근본 원인은 <b>‖·‖의 0점 NaN 그래디언트</b>
(정지 프레임의 완전 동일 토큰 쌍이 norm 0을 침). proprio 버전이 살아남은 이유도 이것(proprio가 중복을 깨줌).
ε-safe distance로 해결, 두 변형 재학습 완료(loss 0.77/0.75). ② 완료 마커를 `;`로 체인하면 실패도 DONE으로
찍힌다 — `&&` 게이팅으로 교정.</p>
<h3>검증 배터리 (60ep × 40프레임, latent_semantics 지표)</h3>
<table class='num'><tr><th>지표</th><th>raw 2048</th><th>PCA-128 (차원 통제)</th><th>φ-128</th><th>φ-128+proprio</th></tr>
<tr><td>kNN episode purity ↓ (우연 .016)</td><td>.601</td><td>.517</td><td><b>.388</b></td><td>.396</td></tr>
<tr><td>교차-ep phase 오차 ↓ (우연 .342)</td><td>.102</td><td>.102</td><td><b>.089</b></td><td>.089</td></tr>
<tr><td>episode 판별 acc ↓ (우연 .017)</td><td>.990</td><td>.920</td><td><b>.756</b></td><td>.753</td></tr>
<tr><td>progress R² (에피소드 held-out) ↑</td><td>.760</td><td>.749</td><td>.747</td><td>.732</td></tr>
<tr><td>Kendall τ (교차-비디오 정렬) ↑</td><td>.545</td><td>.568</td><td>.480</td><td>.496</td></tr></table>
<p><b>판독.</b> ① <b>"작아서"는 기각</b> — PCA-128은 episode 정체를 조금만 지운다(acc .92). φ는 같은 차원에서
훨씬 깊게 지우며(.756) progress를 보존(.747): 효과는 차원이 아니라 TD 기하다. ② 단 워커A 보고치(ep .28,
stitch .78)에는 못 미치고 Kendall τ는 raw보다 낮다 — 재현 레시피의 차이가 있고, 원본 체크포인트 교차검증(경로 A)이
필요하다. ③ proprio 유무는 무차이. <h3>디코더 프로브 — φ가 무엇을 버렸나 (사용자 질문의 답)</h3>
<p>임베딩이 이미지의 어떤 정보를 보존/폐기하는지 보려고, 임베딩→이미지 <b>디코더</b>를 학습해 held-out
프레임을 재구성시켰다(kroll 6만 프레임, 20k스텝; φ-128용과 raw-2048용을 같은 조건으로 두 벌).
산출물 4종은 각각 다음을 보여준다:</p>
<table class='num'><tr><th>산출물</th><th>무엇을 보는가</th><th>읽는 법</th></tr>
<tr><td>φ 재구성 패널 (아래 왼쪽)</td><td>held-out 원본(위 행) vs φ-128에서 재구성(아래 행)</td><td>φ에 남은 정보만 복원됨 — 주방 배치·로봇 팔 위치는 남고, 그리퍼 각도·물체 디테일이 뭉개지면 그 정보가 φ에서 사라졌다는 뜻</td></tr>
<tr><td>raw 재구성 패널 (아래 오른쪽)</td><td>같은 프레임을 raw 2048 토큰에서 재구성</td><td>φ 패널과의 차이가 곧 "φ가 버린 것" — raw 쪽이 그리퍼·물체를 더 보존</td></tr>
<tr><td><a href="videos/decoder/22_decoder_phi128_walk.mp4" target="_blank">φ 임베딩 워크(영상)</a></td><td>φ 공간의 두 실제 프레임 임베딩 사이를 선형 보간하며 각 점을 디코딩</td><td>중간 프레임들이 그럴듯한 "장면 변화"로 이어지면 φ 공간이 매끄럽게 의미를 인코딩한다는 것. 뚝뚝 끊기면 임베딩 공간에 구멍이 있다는 것</td></tr>
<tr><td><a href="videos/decoder/22_decoder_phi128_ride.mp4" target="_blank">교차-에피소드 ride(영상)</a></td><td>다른 에피소드의 궤적을 따라 φ를 뽑아 순서대로 디코딩</td><td>학습에 안 쓴 에피소드에서도 진행 상황이 복원되면 φ가 에피소드 정체성이 아니라 "과제 진행"을 일반화해 인코딩한다는 증거</td></tr></table>
<p style='display:flex;gap:8px;flex-wrap:wrap'>
<img src="videos/decoder/22_decoder_phi128_recon.png" alt="φ-128 재구성 패널" style='max-width:49%'>
<img src="videos/decoder/22_decoder_raw_recon.png" alt="raw 2048 재구성 패널" style='max-width:49%'></p>
<p><b>판정(프로그램적):</b> proprio 차원별 선형 프로브 R² 평균이 <b>raw .760 → PCA-128 .653 → φ-128 .546</b> —
φ는 로봇 자체 상태(관절·그리퍼) 정보를 raw 대비 28% 상대 손실한다. 즉 φ가 버린 것은 <b>단거리 행동-관련 정보</b>이며,
"거리만 남기면 디테일이 사라지는 것 아닌가"라는 우려의 정량 확인이자 φ-critic이 BoN을 못 여는 이유의 기전적 설명이다.
(재구성 패널·워크 영상은 정성 참고 자료로만 첨부 — 육안 인상은 판정 근거로 쓰지 않는다.)</p>
<p class='sub'><b>화질에 대한 주의(08-10 사용자 피드백 반영):</b> 위 패널이 전반적으로 흐릿한 것은 임베딩 탓만이
아니라 <b>L2 회귀 디코더의 구조적 한계</b>다 — 픽셀 MSE를 최소화하는 예측은 "가능한 이미지들의 평균"이라 항상
뭉개진다. 임베딩이 실제로 보존한 정보의 상한을 보려면 조건부 생성모델이 맞는 도구다.
→ <b>후속(완료): φ-조건부 diffusion 디코더</b>(같은 kroll 데이터·같은 프로토콜, φ-128 vs raw-2048 조건,
조건부 DDPM 30k). L2의 평균-예측 한계를 제거해 임베딩이 보존한 정보의 상한을 본다.</p>
<h3>diffusion 디코더 판정 (08-11, 프로그램적)</h3>
<p>held-out 128프레임에서 best-of-4 샘플의 픽셀 MSE(낮을수록 조건 임베딩에 정보가 더 남아있음):</p>
<table class='num'><tr><th>조건 임베딩</th><th>best-of-4 MSE ↓</th><th>해석</th></tr>
<tr><td>raw 2048</td><td><b>0.0193</b></td><td>기준 — 원 토큰이 픽셀을 가장 잘 복원</td></tr>
<tr><td>φ-128</td><td>0.0602</td><td>raw의 <b>3.1배</b> — φ 조건만으로는 같은 프레임을 훨씬 못 만든다</td></tr></table>
<p><b>판정:</b> 생성모델(평균-예측 편향 제거)로 봐도 φ는 raw 대비 재구성 정보가 크게 부족하다 —
L2 proprio 프로브(R² raw .760→φ .546)와 <b>독립 지표에서 같은 방향</b>. 즉 앞선 "φ가 단거리 정보를 버린다"는
결론이 디코더 종류(회귀 vs 생성)에 무관하게 성립. 패널·워크 영상은
<a href="videos/decoder/23_diff_raw_recon.png" target="_blank">raw 패널</a> ·
<a href="videos/decoder/23_diff_phi128_recon.png" target="_blank">φ 패널</a> ·
<a href="videos/decoder/23_diff_raw_walk.mp4" target="_blank">raw 워크</a> ·
<a href="videos/decoder/23_diff_phi128_walk.mp4" target="_blank">φ 워크</a> (정성 참고 — 판정은 위 MSE로). (Task#9 완료)</p>
<h3>φ→action BC probe (08-11, 사용자 요청 — 이전에 슬립됐던 실험)</h3>
<p>디코더 probe는 "φ가 이미지·proprio를 얼마나 재구성하나"를 쟀다. 그러나 우리가 정작 알아야 할 건
<b>φ가 행동을 재현할 만큼의 정보를 갖는가</b>다. 그래서 임베딩→데모 action chunk(H=8×12, z-score) BC MLP를
raw/PCA/φ 동일 프로토콜로 학습해 held-out을 비교했다(디코더 probe와 같은 kroll 정렬 데이터):</p>
<table class='num'><tr><th>임베딩</th><th>action R² ↑</th><th>held-out MSE ↓</th></tr>
<tr><td>raw 2048</td><td>0.708</td><td>0.258</td></tr>
<tr><td>PCA-128</td><td>0.697</td><td>0.267</td></tr>
<tr><td><b>φ-128</b></td><td><b>0.682</b></td><td>0.280</td></tr></table>
<p><b>판정 — 디코더 서사의 보정:</b> φ는 <b>행동 정보를 거의 다 보존한다</b>(R² .682 vs raw .708, 상대차 3.6%).
디코더 probe의 "φ가 재구성 정보를 3.1배 잃는다"와 얼핏 모순이지만 해소된다: <b>φ가 버린 것(세밀한 시각 디테일·
재구성용 그리퍼 각도)은 행동 예측에 불필요한 정보</b>였고, 데모 행동을 맞히는 부분공간은 φ가 지켰다. 함의 둘:
① "φ가 정보를 버려서 못 쓴다"는 설명은 <b>기각</b> — φ는 action-sufficient. 따라서 φ-critic이 BoN을 못 연 건
표현 결함이 아니라 <b>반사실 부재(축2)</b>다(우리 결론 강화). ② <b>표현 축은 BC에 거의 평평</b>(raw .708 / PCA
.697 / φ .682 — 셋 다 비슷) — TD-SF-ARQ A단계에서 임베딩 선택(φ vs PCA)은 결정적 변인이 아니며, φ를 써도
행동 정보 손실 걱정은 없다. <b>주의(BC≠판별):</b> 이 probe는 "한 데모 행동 재현"이지 "같은 상태 후보 판별"이
아니다 — φ가 act-sufficient라도 반사실 신호가 없으면 critic은 여전히 막힌다. 두 사실은 함께 "표현은 충분,
데이터가 부족"으로 수렴한다.</p>
④ <b>critic 사다리 확정(n=8):</b> RLT2048 −0.065 / PCA-128 −0.040 / <b>φ-128 −0.010 CI[−0.078,+0.058],
McN +62/−66 p=0.791 — null 확정.</b> n=4의 +0.035와 단조 배열은 새 4시드(−0.02/−0.10/+0.02/−0.12)에서 소멸 —
v11 AQC 전례("n=4 신호를 믿지 말 것")의 재연이며, 그 규칙이 옳았음의 재확인. φ 결합팔(v19: φ×다양화×veto)도
null(+0.005, McN +33/−32). <b>임베딩 교체는 이 과제의 BoN을 열지 못한다</b> — 표현 사다리는 여기서 마감,
남은 생존 후보는 v17b(다양화+σ-veto) 하나다.</p>
""",
)

# ============================================== 08-08 conservatism 종합 프레임
entry(
    "08-07",
    "conservatism",
    "Conservatism 스펙트럼 — 전체 null의 통일 해석과 다음 수",
    "살아있음",
    f"""
<p><b>주장.</b> 지금까지의 모든 판정(FINAL 14팔·v14·calql·randh·워커A 밤샘)은 "보수성 다이얼 위의 어디에 서
있었는가"로 한 번에 설명된다. 그리고 위험은 축이 두 개라서, 한 축의 완벽한 안전이 다른 축을 지켜주지 않는다.</p>
<h3>보수성이 사는 곳 — 방법별 위치</h3>
<table class='num'><tr><th>방법</th><th>보수성의 형태</th><th>위치</th><th>우리 판정</th></tr>
<tr><td>BoN/재순위</td><td>선택형 — π_β 샘플 중 argmax, KL≤log N (N=16이면 2.8nat)</td><td>극안전단</td><td>null (안전한 무익)</td></tr>
<tr><td>IQL</td><td>질의 회피형 — OOD 행동을 아예 평가하지 않음</td><td>안전단</td><td>null</td></tr>
<tr><td>CQL/CalQL</td><td>가치 억압형 — OOD를 명시적으로 누름 (CalQL은 mc 하한 보정)</td><td>중간</td><td>null — 억압할 OOD가 배포 경로에 없음</td></tr>
<tr><td>FQL/velocity steering</td><td>정책 근접형 — BC/flow 앵커가 곧 보수성, 가치 페널티 불필요</td><td>중간 (실행가능)</td><td>미실험 — Q-VGM 리뷰 참조</td></tr>
<tr><td>제약 없는 actor-critic</td><td>없음</td><td>위험단</td><td>워커A full-authority 파국(.300, p=.004)<i>(08-10 갱신: φ 정규화 버그로 원수치 .300/p=.004는 무효였으나, 수정 후 동일 시드 재실행에서 파국이 더 강하게 재현 — .133, +2/−19, p=.0002. 정정된 것은 숫자이지 결론이 아니다)</i>이 근방 증거</td></tr></table>
<h3>핵심 역설 — BoN은 완벽한 BC 정규화인데도 위험하다</h3>
<p>위험의 축이 두 개이기 때문이다. <b>축1 분포 이탈(OOD)</b>: BC 계열이 막는 것 — BoN은 이 축에서 완벽하다(전 후보가
in-support). <b>축2 추정 오차의 착취(optimizer's curse)</b>: argmax는 N개의 잡음 낀 추정 중 가장 과대추정된 것을
체계적으로 고른다 — in-support여도 발생하며, support 제약은 무력하다. max-of-N ≈ N/(N+1) 분위수이므로 τ=0.9
critic이 보증하는 것은 N≈10까지: <b>N을 키울수록 보정 안 된 꼬리를 수확한다.</b></p>
<h3>축2에 대한 처방 (우리 스택 실행 순)</h3>
<table class='num'><tr><th>처방</th><th>원리</th><th>상태</th></tr>
<tr><td><b>LCB 선택</b> argmax(Q̄−k·σ)</td><td>불확실성 비례 비관 — 과대추정 후보를 깎고 고름 (K=2 min은 조악한 버전; K↑)</td><td>워커A σ-veto .767이 독립 증거 — 설계 후보</td></tr>
<tr><td><b>τ↔N 정합</b></td><td>선택 분위수(1−1/N)를 학습 분위수로 — 꼬리를 critic이 보증</td><td>N·τ는 짝 노브 — 다양화와 세트</td></tr>
<tr><td><b>랭킹 손실 critic</b></td><td>절대값 대신 같은 상태 후보쌍의 순서를 학습 — curse가 착취하는 절대 스케일 오차와
familiarity bias(실행 행동만 회귀)를 통째로 소거</td><td><b>v16 후보 1순위</b></td></tr>
<tr><td>soft 선택</td><td>softmax(Q/T) 샘플링 — "가장 시끄러운 추정치"를 항상 잡는 성질 완화</td><td>softcand 모드 기구현</td></tr></table>
<h3>다양화 프로브 완결 — σ_signal은 샘플링 노브로 열린다</h3>
{img(P / "21_cand_diversity.png", "candidate diversity sweep")}
<table class='num'><tr><th>세팅</th><th>행동 스프레드(z)</th><th>Q밴드 q99−q01</th><th>Q max−min</th></tr>
<tr><td>fs10 · ns1.0 (기준)</td><td>0.299</td><td>0.025</td><td>0.026</td></tr>
<tr><td>fs5 · ns1.0</td><td>0.387 (+29%)</td><td>0.038</td><td>0.039</td></tr>
<tr><td>fs3 · ns1.0</td><td>0.526 (+76%)</td><td>0.048</td><td>0.049</td></tr>
<tr><td>fs5 · ns1.5</td><td>1.945</td><td>0.131</td><td>0.138</td></tr>
<tr><td>fs10 · ns1.5</td><td>2.026 (6.8×)</td><td>0.147 (5.8×)</td><td>0.158</td></tr>
<tr><td>fs10 · ns2.0</td><td>7.524 (25×)</td><td>0.280 (11×)</td><td>0.292</td></tr></table>
<p><b>판독.</b> 같은 48프레임에서 샘플링만 바꿔 잰 결과(v12 mixed critic으로 밴드 측정) — 후보 스프레드는 고정 상수가
아니라 노브로 열리는 양이고, critic도 열린 풀에서 후보를 실제로 구분한다(밴드 동반 상승). 단 ns2.0의 z-스프레드
7.5는 매니폴드 밖 행동이 섞였다는 뜻이라, 배포 실험은 <b>안전 코어 8(ns1.0, 후보0 = 순정 VLA 샘플) + 다양 꼬리
8(ns1.5) 혼합 풀</b>로 설계했다: vla 기준선은 완전 순정 유지, critic만 넓은 풀에서 선택(v17, 4시드 페어드 제출됨).</p>
<h3>히스토리 critic 첫 판정 (v15)</h3>
<p>iql_hist5(K=5, stride 8, mixed, FINAL 레시피와 히스토리만 diff): n=4 Δ̄=−0.025 CI[−0.215,+0.165],
McNemar +30/−35 p=0.620 — <b>null</b>. td_max_hist5(n=4): Δ̄=+0.020 CI[−0.152,+0.192], McN +38/−34 p=0.724 —
<b>null. 히스토리 축은 iql·td 모두 null로 마감</b>(후속 없음). SNR 프레임 예측과 정합: 히스토리는 추정 품질(ρ)을
겨냥하지만 σ_signal이 없으면 이득 원천이 없다.</p>
<h3>용어 해설 — 이 리포트를 읽는 데 필요한 두 가지</h3>
<p><b>σ-veto (시그마 거부권).</b> critic은 초기값만 다른 쌍둥이 2개(앙상블)로 학습된다. 어떤 후보에 대해 두 critic의
점수가 비슷하면(0.62 vs 0.60) 데이터에서 충분히 본 상황이라 믿을 만하고, 크게 갈리면(0.80 vs 0.45) 둘 다 찍고 있다는
뜻이다. 최고점 선택(argmax)은 하필 이런 "가장 크게 뻥튀기된 추정"을 골라잡는 성질이 있다(winner's curse). σ-veto는
<b>두 critic의 의견차 |Q₁−Q₂|가 큰 상위 절반을 후보에서 제외하고, 둘 다 동의하는 후보 중에서만 최고점을 고르는 규칙</b> —
면접관 둘의 평가가 극단적으로 갈린 지원자는 뽑지 않는 것과 같다.</p>
<p><b>McNemar 검정.</b> critic과 vla를 완전히 같은 주방 세트에서 돌리므로(페어드) 각 주방은 [둘 다 성공 / 둘 다 실패 /
critic만 성공(+) / vla만 성공(−)] 중 하나다. 앞의 둘은 우열 정보가 없으니 버리고 <b>승부가 갈린 판의 +와 −만 센다.</b>
p값은 "실력 차가 없다면(동전 던지기) 이만큼 쏠릴 확률" — 예: +74/−53이면 갈린 127판 중 74승, p=0.076은 그 쏠림이
우연일 확률 7.6%라는 뜻(관례상 5% 미만이면 우연이 아니라고 판정). 주방마다 난이도가 달라 생기는 운을 짝짓기로 소거하는
것이 요점이며, 시드 평균±신뢰구간(보수적)과 병기한다.</p>
<h3>v17 판정 — 다양화 단독은 실패한다 (프레임의 예측대로)</h3>
<p><b>설계:</b> 후보 풀 = 안전 코어 8(ns1.0, 후보0 = 순정 VLA 샘플) + 다양 꼬리 8(ns1.5), critic = FINAL iql(τ0.9,
불변), vla 기준선 완전 순정, 4시드×50 페어드. <b>결과: n=4 Δ̄=−0.065 CI[−0.203,+0.073], McNemar +31/−44
p=0.165 — null이며 음의 경향.</b> 스프레드를 5.8× 열어줘도(canddiv 실측) 성공률은 안 움직였고 오히려 내려가는
방향 — "보수성 없는 다양화 = curse 증폭"의 실증이다. τ0.9 critic은 ns1.5 꼬리를 보증하지 않으므로 argmax가
과대추정된 꼬리 후보를 집는다. <b>v17b 최종(n=16): Δ̄=+0.019 CI[−0.021,+0.059], McNemar +136/−121 p=0.383 — null 확정 마감.</b>
n=8의 +0.052/p=0.076은 φ의 n=4 신호와 같은 소멸 패턴(시드 ≥0이 9/16 = 동전). 다양화·veto 계열은 여기서 종결 —
"트릭을 덧대지 말고 본질로"라는 방향 전환(model-based 리포트 참조)이 데이터로도 정당화됐다.
n≤8의 유망 신호는 판정으로 취급하지 않는다는 규칙을 재확인.</p>
<h3>v20 — N=4 소표본 BoN (ACSAC 소견 검증, 사용자 질문)</h3>
<p><b>배경.</b> ACSAC은 n=4가 최적이라 보고했고, 우리 프레임에서도 근거가 있다: max-of-4는 ~80분위 질의라
τ=0.9 critic의 보증 범위(N≤10) <b>안</b>이고(N=16은 밖), curse 계수 c₄≈1.0으로 c₁₆≈2.35의 절반 이하.
확인 결과 우리는 전 실험 N=16 고정 — 처음 검증한다. 단일변수: FINAL iql critic 그대로, N만 16→4, 4시드 페어드.</p>
<p><b>판정: Δ̄=−0.015 CI[−0.090,+0.060], McNemar +30/−33 p=0.801 — null.</b> N=16(−0.065)보다 점추정은 덜
나쁘지만(curse 감소 방향과 부호 일치) 이득은 없다. <b>해석: 이 과제의 병목은 τ↔N 부정합이 아니라 σ_signal 부재</b> —
스프레드가 없으면 어떤 N도 0을 곱한다. ACSAC의 n=4 최적은 참 스프레드가 존재하는 과제의 이야기이며,
GR1 파일럿의 초기 측정에 N∈{4,8,16} 스윕을 포함해 그 조건에서 재검한다.</p>
<h3>SNR 조건 — 어떤 선택 규칙도 못 피하는 것</h3>
<p>BoN 기대 이득 ≈ (참 가치 스프레드 상위꼬리) − (오차 상위꼬리). 참 스프레드가 0에 가까우면(우리 측정: demo 밴드
0.002–0.023, rand≈vla) 이득의 원천 자체가 없다. <b>따라서 정답은 한 쌍: 신호를 키우는 후보 다양화(canddiv 프로브)
+ 오차를 억누르는 위 장치들.</b> 다양화 없는 보수성 = 안전한 무익(지금), 보수성 없는 다양화 = curse 증폭.</p>
""",
)

# ============================================== 08-08 교차 워커 배움
entry(
    "08-07",
    "xworker-0808",
    "교차 워커 리뷰 — 워커A에게서 배운 것 (08-08)",
    "완결",
    """
<p class='sub'>같은 허브의 워커A 리포트 7건을 정독하고, 방법·결과·운영 관행에서 배울 것을 우리 스택에 반영한 기록.</p>
<h3>① 서로 재현: BoN은 두 스택 모두에서 무익</h3>
<p>워커A는 전혀 다른 표현(HILP φ 128d, TD readout)·다른 critic(Cal-QL+swap negatives, 오프라인 action-sensitivity
0→0.524 통과)으로 밤샘 롤아웃 판정을 했다: <b>BoN 무익(.700 동률), full-authority critic은 파국(.300, McNemar p=.004)</b><i>(08-10 갱신: φ 정규화 버그로 원수치 .300/p=.004는 무효였으나, 수정 후 동일 시드 재실행에서 파국이 더 강하게 재현 — .133, +2/−19, p=.0002. 정정된 것은 숫자이지 결론이 아니다)</i>,
σ-veto BoN .767(비유의). 우리 FINAL(14팔 null, TD 3팔 유의 해악)과 <b>독립 스택 상호 재현</b> — "action-sensitive critic을
만들어도 BoN으로는 못 이긴다"가 두 워커 공통 결론이 됐다. 그들의 논지(critic에게는 커밋 길이·거부권 같은
좁고 검증가능한 권한이 맞다)는 우리 다음 수(다양화·그래디언트 조향)와 상보적이다.</p>
<h3>② 도입한 것: 트라이얼 페어드 McNemar</h3>
<p>워커A의 판정 표준을 FINAL 탭에 즉시 도입했다(200쌍/팔). 결과: run-level CI로 안 보이던
TD 3팔의 유의한 해악이 확정됐다 — FINAL 탭 참조. 앞으로 모든 페어드 판정에 두 통계를 병기한다.</p>
<h3>③ 표현 병리의 다른 공격로</h3>
<p>워커A: VLA 토큰의 episode-정체성 병리(kNN purity .42, 학습할수록 악화)를 <b>표현 쪽에서</b> 공격 —
DINOv2+15분 헤드(cheap-z, purity .154)와 episode 적대자, TD readout φ(stitch .78). 우리는 같은 병리를
<b>데이터 쪽에서</b> 공격했다(K-per-scene, 혼합결과 주방 45%). 두 접근은 독립이라 결합 가능: φ/cheap-z 위에
우리 FINAL 레시피를 얹는 실험이 유효한 다음 후보다. 또 dynamics 앙상블 disagreement가 "성공 데모만으로도
action을 구분하는 유일한 오프라인 신호"라는 측정은 우리 후보 다양화 프로브와 직접 비교 대상.</p>
<h3>④ 08-09 심야 갱신 — 루프가 양방향이 됐다</h3>
<p>워커A 신규 5건(08-08 야간) 정독: ① 밤샘 판정을 <b>set B(새 장면 30개, 자체 vla 기준선) 재현</b>까지 얹어 확정 —
우리도 v17b 확정 시 재현 세트를 붙이는 게 맞다. ② 신규 <b>MVE critic</b>은 명시적으로 "두 워커의 유의한 부정 결과
둘(전권 파국 p=.004<i>(08-10 갱신: φ 정규화 버그로 원수치 .300/p=.004는 무효였으나, 수정 후 동일 시드 재실행에서 파국이 더 강하게 재현 — .133, +2/−19, p=.0002. 정정된 것은 숫자이지 결론이 아니다)</i> · 우리 calql_v14 과잉억압 p=.001)을 동시에 회피"하도록 설계 — 비관은 in-sample V+min 앙상블에만,
OOD 질의는 구조적으로 0(MAC 계보). 우리 주석의 프레임당 16후보를 'frozen policy 아카이브'로 재사용한다.
<b>서로의 리포트가 서로의 실험 설계에 인용되는 루프가 성립.</b> ③ 그들의 σ-veto는 dynamics-불일치 기반(러닝 미디언×τ),
우리 v17b는 Q-앙상블-불일치 기반 — v17b 확정 시 veto 신호원 비교가 자연스러운 공동 후속. ④ 운영 메모: 워커A는
fix/probe-eval-jit 브랜치에 커밋 중 — 머지 시 조율 필요.</p>
<h3>⑤ 운영 관행</h3>
<p>좀비 잡(squeue R인데 실제 사망)의 로그 mtime 감시 — 우리도 행 걸린 평가 2건을 겪었으므로 감시 루틴에 채택.
체크포인트 자동 아카이브(HF 업로드 검증 후 로컬 삭제)도 디스크 사고 예방책으로 참고.</p>
<h3>⑥ 08-10 갱신 — 워커A 3-태스크 교차 판정이 우리 GR1 설계를 검증하다</h3>
<p>워커A 신규 판정(08-09 "V는 완성, Q는 spread 퍼즐") 정독. 요지: ① YAM 혼합성과 데이터로 <b>역대 최고
V(Spearman .966)</b>를 얻었지만 action sensitivity는 0 — 다만 이는 과학습 정책의 후보 붕괴(spread 1.8%)가 강제한 것.
② <b>퍼즐: GP는 후보가 가장 다양(11%)한데도 sens≈.0002</b> — "후보 spread가 크면 가치 차이도 있겠지"라는
직관(spread→sens 상관)이 깨졌다. ③ 종합: 에피소드 단위 실패 라벨은 '어느 상태가 실패로 가는가'는 가르치지만
같은 상태 후보 구분은 못 가르친다 — 장벽의 원인이 <b>동일-상태 반사실의 부재</b>로 특정됐다.</p>
<p><b>우리 GR1 계획에 주는 함의 둘.</b> (a) phase-1에서 행동공간 spread만 재면 속는다(GP 반례) —
우리 사전등록대로 <b>결과(성공률) 수준의 rand-vs-vla 페어드 비교</b>가 옳은 게이트다. (b) 데모-only 주석으로 critic이
게이트를 통과 못 하면, 다음 수는 트릭이 아니라 <b>--policy-seed형 on-policy 수집</b>(같은 장면 K회 롤아웃 = 장면 수준
반사실 생성)이다 — 워커A의 "on-policy 개입만이 반사실을 준다"는 논증과 우리 v14 K-per-scene 경험이 같은 결론.</p>
<h3>⑦ 08-10 오후 갱신 — 정정 문화·γ-천장·관계 기하</h3>
<p><b>(a) 정정 공지 — 우리 인용도 즉시 주석.</b> 워커A가 φ rollout 정규화 버그(표준화 토큰으로 학습된 phi.pt에
raw 토큰 주입, 출력 오차 45%)로 <b>φ-소비 롤아웃 팔 전체 무효</b>를 공지 — 우리가 여러 엔트리에서 인용한
"전권 파국 .300, p=.004"가 포함된다. 본 허브의 인용 지점 전부에 주의 문구를 달았다(오프라인 결과·대조군은
유효, 동일 시드 재실행 중). 부정 결과의 자기 정정을 즉시 공지하는 관행 자체가 이 허브의 자산이다.</p>
<p><b>(b) sensitivity의 γ-천장 보정 — 우리 게이트에 직접 반영.</b> 참 가치차의 상한은 ΔQ≈V·|lnγ|·Δt인데
기존 sensitivity는 이를 무시했다: calswap의 .29–.52는 천장 초과 = CQL 인공 마진("오프라인 통과, 롤아웃 0"
미스터리 해소), YAM의 0.0000은 γ 때문에 완벽 critic도 .0001인 게이트 결함. <b>새 게이트: 시간해상도 ≥ 천장의
30%.</b> → 우리 TD-SF-ARQ A단계 사전등록 기준에 이 천장 보정을 채택한다(demo_winrate·band에 γ-천장 대비
정규화 병기).</p>
<p><b>(c) 교차-궤적 이웃 판정 — φ의 bridging 축은 관계 기하.</b> expert action chunk를 관계 상태의 증인으로
쓰는 kNN 판정: φ 이웃 act-cos .661 vs stage-only .334(paired +.327), raw 대비 +.026 유의. "φ가 잇는 것은
phase가 아니라 블럭-팔 상대 배치 같은 관계 기하"라는 발견 — 남은 병목을 "그 다리를 건너는 action-조건
backup"으로 특정한 점이 우리 TD-SF-ARQ의 벡터 SF 타깃 논리와 정확히 맞물린다.</p>
<h3>⑧ 08-11 갱신 — 워커A YAM 실기 스케일링이 우리 GR1에 주는 경고</h3>
<p>워커A r51(YAM π0.5, 체크포인트 50k–200k×10트라이얼): <b>완주 0/50, 그리고 결정적 교차</b> — 200k는
milestone ≥1을 10/10 통과하지만 ≥3은 0/10, 즉 <b>과학습이 쉬운 구간만 확실히 하고 어려운 구간은 전멸</b>시킨다.
오프라인에서 잰 200k 후보 스프레드 붕괴(.018)와 정합 — 정책이 사실상 결정론이 되어 후보축이 죽는다. H60(히스토리
60) 변형은 open-loop 교정 상실로 손해(우리 adaptive chunking 동기와 같은 관찰).</p>
<p><b>우리 GR1 이식에 즉시 반영:</b> ① pilot-2가 30k에서 완주 0일 때 "학습 부족"으로만 읽지 말 것 — 워커A는
<b>더 오래 학습한 200k가 어려운 구간에서 오히려 나빠졌다</b>. 우리 phase-1은 헤드룸을 <b>milestone별로</b> 봐야
하며(전 구간 0/50이어도 milestone별 통과율 곡선이 정보), 후보 스프레드가 아직 살아있는 <b>덜 학습된 체크포인트
(예: 20k)</b>가 critic 실험엔 오히려 나을 수 있다. ② 이는 우리 σ_signal 프레임의 실기 재확인이다: 과학습→스프레드
붕괴→선택 신호 소멸. <b>critic 실험용 기준 체크포인트는 스프레드가 살아있는 지점으로 프로그램적 선택</b>(후보
16개 std를 체크포인트별로 재어 최대 지점)하도록 phase-1 절차에 추가한다.</p>
<p><b>⑨ BoN 사전등록 종결 (워커A r53, 08-10 23:00) — test-time BoN 무익 확정.</b> 이전 사이클에서 유보했던
seed 0의 첫 양(+) 신호(.800 vs .700)는 <b>노이즈로 판명</b>: seed 30·60이 정확히 동률, 90쌍 합산
bon .711 vs vla .678(+11/−8, McNemar <b>p=0.65</b>) → 사전등록 기준 p&lt;.05 미달, null. 전권 파국(.133)과
합치면 <b>"전권=해롭고 선택만=무익"이 demo-only critic의 천장</b> — 우리 FINAL 14팔 null과 독립 스택에서
다시 만난다. <b>교훈(양방향):</b> 소표본 양(+) 신호를 사전등록 파워로 닫는 규율이 이번엔 워커A 쪽에서
작동했다. test-time 선택(BoN)은 두 스택·세 데이터셋에서 닫혔고, 남은 문은 학습-시간 개입(벡터-SF critic)과
on-policy 반사실뿐이다.</p>
<h3>⑩ 08-11 — floq(flow-matching critic) 교차 리뷰: TD-SF-ARQ와의 접점</h3>
<p>워커A r56이 floq("critic을 velocity field로, K스텝 적분해 값 읽기", OGBench hard 1.8×)와 후속 해부를 정독.
핵심: 이득의 출처는 distributional 모델링이 아니라 ① test-time recovery(적분이 초기오차 감쇠) ②
<b>plasticity</b>(dense velocity 감독 → 비정상 TD 타깃에 피처 재가중), high-UTD에서 2×/5×. 워커A 판정:
<b>커버리지 축이 아니라 용량·최적화 축</b> — 온라인 phase 유효, demo-only 후보붕괴엔 무효(우리 결론과 정합).</p>
<p><b>우리 TD-SF-ARQ에 주는 함의:</b> ① 우리 <b>벡터 SF 타깃(전이당 128차원)이 floq의 plasticity 메커니즘(dense
감독)을 구조적으로 이미 내장</b> — 스칼라 TD의 1차원 감독 굶주림을 푸는 같은 원리다. ② floq의 velocity-field
critic은 우리 actor-critic 사다리(∂Q/∂a flow 조향)와 같은 계열이므로, phase-2 on-policy에서 critic 형태 후보로
사전등록에 추가. ③ 단 floq도 커버리지는 못 만든다 — 반사실 제조(on-policy)와 상보라는 우리 순서를 재확인.</p>
<p><b>⑪ floq 직접 구현·테스트 → 독립 리포트로 분리:</b> <span class="xref" data-eid="floq">floq — flow-matching critic 실증·시각화</span> (08-12) 참조.</p>

""",
)

# ============================================== 08-08 아침 종합
entry(
    "08-07",
    "morning-0808",
    "밤샘 종합 보고 (08-08 아침) — 무엇이 풀렸고 무엇이 남았나",
    "살아있음",
    """
<p><b>한 줄 요약.</b> 닷새를 막던 TD+mixed 학습 불능이 근본 원인(int32 초과 후보 버퍼) 수준에서 해결되어 FINAL 스윕
전 팔이 재가동됐고, 장면 암기 지름길을 차단한 v14 데이터가 완성됐으며, 보고 체계는 육하원칙·상호연결·백색 카드로 재설계됐다.
성공률 판정에서 VLA를 이긴 팔은 아직 없다.</p>
<h3>과학적 성과</h3>
<table class='num'><tr><th>항목</th><th>내용</th><th>상태</th></tr>
<tr><td>TD+mixed 침묵사</td><td>후보 버퍼 2.48×10⁹ 원소 &gt; INT32_MAX → sm_86/Blackwell gather 코드젠 segfault. 부차 발견: 클로저 캡처로 17GB가 XLA 상수로 복제(메모리 2배). 분할(cand_at)+pytree 인자화 픽스, 수치·RNG 불변</td><td class='good'>해결·실전 확인</td></tr>
<tr><td>qc (mixed, n=4)</td><td>Δ̄=−0.020 CI[−0.174,+0.134]</td><td>null 완결</td></tr>
<tr><td>td_max_demo (n=4)</td><td>Δ̄=−0.190 CI[−0.407,+0.027] — v11 TD 해악(−0.167, n=16)과 방향·크기 일치</td><td>null(검정력 한계)</td></tr>
<tr><td>K-per-scene 수집</td><td>450/450, 주방 45%가 혼합 결과 — 암기 지름길 차단 확인</td><td class='good'>완료</td></tr>
<tr><td>v14 데이터</td><td>demos+kroll 병합 605,684 프레임/964 에피소드 (merge_annot.py, episode offset 처리)</td><td class='good'>완료 · 3팔 학습중(iql_v14 99k)</td></tr>
<tr><td>무결 감사</td><td>v11·v12 공표 수치를 원본 JSON에서 재계산 — 일치</td><td class='good'>통과</td></tr></table>
<h3>인프라·보고 체계</h3>
<p>ft2 6팔(td_max/td_soft/td_aqcmax/a101/a201/online) 65–72k/100k — 오전 중 완주하면 평가 4시드+비디오 체인이 자동 개시된다.
calql_mixed 45k, calql_noprop 학습 완료→평가 4시드 방금 제출. 보고 체계: 전 리포트 육하원칙 헤더 + 상호연결 52개 +
데일리 스레드 + 마인드맵 + 백색 종이 카드 + 실험별 영상 임베드(24개) + plot 표준화(plot_style.py, 간결 타이틀) +
방법론 문서화(repo CLAUDE.md). 그림은 리포트 생성 때마다 원본 JSON에서 자동 재생성.</p>
<h3>오늘 낮 계획</h3>
<p>① ft2 완주 → FINAL 14팔 표 완성·최종 판정문 작성 ② v14 3팔 판정(장면 지름길 제거가 밴드→성공률 전환을 만드는가)
③ calql 2팔 판정(후보축 학습 신호의 첫 성공률 검증) ④ GR1 tabletop Teleop-Sim 데이터 다운로드 개시.</p>
""",
)

# ============================================== 08-07 TD 침묵사 + K-수집
entry(
    "08-07",
    "td-segv",
    "TD+mixed 침묵사 규명 — XLA 컴파일 segfault",
    "진행 중",
    f"""
{
        spec(
            [
                (
                    "증상",
                    "TD·QC·CalQL × mixed(17GB) 학습이 'ARQ critic: 10.22M params' 직후 트레이스백 없이 사망 — r1(96G)/r2(180G)/r3(250G) 전멸",
                ),
                ("판정 도구", "PYTHONFAULTHANDLER=1 + slurm ExitCode"),
                (
                    "공정성",
                    "동일 데이터·동일 배치의 IQL은 같은 노드(node26/28)에서 정상 학습 — 노드가 아니라 TD 프로그램이 원인",
                ),
            ]
        )
    }
<table class='num'><tr><th>가설</th><th>실험</th><th>결과</th></tr>
<tr><td>호스트 RAM 부족</td><td>mem 96→180→250G 증량</td><td class='bad'>기각 — 전부 사망, 1TB 노드(node58)에서도 사망</td></tr>
<tr><td>노드 불량</td><td>IQL을 같은 노드에서 학습</td><td class='bad'>기각 — IQL은 정상</td></tr>
<tr><td>정체 확인</td><td>faulthandler 스택</td><td><b>XLA backend_compile 내부 SIGSEGV(exit 139)</b> — 컴파일러 크래시</td></tr>
<tr><td>공유 컴파일 캐시 오염</td><td>캐시 on/off A/B</td><td class='bad'>기각 — 둘 다 segfault</td></tr>
<tr><td>autotune/병렬컴파일/배치형상</td><td>autotune0 · 직렬컴파일 · b64 3종</td><td class='bad'>기각 — 전부 segfault</td></tr>
<tr><td>cuDNN fMHA / Triton gemm 코드젠</td><td>플래그 off 3종(A6000)</td><td class='bad'>기각 — 전부 segfault</td></tr>
<tr><td><b>후보 버퍼 int32 초과</b></td><td>원소 수 계산 + cand 분할 픽스 A/B</td><td class='good'><b>확정</b> — 아래 참조</td></tr></table>
<p><b>근본 원인 (확정).</b> mixed 데이터의 후보 배열이 807,634×16×16×12 = <b>2.48×10⁹ 원소로 INT32_MAX(2.147×10⁹)를 15% 초과</b>.
이 버퍼를 인덱싱하는 XLA gather의 코드젠이 sm_86(A6000·3090)·Blackwell에서 컴파일 중 segfault하고 sm_89(RTX6000ADA)만 통과한다.
모든 관측이 이 하나로 설명된다: TD+demo는 8.6×10⁸(한도 이하)이라 3090에서도 정상, IQL+mixed는 토큰 버퍼 1.67×10⁹(한도 이하)만
gather하므로 정상, TD/CalQL+mixed(후보 gather)만 사망, 유일 생존 노드는 sm_89.</p>
<p><b>2차 발견 — 상수 복제.</b> 후보 분할로 컴파일을 뚫자 두 번째 문제가 노출됐다: 학습 스텝이 데이터를 클로저로 캡처해
<b>XLA가 17GB 데이터셋 전체를 프로그램 상수로 복제</b>하고 있었다(6.7GB 토큰 상수 할당 실패로 표면화). IQL mixed의 피크
31.8GB(데이터 17.2 + 상수 ~7.3 + 워크스페이스)의 정체이기도 하다. 픽스: cand를 서브-int32 조각으로 분할(<code>Data.cand_at</code>) +
Data를 pytree로 등록해 jit <b>인자</b>로 전달. 수치·RNG 스트림 불변 — 기존 학습 팔과의 공정 비교 유지.</p>
<p><b>검증 — 해결 확정.</b> ① 3090(sm_86): segfault 없이 컴파일 통과, 실행 단계 도달(24GB에는 예상대로 깔끔한 OOM).
② <b>A6000 실전: ft2_td_max가 node47에서 학습 진입 (step 100+, 4 it/s)</b> — 닷새간 어떤 노드에서도 못 밟던 첫 스텝.
8팔 전부 재제출 완료, 큐 순차 진행 중. 예상 학습 시간 팔당 ~7h.</p>
<p><b>아키텍처 의존성.</b> 유일한 통과는 RTX6000ADA(node52, 500스텝 진단 EXIT=0). A6000(sm_86)·RTXPRO6000(Blackwell) 전멸.
IQL과 TD의 차이는 16후보 forward(어텐션·gemm이 후보축으로 16×) — 후보축이 있는 프로그램만 특정 아키텍처 컴파일에서 죽는다.
CalQL도 후보축을 쓰므로 같이 죽는다(demo-only CalQL은 정상 학습 중, node44).</p>
<p><b>영향과 우회.</b> FINAL의 TD 계열 7팔 + calql_mixed가 지연. 우회 경로: ① 플래그 판정 시 해당 플래그로 A6000 함대 재제출,
② 실패 시 node52 파티션(RTX6000ADA×8, 현재 만석) 직렬 통과. 학습 외 파이프(평가·비디오·주석)는 영향 없음.</p>
""",
)

entry(
    "08-07",
    "kper",
    "K-per-scene 수집 완료 — 장면 정체성 지름길 제거 데이터",
    "진행 중",
    f"""
{
        spec(
            [
                (
                    "동기",
                    "기존 mixed는 주방당 롤아웃 1개 → 결과를 장면 정체성으로 외울 수 있음(암기 지름길). 같은 주방을 정책시드만 바꿔 K번 굴리면 지름길 차단",
                ),
                (
                    "수집",
                    "--policy-seed 분리(커밋 922d7d4) · 주방 150개(장면시드 1000–1400×30) × 정책시드 3 = 450 롤아웃 · VLA 동결 그대로",
                ),
                (
                    "사고록",
                    "1차 15잡 중 11잡 bad-node 사망 + 덤프 파일명에 정책시드 누락으로 p1/p2/p3 상호 덮어쓰기 발견 → 잡별 하위디렉토리로 재수집(450/450 완료)",
                ),
            ]
        )
    }
<table class='num'><tr><th>지표</th><th>값</th></tr>
<tr><td>VLA 성공률 (450 롤아웃)</td><td>0.676</td></tr>
<tr><td>혼합 결과 주방 (성공·실패 공존)</td><td><b>68/150 (45%)</b></td></tr>
<tr><td>전부 성공 주방</td><td>63/150</td></tr>
<tr><td>전부 실패 주방</td><td>19/150</td></tr></table>
<p><b>해석.</b> 주방의 45%에서 같은 장면이 성공도 실패도 한다 — 이 68개 주방에서는 critic이 장면 정체성으로 결과를 맞힐 수 없고,
상태·행동에서 신호를 찾아야만 한다. held-out 프로브에서 확인된 보수 편향(낯선 성공 저평가)도 이 데이터가 직접 겨냥한다.
현재 450궤적 × 프레임당 VLA 16후보 주석을 8샤드로 병렬 진행 중(annot/kroll) → 완료 시 데모와 병합해 <b>v14 mixed</b>로
FINAL 승자 방법을 재학습한다.</p>
""",
)


entry(
    "08-07",
    "video-gallery",
    "HUD 롤아웃 비디오 갤러리",
    "살아있음",
    """
<p>대표 롤아웃 영상. HUD 읽는 법: 오른쪽 trace의 회색 밴드 = 후보 16개 Q 분포(q01–q99), 파란 선 = 실행된 chunk의 Q,
빨간 선 = V(z), 왼쪽 grid = Q[후보, 커밋길이]와 선택 칸. 전체 아카이브:
<a href="https://huggingface.co/datasets/jellyho/acrft-rollout-videos" target="_blank">acrft-rollout-videos</a>.</p>
<table class='num'><tr><th>영상</th><th>보는 포인트</th></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/v12_mixed_critic_success.mp4"></video></td><td>혼합(v12) critic 성공 — 열린 밴드와 V의 동행</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/v12_mixed_vla_fail_same_scene.mp4"></video></td><td>같은 장면에서 vla는 실패 — 페어드 비교의 실제 사례</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/v12_mixed_critic_fail_a.mp4"></video></td><td>혼합 critic 실패(머그 이탈형) — 실패 순간 밴드·V 반응</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/heldout_fail_rise_collapse_t05.mp4"></video></td><td>held-out 실패의 상승→붕괴 V (일반화 증거, 프로브 영상)</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/v11_demoonly_critic_success.mp4"></video></td><td>데모-only(v11) — 닫힌 밴드(후보 무구분)와 대조</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/aqc_demoonly_fail.mp4"></video></td><td>AQC 배포 규칙 — h 선택 정상화 후의 commit 패널</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/final_td_max_demo/PrepareCoffee_critic_t00_fail.mp4"></video></td><td>FINAL td_max_demo critic 실패 (t00) — 같은 장면에서 vla는 성공. 6장면 잠정: critic 2/6 vs vla 4/6 (n=6 비디오잡, 판정은 4시드×50 평가로)</td></tr>
<tr><td><video controls preload="none" style="max-width:100%" src="videos/final_td_max_demo/PrepareCoffee_vla_t00_succ.mp4"></video></td><td>위와 동일 장면(t00)의 vla 성공 — v11 TD 유해 시그니처(placed_no_press류)와의 비교용</td></tr>
</table>
""",
)

# ================================================================== 육하원칙 + 상호 연결
# 모든 리포트에 표준 5W1H 헤더를 달고(과학 보고 원칙), 연결된 리포트를 명시한다.
# date: 허브(시간순 정렬)에 쓰는 실제 ISO 날짜. links: 이 리포트가 근거로 삼거나 후속으로 이어지는 eid.
META = {
    "v14": {
        "date": "2026-08-08 08:40",
        "who": "워커B · iql/td_max/calql × mixed_v14",
        "where": "a6000 풀(학습) + 3090 풀(평가)",
        "what": "장면 지름길 제거 데이터(v14)가 판정을 바꾸는지",
        "how": "FINAL 레시피 + 4시드 페어드 평가",
        "why": "held-out 보수 편향의 원인 후보였던 암기 지름길을 제거한 첫 판정",
        "links": ["kper", "v12", "final"],
    },
    "calql": {
        "date": "2026-08-08 08:45",
        "who": "워커B · CalQL(CO-RFT) critic",
        "where": "3090/a6000 풀",
        "what": "최초의 학습-시간 후보축 신호(CQL 항)의 성공률 판정",
        "how": "demo-only 8시드 + mixed(학습중) 페어드 평가",
        "why": "모든 추론-시간 트릭이 실패한 후보 구분을 학습 신호로 만들 수 있는지",
        "links": ["v12", "final", "wcurse"],
    },
    "horizon-probe": {
        "date": "2026-08-11 03:30",
        "who": "워커B(설계) + 사용자(제안·확정)",
        "where": "설계 문서 (구현 전)",
        "what": "커버리지 역방향 ablation — OGBench에서 반사실 밀도만 다이얼해 TD-BoN 붕괴점 특정",
        "how": "TD-BoN이 OGBench서 성공 = 존재증명. 같은 env 반사실 밀도만 감축 → demo-only 레짐 재현 여부",
        "why": "커버리지(반사실)가 OGBench 장점이자 VLA null의 인과 변인 후보 — 한 변인 통제로 확정",
        "links": ["conservatism", "phi-ladder", "tdsf-arq", "papers-byolg", "papers-tdjepa", "gr1-port", "final"],
    },
    "embed-compare": {
        "date": "2026-08-11 06:30",
        "who": "워커B + 사용자(방법 제안)",
        "where": "PrepareCoffee mixed 어노테이션 (frozen RLT 토큰)",
        "what": "raw/PCA/phi/BYOL-gamma/TD-JEPA 구조 배터리 + cross-episode 이웃 이미지 패널 + BC probe 통합",
        "how": "같은 토큰 위 readout 학습, 동일 배터리 채점, kroll 이미지로 이웃 패널",
        "why": "사용자 요청 — 어떤 목적식이 궤적을 가로질러 붙이나. 답: cross-episode 대조가 열쇠",
        "links": [
            "phi-ladder",
            "papers-byolg",
            "papers-tdjepa",
            "papers-dbc",
            "tdsf-arq",
            "xworker-0808",
            "conservatism",
        ],
    },
    "papers-dbc": {
        "date": "2026-08-11 05:10",
        "who": "워커B(리뷰)",
        "where": "arXiv",
        "what": "DBC/bisimulation(2006.10742) 정독 — 우리가 원하는 semantic invariance의 정면 정의",
        "how": "원문 인용 병기, SR/BYOL-γ와 보상축 유무로 대조, invariance=커버리지 해금 종합",
        "why": "사용자 질문 'semantic 비슷하면 임베딩도 비슷하게, 다른 연구 있나?'",
        "links": ["papers-byolg", "papers-tdjepa", "phi-ladder", "tdsf-arq", "xworker-0808"],
    },
    "papers-byolg": {
        "date": "2026-08-11 02:40",
        "who": "워커B(리뷰)",
        "where": "arXiv",
        "what": "BYOL-γ(2506.10137) 정독 — TD-JEPA 계보 확인 + TD-SF-ARQ A단계 MC 팔 도출",
        "how": "원문 정독(인용구 병기), TD-JEPA·워커A bridging 발견과 대조",
        "why": "사용자 질문 'TD-JEPA의 선조격?' — demo-only 설정에 더 가까운 조상 확인",
        "links": ["papers-tdjepa", "tdsf-arq", "phi-ladder", "xworker-0808"],
    },
    "tdsf-arq": {
        "date": "2026-08-10 21:20",
        "who": "워커B(설계) + 사용자(논점 두 개)",
        "where": "설계 문서 (구현 전)",
        "what": "TD-JEPA 증류: ARQ critic 출력을 벡터 SF로 교체하는 단일 TD 손실 + A/B/C 사다리 + actor-critic 확장",
        "how": "TD-JEPA 리뷰 → 사용자 질문 2건 → 우리 진단(+7.3% 압축좌표·스칼라 TD 굶주림)과 결합",
        "why": "phase-2 critic의 사전등록 — 접합부 없는 행동조건 목적식으로 희미한 신호를 최대 추출",
        "links": ["papers-tdjepa", "model-based", "phi-ladder", "gr1-port", "conservatism", "final"],
    },
    "papers-tdjepa": {
        "date": "2026-08-10 18:10",
        "who": "워커B(리뷰)",
        "where": "arXiv",
        "what": "TD-JEPA(arXiv:2510.00739) 정독 — HILP 대비, phase-2 critic 후보 판단",
        "how": "원문 정독 후 우리 판정(합성게이트 null·반사실 부재)과 대조",
        "why": "사용자 질문 'HILP 말고 TD-JEPA는 어때?' — 표현 학습의 다음 수 선택",
        "links": ["phi-ladder", "model-based", "xworker-0808", "gr1-port", "papers-value-steering"],
    },
    "papers-value-steering": {
        "date": "2026-08-08 13:30",
        "who": "워커B(리뷰)",
        "where": "arXiv",
        "what": "Robo-ValueRL(arXiv:2607.09866) 정독 + 인접 3편(V-GPS·Q-VGM·프로빙) 리뷰",
        "how": "원문 정독 후 FINAL 결론·우리 스택과 대조",
        "why": "'robo-value-RL' 탐색 요청 — FINAL null의 해석과 다음 수(그래디언트 조향)의 문헌 근거",
        "links": ["final", "calql", "wcurse"],
    },
    "floq": {
        "date": "2026-08-12 03:00",
        "who": "워커B(구현·실험)",
        "where": "우리 ARQ critic (flow_head) + PrepareCoffee mixed annot",
        "what": "floq 원본 충실 구현(critic.py flow_head) + value-fit 테스트 + flow funnel·궤적 HUD 영상",
        "how": "ARQ 트렁크 그대로, head를 velocity field로, 손실·부트스트랩만 floq(Eq4.2)",
        "why": "사용자 지시 '멋대로 말고 원본대로 우리 critic에' — 워커A floq 해석 상호 재현",
        "links": ["critic-heads", "xworker-0808", "papers-tdjepa", "tdsf-arq", "conservatism", "final"],
    },
    "critic-heads": {
        "date": "2026-08-12 15:30",
        "who": "워커B(구현·실험)",
        "where": "우리 AQC critic (scalar/HL-Gauss/floq head) + PrepareCoffee (오프라인 mixed annot + closed-loop 롤아웃)",
        "what": "critic head 3종 비교 — 오프라인 랭킹 3지표 + closed-loop BoN 성공률(vla/rand/critic별, n=25, N=8)",
        "how": "같은 AQC 트렁크에 head/loss만 3종, γ=0.997·단일샘플 부트스트랩; critic 저장→VLA BoN 롤아웃, scene-paired McNemar",
        "why": "사용자 지시 '실제 evaluation으로 critic이 VLA를 향상시키는지 테스트' + flow냐 categorical이냐 가르기",
        "links": ["floq", "critic-pfx", "conservatism", "embed-compare", "tdsf-arq", "model-based", "final"],
    },
    "critic-pfx": {
        "date": "2026-08-12 19:00",
        "who": "워커B(구현·실험)",
        "where": "우리 AQC critic (per-prefix td-max) + PrepareCoffee closed-loop",
        "what": "부트스트랩 교정 재검 — td-max over 후보 + per-prefix + (후보×prefix) joint argmax, 여전히 critic이 VLA 못 이김",
        "how": "프로덕션 targets() 충실 재현(착지 후보 max·per-prefix·mc하한); scalar/HLG/floq 동일 타깃, joint-argmax 롤아웃 + randh null",
        "why": "사용자 지적 'TD면 샘플 액션 max로 부트스트랩·per-prefix로 내야' — 앞 실험의 데모-부트스트랩 결함 교정",
        "links": ["critic-heads", "deas", "model-based", "conservatism", "calql", "final"],
    },
    "deas": {
        "date": "2026-08-13 02:00",
        "who": "워커B(리뷰·구현) → 워커A에 알림",
        "where": "arXiv:2510.07730 + github DEAS-Isaac-GR00T + 우리 pi05 백본(eval_deas.py)",
        "what": "정정 — 우리 BoN 실패는 td-max 과대평가였다; DEAS의 detached value learning(expectile-V 부트스트랩)을 우리 백본에 이식",
        "how": "DEAS 코드 실측(V=HLGauss+expectile, Q는 V로 부트스트랩, double-min, dual-discount); 우리 백본·주석 유지, 방법론만",
        "why": "사용자 지적 'cand[0]도 VLA 샘플인데 BoN이 그보다 못할 리 없다' — 앞선 coverage 결론의 정정 가능성",
        "links": ["critic-pfx", "critic-heads", "floq", "conservatism", "calql", "model-based", "final"],
    },
    "exp-board": {
        "date": "2026-08-13 15:45",
        "who": "워커B(운영)",
        "where": "허브 living 엔트리",
        "what": "실험 보드 — 계획/진행중/완료를 담당·wandb·리포트와 함께 한자리에",
        "how": "space_add_entry로 같은 eid replace하며 상시 갱신; 리포트는 data-eid xref",
        "why": "사용자 지시 '실험 하려다 까먹고 돌려놓고 정리 안 됨 — 실험 탭 만들자'",
        "links": ["mb-arq", "deas", "tdsf-arq", "gr1-port", "conservatism"],
    },
    "mb-arq": {
        "date": "2026-08-13 09:00",
        "who": "워커B(해설)",
        "where": "개념 설명 (워커A train_mve_critic.py·train_cheapz_dynamics 참조)",
        "what": "model-based critic(MVE)을 비유로 쉽게 설명 — '상상하는 심판' + 수정구슬(월드모델)을 어떻게 배우나·왜 어렵나",
        "how": "수식 최소화, 체스엔진·수정구슬 비유로 처음부터; 우리 coverage·승자의저주·임베딩 findings와 연결",
        "why": "사용자 요청 '너무 어렵다 — model-based ARQ를 포스트 하나로 쉽게 설명해달라'",
        "links": ["model-based", "deas", "embed-compare", "conservatism"],
    },
    "xworker-0808": {
        "date": "2026-08-08 14:10",
        "who": "워커B(리뷰) ← 워커A 리포트 7건",
        "where": "공유 허브",
        "what": "교차 워커 배움 — 상호 재현·McNemar 도입·표현/데이터 공격로 비교",
        "how": "워커A 전 리포트 정독 후 즉시 반영",
        "why": "워커끼리 서로 배우라는 지시 — 독립 스택의 결론 상호 검증",
        "links": ["final", "kper", "papers-value-steering", "gr1-port", "model-based"],
    },
    "conservatism": {
        "date": "2026-08-08 15:20",
        "who": "워커B(종합)",
        "where": "전 판정 결과 재해석",
        "what": "보수성 2축 프레임 — 전 null의 통일 해석과 축2(추정 오차) 처방",
        "how": "방법별 다이얼 위치 분류 + SNR 조건",
        "why": "BoN이 완벽한 BC 정규화인데도 위험한 역설의 해소 — 다음 수(다양화+LCB+랭킹)의 논리적 근거",
        "links": ["final", "calql", "v14", "wcurse", "papers-value-steering", "xworker-0808"],
    },
    "morning-0808": {
        "date": "2026-08-08 06:45",
        "who": "워커B(Claude)",
        "where": "클러스터 전체 + HF Space",
        "what": "밤샘(08-07 밤~08-08 아침) 작업 종합",
        "how": "각 리포트의 완결 결과를 표로 집약",
        "why": "아침에 전체 상황을 한 번에 파악할 수 있도록",
        "links": ["td-segv", "final", "kper", "v12"],
    },
    "flow": {
        "date": "2026-08-08 06:40",
        "who": "워커B(Claude)",
        "where": "요약 뷰 — 클러스터 전체",
        "what": "프로젝트 전체 타임라인과 현재 위치",
        "how": "각 완결 리포트의 결론을 시간축 노드로 요약",
        "why": "개별 실험이 어떤 흐름에서 나왔는지 한눈에 보기 위해",
        "links": ["v11", "v12", "final"],
    },
    "genesis": {
        "date": "2026-07-30 22:00",
        "who": "워커B · VLA π0.5@70k 동결 · TD critic v1–v5",
        "where": "3090 풀",
        "what": "첫 TD critic 세대들과 BoN16 배포의 첫 성적",
        "how": "TD 부트스트랩 학습 → BoN 롤아웃 페어드 평가",
        "why": "RLT 임베딩 위 가치기반 후보 선택이 성립하는지 첫 검증",
        "links": ["vbias", "families"],
    },
    "vbias": {
        "date": "2026-08-01 14:00",
        "who": "워커B · TD critic",
        "where": "3090 풀",
        "what": "TD 타깃이 목표거리 d에 따라 갖는 구조적 바이어스 b(d)",
        "how": "거리별 Q 잔차 프로브 회귀",
        "why": "genesis 적자의 원인 후보를 계통적으로 분리",
        "links": ["genesis", "wcurse"],
    },
    "families": {
        "date": "2026-08-02 20:00",
        "who": "워커B · TD/QC/IQL/AQC critic",
        "where": "3090 풀",
        "what": "방법 패밀리별 롤아웃 총결산",
        "how": "동일 데이터·동일 장면 페어드 비교",
        "why": "방법 선택 근거 — TD 적자 확인 후 IQL 전환의 문서화",
        "links": ["genesis", "v11"],
    },
    "wcurse": {
        "date": "2026-08-03 15:00",
        "who": "워커B",
        "where": "3090 풀 (프로브)",
        "what": "argmax 선택의 winner's curse 정량화",
        "how": "후보 Q 분산 분해(상태 vs 후보축) + 두 argmax(후보/프리픽스) 분리",
        "why": "BoN이 이득을 못 내는 구조적 이유 규명",
        "links": ["vbias", "duel", "aqc"],
    },
    "duel": {
        "date": "2026-08-03 18:00",
        "who": "워커B · dueling ARQ",
        "where": "3090 풀",
        "what": "dueling 게이지 자유도로 인한 학습 실패 2회와 해법",
        "how": "zero-mean advantage로 (V+c, A−c) 게이지 고정",
        "why": "V+A 분해 도입 시 절대 레벨이 유일하게 정의되도록",
        "links": ["wcurse"],
    },
    "singlefit": {
        "date": "2026-08-05 04:00",
        "who": "워커B · TD critic",
        "where": "3090 1노드",
        "what": "단일 궤적 과적합으로 terminal 처리 검증",
        "how": "1궤적 fit 후 Q@goal, corr(Q,mc) 확인",
        "why": "--terminal-uses-mc 누락이 학습 전체를 망치던 버그의 최소 재현",
        "links": ["ladders", "fullfit"],
    },
    "ladders": {
        "date": "2026-08-05 11:00",
        "who": "워커B",
        "where": "3090 풀 (스윕)",
        "what": "데이터 사다리 1→64 에피소드 × objective × γ",
        "how": "각 조합 학습 후 fit 지표 격자",
        "why": "필요 데이터 규모와 discount 선택 근거",
        "links": ["singlefit", "fullfit", "v11"],
    },
    "fullfit": {
        "date": "2026-08-05 15:00",
        "who": "워커B",
        "where": "3090 풀",
        "what": "full-data critic 품질 검수",
        "how": "fit 지표 + 궤적 시각화 게이트",
        "why": "롤아웃 평가 투입 전 최소 품질 게이트",
        "links": ["ladders", "highpower"],
    },
    "highpower": {
        "date": "2026-08-05 21:00",
        "who": "워커B",
        "where": "3090 풀 (n↑ 롤아웃)",
        "what": "고검정력 롤아웃 판정 (softcand·e70 재현·softmax)",
        "how": "시드·트라이얼 수 증대로 CI 폭 축소",
        "why": "작은 효과도 걸러낼 검정력 확보",
        "links": ["fullfit", "randh", "v11"],
    },
    "randh": {
        "date": "2026-08-06 10:00",
        "who": "워커B · critic vs 동전던지기",
        "where": "3090 풀",
        "what": "랜덤 h 대조 실험 — critic의 능동 손실 분리",
        "how": "randh 모드 페어드 롤아웃",
        "why": "critic 선택이 무작위보다 못한지(능동적 해악) 판정",
        "links": ["highpower", "autopsy"],
    },
    "aqc": {
        "date": "2026-08-06 14:00",
        "who": "워커B · AQC critic",
        "where": "3090 풀",
        "what": "AQC(베이스라인 보정 argmax) 구현과 demo-only 판정",
        "how": "b_h 학습 + z_ε(Q−b) 배포, h-collapse 교정",
        "why": "프리픽스 헤드 간 계통 바이어스 제거",
        "links": ["wcurse", "v11"],
    },
    "autopsy": {
        "date": "2026-08-06 17:00",
        "who": "워커B",
        "where": "평가 로그 (3090 풀)",
        "what": "실패 유형 부검",
        "how": "env 술어 단계 로그(stage_flags)로 프로그램적 분류",
        "why": "어디서 지는지 — grasp 0%·엔드게임 2/3 — 개선 표적화",
        "links": ["randh", "failpipe", "kper"],
    },
    "pools": {
        "date": "2026-08-06 19:00",
        "who": "워커B",
        "where": "평가 JSON 재분석",
        "what": "장면 풀이 성공률에 주는 효과",
        "how": "한 체크포인트를 풀별로 분해",
        "why": "풀 혼동이 ±0.1 흔들던 비교 방법론의 교정",
        "links": ["v11", "highpower"],
    },
    "failpipe": {
        "date": "2026-08-06 22:00",
        "who": "워커B",
        "where": "3090 풀",
        "what": "실패 롤아웃 수집·주석 파이프라인 + in-dist 장면 재현",
        "how": "dump-traj → annotate_rollouts → memmap",
        "why": "v12 mixed 데이터의 재료 — 실패를 본 critic 만들기",
        "links": ["autopsy", "v12"],
    },
    "v11": {
        "date": "2026-08-07 03:00",
        "who": "워커B · 4방법 × 16시드",
        "where": "3090 풀 (64런)",
        "what": "demo-only 공정 비교 완결",
        "how": "method-only-diff 체크포인트, in-job 페어드, run-level 95% t-CI",
        "why": "사전등록 판정 — TD 확실 해악, IQL/QC/AQC null → 남은 지렛대는 데이터",
        "links": ["families", "aqc", "pools", "final"],
    },
    "v12": {
        "date": "2026-08-07 10:00",
        "who": "워커B · iql/aqc × mixed",
        "where": "a6000 풀 (17GB 상주)",
        "what": "혼합 데이터 판정 — 밴드 개방과 성공률",
        "how": "v11 프로토콜 + held-out 프로브(시드 9100)",
        "why": "실패 데이터가 후보 구분을 여는지 — 열림(10–30×) but 성공률 null",
        "links": ["failpipe", "v11", "kper", "final"],
    },
    "final": {
        "date": "2026-08-08 12:50",
        "who": "워커B · 14팔 × 4시드",
        "where": "a6000(학습)+3090(평가) 풀",
        "what": "전 요인 사전등록 스윕 (방법×부트스트랩×atoms×타깃넷×데이터)",
        "how": "공통 레시피 고정, 팔당 4×50 페어드, 동일 장면",
        "why": "파편화된 실험을 하나의 정당한 비교로 — 최종 판정",
        "links": ["v11", "v12", "td-segv", "video-gallery"],
    },
    "td-segv": {
        "date": "2026-08-08 02:50",
        "who": "워커B",
        "where": "A6000·3090·PRO6000·RTX6000ADA 교차 검증",
        "what": "TD+mixed 학습 침묵사의 근본 원인",
        "how": "가설 기각 사다리(6단계) + faulthandler + A/B 진단",
        "why": "FINAL의 TD 계열 7팔이 전멸하던 인프라 병목 제거",
        "links": ["final"],
    },
    "kper": {
        "date": "2026-08-08 00:40",
        "who": "워커B · VLA 동결",
        "where": "a6000 풀 (수집) + 3090 풀 (주석)",
        "what": "K-per-scene 데이터 — 주방당 정책시드 3롤아웃",
        "how": "--policy-seed 분리, 150주방 × 3, 8샤드 주석",
        "why": "장면 정체성 암기 지름길 차단 (혼합결과 주방 45%)",
        "links": ["v12", "autopsy"],
    },
    "video-gallery": {
        "date": "2026-08-08 09:00",
        "who": "워커B",
        "where": "HF Space 서빙",
        "what": "대표 롤아웃 HUD 비디오",
        "how": "팔당 6장면 fvid 잡, 성공/실패 페어 선별",
        "why": "숫자 판정을 눈으로 검증 — 밴드·V·commit 패널 동행 확인",
        "links": ["v11", "v12", "final"],
    },
}


def _git_stamp() -> str:
    """branch@hash(+dirty) at publish time — every posted report carries the code state."""
    import subprocess

    def g(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=pathlib.Path(__file__).parent.parent, check=False
        ).stdout.strip()

    branch = g("rev-parse", "--abbrev-ref", "HEAD") or "?"
    sha = g("rev-parse", "--short", "HEAD") or "?"
    dirty = "+dirty" if g("status", "--porcelain") else ""
    return f"{branch}@{sha}{dirty}"


GIT_STAMP = _git_stamp()

_titles = {e[1]: e[2] for e in ENTRIES}

# 실험별 관련 영상 (Space videos/ 에서 서빙). 항목: (경로, 한 줄 설명)
VIDEOS = {
    "v11": [
        ("videos/v11_demoonly_critic_success.mp4", "demo-only critic 성공 — 닫힌 밴드(후보 무구분)의 전형"),
        ("videos/v11_demoonly_critic_fail.mp4", "demo-only critic 실패 — 같은 닫힌 밴드에서의 실패 사례"),
    ],
    "aqc": [
        ("videos/aqc_demoonly_success.mp4", "AQC 배포 규칙 성공 — h-collapse 교정 후 commit 패널"),
        ("videos/aqc_demoonly_fail.mp4", "AQC 실패 사례 — 같은 규칙의 한계"),
    ],
    "v12": [
        ("videos/v12_mixed_critic_success.mp4", "mixed critic 성공 — 열린 밴드와 V의 동행"),
        ("videos/v12_mixed_vla_fail_same_scene.mp4", "같은 장면의 vla 실패 — 페어드 비교 실사례"),
        ("videos/v12_mixed_critic_fail_a.mp4", "mixed critic 실패(머그 이탈형) — 실패 순간 밴드·V 반응"),
        ("videos/heldout_fail_rise_collapse_t05.mp4", "held-out 실패의 상승→붕괴 V — 일반화 증거"),
        ("videos/heldout_fail_rise_collapse_t07.mp4", "held-out 상승→붕괴 두 번째 사례"),
    ],
    "final": [
        ("videos/final_td_max_demo/PrepareCoffee_critic_t00_fail.mp4", "td_max_demo critic 실패 (t00)"),
        ("videos/final_td_max_demo/PrepareCoffee_vla_t00_succ.mp4", "같은 장면(t00) vla 성공 — TD 해악 시그니처 비교"),
        ("videos/final_td_max_demo/PrepareCoffee_critic_t03_succ.mp4", "td_max_demo critic 성공 사례 (t03)"),
        ("videos/final_qc/PrepareCoffee_critic_t01_succ.mp4", "qc(mixed) critic 성공 (t01) — 같은 장면 vla는 실패"),
        ("videos/final_qc/PrepareCoffee_vla_t01_fail.mp4", "같은 장면(t01) vla 실패 — qc가 이긴 페어"),
        ("videos/final_qc/PrepareCoffee_critic_t05_fail.mp4", "qc critic 실패 (t05) — vla는 성공한 장면"),
    ],
}
VIDEOS["autopsy"] = VIDEOS["v12"][2:3]  # 실패 유형 실사례
VIDEOS["kper"] = [
    ("videos/v12_mixed_vla_fail_same_scene.mp4", "같은 주방이 실패하는 사례 — K-per-scene이 겨냥하는 혼합 결과의 실체")
]


def _video_block(eid):
    vids = VIDEOS.get(eid)
    if not vids:
        return ""
    rows = "".join(
        f"<tr><td><video controls preload='none' style='max-width:100%' src='{src}'></video></td><td>{cap}</td></tr>"
        for src, cap in vids
    )
    return (
        "<h3>관련 영상</h3><p class='sub'>HUD 읽는 법: 회색 밴드 = 후보 16개 Q 분포(q01–q99), 파란 선 = 실행 chunk의 Q, "
        "빨간 선 = V(z). 전체 아카이브는 '비디오 갤러리' 리포트 참조.</p>"
        f"<table class='num'><tr><th>영상</th><th>보는 포인트</th></tr>{rows}</table>"
    )


def _decorate(eid, body):
    m = META.get(eid)
    if not m:
        return body
    w6 = (
        "<table class='spec w6'>"
        + "".join(
            f"<tr><th>{k}</th><td>{m[f]}</td></tr>"
            for k, f in [
                ("누가", "who"),
                ("언제", "date"),
                ("어디서", "where"),
                ("무엇을", "what"),
                ("어떻게", "how"),
                ("왜", "why"),
            ]
        )
        + f"<tr><th>코드</th><td><code>{GIT_STAMP}</code> (게시 시점 repo 상태 — branch@hash)</td></tr>"
        + "</table>"
    )
    links = "".join(
        f"<span class='xref' data-eid='{lk}'>{_titles.get(lk, lk)}</span>" for lk in m.get("links", []) if lk in _titles
    )
    tail = _video_block(eid)
    tail += f"<p class='xrefs'><b>연결된 리포트</b> {links}</p>" if links else ""
    return w6 + body + tail


ENTRIES[:] = [(d, eid, t, st, _decorate(eid, b)) for d, eid, t, st, b in ENTRIES]

# ================================================================== English versions (KO/EN toggle)
en(
    "flow",
    "Timeline · Takeaways",
    """
<p>The project's arc as takeaway cards: pipeline built (annotation → critic → paired rollouts) → early TD deficits
→ terminal-handling bug found via single-trajectory overfit → data ladders and full-data gates → the demo-only
verdict (v11: TD definitively harmful −0.167; IQL/QC/AQC null, 16 seeds) → mixed data opens the candidate band
10–30x but success rates do not move (v12) → the FINAL 14-arm preregistered sweep (all null) → segfault
root-caused to an INT32 overflow → K-per-scene data kills the scene-identity shortcut → conservatism synthesis →
model-based route closed offline → GR1 port. Current position: every offline route on PrepareCoffee is closed;
the next arena is GR1 (two decisions pending).</p>
""",
)

en(
    "genesis",
    "The first TD critic generations and the first deficit",
    """
<p><b>What/why.</b> First test of value-based candidate selection on frozen RLT embeddings: TD-bootstrap critics
(v1–v5) deployed Best-of-16. <b>Result: consistent deficits vs the VLA</b>, opening the diagnostic program:
targets, values, or selection rule?</p>
""",
)

en(
    "vbias",
    "Distance-structure bias of TD targets (the b(d) probe)",
    """
<p><b>What.</b> TD-critic error regressed against distance-to-goal d shows a systematic bias b(d) growing with
distance — long-horizon states are misvalued in a structured way; an early candidate explanation for the
genesis deficits.</p>
""",
)

en(
    "families",
    "Rollout totals across method families (the TD→IQL pivot)",
    """
<p><b>What.</b> Paired rollout totals for TD/QC/IQL/AQC on identical scenes. TD kept losing; IQL variants were
least harmful — the documented basis for pivoting to IQL.</p>
""",
)

en(
    "wcurse",
    "Anatomy of the winner's curse — variance decomposition, two argmaxes",
    """
<p><b>What.</b> Candidate-Q error decomposes into a state-axis part (shared by all candidates, ~88% on the
prefix axis) and a candidate-axis part (what selection uses). Within-row comparisons are paired-safe;
cross-candidate argmax harvests noise. Later the backbone of the conservatism synthesis.</p>
""",
)

en(
    "duel",
    "Two dueling-gauge failures and the zero-mean fix",
    """
<p><b>What.</b> Q=V+A failed twice because (V+c, A−c) is a gauge freedom. Fix: zero-mean advantage per state,
pinning the absolute level to V (scalar ARQ only).</p>
""",
)

en(
    "singlefit",
    "Single-trajectory fit — validating terminal handling",
    """
<p><b>What.</b> Overfitting one trajectory exposed the missing terminal reward in the bootstrap
(--terminal-uses-mc): corr(Q, mc) −0.75 → +0.983, Q at goal 0.00 → 0.84 within 6k steps. One flag was the
entire failure.</p>
""",
)

en(
    "ladders",
    "Data ladders (1→64 episodes) × objective × γ",
    """
<p><b>What.</b> Fit metrics across the grid: data requirements per objective and the value-scale effect of γ.
Basis for the γ=0.995 default and the recipes used from v11 onward.</p>
""",
)

en(
    "fullfit",
    "Full-data critic inspection",
    """
<p><b>What.</b> The pre-rollout quality gate: fit metrics and trajectory visualisations a critic must pass
before evaluation GPU is spent.</p>
""",
)

en(
    "highpower",
    "High-power rollout verdicts (softcand / e70 replication / softmax)",
    """
<p><b>What.</b> Seeds and trials raised until CIs could detect ~0.05 effects: softcand, e70 replication and
smooth-max — all null. Early apparent gains did not survive statistical power.</p>
""",
)

en(
    "randh",
    "The coin-flip experiment — measuring active harm",
    """
<p><b>What.</b> rand (uniform over the 16 candidates — structurally the VLA's own distribution; n=71,
Δ̄ −0.020 CI[−0.054,+0.015]) and randh (random commit length too) as honest nulls. A critic below rand shows
active harm — TD did. rand≈vla later anchored the SNR argument.</p>
""",
)

en(
    "aqc",
    "AQC implementation and the demo-only verdict",
    """
<p><b>What.</b> AQC-style deployment: per-prefix baselines b_h + epsilon'd z-score across commit lengths;
cured the h-collapse (61% h=2 → mean h≈11). Demo-only verdict, 16 seeds: null (0.000).</p>
""",
)

en(
    "autopsy",
    "Failure autopsy — programmatic stage predicates",
    """
<p><b>What.</b> Env-predicate stages (grasped/placed/machine_on) logged every step — no eyeball classification.
Failures ~2/3 endgame; grasp ~0%; TD's signature is placed-no-press dithering; the button press is nearly
invisible in a single frame (a structural ambiguity later cited by the history experiments).</p>
""",
)

en(
    "pools",
    "Scene-pool effects — fixing the evaluation methodology",
    """
<p><b>What.</b> The same checkpoint swings ±0.1 across scene pools; only within-pool, in-job-paired comparisons
are valid. Retired several earlier cross-pool claims; the protocol used everywhere since.</p>
""",
)

en(
    "failpipe",
    "Failure-data pipeline + in-distribution scene replay",
    """
<p><b>What.</b> dump-traj → annotate_rollouts → memmap: failure rollouts became training data (v12 mixed);
stored ep_meta replays exact scenes for in-distribution evaluation.</p>
""",
)

en(
    "v11",
    "v11 fair comparison — 16-seed CIs, complete",
    f"""
<p><b>What.</b> The demo-only preregistered verdict: 4 methods × 16 seeds × 50 scenes, method-only-diff
checkpoints, in-job pairing.</p>
{img(P / "16_run_level.png", "v11 forest")}
<p><b>TD: definitively harmful</b> (Δ̄=−0.167 CI[−0.214,−0.119], 16/16 negative); QC −0.038, IQL +0.004,
AQC 0.000 — null. On demo-only data no method helps and TD actively hurts; the remaining lever is data.</p>
""",
)

en(
    "v12",
    "v12 mixed data — band opening vs the success-rate judge",
    """
<p><b>What.</b> Failure rollouts added to training. The mechanism moved: candidate band opened 10–30x
(0.065–0.107 vs 0.002–0.023). Success did not: iql Δ̄=−0.017 CI[−0.068,+0.035], aqc −0.019 (n=16) — null.
Held-out probe (seed 9100): genuine rise-then-collapse V on unseen failures, plus a conservative bias on unseen
successes — later targeted by K-per-scene data.</p>
""",
)

en(
    "final",
    "FINAL campaign — the preregistered all-factor sweep",
    f"""
<p><b>What.</b> 14 arms (method × bootstrap-op × atoms × target-net × data), one fixed recipe, 4 seeds × 50
scenes per arm, identical scenes across arms; verdict = run-level 95% t-CI + trial-paired McNemar.</p>
{img(P / "20_final_forest.png", "FINAL forest")}
<p><b>Verdict (14/14): no factor combination beats the VLA.</b> Point estimates −0.190…+0.040 (mean −0.051),
all CIs cover zero; McNemar sharpens three TD arms into significant harm (td_max_demo p&lt;0.001, td_aqcmax
p=0.007, td_max_online p=0.026); zero significant wins. Structural reading: rand≈vla by construction, all
critic argmaxes ≈ vla, candidate-axis variance ≪ state-axis. Side finding: mixed data softens TD's demo-only
harm (−0.167 → −0.050).</p>
""",
)

en(
    "td-segv",
    "Root-causing the silent TD+mixed deaths — an XLA compile segfault",
    """
<p><b>What.</b> TD/QC/CalQL × mixed died after model init with no traceback. Hypothesis ladder: RAM (rejected),
bad nodes (rejected: IQL trains there), faulthandler → SIGSEGV inside XLA backend_compile, cache (rejected),
autotune/parallel/batch (rejected), fMHA/Triton flags (rejected). <b>Root cause: the candidate buffer is
2.48e9 elements &gt; INT32_MAX</b>; XLA gather codegen segfaults on sm_86/Blackwell, surviving only on sm_89.
Fix: sub-int32 buffer split (cand_at) — which exposed the dataset being baked into the program as constants
(17GB duplicated on device); fixed by passing Data as a traced pytree argument. Numerics unchanged;
field-confirmed by the first-ever TD+mixed steps on the A6000 fleet.</p>
""",
)

en(
    "kper",
    "K-per-scene collection — removing the scene-identity shortcut",
    """
<p><b>What.</b> One rollout per kitchen lets a critic regress outcomes off scene identity. Fix: --policy-seed
decouples sampling from the scene; 150 kitchens × 3 policy seeds = 450 rollouts (VLA success 0.676).
<b>45% of kitchens have mixed outcomes</b> — identity cannot predict outcome there. Ledger: 11/15 first-round
jobs died on bad nodes; dump filenames collided across policy seeds; re-collected into per-job directories.</p>
""",
)

en(
    "video-gallery",
    "HUD rollout video gallery",
    """
<p><b>What.</b> Representative rollouts with the critic HUD (grey band = 16-candidate Q spread q01–q99,
blue = executed chunk's Q, red = V(z)); served from the Space, full archive in the acrft-rollout-videos dataset.</p>
""",
)

en(
    "papers-value-steering",
    "Paper review — Robo-ValueRL and value-guided VLA steering",
    """
<p><b>Robo-ValueRL (2607.09866).</b> Same value-head family as ours (frozen VLM + light transformer + HL-Gauss),
but <b>no Best-of-N anywhere</b>: value is a training-time interface — ΔV → quality labels as text prompts
(+26/+34% over BC) and value-filtered online data training a residual adapter on a frozen base (46%→86%).
History: 5 frames beat none and 30 (single frames ambiguous under occlusion/repetition; 28/46/30%).
Adoptables: short history (tested: null here), failure-penalty targets, the adapter route (needs a user call on
the attribution rule). <b>V-GPS</b>: K=50 Cal-QL reranking — gains where the base is suboptimal vs its data.
<b>Q-VGM</b>: critic gradients steer flow denoising; "discrete reranking cannot refine candidates".
<b>Frozen-VLA probing</b>: value decodes linearly from frozen features (R²=.55); gains only with headroom.</p>
""",
)

en(
    "papers-tdjepa",
    "Paper review — TD-JEPA, a strict upgrade over HILP?",
    """
<p class='sub'>User-requested review: "what about TD-JEPA instead of HILP?" (arXiv:2510.00739, Bagatella,
Pirotta, Touati, Lazaric, Tirinzoni — FAIR/Meta, 2025-10-02)</p>

<h3>① The problem — what zero-shot RL means</h3>
<p>Pretrain on reward-free transitions D={(s,a,s′)}; at test time an arbitrary reward r arrives and the agent
must maximize it <b>with no further training</b>. The classical key is the <b>successor measure</b>
M^π(X|s,a) = Σ_t γ^t Pr(s_{t+1}∈X|s,a,π) — a <b>discounted visitation ledger</b> of where the policy will go.
Any reward then assembles instantly: Q^π_r(s,a) = ∫ M^π(ds⁺|s,a)·r(s⁺) — <b>Q = ledger × reward</b>. The
catch: M is a huge object, so the game (FB, HILP, TD-JEPA) is about learning a low-rank factorization.</p>

<h3>② The method — deriving the loss (MC → TD)</h3>
<p><b>Step 1 (idealized, MC-JEPA).</b> For a policy family {π_z}, make a predictor hit "embeddings of states
π_z will visit": E‖T(φ(s),a,z) − φ(s⁺)‖², s⁺~M^{π_z}. Its optimal predictor is exactly <b>φ's successor
features</b> (Prop.1) — but sampling s⁺ requires rolling out every π_z (on-policy), impossible offline.</p>
<p><b>Step 2 (the contribution).</b> Successor features obey a Bellman equation, so the loss becomes TD:</p>
<p style='text-align:center'><b>L = E‖T(φ(s),a,z) − φ̄(s′) − γ·T̄(φ̄(s′),a′,z)‖²</b>, a′~π_z(s′), bar = target nets</p>
<p>Now only single-step transitions are needed — offline, off-policy. a′ comes from the current policy network,
the same bootstrap structure as our IQL/TD runs.</p>
<p><b>Step 3 (asymmetric dual encoders).</b> A state encoder φ (low-level control information) and a task
encoder ψ (reward-defining information) are trained separately with two predictors predicting each other
(T_φ: φ→ψ space, T_ψ: ψ→φ space). Robot analogy: φ = joints/velocities, ψ = building topology.</p>
<p><b>Step 4 (policies and zero-shot inference).</b> Latent policies π(φ(s), z) are trained actor-critic-style
to be optimal for linear rewards r_z(s)=⟨ψ(s),z⟩, z~Z. At test time, one linear regression:
ω_r = (ψᵀψ)⁻¹ψᵀr, then run π(·, z=ω_r). That's all — no retraining.</p>

<h3>③ Theory — why it doesn't collapse and why it's grounded</h3>
<table class='num'><tr><th>Thm</th><th>Statement</th><th>Meaning</th></tr>
<tr><td>1/3</td><td>optimal predictors/gradients of (MC/TD)-JEPA match those of the successor-measure factorization loss ‖φT_zψᵀ − M^{π_z}‖² (oblique projection in the TD case)</td><td>latent prediction is not a trick — it genuinely learns a <b>low-rank factorization of M</b></td></tr>
<tr><td>2</td><td>if predictors learn faster than encoders, the covariances φᵀφ, ψᵀψ are invariant over training</td><td>initialize with unit covariance → <b>no collapse to zero</b>; with orthonormality regularization and target nets, the practical stabilizer set</td></tr>
<tr><td>4</td><td>policy-evaluation error over all unit-norm rewards ≤ 2·L_SM ≤ c·L_TD</td><td>reducing the TD loss <b>bounds the zero-shot evaluation error</b> — the loss is the guarantee</td></tr></table>
<p class='sub'>Author-stated limit: guarantees assume symmetric P^π — real robot dynamics are asymmetric, so
theory is directional and experiments decide.</p>

<h3>④ Results — where it wins and where it loses (13 datasets, 65 tasks, honest reading)</h3>
<table class='num'><tr><th>Bench</th><th>HILP</th><th>FB</th><th>TD-JEPA</th><th>Reading</th></tr>
<tr><td>DMC pixels (return, avg)</td><td>391.2±23.8</td><td>456.2±8.6</td><td><b>628.8±5.5</b></td><td>dominant from pixels — first in every domain (walker 738.9)</td></tr>
<tr><td>DMC proprio (avg)</td><td>620.1±8.4</td><td>648.2±4.1</td><td><b>661.2±6.3</b></td><td>top tier, margin shrinks on low-dim inputs</td></tr>
<tr><td>OGBench pixels (success, avg)</td><td>32.6±0.9</td><td>39.9±0.5</td><td><b>41.3±0.5</b></td><td>co-leader with BYOL-γ (41.6)</td></tr>
<tr><td>OGBench proprio (avg)</td><td>38.0±1.1</td><td><b>39.0±0.7</b></td><td>38.0±0.8</td><td>tie — and <b>HILP crushes it on cube-single, 74.2 vs 34.2</b></td></tr></table>
<p><b>Honest note:</b> on proprioceptive manipulation (the cube family) HILP still wins big in places — "a
strict upgrade over HILP" is an overstatement; the accurate claim is <b>"an upgrade from pixels with broad
coverage; domain-dependent on proprio manipulation."</b> Our GR1 setting is ego-pixel-centric, the favorable
side. Ablations: ① the prediction-target ablation (§5.2) moves along two axes — horizon (one-step BYOL* → multi-step
BYOL-γ*: DMC-pixels 513.8→582.4) and whose future (behavior policy → policy-conditional TD-JEPA:
582.4→628.8) — a <b>cumulative average benefit, not a collapse</b>, with the caveat that behavior-dynamics
approximation can suffice on expert-like data (our GR1 demos are that case): <i>"directly modeling
policy-conditional successor measures is on average beneficial"</i>; ② separate encoders beat a shared
one; ③ a <b>frozen</b> pretrained φ still enables much faster TD3 finetuning than from-scratch — evidence the
representation retains real information.</p>

<h3>⑤ Structural comparison vs HILP — 1:1 against the limits we measured</h3>
<table class='num'><tr><th></th><th>HILP (what we used)</th><th>TD-JEPA</th></tr>
<tr><td>Learns</td><td>reachability-<i>distance</i> embedding (states only)</td><td>embedding + action-conditioned long-horizon prediction + policies</td></tr>
<tr><td>Action info</td><td>none → we bolted a supervised f(z,a) on top</td><td>built into T(φ(s),<b>a</b>,z), propagated long-horizon by TD</td></tr>
<tr><td>Task space</td><td>specialized to distance/goal-reaching</td><td>all linear rewards in ψ-span — beyond goals</td></tr>
<tr><td>Our measured weakness</td><td>"only distance survives" (decoder probe: proprio R² raw .760→φ .546); composition gate action-blind (.487)</td><td>SF approximation forces value-relevant retention; the frozen-adaptation result demonstrates it</td></tr>
<tr><td>Structure</td><td>three-piece: HILP φ + supervised f + IQL V — every seam adds error</td><td><b>one TD loss</b> — seams gone</td></tr>
<tr><td>Caveats</td><td>still strong on proprio manipulation (cube-single 74.2)</td><td>hyperparameter sensitivity, symmetry assumption, needs z-policy diversity</td></tr></table>

<h3>⑥ Crossing with our verdicts — where it actually plugs in</h3>
<p>The wall both workers converged on (demo-only data carries no same-state counterfactuals) is not crossed
for free: z-conditioned prediction only means something if <b>distinct behavior modes exist in the data</b> —
on a single teleop mode the z space collapses to one policy (the paper's training sets are ExoRL exploratory
data and diverse-coverage OGBench; policy diversity is baked in). Flipped around, z-conditioning is precisely
a machine for manufacturing "same state, different policy → different future" counterfactuals along the policy
axis — and our planned phase-2 on-policy K-per-scene collection (vla/rand/noise-scale variants) supplies a
<b>real policy family</b>. Once z genuinely varies in the data, the frame locks onto our actual problem
(valuing candidate chunks): treat each candidate chunk as a short-horizon policy and score it by the landing
SF of T(φ(s), chunk, z) — the principled completion of our model-based attempt Q=γ^h·V(f(z,a)).</p>
<p><b>→ The design proposal now lives in its own report:</b> <span class='xref' data-eid='tdsf-arq'>TD-SF-ARQ design</span> — the single-task distillation (⑦) and the actor-critic ladder (⑧) are treated there in full.</p>
""",
)

en(
    "tdsf-arq",
    "TD-SF-ARQ design — a single-task critic with vector SF targets (preregistration)",
    """
<p class='sub'>Design proposal (preregistration document). Grew out of two user questions on the
<span class='xref' data-eid='papers-tdjepa'>TD-JEPA review</span> ("can we keep the JEPA benefit in a
single-task dataset where z is only locally meaningful?", "does it work as an actor-critic?").
This is the complete write-up.</p>
<h3>Motivation — our diagnosis in brief</h3>
<p>Two days of verdicts established: ① value differences between candidate chunks are extremely faint in
demo-only data (σ_signal≈0; action information only +7.3%, and only in compressed coordinates); ② even that
faint signal starves under scalar TD — one gradient dimension per transition; ③ the three-piece composition
(HILP φ + supervised f + IQL V) accumulated seam errors and gated out action-blind. The prescription is a
single loss with action-conditioned prediction built in — TD-JEPA's core structure. But we want a single-task
critic, not a behavior foundation model, so the frame must be distilled.</p>
<h3>The design — what to drop, what to keep</h3>
<p><b>The user's point.</b> We are not building a behavior foundation model — one task, a dataset where z is
only locally meaningful. How do we keep the JEPA-style representation benefit while preserving our adaptive
transformer critic?</p>
<p><b>Answer: split what to drop from what to keep.</b> The task encoder ψ, the global z-space, zero-shot
inference (ω_r regression) and latent policy learning are all machinery for "many rewards" — dropped. What
survives is exactly one thing: <b>replace the critic's scalar Q output with a vector successor feature and
make the TD target a vector too</b>:</p>
<p style='text-align:center'><b>F(s, chunk) ≈ φ̄(s′) + γ·F̄(s′, chunk′)</b> (chunk-level MDP, s′ = state after
executing the chunk), &nbsp; BoN score Q = ⟨F, w⟩ (w = closed-form ridge regression of progress/success
labels on φ)</p>
<p><b>Why this is the prescription for our diagnosis.</b> The measured bottleneck was "action information
exists but is faint" (+7.3% in compressed coordinates). Scalar TD feeds that faint action-conditioned pathway
<b>one gradient dimension per transition</b> — a starvation structure (the learning-signal version of the GR1
gripper-normalization incident). A vector SF target pours <b>128 dimensions of dense supervision per
transition</b> straight into the action-conditioned path. TD-JEPA's §5.2 ablation points the same way — one-step
behavior-policy targets → policy-conditional multi-step gives DMC-pixels 513.8→582.4→628.8, "directly
modeling policy-conditional successor measures is on average beneficial" (verbatim) — a cumulative gain,
not a collapse, and the paper cautions the gap can shrink on expert-like data; which is why we preregister
this design as a hypothesis, not a guarantee. It also achieves what our model-based attempt Q=γ^h·V(f(z,a)) was after ("value of the
landing point") in a single TD loss with no f/V seams. <b>The ARQ transformer keeps its architecture; only
the output head changes</b> (HL-Gauss can stay as an auxiliary head).</p>
<table class='num'><tr><th>Stage</th><th>What</th><th>Control</th></tr>
<tr><td>A</td><td>φ = <b>frozen</b> PCA-128; swap ARQ output to F; chunk-level TD + w regression</td><td>collapse ruled out by construction, one variable. Verdict = offline gate (demo_winrate, band) side-by-side with the IQL critic</td></tr>
<tr><td>B</td><td>unfreeze φ for joint learning</td><td>port the anti-collapse trio: unit-covariance init, orthonormality regularization, predictor lr &gt; encoder lr (Th.2)</td></tr>
<tr><td>C</td><td>after on-policy K-per-scene: local z-conditioning on policy variant (vla/rand/noise)</td><td>introduce z only when counterfactuals actually ride on it</td></tr></table>
<p><b>Honest expectation:</b> the same-state-counterfactual wall stands through A and B — this is a better
pipe for a weak signal, not a guarantee. But the +7.3% we measured lives exactly in compressed-coordinate
prediction, and this objective consumes that signal head-on: the highest-value single change available.
(Task#8; stage A preregistered once the GR1 phase-1 gates pass.)</p>
<h3>Actor-critic extension — the intervention-strength ladder</h3>
<p><b>Yes — natively.</b> The original TD-JEPA already trains its latent policies by DPG against Q. In our
version Q=⟨F(s,chunk),w⟩ is differentiable in the chunk, with a clean interpretation:
∂Q/∂chunk = wᵀ·∂F/∂chunk = <b>"the direction that moves the predicted landing embedding up the value
gradient in φ-space"</b> — continuous refinement, the step beyond BoN's discrete pick-of-16.</p>
<table class='num'><tr><th>Form (weak→strong)</th><th>What</th><th>Training</th><th>Risk</th></tr>
<tr><td>∂Q/∂a flow steering</td><td>velocity correction injected into the VLA's flow denoising</td><td>none (test-time)</td><td>low — step-size-controlled; a free arm once the critic exists</td></tr>
<tr><td>AWR residual adapter</td><td>small adapter trained by exp(A/β)-weighted BC (A=Q−V)</td><td>light</td><td>low — weights attach only to data chunks, in-support by construction (Robo-ValueRL family)</td></tr>
<tr><td>DPG residual actor</td><td>ascend δ(φ(s),chunk) directly along ∂Q/∂a</td><td>yes</td><td><b>high</b> — below</td></tr></table>
<p><b>But our two-axis conservatism diagnosis returns exactly here.</b> Even critic-argmax (the TD arms)
significantly harmed by stepping on estimation errors (McNemar p&lt;0.01). Gradient ascent on ∂Q/∂a digs into
overestimated directions <b>continuously</b>, exploiting axis 2 (winner's curse) harder than any discrete
pick. It works in TD-JEPA's benchmarks because exploratory data calibrates Q broadly; on demo-only data a few
steps already leave support. <b>Intervention ladder: BoN gate (signal first) → flow steering and AWR adapter
(BC anchors built in) → DPG actor only after on-policy collection calibrates Q on its own distribution
(post stage C).</b></p>
<h3>Preregistration — stage-A verdict criteria</h3>
<p>Once the GR1 phase-1 gates (headroom, rand-vs-vla) pass, stage A is preregistered as: train the IQL critic
and TD-SF-ARQ side by side on identical annotations; promote to a rollout arm only if the offline gate
(held-out demo_winrate, band) shows ① demo_winrate significantly off 0.5, ② improvement over IQL, and
③ <b>temporal resolution ≥ 30% of the γ-ceiling</b> (ΔQ≈V·|lnγ|·Δt, worker A's 08-10 correction) —
sensitivity above the ceiling is treated as an artificial margin and rejected.
Otherwise record null and move to the stage-C (on-policy) precondition — no tricks added.</p>
""",
)

en(
    "papers-byolg",
    "Paper review — BYOL-γ, TD-JEPA's ancestor (and an MC arm for our stage A)",
    """
<p class='sub'>User-requested review (arXiv:2506.10137, Lawson, Hugessen, Cloutier, Berseth, Khetarpal —
Mila/DeepMind, ICLR 2026)</p>
<h3>Problem — stitching (combinatorial generalization)</h3>
<p>Goal-conditioned BC handles seen (state, goal) combinations but fails on novel ones absent as complete
trajectories: train on s₀→s_h and s_b→s_f crossing at w, evaluate on s₀→s_f (must stitch).</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"while BC methods can
perform well on tasks directly observed in the dataset, they often fail to perform zero-shot transfer to tasks
requiring novel combinations of in-distribution behavior, known as combinatorial generalization." (§1)</blockquote>
<p>BC lacks the MDP inductive bias; TD has it but is "challenging to scale due to the instability of
bootstrapping in TD learning when combined with fully offline training" (§1).</p>
<h3>The trick — latent self-prediction at geometric offsets (successor approximation without TD)</h3>
<p>Approximate the successor measure not by bootstrapping but by <b>sampling the prediction offset
k~Geom(1−γ)</b>: forward predictor ψ_f(φ(s_t), a_t) → sg[φ(s_{t+k})], plus a backward predictor
ψ_b(φ(s_{t+k})) → φ(s_t) (a cousin of FB's backward map). Used as an auxiliary loss for BC:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"With ϕ affected by
both terms, the BC loss ensures that the representation is sufficient for action prediction, preventing
collapse." (§4.2)</blockquote>
<p>Theory: approximates the SR in the finite single-policy case; on mixture data MC methods capture a
mixture of SRs (TD-SR captures the mixture policy's SM), with a decisive contrast to contrastive learning:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"CL as used in TRA
leads to pessimism in the relationship between states sampled by different policies. … this pessimism is not
encountered with BYOL-γ, which does not utilize negative examples." (§4.2)</blockquote>
<p>No negatives → representations that <b>optimistically bridge states across trajectories</b> — the
objective-side counterpart of worker A's 08-10 "φ bridges via relational geometry" measurement.</p>
<h3>Results — honest reading</h3>
<p>On OGBench stitch datasets BYOL-γ+GCBC beats GCBC, TRA and offline RL (IQL/QRL/CRL) on average; but
<b>TD-SR wins on small state spaces (antmaze)</b> while BYOL-γ wins on large ones (humanoidmaze, visual) —
"BYOL-γ's simpler training procedure is beneficial in environments with larger state spaces" (§5.2).
Cost: BYOL-γ O(B) vs O(B²) for CL negatives / TD-SR bootstrap terms.</p>
<h3>Lineage and our takeaway — an MC arm for stage A</h3>
<p><b>Lineage:</b> BYOL (1-step) → BYOL-γ (geometric-offset MC, behavior policy) → TD-JEPA (TD off-policy +
policy-conditioning z + task encoder = SFs of arbitrary policies). In TD-JEPA's tables BYOL-γ* was the
strongest pixel baseline (DMC-RGB 582.4 vs 628.8).</p>
<p><b>Takeaway:</b> BYOL-γ needs (a) a single behavior mixture, (b) no TD, (c) no policy conditioning —
<b>closer to our demo-only setting than TD-JEPA itself</b>. Action: add an <b>MC-geometric arm</b> to
<span class='xref' data-eid='tdsf-arq'>TD-SF-ARQ</span> stage A (T(φ(s),chunk) predicting φ(s_{t+k}),
k~Geom(1−γ) — no bootstrap, O(B)) as a one-variable comparison against the TD arm. The "BC loss prevents
collapse" observation is a stage-B stabilizer candidate; the TRA-pessimism analysis grounds excluding
contrastive variants.</p>
""",
)

en(
    "embed-compare",
    "Embedding-structure comparison — raw/PCA/phi/BYOL-gamma/TD-JEPA (what bridges across trajectories)",
    """
<p class='sub'>User request: "visualize TD-JEPA and BYOL-gamma like the HILP-vs-cheap-z comparison" + "take one
embedding from a trajectory and show its nearest neighbors from other trajectories as images (I'll judge)" +
consolidate BC probing. Data: PrepareCoffee mixed annotation (807,634 token-frames, 1,234 episodes, median 582
tokens/episode — not short).</p>
<p class='missing'><b>Headline (TL;DR) — DiT closed-loop probe.</b> Every offline metric (BC R² 0.64–0.71,
stage-purity 0.76–0.83) is flat and failed to predict this collapse — <b>only closed-loop revealed it</b>. PCA-128
and φ-128 are both 128-d yet differ 14× → <b>not dimension but the learned readout's geometry destroys control
information</b>.</p>
<table class='num'><tr><th>Embedding</th><th>DiT closed-loop success (25 trials) ↑</th></tr>
<tr><td>raw 2048</td><td><b>0.60</b></td></tr>
<tr><td>PCA-128</td><td><b>0.40</b></td></tr>
<tr><td>φ (HILP)</td><td>0.04</td></tr>
<tr><td>TD-JEPA</td><td>0.04</td></tr>
<tr><td>BYOL-γ</td><td>0.00</td></tr></table>
<h3>Setup</h3>
<p>Train TD-JEPA and BYOL-gamma as readouts on the <b>same frozen RLT tokens</b> phi(HILP) used (identical MLP
arch — objective only differs). phi = goal-conditioned expectile TD (cross-episode goal sampling); BYOL-gamma =
geometric-offset self-prediction (EMA target); TD-JEPA = action-free TD successor. Same battery: kNN episode
purity (↓ = less episode identity), cross-episode phase error (↓), progress R² (↑).</p>
<h3>Quantitative (battery)</h3>
<table class='num'><tr><th>Embedding</th><th>purity ↓</th><th>phase_err ↓</th><th>prog R² ↑</th><th>Reading</th></tr>
<tr><td>raw 2048</td><td>0.589</td><td>0.114</td><td>0.690</td><td>baseline</td></tr>
<tr><td>PCA-128</td><td>0.511</td><td>0.111</td><td>0.694</td><td>dimension only</td></tr>
<tr><td><b>phi (HILP)</b></td><td><b>0.382</b></td><td><b>0.097</b></td><td><b>0.715</b></td><td><b>the only one that removes episode identity</b></td></tr>
<tr><td>BYOL-gamma</td><td>0.917</td><td>0.175</td><td>0.180</td><td><b>amplifies</b> identity (γ=0.9; at γ=0.98 it collapses, R²=−3.7)</td></tr>
<tr><td>TD-JEPA (SR)</td><td>0.610</td><td>0.126</td><td>0.650</td><td>≈ raw (fails to remove)</td></tr></table>
<p><img src="videos/embed/24_embed_compare.png" alt="embedding battery + PCA-2 projection"></p>
<p class='sub'>The scatter above is a <b>PCA-2</b> (linear) projection. Below is a <b>t-SNE</b> (nonlinear)
projection of the same embeddings, color = task progress:</p>
<p><img src="videos/embed/26_tsne_embed.png" alt="t-SNE of 5 embeddings colored by progress"></p>
<p><b>t-SNE reading:</b> phi flows as <b>one coherent progress manifold</b> (task phase aligned across
trajectories). <b>BYOL-gamma fragments into ~60 small islands</b> — each with its own internal progress
gradient = per-episode timelines (the visual proof of purity .92). raw/PCA have a gradient but less organized;
TD-JEPA is intermediate.</p>
<h3>Key insight — not TD vs MC, but whether the objective contrasts across episodes</h3>
<p>BYOL-gamma (future self) and the TD-JEPA readout (TD successor, action-free) both predict <b>within-trajectory
futures only</b> → no pressure to merge across trajectories. Our RLT tokens already carry strong episode identity
(raw purity .59), so pure self-prediction locks onto <b>the most predictable direction = the slowly-varying
identity feature</b> and amplifies it (BYOL-gamma purity .92). Only phi (HILP) samples goals <b>cross-episode</b>,
forcing a contrast between trajectories that strips identity (.38). <b>DBC follows the same pattern</b> — |r_i−r_j|
+ W₂ over permuted-batch arbitrary pairs is a cross-trajectory contrast. <b>Conclusion: the cross-trajectory
invariance we want requires a cross-trajectory contrast / goal sampling in the objective.</b> Pure self-prediction
(BYOL-gamma, TD-SR) lacks it.</p>
<p class='sub'><b>Correction:</b> I first misdiagnosed the BYOL-gamma collapse as "short episodes + end-clamp,"
but episodes are long (median 582 tokens). Lowering γ 0.98→0.9 still leaves purity .92 (only softens the
collapse), confirming the cause is not horizon but the <b>self-prediction identity amplification</b> above.</p>
<h3>Qualitative neighbor matrix (user's design) — columns = one trajectory</h3>
<p>Layout: <b>rows = 7 query episodes</b> (one random frame each), <b>columns = 7 fixed other episodes</b> (same
across all rows), cell[i,j] = the frame in column-episode j closest to query i. Each cell is <b>top agentview /
bottom wrist</b>, number = cosine similarity. Because each column is always the same trajectory, you can directly
compare "what each embedding considers equal to the query in other trajectories."</p>
<p><b>phi (HILP)</b><br><img src="videos/embed/25_xneighbor_phi128.png" alt="phi neighbor matrix"></p>
<p><b>raw 2048</b><br><img src="videos/embed/25_xneighbor_raw2048.png" alt="raw neighbor matrix"></p>
<p><b>BYOL-gamma</b><br><img src="videos/embed/25_xneighbor_byolg128.png" alt="byolg neighbor matrix"></p>
<p><b>PCA-128</b><br><img src="videos/embed/25_xneighbor_pca128.png" alt="pca neighbor matrix"></p>
<p><b>TD-JEPA</b><br><img src="videos/embed/25_xneighbor_tdjepa128.png" alt="tdjepa neighbor matrix"></p>
<p>phi's neighbors tend to be the same task moment across trajectories (same arm-object layout, cups/mugs in the
wrist view). <b>Two honest limits:</b> ① PrepareCoffee reuses kitchen scenes, so <b>every</b> embedding (raw
included) finds plausible cross-trajectory neighbors — the panel alone does not strongly separate methods (the
discriminators are the quantitative purity and t-SNE). ② The neighbors are closer to near-duplicates than
"same-semantics, different-visuals," so this data does not fully stress-test the invariance we want — <b>visually
diverse data (GR1 novel objects, OGBench visual variants)</b> is that test bed.</p>
<h3>DiT closed-loop probe (the verdict, 08-11) — it overturns the offline metrics</h3>
<p>Offline BC MSE cannot see compounding error. So per embedding we train a <b>DiT policy head</b> (action chunk
as H tokens, temporal self-attention + AdaLN-Zero, rectified flow) and measure <b>closed-loop success</b> in the
PrepareCoffee sim (25 trials, same VLA backbone token):</p>
<table class='num'><tr><th>Embedding</th><th>DiT closed-loop success ↑</th><th>(ref) BC R²</th><th>(ref) stage-purity</th></tr>
<tr><td>raw 2048</td><td><b>0.60</b></td><td>0.708</td><td>0.825</td></tr>
<tr><td>PCA-128</td><td><b>0.40</b></td><td>0.697</td><td>0.828</td></tr>
<tr><td>φ (HILP)</td><td>0.04</td><td>0.682</td><td>0.815</td></tr>
<tr><td>TD-JEPA</td><td>0.04</td><td>0.688</td><td>0.807</td></tr>
<tr><td>BYOL-γ</td><td>0.00</td><td>0.640</td><td>0.762</td></tr></table>
<p><b>Verdict — three decisive points:</b> ① <b>Geometry, not dimension, is the culprit.</b> PCA-128 (0.40) and
φ-128 (0.04) are both 128-d yet differ 14× — a variance-preserving linear projection keeps control, while the
<b>learned readouts' (φ/TD-JEPA/BYOL-γ) reachability/self-predictive geometry destroys the fine control
information</b>. ② <b>Offline metrics all fail.</b> Neither BC R² (0.64–0.71, flat) nor stage-purity (0.76–0.83,
flat) predicts the collapse — <b>only closed-loop reveals it</b> (compounding error). ③ <b>"φ is sufficient as a
representation" is firmly rejected</b> — φ handles single-step BC and coarse stage but is insufficient for
closed-loop control (the decoder probe's proprio loss .760→.546 was control-critical after all). <b>TD-SF-ARQ
implication:</b> use <b>PCA-128/raw, not φ</b>, for the critic embedding — learned readouts throw away the fine
control signal candidate discrimination needs. (The user's insistence on closed-loop + DiT was exactly right.)</p>
<h3>BC probing consolidated (5 embeddings) — BC-sufficiency ⊥ invariance</h3>
<p>Embedding → demo-action-chunk BC under the identical protocol for all five (kroll, held-out):</p>
<table class='num'><tr><th>Embedding</th><th>BC action R² ↑</th><th>(ref) structure purity ↓</th></tr>
<tr><td>raw 2048</td><td>0.708</td><td>0.589</td></tr>
<tr><td>PCA-128</td><td>0.697</td><td>0.511</td></tr>
<tr><td>phi (HILP)</td><td>0.682</td><td><b>0.382</b></td></tr>
<tr><td>TD-JEPA</td><td>0.688</td><td>0.610</td></tr>
<tr><td>BYOL-gamma</td><td>0.640</td><td>0.917</td></tr></table>
<p><b>Reading:</b> BC R² is <b>nearly flat across all embeddings (0.64–0.71)</b> — even the episode-identity-
collapsed BYOL-gamma retains most action info (0.640). So <b>action prediction (BC) and cross-trajectory
invariance are orthogonal</b>: BYOL-gamma does BC (0.64) but is the worst at bridging (purity 0.92), while phi
does both (0.68 / 0.38). This quantifies "good at BC ≠ good embedding" — our goal (same-state candidate
discrimination) needs invariance, not BC-sufficiency, and only the cross-episode-contrast objective (phi)
provides it. Together phi (a) retains action info (BC .68), (b) removes identity (purity .38), (c) bridges
relationally (worker A act-cos .661) — <b>sufficient as a representation</b>. phi-critic's BoN failure is thus
not a representation defect but the <b>absent same-state counterfactual (data)</b>. Decoder, BC, and structure
probes converge on "representation sufficient, data insufficient."</p>
""",
)

en(
    "papers-dbc",
    "Paper review — DBC (bisimulation): the head-on definition of the invariance we want",
    """
<p class='sub'>From the user's question: "the invariance we want is that semantically similar scenes (approaching
the cup by its handle), visually different, get similar embeddings — does BYOL-γ help, or is there other work?"
→ the lineage that defines that invariance head-on is bisimulation. (arXiv:2006.10742, Zhang, McAllister,
Calandra, Gal, Levine, ICLR 2021 oral)</p>
<h3>What it defines — behavioral equivalence</h3>
<p>Bisimulation calls two states equal not by "similar pixels" but by <b>"behaviorally indistinguishable"</b>:
same reward, and under every action transition to a bisimulation-equivalent distribution of next states. DBC
plants that distance <b>directly</b> as latent distance:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"Our method trains
encoders such that distances in latent space equal bisimulation distances in state space." (Abstract)</blockquote>
<p style='text-align:center'><b>J(φ) = ( ‖z_i−z_j‖₁ − |r_i−r_j| − γ·W₂(P̂(·|z̄_i,a_i), P̂(·|z̄_j,a_j)) )²</b></p>
<p>Latent distance is regressed onto <b>reward difference + Wasserstein distance of next-state distributions</b>
(z̄ = stop-grad, dynamics model P̂ trained jointly). No reconstruction at all, so task-irrelevant visuals are
never represented:</p>
<blockquote class="sub" style="border-left:3px solid #c7cbd4;margin:8px 0;padding:4px 12px">"state elements are
relevant not only if they influence the current reward, but also if they influence state elements in the future
that in turn influence future rewards … an ideal representation is one that is predictive of reward, and also
predictive of itself in the future." (§1)</blockquote>
<p>Experiments are exactly our concern: invariant to moving-distractor / natural-video backgrounds in MuJoCo,
and to clouds/weather/time-of-day in driving.</p>
<h3>Relation to SR/BYOL-γ — the reward axis</h3>
<table class='num'><tr><th></th><th>SR/BYOL-γ (successor)</th><th>bisimulation/DBC</th></tr>
<tr><td>Equality criterion</td><td>same future <b>visitation</b></td><td>same future <b>reward outcome</b> (reward+transition)</td></tr>
<tr><td>Reward axis</td><td>none (dynamics-only)</td><td>yes — the |r_i−r_j| term</td></tr>
<tr><td>Distance to our want</td><td>proxy (tight when semantics≡future outcome)</td><td><b>the definition itself</b></td></tr>
<tr><td>Practical risk</td><td>collapse (BYOL-γ prevents via BC anchor)</td><td>W₂ + joint-dynamics instability (paper notes it)</td></tr></table>
<p>BYOL-γ is the <b>reward-free cousin</b> ("same visitation → merge"), DBC the canonical form ("same reward
outcome → merge"). Our "handle-approach equal despite background" — as long as handle-approach leads to the same
future reward (grasp success) — is caught by both, DBC more directly.</p>
<h3>Adjacent lineage (same goal, different handle)</h3>
<table class='num'><tr><th>Family</th><th>Representative</th><th>Distance means</th></tr>
<tr><td>Bisimulation</td><td>DBC(2021), DeepMDP(2019)</td><td>reward+transition equivalence</td></tr>
<tr><td>Value-implicit</td><td>VIP(2023)</td><td>goal progress</td></tr>
<tr><td>Quasimetric</td><td>QRL(2023)</td><td>optimal steps-to-reach (asymmetric)</td></tr>
<tr><td>Temporal-contrastive</td><td>TRA(2025), CRL, TCN</td><td>temporal proximity / same goal</td></tr>
<tr><td>Object-centric</td><td>slot-attention family</td><td>relational layout directly (background-agnostic)</td></tr></table>
<h3>Connection to our conclusion — invariance = unlocking coverage</h3>
<p>Key insight: better semantic invariance is a <b>cross-trajectory counterfactual-sharing</b> device. If φ
merges trajectory A's "handle approach" with B's, then A(a₁→success) and B(a₂→failure) become a
<b>same-embedding-state counterfactual pair</b> despite different source trajectories. So invariance unlocks
latent counterfactuals in offline data, raising <b>effective coverage</b> (why OGBench stitch works).
<b>Two limits:</b> ① there must be latent counterfactuals to unlock (a single demo mode has zero); ② our
downstream (candidate discrimination) needs the counterfactuals to exist somewhere. So invariance (unlock
latent) and on-policy (manufacture new) are complementary. <b>Our evidence:</b> φ (a successor-family readout)
already bridges cross-trajectory via relational geometry (worker A act-cos .661 vs .334) — this lineage
demonstrably works in our stack. <b>Next:</b> compare TD-JEPA/BYOL-γ readouts against φ/raw on the same battery
(embed-compare entry); bisimulation (adding the reward axis) is tryable on GR1 with a progress reward.</p>
""",
)


en(
    "horizon-probe",
    "Design — is coverage the culprit? A reverse ablation on OGBench (the counterfactual dial)",
    """
<p class='missing'><b>Shelved (08-11, user call):</b> re-confirming on OGBench adds little — "coverage is
OGBench's advantage" is already explained by our two-axis frame. We drop the experiment and keep only its
conclusion (coverage = causal variable; prescription = manufacture coverage on-policy) in the roadmap. The
main line is GR1 phase-1 + phase-2 on-policy.</p>
<p class='sub'>User proposal (08-11) + a decisive fact: "TD-BoN (our original version) works really well on
OGBench." And OGBench's advantage is the <b>counterfactual richness of its diverse/play data (coverage)</b>.
This entry turns that into a culprit-isolation experiment.</p>
<h3>Reframe — the diagnosis flipped</h3>
<p>That TD-BoN works on OGBench <b>already answers</b> "does our pipeline capture signal when present?" — yes.
So the PrepareCoffee/YAM nulls are <b>not an embedding/critic defect but the absence of signal (counterfactuals)
in that data</b> — confirming our two-axis conclusion (axis 2 = absent same-state counterfactuals) from the
opposite side, and <b>exonerating our method</b>. So the question changes:</p>
<p style='text-align:center'><b>What single property, present in OGBench and absent in VLA manipulation, kills TD-BoN?</b></p>
<p>User-confirmed: <b>coverage (counterfactual density)</b> is the prime suspect and OGBench's core advantage.
The hypothesis is not "the embedding scatters" but "there is no data of the same state traversed by different
actions," so candidates cannot be told apart.</p>
<h3>Decisive experiment — dial only coverage in the same environment (one variable)</h3>
<p>No new environment needed. <b>Thin coverage within OGBench itself</b>: hold env, horizon, observation and
algorithm fixed, and reduce only the <b>counterfactual density</b> of the data (programmatically cut the number
of distinct trajectories/actions passing through the same state → converge to a demo-only regime). Counterfactual
density is measurable — the number of distinct next-actions per state present in the data.</p>
<table class='num'><tr><th>Observation</th><th>Conclusion</th></tr>
<tr><td>as coverage↓, TD-BoN sensitivity/success <b>monotonically collapses</b>, reproducing our VLA null in the demo-only regime</td><td><b>coverage confirmed as the causal variable</b> — the root of two years of nulls, proven under control in the same env. Prescription is clear: for VLA, counterfactuals must be <b>manufactured</b> (on-policy)</td></tr>
<tr><td>TD-BoN survives even at thin coverage</td><td>coverage is not the culprit → move the dial to second-tier suspects (horizon, observation dimensionality)</td></tr>
<tr><td>at equal coverage, raw works but φ collapses</td><td>representation is a secondary variable — the information φ discards is decisive in this regime (contrast decoder probe .546)</td></tr></table>
<h3>Strategic corollary — if this holds, our roadmap is settled</h3>
<p>If coverage is confirmed causal: ① the <b>offline-only embedding line (φ, TD-SF-ARQ stages A/B) has a
ceiling</b> — with no counterfactuals in the data, no objective change lifts it far. ② The real lever is
<b>manufacturing coverage = on-policy K-per-scene collection</b> (traverse the same scene differently via
vla/rand/noise variants) — already planned as GR1 phase-2 stage C. OGBench is the existence proof: "give it
counterfactuals and TD-BoN works." TD-SF-ARQ then sits as the objective that <b>absorbs those counterfactuals
most efficiently</b> (128-dim SF supervision per transition).</p>
<p><b>Order:</b> parallel with GR1 phase-1 (different resources). Fast to start — our original TD-BoN code +
the OGBench loader. First output: a 3–4-point counterfactual-density sweep. (Task#10)</p>
""",
)


en(
    "floq",
    "floq — flow-matching critic (implementation, test, visualization)",
    """
<p class='sub'>Per user request, floq (arXiv:2509.06863, "Training Critics via Flow-Matching") was implemented
<b>directly on our critic</b>, tested and visualized. Grew from worker A's r56 floq review
(<span class='xref' data-eid='xworker-0808'>cross-worker review</span>): we verify the "gain is capacity/plasticity,
not distributional" reading on our data and an independent implementation.</p>
<p><b>⑪ floq implemented & tested (08-12, user request — faithful to the paper).</b> Discarded the toy and
<b>kept our ARQ critic architecture, swapping only bootstrap+head to floq</b>: same causal-transformer trunk,
head becomes a velocity field (critic.py <code>flow_head</code>, non-breaking commit), loss is the paper's
Eq 4.2 <code>‖v_θ(t,z(t)|s,a)−(y−z)‖²</code>, z(t)=(1−t)z+t·y, and the TD target y=r+γ^H·V_next bootstraps by
<b>integrating the target velocity field K steps</b>. Both collapse-prevention choices ported: initial noise
<b>Uniform[−0.5,1]</b> (u=Q_max), interpolant z as a <b>categorical</b> encoding, time t as <b>Fourier</b>
features. Held-out by episode:</p>
<table class='num'><tr><th>Metric</th><th>scalar ARQ (expectile TD)</th><th>floq ARQ</th></tr>
<tr><td>Spearman(Q, mc) — value fit</td><td>0.395</td><td><b>0.521</b> (+32%)</td></tr>
<tr><td>action-sensitivity</td><td>0.024</td><td>0.071</td></tr>
<tr><td>demo_winrate (demo vs VLA cand, optimistic)</td><td>0.279</td><td>0.781</td></tr></table>
<p><b>Verdict — capacity yes, coverage no (strongly).</b> ① floq's <b>capacity gain confirmed</b>: on our real
ARQ critic, value-fit +0.13 (+32% rel), larger than the toy. ② But <b>deployment discrimination is still
absent</b>: sensitivity 0.071 exceeds the γ-ceiling (γ=0.99, V~0.11, Δt~16 → ΔQ ceiling ≈ <b>0.018</b>), so it
is action style/distribution discrimination (artificial margin), not true value; and demo_winrate 0.78 is
optimistic (it includes the demo, absent from the deployment BoN pool). <b>Conclusion: floq is a genuine
critic-capacity / value-fit upgrade (worth adopting as the TD-SF-ARQ head), but does not solve the binding
constraint, coverage — a mutual replication of worker A's floq reading on an independent implementation.</b>
(Simplification: single-commit TD to isolate floq-vs-scalar; the ARQ prefix/ensemble machinery is orthogonal.)</p>
<p><b>Flow visualization.</b> How Q is read: q(t) integrating along the velocity field from noise at t=0
(uniform [−0.5,1]) to the Q-value at t=1 — a <b>funnel</b>. Color = true mc_return. Different noise seeds for the
same state converge to the same Q (the velocity's z-dependence, floq's capacity source): most (low mc) land near
0, a few high-mc states split to 0.75–0.85.
<a href="videos/floq/27_floq_flow.mp4" target="_blank">▶ animation</a></p>
<p><img src="videos/floq/27_floq_flow.png" alt="floq flow: q(t) integration funnel"></p>
<p class='sub'>Curvature 0.0135 is small — rectified flow is near-straight per trajectory (our curvature metric is
per-trajectory deviation from its own straight line, trivially small for rectified flow); the capacity comes from
the <b>spread across noise seeds (z-conditioning)</b>, not dramatic bending.</p>
<p><b>Trajectory HUD video (multi-peak-experiment style).</b> Along a successful held-out kroll trajectory:
<b>left the robot ego view</b> + <b>right the floq return distribution</b> (256 integrated noise samples), with
the floq mean and the scalar Q (dashed) for contrast. <a href="videos/floq/28_floq_traj.mp4" target="_blank">▶
HUD video</a></p>
<p><img src="videos/floq/28_floq_traj.png" alt="floq trajectory HUD: robot + return distribution"></p>
<p class='sub'>Reading: the distribution is mostly <b>unimodal and peaked</b> (sparse binary reward makes the return
near-degenerate; dramatic multi-peak only at genuinely uncertain outcomes). The floq mean and scalar Q can
diverge on some frames (the scalar ARQ is less calibrated in this simplified setup). The qualitative difference
is that floq gives a <b>distribution</b>, not a point.</p>""",
)

en(
    "critic-heads",
    "Critic heads (scalar/HL-Gauss/floq) — offline ranking vs closed-loop BoN verdict",
    """
<p class='sub'>Per user request: after putting floq on our critic, separate whether the gain comes from the
<b>flow mechanism</b> or the <b>categorical representation</b>, and then decide by closed loop whether
<b>this critic actually improves the VLA</b>. Direct follow-up to the <span class='xref' data-eid='floq'>floq
implementation post</span>.</p>

<h3>Design — same trunk, only the head/loss differ (method-only-diff)</h3>
<p>Our standard <b>AQC critic</b> (ARQCritic: a causal transformer over [obs token, action-macro tokens],
macro_group_size=2, scored at the full-chunk prefix) has its <b>trunk left untouched</b>; only the output head
and training loss are swapped into three variants, trained side by side:</p>
<table class='num'><tr><th>head</th><th>representation</th><th>loss</th><th>read Q</th></tr>
<tr><td><b>scalar</b></td><td>one scalar</td><td>expectile-TD(0.9) regression</td><td>the scalar</td></tr>
<tr><td><b>HL-Gauss</b></td><td>51-atom logits</td><td>cross-entropy to a Gaussian-smoothed target (classification)</td><td>Σ softmax·centers</td></tr>
<tr><td><b>floq</b></td><td>velocity field v(t,z|s,a)</td><td>flow-matching (Eq 4.2)</td><td>integrate noise K steps</td></tr></table>
<p class='sub'>The three form a <b>regression → classification → flow</b> ladder. HL-Gauss is the decisive middle
term: floq's collapse-prevention also encodes z categorically, so inserting HL-Gauss ("same 51-atom
classification but no flow") <b>splits whether the gain is categorical or flow</b>. Shared setup: γ=0.997 (longer
effective horizon), <b>single-sample bootstrap</b> (faithful floq — averaging washes out the distribution),
PrepareCoffee mixed annotation, held-out by episode. 30k steps per head.</p>

<h3>① Offline — does the head rank actions?</h3>
<table class='num'><tr><th>metric</th><th>scalar (regression)</th><th>HL-Gauss (classification)</th><th>floq (flow)</th></tr>
<tr><td>Spearman(Q, mc) — value fit</td><td>0.398</td><td><b>0.594</b></td><td>0.518</td></tr>
<tr><td>action-sensitivity</td><td>0.018</td><td>0.011</td><td><b>0.281</b></td></tr>
<tr><td>demo_winrate (demo vs VLA cand, optimistic)</td><td>0.226</td><td><b>0.905</b></td><td>0.851</td></tr></table>
<p><b>Offline reading.</b> ① <b>Most of the gain is the categorical representation</b>: HL-Gauss lifts winrate
0.23→0.91 and Spearman 0.40→0.59, ranking <b>as well as (better than) floq without any flow</b>. floq beating
scalar was largely the categorical head, not the flow mechanism. ② floq's sensitivity 0.281 exceeds the
<b>γ-ceiling</b> (γ=0.99, V~0.11, Δt~16 → true value-diff ceiling ΔQ≈<b>0.018</b>, worker A's correction) by
<b>14×</b> — the single-sample bootstrap raised action-sensitivity, but this is <b>action-style discrimination
artifact</b>, not true value. And winrate 0.85–0.91 is optimistic (the demo is in the pool; the deployment BoN has none).</p>

<h3>② Closed-loop — does BoN actually beat the VLA? (verdict: no)</h3>
<p>The three saved critics are placed in a <b>real BoN deployment</b>: PrepareCoffee, 25 scenes, and at each
replan the VLA samples <b>N=8 candidates</b> → the critic scores → the argmax candidate is executed. Two controls —
<b>VLA baseline</b> (execute candidate 0) and a <b>rand null</b> (execute a random candidate). rand is the key
control: BoN beating VLA but not rand would only show resampling helps, not that the critic picked. run_trials
pins the scene by (seed, trial), so <b>every mode faces the identical 25 scenes</b> (scene-paired).</p>
<table class='num'><tr><th>mode</th><th>success</th><th>rate</th><th>Wilson 95% CI</th><th>vs VLA (paired McNemar: win/loss)</th></tr>
<tr><td><b>VLA (baseline)</b></td><td>20/25</td><td><b>0.80</b></td><td>[0.61, 0.91]</td><td>—</td></tr>
<tr><td>rand (null)</td><td>17/25</td><td>0.68</td><td>[0.48, 0.83]</td><td>3 / 6</td></tr>
<tr><td>scalar BoN</td><td>10/25</td><td>0.40</td><td>[0.23, 0.59]</td><td><b>0 / 10</b></td></tr>
<tr><td>HL-Gauss BoN</td><td>16/25</td><td>0.64</td><td>[0.45, 0.80]</td><td>3 / 7</td></tr>
<tr><td>floq BoN</td><td>11/25</td><td>0.44</td><td>[0.27, 0.63]</td><td><b>1 / 10</b></td></tr></table>
<p><img src="videos/critic-heads/29_bon_compare.png" alt="offline demo-winrate vs closed-loop BoN success"></p>
<p class='sub'>Left: offline winrate (HL-Gauss·floq dominate at 0.91·0.85). Right: closed-loop success — the black
dashed line is the VLA baseline. Colors: gray=scalar, green=HL-Gauss, red=floq, black=VLA, gray=rand null. Error
bars are Wilson 95%.</p>

<h3>Verdict</h3>
<p><b>No critic head improves the VLA via BoN.</b></p>
<p>① <b>scalar and floq fall below the random null</b> (0.40·0.44 &lt; rand 0.68). By paired McNemar, scalar is
<b>0 wins / 10 losses</b> vs VLA and floq <b>1 / 10</b> — significantly harmful. The critic's argmax picks a
<b>worse</b> candidate than random: among N candidates it takes the one it <b>most overestimates</b>, and that
overestimation correlates with estimation error / off-distribution, not true value (the <b>winner's curse</b> =
estimation-error exploitation, exactly the failure mode of <span class='xref' data-eid='conservatism'>two-axis
conservatism</span>). The candidates are all decent (rand only drops to 0.68), so argmax singling out the worst
drops further.</p>
<p>② <b>Offline winrate badly mispredicted the closed loop.</b> HL-Gauss winrate 0.905, BoN 0.64. This is the
<b>second "offline → closed-loop" disconnect</b>, after the <span class='xref' data-eid='embed-compare'>embedding-DiT
overturn</span> (offline BC/decoder said "representation sufficient"; closed loop overturned it). Reason: winrate
scores demo vs candidates at <b>in-dist demo states</b>, but deployment argmaxes over <b>the VLA's own candidates at
the states the VLA visits</b> — a different distribution, where the argmax amplifies error. Offline ranking metrics
are not a trustworthy proxy for deployment success.</p>
<p>③ <b>Head-level conclusion</b>: categorical (HL-Gauss) is cheaper than flow and better both offline and closed
loop (BoN 0.64, the least harmful, the only one near rand). floq's flow only inflated sensitivity to 14× the
γ-ceiling, <b>splitting candidates on style artifacts and hurting</b>. → the <span class='xref'
data-eid='tdsf-arq'>TD-SF-ARQ</span> head should be <b>HL-Gauss</b>, not the floq flow.</p>
<p>④ <b>Root cause restated</b>: the problem is not the head but <b>the deployment scheme and the data</b>. Using a
demo-only critic to argmax over its own candidates (BoN) loses to the winner's curse without conservatism. No head
is saved without <b>coverage (counterfactual candidates)</b> — an independent path back to the same conclusion as
<span class='xref' data-eid='model-based'>the missing candidate-axis signal</span> and
<span class='xref' data-eid='conservatism'>two-axis conservatism</span>.</p>

<h3>Limits (honest)</h3>
<p>n=25, single seed, so <b>provisional</b>. The VLA is already strong (0.80), a ceiling effect that leaves <b>little
room to rise, much to fall</b> (needs a rerun on a weaker-baseline task). Full-chunk commit only (no ARQ prefix
selection). HL-Gauss CI [0.45,0.80] overlaps VLA [0.61,0.91], so "<b>HL-Gauss &lt; VLA</b>" is unconfirmed (McNemar
p≈0.34) — what is <b>confirmed is the harm of scalar and floq</b>.</p>

<h3>Aside — flow visualization (γ=0.997, single-sample bootstrap)</h3>
<p>The floq flow from the same retrain. Raising γ 0.99→0.997 grew the flow <b>curvature 0.0135→0.0317</b> (the
longer horizon bends the velocity path more). Trajectory HUD: left the robot ego view, right the floq return
distribution (256 integrated noise samples). With the single-sample bootstrap the distribution spreads more at
uncertain-outcome states. <a href="videos/critic-heads/28_floq_traj.mp4" target="_blank">▶ HUD video</a> ·
<a href="videos/critic-heads/27_floq_flow.mp4" target="_blank">▶ funnel video</a></p>
<p><img src="videos/critic-heads/28_floq_traj.png" alt="floq HUD gamma=0.997 single-sample bootstrap"></p>

<p class='sub'>Reproduce: offline 3-way <code>probes/floq_critic.py</code> (HL-Gauss head added), closed-loop
<code>probes/eval_bon.py</code> (save critics → VLA BoN rollout), figure <code>probes/plot_bon.py</code>
(JSON→figure). Result JSONs: <code>floq_critic.json</code>·<code>bon_critic_compare.json</code>. All committed.</p>""",
)

en(
    "critic-pfx",
    "Bootstrap fix — per-prefix TD-max + joint arg-max still can't beat the VLA",
    """
<p class='sub'>A re-run that fixes the flaw in the previous <span class='xref' data-eid='critic-heads'>critic-head
comparison</span>. Per the user: "if you do TD, you must sample dataset actions and take the <b>max</b> to
bootstrap, and the critic must output Q <b>per prefix</b>." The previous bootstrap used the <b>demo's next
action</b> (SARSA-style), which is the demo-policy value, not the optimal. Here both are corrected — does the
conclusion change?</p>

<h3>What changed (faithful to the production trainer's targets())</h3>
<ul>
<li><b>Bootstrap = TD-max over candidates</b>: at the landing state, over the stored VLA candidates (8 of the 16
base_action samples), <b>V(s′)=max_j Q(s′, cand_j)</b>. The only way to form max_a Q offline — each state has one
demo, so you need "several actions", and the candidate pool is that material.</li>
<li><b>per-prefix native</b>: for each prefix p (commit length), <b>y_p = Σ_{i&lt;p}γ^i r + γ^p·(1−ended_p)·V_next(landing_p)</b>,
with an mc floor. floq integrates per prefix to read Q.</li>
<li><b>Deploy = joint (candidate × prefix) arg-max</b>: pick the candidate AND the commit length n_exec together (the
real AQC deployment rule). Null control adds <b>randh</b> (random candidate AND random prefix — the honest null for a
joint arg-max).</li>
</ul>
<p class='sub'>Three heads (scalar/HL-Gauss/floq) trained against the <b>same td-max target</b> (representation only
differs). γ=0.997, single critic (no ensemble, an explicit simplification). PrepareCoffee, N=8, 25 scene-paired
trials, Wilson 95% + paired McNemar.</p>

<h3>Result — still no critic beats the VLA</h3>
<table class='num'><tr><th>mode</th><th>success</th><th>rate</th><th>Wilson 95%</th><th>vs VLA (win/loss)</th></tr>
<tr><td><b>VLA (baseline)</b></td><td>14/25</td><td><b>0.56</b></td><td>[0.37, 0.73]</td><td>—</td></tr>
<tr><td>rand (null)</td><td>17/25</td><td><b>0.68</b></td><td>[0.48, 0.83]</td><td>6 / 3 (+0.12)</td></tr>
<tr><td>randh (joint null)</td><td>13/25</td><td>0.52</td><td>[0.33, 0.70]</td><td>4 / 5</td></tr>
<tr><td>scalar BoN</td><td>9/25</td><td>0.36</td><td>[0.20, 0.55]</td><td>3 / 8 (−0.20)</td></tr>
<tr><td>HL-Gauss BoN</td><td>11/25</td><td>0.44</td><td>[0.27, 0.63]</td><td>3 / 6 (−0.12)</td></tr>
<tr><td>floq BoN</td><td>6/25</td><td><b>0.24</b></td><td>[0.11, 0.43]</td><td>2 / <b>10</b> (−0.32)</td></tr></table>
<p><img src="videos/critic-pfx/30_pfx_bon.png" alt="per-prefix td-max joint-argmax BoN success"></p>

<h3>Verdict</h3>
<p>① <b>Fixing the bootstrap and the selection rule changes nothing.</b> scalar, HL-Gauss and floq all sit below the
VLA, floq worst (0.24). <b>rand (0.68) tops the critics</b> — resampling helps, critic arg-max hurts. The
<b>winner's curse</b> survives td-max and per-prefix.</p>
<p>② <b>Why — the binding constraint is neither the head nor the deployment rule, it is coverage.</b> Even with the
max, all 16 candidates are <b>the same VLA's near-demo samples</b>, so no <b>counterfactual action</b> far from the
demo is ever manufactured. The max then just picks "whichever near-demo candidate the critic most overvalues" — exactly
the failure that <span class='xref' data-eid='model-based'>the missing candidate-axis signal</span> and
<span class='xref' data-eid='conservatism'>two-axis conservatism</span> named. <b>Two independent deployment designs
(demo-bootstrap/full-chunk and td-max/per-prefix joint) converge on the same negative conclusion.</b></p>

<h3>Honest caveat — n=25 single seed is underpowered</h3>
<p>The VLA baseline swung <b>0.80 (prior) → 0.56 (here)</b> at the same seed and N. A long-horizon sim plus bf16 VLA
inference differing across GPU nodes flips borderline scenes. So <b>absolute rates are unstable at n=25</b>; the
trustworthy signal is the <b>within-run paired direction</b> (critic arg-max &lt; resample &lt; VLA, by McNemar). A
confirmed verdict is deferred to a <b>multi-seed run-level CI</b> (follow-up). Single critic (no ensemble) is also a
simplification — min-ensemble pessimism is an orthogonal axis that curbs the winner's curse, a natural next lever.</p>

<h3>So next</h3>
<p>Neither the head (HL-Gauss), the deployment (joint arg-max), nor the bootstrap (td-max) manufactures coverage. The
remaining head-on move is <b>training-time conservatism that pushes the candidate axis down</b> (CalQL-style: push
counterfactual candidates below the demo) and <b>on-policy counterfactual generation</b> — reconverging on the
<span class='xref' data-eid='conservatism'>conservatism</span> / <span class='xref' data-eid='calql'>Cal-QL</span> path.</p>
<p class='sub'>Reproduce: <code>probes/eval_bon_pfx.py</code> (per-prefix td-max train → VLA joint-argmax rollout),
figure <code>probes/plot_pfx.py</code>. Result <code>bon_pfx_compare.json</code> committed.</p>""",
)

en(
    "deas",
    "DEAS reproduction + correction — the critic ties the VLA (neither beats nor hurts); our earlier 'critic harmful' verdict was n=25 noise",
    """
<p class='sub'><b>A note to worker A + a correction of our earlier verdict.</b> In
<span class='xref' data-eid='critic-heads'>critic-heads</span> and <span class='xref' data-eid='critic-pfx'>critic-pfx</span>
we concluded "no critic beats the VLA via BoN → the binding constraint is coverage." The user doubted this
("cand[0] is itself a VLA sample, so an arg-max shouldn't do worse"). A literature search says — <b>our conclusion
may be partly wrong. The cause was not coverage but the bootstrap operator we chose (td-max).</b></p>

<h3>DEAS (arXiv:2510.07730) — our exact domain names this failure</h3>
<p>DEAS ("DEtached value learning with Action Sequence", Changyeon Kim, Younggyo Seo, Kimin Lee, Yuke Zhu) is
<b>VLA + RoboCasa Kitchen + action sequences (chunks) + distributional value + BoN</b> — our stack. Verbatim:</p>
<blockquote class='sub'>"directly adopting such sequences in actor-critic algorithms introduces <b>excessive value
overestimation</b>, which we address through <b>detached value learning that steers value estimates toward
in-distribution actions</b> that achieve high return in the offline dataset."</blockquote>
<p><b>Our critic-pfx bootstrap `V_next = max_j Q(s′, cand_j)` (td-max) is exactly this excessive overestimation.</b>
Among 8–16 near-demo candidates, the max feeds the one the critic <b>most overestimates</b> into the bootstrap;
the value inflates and the deployment arg-max then executes that inflated candidate — the winner's curse, amplified.
DEAS <b>drops the max</b> and bootstraps from an expectile state value V. And <b>DEAS beats the GR00T baseline on
RoboCasa</b> (e.g. PnPCounterToMicrowave ~45%→~65%) — a critic BoN CAN beat the VLA, in our own domain.</p>

<h3>DEAS methodology (from the original <code>deas_critic.py</code>)</h3>
<p><b>① V loss = expectile + HL-Gauss together</b> — the expectile is a scalar weight on the loss, so it composes
with categorical cross-entropy:</p>
<table class='num'><tr><td><code>q_demo = min(Q1_tgt, Q2_tgt)(s, a_demo)</code>  # in-distribution demo action, detached</td></tr>
<tr><td><code>g = where(q_demo &gt;= V, τ, 1−τ)</code>  ;  <code>L_V = mean( g · CE(V_logits, HLGauss(q_demo)) )</code></td></tr></table>
<p><b>② Q loss = bootstrap from V (not a candidate max)</b> — dual discount γ1 (within-chunk reward) and γ2 (across-chunk):</p>
<table class='num'><tr><td><code>target = Σ_i γ1^i·r_i + γ2^(nH)·(1−done)·V(s′)</code>  ;  <code>L_Q = (HLGauss_CE(Q1,target)+HLGauss_CE(Q2,target))/2</code></td></tr></table>
<p>double critic min + EMA targets. Deploy BoN: <code>score=min(Q1,Q2)(z,cand)</code>, arg-max.</p>

<h3>Result (1) — methodology ported to our backbone (DEAS's GR00T values verbatim)</h3>
<p>Instead of the whole Isaac-GR00T stack, we ported <b>only the DEAS methodology onto our pi05 backbone, mixed
annotation and AQC trunk</b> (<code>probes/eval_deas.py</code>): V = HL-Gauss + expectile, Q bootstraps from V (td-max
dropped), double-min, <b>dual-discount γ1=0.9/γ2=0.99, negative reward (−1/step, support [−100,0]), τ=0.7</b> — exactly
DEAS's RoboCasa reproduction values. <b>Single task, PrepareCoffee</b>, N=10 arg-max.</p>
<p><b>Axis 1 — DEAS fixed, sweep the head (n=25, provisional):</b> scalar 0.52 / <b>HL-Gauss 0.64 = VLA 0.64 (tie)</b> /
floq 0.36. HL-Gauss is best. floq diverged on the [−100,0] support (flow velocities ~100, q_loss 58.8); <b>normalizing
the value into [−1,0] fixed convergence entirely (q_loss 0.008)</b> — a pure scale issue (the user's point). Even so,
normalized floq's n=25 BoN is 0.40, below HL-Gauss, but n=25 is inconclusive.</p>

<h3>Result (2) — our td-max vs DEAS, one paired run, high power</h3>
<p>n=25 single-seed swung 0.52–0.64 run to run (same critic, different node) — undecidable. So we <b>fixed the head
(HL-Gauss) and varied only the bootstrap</b> (td-max = candidate max / DEAS = expectile-V), <b>6 seeds × 25 = 150
trials/arm</b>, scene-paired, run-level t-CI:</p>
<table class='num'><tr><th>arm</th><th>bootstrap</th><th>run-level rate</th><th>±95% t-CI</th></tr>
<tr><td>VLA (baseline)</td><td>—</td><td>0.640</td><td>±0.084</td></tr>
<tr><td>rand (null)</td><td>—</td><td>0.553</td><td>±0.062</td></tr>
<tr><td><b>td-max (ours)</b></td><td>max_j Q(s′,cand_j)</td><td><b>0.660</b></td><td>±0.115</td></tr>
<tr><td><b>DEAS</b></td><td>expectile-V</td><td><b>0.633</b></td><td>±0.097</td></tr></table>
<table class='num'><tr><th>paired Δ̄ (per-seed diff)</th><th>value</th><th>95% t-CI</th><th>significant?</th></tr>
<tr><td>td-max − VLA</td><td>+0.02</td><td>±0.10</td><td>no (CI spans 0)</td></tr>
<tr><td>DEAS − VLA</td><td>−0.01</td><td>±0.13</td><td>no</td></tr>
<tr><td>DEAS − td-max</td><td>−0.03</td><td>±0.14</td><td>no</td></tr></table>
<p><img src="videos/deas/31_runlevel_cmp.png" alt="run-level 6-seed comparison vla/rand/tdmax/deas"></p>
<p class='sub'>Bars = run-level mean ±95% t-CI, dots = per-seed rates (that spread is why single-seed n=25 could not be trusted).</p>

<h3>Verdict — all three are statistically indistinguishable from the VLA (with a correction)</h3>
<p>① <b>the critic (td-max and DEAS) ties the VLA — it does not hurt.</b> ② <b>nor does it beat it</b> (Δ̄≈0, CI spans 0).
③ <b>td-max ≈ DEAS</b> — the bootstrap operator (max vs expectile-V) makes no difference here. Both my "td-max was THE
problem" and "DEAS fixes it" are <b>unsupported</b>. ④ <b>only rand (0.55) dips</b> — the critic out-ranks random
(0.66>0.55) but cannot exceed the VLA's own top sample.</p>
<p><b>Important correction.</b> The claim in <span class='xref' data-eid='critic-heads'>critic-heads</span> and
<span class='xref' data-eid='critic-pfx'>critic-pfx</span> that "the critic is <b>significantly worse</b> than the VLA
and than random (McNemar 0/10 etc.)" was an <b>n=25 single-seed noise artifact</b>. At power, a good head (HL-Gauss) +
double-min ties the VLA. A correction banner was added to both posts.</p>
<p>⑤ <b>coverage, restated.</b> On a single task (PrepareCoffee, VLA already 0.64) BoN has little headroom. DEAS beats
GR00T because of <b>24-task diversity = broad coverage</b>, unlike our <b>single-task near-demo</b> pool — consistent
with our standing binding-constraint = coverage.</p>
<p class='sub'><b>Meta-lesson.</b> Five n=25 single-seed closed-loop verdicts were all noise. From now, verdicts start at
<b>run-level multi-seed</b>. Reproduce: <code>probes/eval_deas.py</code> (DEAS), <code>eval_compare.py</code> (high-power
td-max vs DEAS), <code>plot_cmp.py</code>. DEAS code <code>github.com/csmile-1006/DEAS-Isaac-GR00T</code> read in full.</p>""",
)

en(
    "exp-board",
    "🧭 Experiment board — planned / running / done (owner · wandb · report)",
    """
<p class='sub'>A <b>living board</b> so experiments don't get forgotten (started-then-dropped) or run-without-tracking.
Flow: planned → running → done. <b>Update rule</b>: whenever an experiment is submitted/finished, update this board in the
same cycle (replace the same eid via <code>space_add_entry.py</code>). Report chips navigate on click.</p>

<h3>🟡 Running</h3>
<table class='num'><tr><th>experiment</th><th>owner</th><th>note</th><th>wandb</th><th>report</th></tr>
<tr><td colspan="5" class="pending">No job running now (last batch finished). Next from Planned below.</td></tr></table>

<h3>🔵 Planned — next candidates</h3>
<table class='num'><tr><th>experiment</th><th>owner</th><th>note</th><th>report</th></tr>
<tr><td><b>MVE gate</b> — does the crystal ball separate candidates?</td><td>B</td><td>gate worker A's <code>train_mve_critic</code>: candidate resolution + disagreement pessimism → run-level rollout if it passes</td><td><span class='xref' data-eid='mb-arq'>mb-arq</span></td></tr>
<tr><td><b>N sweep + λ-weighted min/max</b></td><td>B</td><td>EMaQ guidance: revisit small N(5), λ toward min (we were fixed at 8–10)</td><td><span class='xref' data-eid='deas'>deas</span></td></tr>
<tr><td><b>TD-SF-ARQ design</b> (phase-2 preregistration)</td><td>B</td><td>head=HL-Gauss+expectile-V settled, wall=coverage — write the preregistration</td><td><span class='xref' data-eid='tdsf-arq'>tdsf-arq</span></td></tr>
<tr><td><b>on-policy counterfactual generation</b></td><td>B</td><td>attack the coverage root — sim rollouts to make counterfactual data</td><td><span class='xref' data-eid='conservatism'>conservatism</span></td></tr>
<tr><td><b>GR1 port</b> (pi05 training)</td><td>B→node200</td><td>awaiting external run on node200/B200; handoff/norm-stats/config committed</td><td><span class='xref' data-eid='gr1-port'>gr1-port</span></td></tr></table>

<h3>🟢 Done — worker B</h3>
<table class='num'><tr><th>experiment</th><th>status</th><th>headline</th><th>wandb</th><th>report</th></tr>
<tr><td>DEAS reproduction + high-power</td><td>final</td><td>critic <b>ties</b> the VLA (neither beats nor hurts); n=25 verdicts were noise</td><td>offline</td><td><span class='xref' data-eid='deas'>deas</span></td></tr>
<tr><td>critic heads ×3 (scalar/HLG/floq)</td><td>final(corrected)</td><td>categorical carries most gain — closed-loop was n=25 noise</td><td>offline</td><td><span class='xref' data-eid='critic-heads'>critic-heads</span></td></tr>
<tr><td>per-prefix td-max + joint argmax</td><td>final(corrected)</td><td>operator change = same — noise</td><td>offline</td><td><span class='xref' data-eid='critic-pfx'>critic-pfx</span></td></tr>
<tr><td>floq (flow-matching critic)</td><td>final</td><td>capacity yes, coverage no; value [−1,0] normalization fixed convergence</td><td>offline</td><td><span class='xref' data-eid='floq'>floq</span></td></tr>
<tr><td>embedding comparison + DiT probe policy</td><td>final</td><td>offline metrics fail to predict closed-loop (overturned)</td><td>offline</td><td><span class='xref' data-eid='embed-compare'>embed-compare</span></td></tr>
<tr><td>model-based gate</td><td>shelved</td><td>candidate-axis action info +7.3% — weak</td><td>offline</td><td><span class='xref' data-eid='model-based'>model-based</span></td></tr>
<tr><td>horizon-decisive diagnostic</td><td>final(shelved)</td><td>embedding-defect vs no-signal discriminator env</td><td>offline</td><td><span class='xref' data-eid='horizon-probe'>horizon-probe</span></td></tr>
<tr><td>two-axis conservatism synthesis</td><td>living</td><td>distribution-shift vs estimation-error exploit — matches EMaQ</td><td>—</td><td><span class='xref' data-eid='conservatism'>conservatism</span></td></tr></table>
<p class='sub'>Earlier ladder (v11/v12/final/aqc/families/…) is in the report list / mind-map.</p>

<h3>🟣 Worker A experiments (reference — owned/updated by worker A)</h3>
<table class='num'><tr><th>experiment</th><th>owner</th><th>note</th><th>wandb</th><th>report</th></tr>
<tr><td>patch-critic (frozen DINOv2 + distributional ARQ IQL)</td><td>A</td><td>cost-to-goal reward, adaptive-K/BoN deploy</td><td>acrft / patch-critic</td><td><span class='xref' data-eid='wa-patchcritic-method'>patch-critic</span></td></tr>
<tr><td>MVE critic / cheap-z dynamics</td><td>A</td><td>model-based value expansion + 5-member ensemble dynamics</td><td>—</td><td>code <code>train_mve_critic.py</code></td></tr>
<tr><td>EMaQ paper review</td><td>A</td><td>BoN as a Bellman operator; large N exploits critic error</td><td>—</td><td><span class='xref' data-eid='wa-emaq-bon'>EMaQ</span></td></tr>
<tr><td>policy server / value-guided serving</td><td>A</td><td>server-side BoN, HUD provenance</td><td>—</td><td>code <code>serve_policy.py</code></td></tr></table>

<p class='sub'>A true template-level <b>Experiments tab</b> needs worker A's fixed <code>index.html</code> — <b>coordinate with worker A</b>
(proposal: a shared <code>experiments.json</code> + a tab). This board is the living stand-in until then; each worker maintains its rows.</p>""",
)

en(
    "mb-arq",
    "Model-based critic, in plain words — can an 'imagining judge' beat the VLA?",
    """
<p class='sub'>This post explains one thing — the <b>model-based critic (model-based ARQ / MVE)</b> — from scratch,
by analogy, with almost no math: why we try it, what it is, and whether it is likely to work.</p>

<h3>0. The stage — robot, candidates, judge</h3>
<p>Our robot policy (<b>VLA</b>) proposes, at every moment, <b>several candidate moves</b> (say 8–16) — small variations
of the same situation. We want a <b>judge (critic) to pick the best one</b> — "propose N, keep the best" is
<b>Best-of-N (BoN)</b>. A good judge makes the robot better.</p>

<h3>1. What we found tonight — the judge can't beat the robot</h3>
<p>We built several judges (<span class='xref' data-eid='deas'>DEAS reproduction/correction</span>) and, measured
properly, <b>the success rate with a judge equalled just trusting the robot (execute the first candidate).</b> Why?
Two reasons:</p>
<ul>
<li><b>The candidates are all alike</b> — variations from the same robot, so telling "slightly better" apart is thin.</li>
<li><b>The judge never saw the alternatives</b>: the training data (human demos) has only the <b>one action actually taken</b>
per situation, never "what if a different action." So the judge essentially <b>guesses</b> on unseen candidates — and when
guessing it tends to pick the one it <b>over-values by mistake</b> (the <b>winner's curse</b>), so it loses.</li>
</ul>
<p class='sub'>We call this binding gap <b>coverage</b> — the data hole of "not knowing the outcome of actions never tried."</p>

<h3>2. The idea — give the judge a 'crystal ball' (this is model-based)</h3>
<p>Instead of ranking candidates by gut, give the judge a <b>model that mimics how the world works (a crystal ball)</b>,
let it <b>imagine what happens if each candidate is executed</b>, then score <b>how good that imagined result is</b>.
Like a chess engine: "if I play here → the board becomes this → that board is winning." We score
<code>candidate = imagined immediate reward + γ·V(imagined landing state)</code>, where V is the usual "how good is this state" judge.</p>

<h3>3. Avoiding over-valuation — 5 crystal balls, 'be pessimistic when unsure'</h3>
<p>Imagination can be wrong. So keep <b>5 differently-trained crystal balls</b>; if they <b>disagree a lot</b> about a
candidate (uncertain), <b>take the worst value</b> (min). Only candidates the ensemble confidently predicts score high —
structurally blocking the winner's curse of over-valuing unknown candidates. This safeguard is what we lacked tonight.
(This is <b>MVE, model-based value expansion</b>; worker A already implemented it in <code>train_mve_critic.py</code>.)</p>

<h3>4. But — <u>how do you build the crystal ball?</u> (the real crux)</h3>
<p>The model learns from <b>what the robot actually did</b>: collect "in this situation, this action led here" and fit
(situation, action) → (landing). The same problem hits here:</p>
<ul>
<li>The data has <b>one action per situation</b>. So the model learns "this situation usually goes here" well, but
<b>barely learns "how changing the action changes the landing"</b> (when we measured it, the action added only +7.3% of
predictive info). → the <b>same coverage wall</b> lives inside model learning.</li>
<li>In particular <b>"does this candidate actually grasp the mug?"</b> — object-interaction outcomes for untried actions —
<b>cannot be learned in principle</b>, because there is no such data.</li>
</ul>

<h3>5. Why it might still work — the physics (arm motion) is cheap to learn</h3>
<p>One thing the crystal ball <b>does</b> learn well: <b>the arm's physics</b>. "Command right → the arm goes right" is
deterministic and always visible, so it is learned reliably. So the model can <b>clearly imagine 'where the arm heads'</b>
for each candidate. If good-vs-bad candidates are decided mostly by <b>"is the arm heading correctly toward the mug,"</b>
then even without predicting object outcomes, <b>trajectory quality</b> can separate them. <b>Condition</b>: the
representation the model runs in must <b>preserve that arm/position detail</b>. As our
<span class='xref' data-eid='embed-compare'>embedding comparison</span> showed, a progress-only representation (φ) that keeps
only "how close to done" <b>destroys control information</b> — learn the model there and it cannot separate candidates.
<b>The representation choice decides it.</b></p>

<h3>6. Honest conclusion & the next single step</h3>
<p>The model-based critic is a principled upgrade — "rank by gut" becomes "imagine then rank + be pessimistic when unsure."
<b>But</b> the crystal ball inherits the demo data's hole (coverage): <b>it can imagine the arm's path, but not the object
outcome of untried actions.</b> So:</p>
<ul>
<li><b>Could work</b>: if candidate quality is mostly "is the arm heading right" — the trajectory channel + ensemble pessimism
leaves room to beat the robot.</li>
<li><b>Won't</b>: if predicting object outcomes is essential — then you must <b>actually try candidates in sim</b> to make the
data (on-policy), the conclusion we keep reaching.</li>
</ul>
<p><b>So the next experiment</b> is a <b>gate</b>: does the trained model actually separate candidates — (a) confident
candidates get distinct values, (b) unknown candidates make the 5 disagree and fall to pessimism? Only if it passes do we go
to real rollouts (and averaged over <b>several seeds</b> — tonight taught us a single n=25 is noise). Continues
<span class='xref' data-eid='model-based'>the earlier model-based work</span>.</p>""",
)

en(
    "xworker-0808",
    "Cross-worker review — learning from worker A",
    """
<p><b>What.</b> Adopted from worker A: the trial-paired McNemar test (immediately revealed significant TD harm
in three FINAL arms); mutual replication (their independent φ + Cal-QL stack: BoN tie .700, full authority
.300 p=.004 <i>(update 08-10: the original .300/p=.004 was voided by a φ normalization bug, but the fixed identical-seed rerun reproduces the catastrophe more strongly — .133, +2/−19, p=.0002; the number was corrected, not the conclusion)</i>); their representation-side attack on episode identity vs our data-side one (complementary);
ops practices (zombie-job mtime detection, checkpoint archiving). The loop is now bidirectional — their MVE
critic cites both workers' significant negatives as design constraints.</p>
<p><b>08-10 update — worker A's 3-task cross-verdict validates our GR1 design.</b> Their new verdict ("V is
solved, Q is a spread puzzle"): ① YAM mixed-outcome data yields their <b>best-ever V (Spearman .966)</b> yet
action sensitivity 0 — forced by candidate collapse (1.8% spread from an overtrained policy). ② <b>Puzzle: GP
has the MOST diverse candidates (11%) yet sens≈.0002</b> — the intuition that action-space spread implies value
spread is broken. ③ Episode-level failure labels teach which states fail, not which same-state chunk to pick:
the barrier is pinned to the <b>absence of same-state counterfactuals</b>. Two implications for our GR1 plan:
(a) measuring action-space spread alone is misleading (the GP counterexample) — our preregistered
<b>outcome-level paired rand-vs-vla comparison</b> is the right gate; (b) if a demo-only critic fails that gate,
the next move is not another trick but <b>--policy-seed-style on-policy collection</b> (K rollouts per scene =
scene-level counterfactuals), converging with worker A's "only on-policy intervention provides them."</p>
<p><b>08-10 afternoon update.</b> (a) <b>Retraction handled:</b> worker A voided all φ-consuming
rollout arms (raw tokens fed to a phi trained on standardized ones; 45% output error) — including the
".300, p=.004 full-authority catastrophe" we cite; every citation on this hub now carries a caveat (offline
results and controls stand; identical-seed reruns underway). (b) <b>γ-ceiling sensitivity correction
adopted:</b> the true value-difference ceiling is ΔQ≈V·|lnγ|·Δt; calswap's .29–.52 exceeded it (CQL artifact,
resolving "passes offline, zero in rollouts"), YAM's 0.0000 was a gate defect. New gate: temporal resolution
≥ 30% of ceiling — adopted into our TD-SF-ARQ stage-A preregistration. (c) <b>Cross-trajectory neighbor
verdict:</b> φ's bridging axis is relational geometry, not stage (act-cos .661 vs stage-only .334); the
remaining bottleneck named as "the action-conditioned backup that crosses that bridge" — exactly the axis our
vector-SF target feeds.</p>
<p><b>08-11 update — worker A's YAM real-robot scaling warns our GR1.</b> r51 (YAM π0.5, checkpoints
50k–200k ×10 trials): <b>0/50 completions and a decisive crossing</b> — 200k passes milestone ≥1 at 10/10 but
≥3 at 0/10; overtraining sharpens the easy segment and wipes out the hard one, matching the offline 200k
candidate-spread collapse (.018) — the policy goes near-deterministic and the candidate axis dies. The H60
(history-60) variant loses via lost open-loop correction (same observation motivating our adaptive chunking).
<b>Immediate GR1 implications:</b> ① a 0/N at 30k is not only "undertrained" — worker A's longer-trained 200k
got worse on hard milestones; our phase-1 must read headroom <b>per milestone</b> (a per-milestone pass-rate
curve is informative even at 0/50 overall), and a <b>less-trained checkpoint (e.g. 20k)</b> with live candidate
spread may be better for critic experiments. ② This is a real-robot re-confirmation of our σ_signal frame:
overtraining → spread collapse → selection signal vanishes. We add to phase-1: <b>programmatically pick the
critic checkpoint at the spread-maximizing point</b> (measure 16-candidate std per checkpoint).

""",
)

en(
    "conservatism",
    "The conservatism spectrum — one frame for every null",
    f"""
<p><b>Claim.</b> Every verdict fits one question: where does a method sit on the conservatism dial, and which of
the <b>two risk axes</b> does it guard? Axis 1 — distribution shift (OOD): BC-style constraints handle it;
BoN is perfect here (all candidates in-support). Axis 2 — <b>estimation-error exploitation (optimizer's
curse)</b>: argmax picks the most overestimated of N noisy scores, in-support or not; support constraints are
powerless. max-of-N queries the N/(N+1) quantile, so a τ=0.9 critic certifies only up to N≈10.</p>
<p><b>Positions:</b> BoN = selection-type safety (KL ≤ log N; the safe-and-useless extreme where we sat);
IQL = query avoidance; CQL/CalQL = value suppression (idle: no OOD reaches our deployment);
FQL/velocity-steering = policy proximity; unconstrained actor-critic = the dangerous end (worker A's .300,
p=.004 <i>(update 08-10: the original .300/p=.004 was voided by a φ normalization bug, but the fixed identical-seed rerun reproduces the catastrophe more strongly — .133, +2/−19, p=.0002; the number was corrected, not the conclusion)</i>).</p>
<p><b>v20 — small-N BoN (N=4; ACSAC check, user question).</b> ACSAC reports n=4 as optimal, and our frame
agrees on the mechanism: max-of-4 queries the ~80th percentile — INSIDE a τ=0.9 critic's certified range
(N≤10), unlike N=16 — with less than half the curse coefficient. We had never tested it (all runs N=16).
Single variable: same FINAL iql critic, N 16→4, 4 paired seeds. <b>Verdict: mean −0.015, CI [−0.090,+0.060],
McNemar +30/−33 p=0.801 — null.</b> Less negative than N=16 (−0.065), sign-consistent with reduced curse, but
no gain: the binding constraint here is the absence of σ_signal, not τ↔N mismatch. The GR1 pilot will include
an N∈{{4,8,16}} sweep where real spread may exist.</p>
<p><b>SNR condition.</b> BoN gain ≈ c_N · σ_signal · ρ, c_N≈√(2 ln N): the true candidate spread is the sole
source of gain. Measured: demo-only band 0.002–0.023 (at the HL-Gauss bin width), rand≈vla → gain ≈ 0 for any
selector. Diversification opens the spread (×6.8 at noise 1.5, ×25 at 2.0, critic band following):</p>
{img(P / "21_cand_diversity.png", "diversity sweep")}
<p>v17 (diversified, no error-conservatism) went negative as predicted; v17b (σ-veto on ensemble disagreement)
flipped positive at n=8 (+0.052, p=0.076) then <b>washed out at n=16 (+0.019, p=0.383) — the trick family is
closed</b>, reaffirming the n≥8 rule. History conditioning: null for iql and td. Principled prescriptions
(LCB/veto, τ↔N matching, ranking losses) remain contingent on a task where σ_signal exists at all.</p>
""",
)

en(
    "v14",
    "v14 — the shortcut-free data verdict",
    """
<p><b>What.</b> Retraining on demos + K-per-scene rollouts (605,684 frames; shortcut removed). iql_v14
−0.035 CI[−0.083,+0.013] (tight null); td_max_v14 −0.110 (null, negative-leaning); calql_v14 −0.140,
McNemar p=0.001 — significant harm (CQL over-suppression on failure-heavy data is the working hypothesis).
Removing the memorization shortcut alone does not convert band opening into success.</p>
""",
)

en(
    "calql",
    "CalQL (CO-RFT) — the first training-time candidate-axis signal",
    """
<p><b>What.</b> TD + mc-floor + a CQL term pushing the 16 candidates down vs the demo chunk. calql_noprop
(n=8): −0.018 CI[−0.103,+0.068] (early bimodality washed out). calql_mixed (n=4): +0.015, McNemar p=0.813 —
the only positive-pointing mixed arm, still null; on v14 data it flipped to significant harm. With every
deployed candidate in-support, the suppression dial spins idle.</p>
""",
)

en(
    "phi-ladder",
    "The embedding ladder — dimension or geometry? (HILP φ replication)",
    """
<p><b>What.</b> Worker A's TD-only readout collapses episode identity; two user questions shaped the controls:
PCA-128 (dimension) and decoder/probes (information loss). Battery: episode-probe accuracy .99 raw / .92 PCA /
<b>.756 φ</b> with progress R² preserved — geometry, not dimension (our φ still weaker than worker A's).
Incidents: NaN divergence from the ‖·‖ gradient at 0 (duplicate static frames; ε-safe distance fix);
completion markers must be &amp;&amp;-gated. Information loss, programmatically: per-dim proprio R²
raw .760 → PCA .653 → φ .546 (φ+proprio .617) — φ discards short-range action-relevant information, the
mechanistic explanation for its BoN failure. Critic ladder at n=8: φ-128 −0.010 null; the n=4 monotone
signal was noise. Closed.</p>
<h3>Decoder artifacts — what each one shows</h3>
<table class='num'><tr><th>Artifact</th><th>What it is</th><th>How to read it</th></tr>
<tr><td>φ reconstruction panel (below left)</td><td>held-out originals (top row) vs reconstructions from φ-128 (bottom row)</td><td>only information retained in φ can be reconstructed — kitchen layout and arm pose survive, blurred gripper/objects mean that information is gone from φ</td></tr>
<tr><td>raw reconstruction panel (below right)</td><td>the same frames reconstructed from the raw 2048-d token</td><td>the difference vs the φ panel IS what φ discarded</td></tr>
<tr><td><a href="videos/decoder/22_decoder_phi128_walk.mp4" target="_blank">φ embedding walk (video)</a></td><td>linear interpolation between two real frame embeddings, each point decoded</td><td>smooth plausible scene morphing = the φ space encodes meaning continuously; jumps = holes in the space</td></tr>
<tr><td><a href="videos/decoder/22_decoder_phi128_ride.mp4" target="_blank">cross-episode ride (video)</a></td><td>φ extracted along an unseen episode's trajectory, decoded in order</td><td>if task progress is recovered on unseen episodes, φ generalizes progress rather than memorizing episode identity</td></tr></table>
<p style='display:flex;gap:8px;flex-wrap:wrap'>
<img src="videos/decoder/22_decoder_phi128_recon.png" alt="phi-128 reconstruction panel" style='max-width:49%'>
<img src="videos/decoder/22_decoder_raw_recon.png" alt="raw-2048 reconstruction panel" style='max-width:49%'></p>
<p class='sub'><b>On image quality (08-10 user feedback):</b> the blur is not only the embedding's fault — an
L2 regression decoder structurally predicts the mean of all compatible images. To see the true upper bound of
what the embedding retains, a conditional generative model is the right tool. → <b>Follow-up (done): a
φ-conditioned diffusion decoder</b> (same kroll data and protocol, φ-128 vs raw-2048 conditioning, conditional
DDPM 30k).</p>
<h3>Diffusion decoder verdict (08-11, programmatic)</h3>
<p>Best-of-4 pixel MSE over 128 held-out frames (lower = more information retained in the conditioning embedding):</p>
<table class='num'><tr><th>Conditioning</th><th>best-of-4 MSE ↓</th><th>Reading</th></tr>
<tr><td>raw 2048</td><td><b>0.0193</b></td><td>baseline — the raw token reconstructs pixels best</td></tr>
<tr><td>φ-128</td><td>0.0602</td><td><b>3.1× worse</b> than raw — φ-conditioning alone makes far worse frames</td></tr></table>
<p><b>Verdict:</b> even with a generative model (mean-prediction bias removed) φ retains substantially less
reconstruction information than raw — the <b>same direction on an independent metric</b> as the L2 proprio
probe (R² raw .760 → φ .546). So "φ discards short-range information" holds regardless of decoder type
(regression vs generative). Panels/walks:
<a href="videos/decoder/23_diff_raw_recon.png" target="_blank">raw panel</a> ·
<a href="videos/decoder/23_diff_phi128_recon.png" target="_blank">φ panel</a> ·
<a href="videos/decoder/23_diff_raw_walk.mp4" target="_blank">raw walk</a> ·
<a href="videos/decoder/23_diff_phi128_walk.mp4" target="_blank">φ walk</a> (qualitative; verdict is the MSE). (Task#9 done)</p>
<h3>φ→action BC probe (08-11, user-requested — an experiment that had slipped)</h3>
<p>The decoder probe measured how well φ reconstructs images/proprio. But what we actually need is whether
<b>φ carries enough information to reproduce the action</b>. So we trained an embedding→demo-action-chunk
(H=8×12, z-scored) BC MLP under the identical protocol for raw/PCA/φ and compared held-out (same kroll
alignment as the decoder probe):</p>
<table class='num'><tr><th>Embedding</th><th>action R² ↑</th><th>held-out MSE ↓</th></tr>
<tr><td>raw 2048</td><td>0.708</td><td>0.258</td></tr>
<tr><td>PCA-128</td><td>0.697</td><td>0.267</td></tr>
<tr><td><b>φ-128</b></td><td><b>0.682</b></td><td>0.280</td></tr></table>
<p><b>Verdict — correcting the decoder narrative:</b> φ <b>retains nearly all the action information</b>
(R² .682 vs raw .708, 3.6% relative). This seems to contradict the decoder probe's "φ loses 3.1× the
reconstruction info," but it resolves: <b>what φ discards (fine visual detail, reconstruction-grade gripper
angle) is not needed to predict the action</b>; φ keeps the action-relevant subspace. Two implications:
① the "φ threw away info so it can't be used" explanation is <b>rejected</b> — φ is action-sufficient, so
φ-critic's failure to open BoN is not a representation defect but the <b>absent counterfactual (axis 2)</b>
(strengthening our conclusion). ② the <b>representation axis is nearly flat for BC</b> (raw .708 / PCA .697 /
φ .682) — the embedding choice (φ vs PCA) is not a decisive variable for TD-SF-ARQ stage A, and using φ costs
no action information. <b>Caveat (BC≠discrimination):</b> this probe reproduces one demo action, not
same-state candidate discrimination — φ being act-sufficient doesn't lift the critic if the counterfactual
signal is absent. Together the two facts converge on "representation sufficient, data insufficient."</p>
""",
)

en(
    "morning-0808",
    "Morning synthesis (08-08)",
    """
<p><b>What.</b> Overnight: TD+mixed silent deaths root-caused and fixed (INT32 buffer + constants duplication);
qc(mixed) first through the post-fix pipeline (null); td_max_demo −0.190; K-per-scene complete (45% mixed
kitchens); v14 merged (605k frames); raw-JSON audit passed; reporting system rebuilt (5W1H, cross-links,
thread, mindmap, white theme, paper-style figures).</p>
""",
)

en(
    "morning-0809",
    "Morning synthesis (08-09) — the night of convergence",
    """
<p><b>One line.</b> Every branch that was open overnight closed honestly — the trick family (v17b),
embeddings (phi, phi+proprio), history conditioning, and the offline model-based composition. Two days of
experiments converge on a single structural fact: <b>on this task, with this task-finetuned VLA, there is no
value difference among candidates for selection to harvest.</b> The essential next move is the GR1 transfer;
two user decisions are pending.</p>
<table class='num'><tr><th>Experiment</th><th>Verdict</th><th>Numbers</th></tr>
<tr><td>v17b diversified + sigma-veto (n=16)</td><td>null — trick family closed</td><td>mean +0.019, CI [−0.021,+0.059], McNemar +136/−121 p=0.383</td></tr>
<tr><td>phi-128 embedding ladder (n=8)</td><td>null — the n=4 signal was noise</td><td>mean −0.010</td></tr>
<tr><td>History critic (iql, td)</td><td>both null — axis closed</td><td>−0.025 / +0.020</td></tr>
<tr><td>Model-based composition gate (4 coords)</td><td>rejected offline, zero rollouts spent</td><td>demo winrate .479–.487 (0.5 = blind)</td></tr>
<tr><td>phi+proprio (user question)</td><td>better proprio retention (R² .546→.617) but gate equally closed</td><td>winrate .485</td></tr></table>
<p><b>Why this is progress.</b> Eight independent negative routes point to the same structural fact, each closed
with preregistered criteria and paired statistics, mutually replicated with worker A's independent stack.
The map of "what does not work" is complete, and it dictates the design of the next stage (GR1 pilot:
measure headroom and candidate spread first).</p>
<p><b>Pending decisions:</b> (1) GR1 training compute — A: node200/B200 (fast, needs approval) vs
B: this cluster's PRO6000 (96GB); (2) merging PR#4 (110+ commits) into master.</p>
""",
)

en(
    "model-based",
    "Back to essentials with model-based — Q(z,a) = γ^h · V(f(z,a))",
    """
<p><b>Direction (user directive, 08-09).</b> Instead of stacking tricks, one simple construction that attacks
the measured root cause (the candidate axis receives no training signal): <b>Q(z,a) = γ^h · V(f(z,a))</b>,
where f is a latent dynamics model trained by plain supervision on (z, chunk, z′) pairs already present in the
annotation, and V is the existing IQL value network. Deployment = argmax of V(landing) over the standard 16
candidates. No CQL, no veto, no noise pools.</p>
<h3>Gate 1 — does the token pair carry action information?</h3>
<table class='num'><tr><th>Coordinates</th><th>identity</th><th>no-action</th><th>action-cond.</th><th>action info</th></tr>
<tr><td>raw 2048</td><td>0.197</td><td>0.128</td><td>0.125</td><td>+2.6% — faint</td></tr>
<tr><td>PCA-128</td><td>1.225</td><td>0.649</td><td>0.607</td><td>+6.5%</td></tr>
<tr><td>phi-128</td><td>1.082</td><td>0.579</td><td>0.537</td><td>+7.3%</td></tr></table>
<p>Action information lives in compressed coordinates (~3x the relative share), but even +7.3% is an existence
proof, not a dominant signal.</p>
<h3>Gate 2 — does the composed Q rank candidates? (verdict: no)</h3>
<p>On 2,000 held-out frames, compute y = V(f(z,a)) for the executed demo chunk and the 16 stored candidates.
demo_winrate (0.5 = action-blind): raw .479 / PCA .481 / phi .487 / phi+proprio .485.
<b>In-distribution control (user question): coin-flip even on the training frames themselves</b>
(.482–.489, bands narrower still) — a fundamental failure, not a generalization gap. The bottleneck is the
size of the supervision f gets for its action dependence, not memorization; more data would not fix it.</p>
<p><b>Two-day convergence.</b> Selection tricks, embeddings, history, CalQL and model-based composition all
independently confirm the same structural fact. The essential response is to change the stage: GR1 tabletop,
where action selection genuinely separates outcomes — that is finding a valid arena for the question, not a trick.</p>
""",
)

en(
    "gr1-port",
    "GR1 tabletop port plan — moving to an arena where the question is valid",
    """
<p><b>Why GR1.</b> All routes converged on "nothing to harvest on PrepareCoffee with a task-finetuned VLA".
GR1 tabletop (the AQC paper's setting) offers fixed tabletop scenes, bimanual precision manipulation where
action choice separates outcomes, and base success rates with headroom.</p>
<table class='num'><tr><th>Item</th><th>Status</th></tr>
<tr><td>Simulator (robosuite-gr1 + tabletop fork)</td><td>smoke passed 08-07 (.venv-gr1)</td></tr>
<tr><td>Data (Teleop-Sim, 5 tasks)</td><td>12G verified — 1,000 episodes/task, LeRobot, ego_view 256², state/action 44d</td></tr>
<tr><td>Data pipeline code</td><td>gr1_policy + LeRobotGR1DataConfig implemented and committed; TrainConfig draft ready</td></tr>
<tr><td>Eval harness</td><td>Decided: policy-server split (main venv serves VLA+critic via the standard infer
protocol — BoN lives in a server-side Policy adapter, commit length expressed by truncating the returned chunk;
.venv-gr1 runs the env client). No protocol extension needed.</td></tr></table>
<p><b>Launched (08-09, compute B: PRO6000).</b> Config registered (action_dim 48, 30k pilot) → dataset
converted v2.0→v3.0 → norm stats <b>passed (15:00)</b> → pilot finetune auto-started. <b>Port incident ledger
(the checklist for the other 4 tasks):</b> ① LeRobot v2.0 format → v2.1 tag + official converter,
② the converter's --root means the PARENT directory, ③ episodes_stats.jsonl is loaded, not generated — wrote a
generator (real numeric stats from parquet; neutral image placeholder, unused by openpi), ④ state/action dtype
recorded as 'object' → float32, ⑤ no frame_index column → AddProgress fallback from the global index (committed),
⑥ torchcodec fails without FFmpeg shared libs on the nodes → pyav backend env passthrough (committed).</p>
<p><b>Harness complete (17:00):</b> server (serve_bon_policy.py, standard infer contract over eval_critic) +
client (rollout_client.py, .venv-gr1, paired seeds) both committed; all 5 tasks converted to v3. Two more compute incidents: ⑦ the base checkpoint carries (32,·) action projections — shape clash → loader now
drops mismatched entries and keeps fresh params (committed), ⑧ A6000 48GB OOMs even at batch 16 (rejected),
⑨ PRO6000 96GB OOMs at batch 32 → batch 16 adopted. <b>Training entered (21:15, node57 PRO6000): 1.7 s/it,
~13.7 h to 30k (done by mid-morning 08-10).</b> All nine incidents recorded as cause-fix pairs — the port is
complete.</p>
<p><b>Harness contract pinned down (night of 08-09, two GPU-node smokes).</b> The login node has no GPU so EGL
offscreen rendering fails there; the smokes ran as Slurm jobs on 3090 nodes. Confirmed contract: ① obs image key
<code>video.ego_view_pad_res256_freq20</code> (256²×3 uint8); ② the env action is not a flat vector but a
<b>per-part Dict</b> (left/right arm 7 + left/right hand 6 + waist 3 = 29ch) — in the dataset's 44-d layout
(modality.json: arm7·hand6·<i>leg6</i>·<i>neck3</i>·arm7·hand6·<i>leg6</i>·waist3) the legs and neck channels are
exactly 0 throughout the demos (verified from stats), so state is assembled as zero-filled 44-d and the flat 44-d
action is split into the 5 Dict slices; ③ the training prompt is the literal task string
<b>"PnPCanToDrawerClose"</b> (not the natural-language instruction — eval uses the identical string); ④ success is
read programmatically from <code>info["success"]</code>. rollout_client updated and committed; openpi-client
installed into .venv-gr1.</p>
<p><b>Incident ⑩ — pre-empted (23:35).</b> The pilot's checkpoint destination defaulted to <code>/home</code>
(1.4G free of 200G, 100% full) with save_interval=10k, so the first save (step 10k, ~01:58) would have written
~30GB and killed the 13.7-hour run mid-flight with ENOSPC. Inside the safe window before any save, the checkpoint
directory was symlink-swapped to /scratch (12T free) with no interruption — a watcher confirms the 10k save, and
the moment it lands we run a server+client end-to-end smoke on that intermediate checkpoint, validating the whole
evaluation loop before 30k completes.</p>
<h3>E2E harness boot ladder (early 08-10, on the 10k intermediate checkpoint)</h3>
<p>The 10k save landed safely through the /scratch symlink (incident ⑩'s bypass verified). Booting the full
server+client loop then surfaced four incidents, resolved ladder-style — all fixes pre-applied to the phase-1
eval template:</p>
<table class='num'><tr><th>#</th><th>Symptom</th><th>Cause</th><th>Fix</th></tr>
<tr><td>⑪</td><td>Server dies instantly: HF 404 (RepositoryNotFoundError)</td><td>job lacked HF_LEROBOT_HOME → norm-stats loading queried the HF remote instead of the local dataset</td><td>inject the same two env vars as the training job</td></tr>
<tr><td>⑫</td><td>_METADATA not found (…/params/params)</td><td>--checkpoint takes the step directory but we appended /params, doubling the path</td><td>standardize on the step directory</td></tr>
<tr><td>⑬</td><td>server up yet readiness timeout</td><td>the print marker was stuck in the stdout buffer</td><td>python -u + use the websockets log line as the marker</td></tr>
<tr><td>⑭</td><td>client dies with keepalive 1011</td><td>first-request JIT compilation (minutes) monopolizes the ws event loop → pings unanswered</td><td>warm-up inference before serving (committed)</td></tr></table>
<p>Side findings: two 3090-pool nodes with cuInit CUDA_ERROR_UNKNOWN (one was already in bad_nodes.txt — a
missed lookup, now applied as a standing exclude list). The 24GB 3090 proved structurally tight for this serving
config (π0.5 + 16-candidate flow): raising MEM_FRACTION 0.80→0.92 moved the OOM from a 2.4GB to a 267MB
constant (nearly fits, but not quite) → <b>switched to A6000 48GB + PREALLOCATE=false</b>.</p>
<p><b>✅ E2E smoke passed (05:45, node25 A6000).</b> Server warm-up (JIT) → client completing 2 full trials
(720 steps each, paired seeds 5000/5001) → JSON written: the entire loop is verified. Both trials failing is
expected for the 10k (one-third-trained) intermediate checkpoint — the success-rate verdict belongs to the 30k
phase-1. The phase-1 template is finalized: A6000, 24h, seed-split (2 jobs × 25 trials per arm, identical seed
pairing across arms).</p>
<h3>20k diagnostic — 0/25 and the discrimination plan (08-10 morning)</h3>
<p>While waiting for 30k we ran a 25-trial vla diagnostic on the 20k intermediate checkpoint (learning-curve
point + success-plumbing verification). Two more incidents en route: <b>⑮</b> the phase-1 script rewrite dropped
<code>unset LD_LIBRARY_PATH</code> → miniconda libcrypto pollution killed the server instantly. <b>⑯</b>
(instructive) the readiness marker "serving" false-matched the string "openpi.<b>serving</b>" inside the server's
<i>traceback</i> → SERVER_UP misjudged, client waited an hour on nothing. Success markers must never overlap
failure output (replaced with <code>policy on :</code>).</p>
<p><b>Result: 0/25, every trial hitting the 720-step cap.</b> Two readings — (a) 20k is undertrained,
(b) an action-execution semantics mismatch in the harness. Investigation: the env controller runs
<code>control_delta=False</code> (absolute joint angles) and the declared action_space is a nominal [-1,1],
yet measured demo actions span ±3.0 rad — since demos were collected in this very env, raw radians passing
through is the likely truth, but it needs confirmation. Two discriminator jobs submitted: an <b>open-loop
probe</b> (24 demo frames → predicted chunk vs demo-action MSE against a hold-still baseline — settles serving
-stack health) and a <b>2-trial video job</b> (--video-dir added to the client, committed — failure morphology
for the record). Phase-1 stays queued until the probe reads out — we don't burn 4 jobs × 25 trials on a
possibly-broken harness.</p>
<h3>Root cause found — gripper-dim quantile collapse (08-10 afternoon, discrimination ladder complete)</h3>
<p>Even after the 30k finish we held phase-1 and pushed the discrimination to the end. The ladder:
① open-loop probe — on demo frames, prediction MSE 1.82 &gt; hold-still 0.88, predicted |max| 10.8 vs demo
±3.0 → something broken. ② loader probe (the training pipeline itself) — normalized-space MSE 1.5 vs a
zero-baseline of 65,315, which "looked perfect". ③ But that contrast was an illusion: the 65,315 baseline is
dominated by <b>near-constant dims whose normalization explodes to ±1198</b>; excluding those easy dims, the
real-joint error matches the open-loop number. ④ A/B probe (serving candidate path vs direct sample_actions
on the same element) — the two agree (MSE 1.3–2.0): serving is innocent, <b>the model itself never learned
the meaningful dims</b>.</p>
<p><b>The culprit: the hands' last joints (gripper open/close, flat-44 dims 12/34).</b> Parked at 3.0 in
&gt;99% of demo frames → q01=q99=2.9994 (zero span), yet the joint swings 0↔3 during a grasp. Quantile
normalization y=2(x−q01)/(span+1e-6)−1 blows the grasp moments up <b>by a factor of millions</b>, dominating
the flow loss — the most task-critical dimension (grasping) was the most pathological one, and the policy
never learned to close its hands. 0/25 fully explained.</p>
<p><b>Repair &amp; prevention:</b> <code>slurm/repair_gr1_norm_stats.py</code> (committed) — dims with span
&lt; 0.1 get q01/q99 widened to the dataset's true min/max (grippers 0..3, waist-yaw ±0.1; the constant-zero
legs/neck are harmless and untouched). Also fixed: GR1Outputs assumed 2D so its truncate was a no-op on 3D
candidate arrays (committed). <b>pilot-2 retrain launched</b> (gigabyte_pro6000, repaired stats, checkpoints
pointed straight at /scratch this time — incident ⑩ prevented at the source). The same repair will be applied
to the other four task datasets.</p>
<p><b>Next:</b> pilot-2 training → 20k mid-flight diagnostic (open-loop MSE + 25 trials) → 30k phase-1
(vla/rand × seed split) → if valid, annotate/critic/paired verdicts. PR#4 awaits the user's merge.</p>
<h3>Training-resource decision — off this cluster, onto node200/B200 (08-11)</h3>
<p><b>Why.</b> Repeated attempts to finetune the pi05 (3B) GR1 pilot on this cluster's PRO6000 (96GB) kept
hitting <b>shared-card OOM</b>: the nodes <b>oversubscribe GPUs</b> (gres/gpu:1 but the physical card is shared
with other jobs), so the batch-16 working set (~22GB) doesn't fit on a card already >74GB occupied. Dropping to
batch-8 gets in but risks undertraining/instability. <b>User's call (08-11): don't train here</b> — move to the
original option A, node200/B200 (train_rlt.slurm, /data5, the existing VLA-finetune infra).</p>
<p><b>Handoff ready (all committed to the repo).</b> ① config <code>pi05_gr1_rlt</code> (action_dim 48, 30k)
registered; ② <b>repaired norm stats</b> (<code>assets/pi05_gr1_rlt/…/norm_stats.json</code> — the collapsed
quantile span on gripper dims 12/34 widened to the dataset's true min/max, see incident ledger) committed;
③ GR1 policy transforms (gr1_policy, weight_loaders shape-drop) committed; ④ v3 data (PnPCanToDrawerClose 2.9G +
the 5 tasks) — needs copying to node200-accessible storage; ⑤ point checkpoints at /scratch-like free disk
(incident ⑩: /home ENOSPC). <b>Then:</b> train on node200 → at completion, phase-1 via the committed eval harness
(serve_bon_policy + rollout_client, E2E-verified on A6000 with PREALLOCATE=false).</p>
""",
)

# ------------------------------------------------------------------ assemble
dates = list(dict.fromkeys(d for d, *_ in ENTRIES))
date_btns = "".join(
    f"<button class='dbtn' id='db-{i}' onclick=\"showDate({i})\">{d}</button>" for i, d in enumerate(dates)
)
panes, exp_bars = [], []
for i, d in enumerate(dates):
    es = [(eid, t, st, b) for dd, eid, t, st, b in ENTRIES if dd == d]
    bar = "".join(
        f"<button class='ebtn' id='eb-{i}-{k}' onclick=\"showExp({i},{k})\">{t}"
        f"<span class='chip {'run' if st == '진행 중' else ('live' if st == '살아있음' else 'done')}'>{st}</span></button>"
        for k, (eid, t, st, b) in enumerate(es)
    )
    exp_bars.append(f"<div class='ebar' id='ebar-{i}'>{bar}</div>")
    panes.append(
        "".join(
            f"<section class='pane' id='pane-{i}-{k}'><h2>{t} <span class='chip {'run' if st == '진행 중' else ('live' if st == '살아있음' else 'done')}'>{st}</span></h2>{b}</section>"
            for k, (eid, t, st, b) in enumerate(es)
        )
    )
css = """
/* 프로페셔널·깔끔: 배경은 항상 백색 (다크 테마 비활성 — 2026-08-08 지시) */
:root,:root[data-theme="dark"],:root[data-theme="light"]{--bg:#ffffff;--ink:#1a1a1a;--muted:#5f6b7a;--line:#e2e2e2;--acc:#3730a3;--accink:#eef2ff;--panel:#fff}
body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px 16px;color:var(--ink);background:var(--bg);line-height:1.65}
h1{font-size:1.45em;margin:0 0 4px}h2{font-size:1.2em;margin:0.2em 0 0.6em}
.sub{color:var(--muted);font-size:.92em;margin-bottom:14px}
.dbar,.ebar{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.dbtn,.ebtn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:9px;padding:7px 13px;font-size:.92em;cursor:pointer}
.dbtn.on{background:var(--acc);border-color:var(--acc);color:#fff}
.ebtn.on{background:var(--accink);border-color:var(--acc);color:var(--ink)}
.ebar{display:none;border-top:1px solid var(--line);padding-top:8px}.ebar.on{display:flex}
.pane{display:none;padding:14px 0}.pane.on{display:block}
img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:8px 0}
table{border-collapse:collapse;font-size:.9em;margin:10px 0;max-width:100%}
td,th{border:1px solid var(--line);padding:5px 10px;text-align:left;vertical-align:top}
th{background:var(--accink)}.spec th{width:110px}.num td:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}
.chip{font-size:.72em;border-radius:99px;padding:2px 8px;margin-left:6px;vertical-align:middle}
.chip.done{background:#dcfce7;color:#15803d}.chip.run{background:#fef9c3;color:#a16207}
.tl{position:relative;margin:14px 0 14px 8px;border-left:3px solid var(--acc);padding-left:22px}
.node{position:relative;margin-bottom:20px}
.node:before{content:'';position:absolute;left:-31px;top:6px;width:13px;height:13px;border-radius:50%;background:var(--acc)}
.when{font-size:.82em;color:var(--muted);margin-bottom:2px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.card h3{margin:0 0 6px;font-size:1.02em}.card p{margin:0;font-size:.95em}
.next{margin-top:8px;font-size:.88em;color:var(--acc);font-weight:600}
.now{border:2px solid var(--acc);border-radius:12px;padding:14px 18px;margin-top:10px;background:var(--accink)}
.chip.live{background:#e0e7ff;color:#3730a3}
.good{color:#15803d}.bad{color:#b91c1c}.missing{background:#fef9c3;color:#713f12;padding:6px 10px;border-radius:6px}
code{background:var(--accink);padding:1px 5px;border-radius:4px}
h1,h2,h3{font-family:Georgia,'Times New Roman',serif;letter-spacing:-.01em}
.w6 th{width:64px;font-weight:600}.w6{border-left:3px solid var(--acc)}
td.pending{color:var(--muted);font-style:italic}
.xrefs{margin-top:14px;border-top:1px dashed var(--line);padding-top:8px}
.xref{display:inline-block;border:1px solid var(--acc);color:var(--acc);border-radius:99px;padding:2px 11px;margin:2px 4px 2px 0;font-size:.85em;cursor:pointer}
.xref:hover{background:var(--accink)}
"""
js = f"""
const ND={[len([1 for dd, *_ in ENTRIES if dd == d]) for d in dates]};
function showDate(i){{
  document.querySelectorAll('.dbtn').forEach(e=>e.classList.remove('on'));
  document.querySelectorAll('.ebar').forEach(e=>e.classList.remove('on'));
  document.getElementById('db-'+i).classList.add('on');
  document.getElementById('ebar-'+i).classList.add('on');
  showExp(i,0);
}}
function showExp(i,k){{
  document.querySelectorAll('.pane').forEach(e=>e.classList.remove('on'));
  document.querySelectorAll('.ebtn').forEach(e=>e.classList.remove('on'));
  document.getElementById('pane-'+i+'-'+k).classList.add('on');
  document.getElementById('eb-'+i+'-'+k).classList.add('on');
}}
window.addEventListener('load',()=>showDate(0));
const EIDMAP={{{",".join(f'"{eid}":[{i},{k}]' for i, d in enumerate(dates) for k, eid in enumerate([e[1] for e in ENTRIES if e[0] == d]))}}};
document.addEventListener('click',e=>{{
  const x=e.target.closest('.xref'); if(!x) return;
  const m=EIDMAP[x.dataset.eid]; if(m){{showDate(m[0]);showExp(m[0],m[1]);window.scrollTo(0,0);}}
}});
"""
content = (
    f"<h1>RLT critic — 실험 마스터 리포트</h1>"
    f"<div class='sub'>날짜를 고르고, 그 날의 실험을 고르세요. 각 실험 = 명세 → 질문 → 그림 → 산문 해석. "
    f"정량 표는 생성 시점에 원시 JSON에서 재계산됩니다 (<code>slurm/make_master_report.py</code>).</div>"
    f"<div class='dbar'>{date_btns}</div>{''.join(exp_bars)}{''.join(panes)}<script>{js}</script>"
)
(C / "master_report.html").write_text(
    f"<!doctype html><meta charset='utf-8'><title>RLT critic master report</title><style>{css}</style>{content}"
)
(C / "master_report_artifact.html").write_text(
    f"<title>RLT critic — 실험 마스터 리포트</title><style>{css}</style>{content}"
)

# ------------------------------------------------------------------ timeline page
tl_items = "".join(
    f"<div class='node'><div class='when'>{d}</div><div class='card'><h3>{t}</h3><p>{body}</p>"
    + (f"<div class='next'>{nxt}</div>" if nxt else "")
    + "</div></div>"
    for d, t, body, nxt in FLOW
)
tl_css = (
    css
    + """
.tl{position:relative;margin:20px 0 20px 8px;border-left:3px solid var(--acc);padding-left:22px}
.node{position:relative;margin-bottom:22px}
.node:before{content:'';position:absolute;left:-31px;top:6px;width:13px;height:13px;border-radius:50%;background:var(--acc)}
.when{font-size:.82em;color:var(--muted);font-variant-numeric:tabular-nums;margin-bottom:2px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.card h3{margin:0 0 6px;font-size:1.02em}.card p{margin:0;font-size:.95em}
.next{margin-top:8px;font-size:.88em;color:var(--acc);font-weight:600}
.now{border:2px solid var(--acc);border-radius:12px;padding:14px 18px;margin-top:10px;background:var(--accink)}
"""
)

print(
    f"wrote master_report(.html/_artifact.html): {(C / 'master_report_artifact.html').stat().st_size // 1024} KB, "
    f"{len(dates)} dates / {len(ENTRIES)} experiments"
)
