"""Build the project's master experiment report — every experiment, grouped by DATE, then by
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
워커A의 독립 스택(HILP φ + Cal-QL+swap, McNemar) 판정과 서로 재현 관계다 — full-authority 해악(p=.004), BoN 무익.</p>
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
③ +7.3%는 존재 증명이지 지배 신호는 아니다 — 그래서 다음 게이트는 %오차가 아니라 <b>합성 Q의 후보 구별력</b>을 직접 잰다:
held-out 프레임에서 y=V(f(φ,a))를 16후보+데모에 계산해 (a) 데모>후보 랭킹(action-sensitivity), (b) 후보 간 밴드 개방을
측정. f(φ)·V(φ) 모두 기학습분 재사용 — 새 학습 없이 조립만. 통과 시에만 페어드 롤아웃(mb 모드)으로 간다.</p>
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
<p>φ→이미지 디코더(kroll 6만 프레임, 20k스텝)와 raw 디코더를 같은 조건에서 학습해 held-out 재구성을 비교
(<a href="videos/decoder/22_decoder_phi128_recon.png" target="_blank">φ 패널</a> ·
<a href="videos/decoder/22_decoder_raw_recon.png" target="_blank">raw 패널</a> ·
<a href="videos/decoder/22_decoder_phi128_walk.mp4" target="_blank">φ 임베딩 워크</a> ·
<a href="videos/decoder/22_decoder_phi128_ride.mp4" target="_blank">교차-에피소드 ride</a>).
<b>결과: φ는 주방 외형(색·배치)은 예상보다 많이 유지하면서, 로봇 팔·그리퍼 자세를 고스트처럼 뭉갠다</b> —
raw 재구성은 팔이 선명하다. 정량 대응: proprio 차원별 R² 평균 raw .760 → PCA .653 → φ .546.
즉 φ가 버린 것은 외형이 아니라 <b>단거리 행동-관련 정보</b> — "거리만 남기면 디테일이 사라지는 것 아닌가"라는
우려가 정확했고, φ-critic이 BoN을 못 여는 이유의 기전적 설명이다.</p>
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
<tr><td>제약 없는 actor-critic</td><td>없음</td><td>위험단</td><td>워커A full-authority 파국(.300, p=.004)이 근방 증거</td></tr></table>
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
McNemar +30/−35 p=0.620 — <b>null</b>. 시드 3/4이 0 이상이나 s5300 −0.20이 끌어내림. SNR 프레임 예측과 정합:
히스토리는 추정 품질(ρ)을 겨냥하지만 기준 샘플링의 σ_signal이 그대로면 이득 원천이 없다 — 다양화(v17)와의
결합이 다음 단계. td_max_hist5는 학습 중(21k).</p>
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
0→0.524 통과)으로 밤샘 롤아웃 판정을 했다: <b>BoN 무익(.700 동률), full-authority critic은 파국(.300, McNemar p=.004)</b>,
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
둘(전권 파국 p=.004 · 우리 calql_v14 과잉억압 p=.001)을 동시에 회피"하도록 설계 — 비관은 in-sample V+min 앙상블에만,
OOD 질의는 구조적으로 0(MAC 계보). 우리 주석의 프레임당 16후보를 'frozen policy 아카이브'로 재사용한다.
<b>서로의 리포트가 서로의 실험 설계에 인용되는 루프가 성립.</b> ③ 그들의 σ-veto는 dynamics-불일치 기반(러닝 미디언×τ),
우리 v17b는 Q-앙상블-불일치 기반 — v17b 확정 시 veto 신호원 비교가 자연스러운 공동 후속. ④ 운영 메모: 워커A는
fix/probe-eval-jit 브랜치에 커밋 중 — 머지 시 조율 필요.</p>
<h3>⑤ 운영 관행</h3>
<p>좀비 잡(squeue R인데 실제 사망)의 로그 mtime 감시 — 우리도 행 걸린 평가 2건을 겪었으므로 감시 루틴에 채택.
체크포인트 자동 아카이브(HF 업로드 검증 후 로컬 삭제)도 디스크 사고 예방책으로 참고.</p>
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
    "papers-value-steering": {
        "date": "2026-08-08 13:30",
        "who": "워커B(리뷰)",
        "where": "arXiv",
        "what": "Robo-ValueRL(arXiv:2607.09866) 정독 + 인접 3편(V-GPS·Q-VGM·프로빙) 리뷰",
        "how": "원문 정독 후 FINAL 결론·우리 스택과 대조",
        "why": "'robo-value-RL' 탐색 요청 — FINAL null의 해석과 다음 수(그래디언트 조향)의 문헌 근거",
        "links": ["final", "calql", "wcurse"],
    },
    "xworker-0808": {
        "date": "2026-08-08 14:10",
        "who": "워커B(리뷰) ← 워커A 리포트 7건",
        "where": "공유 허브",
        "what": "교차 워커 배움 — 상호 재현·McNemar 도입·표현/데이터 공격로 비교",
        "how": "워커A 전 리포트 정독 후 즉시 반영",
        "why": "워커끼리 서로 배우라는 지시 — 독립 스택의 결론 상호 검증",
        "links": ["final", "kper", "papers-value-steering"],
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
