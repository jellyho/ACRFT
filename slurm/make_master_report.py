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
from collections import Counter

import numpy as np
from PIL import Image

C = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
TCRIT = {2: 12.7, 3: 4.30, 4: 3.18, 8: 2.36, 10: 2.26, 15: 2.14, 16: 2.13}


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
        return f"<tr><td>{name}</td><td colspan=4>n={len(ds)} — 수집 중</td></tr>"
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
{img(P / "2_value_bias.png", "value bias by distance")}
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
    f"""
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
오른쪽 패널: 같은 규칙·같은 구풀에서 체크포인트 세대만 바꾸면 −0.067→+0.100 — 풀·세대 교란의 위력이자
공정 비교가 필요했던 이유.</p>
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
                ("학습", "v11과 동일 레시피, method=iql/aqc — a6000(데이터 17GB 상주)"),
                ("평가", "v11 동일 프로토콜, 시드 16개 목표 (수집 중)"),
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
최종 심판은 성공률 CI: 0 위로 뜨면 목표 달성, 0을 포개면 "구분하되 잘못 구분" → margin 순위 손실이 다음 수.</p>
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
        f"<span class='chip {'run' if st == '진행 중' else 'done'}'>{st}</span></button>"
        for k, (eid, t, st, b) in enumerate(es)
    )
    exp_bars.append(f"<div class='ebar' id='ebar-{i}'>{bar}</div>")
    panes.append(
        "".join(
            f"<section class='pane' id='pane-{i}-{k}'><h2>{t} <span class='chip {'run' if st == '진행 중' else 'done'}'>{st}</span></h2>{b}</section>"
            for k, (eid, t, st, b) in enumerate(es)
        )
    )
css = """
:root{--bg:#fafaf9;--ink:#1c1917;--muted:#64748b;--line:#e7e5e4;--acc:#3730a3;--accink:#eef2ff;--panel:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#131316;--ink:#e7e5e4;--muted:#94a3b8;--line:#33333a;--acc:#818cf8;--accink:#1e1b4b;--panel:#1b1b1f}}
:root[data-theme="dark"]{--bg:#131316;--ink:#e7e5e4;--muted:#94a3b8;--line:#33333a;--acc:#818cf8;--accink:#1e1b4b;--panel:#1b1b1f}
:root[data-theme="light"]{--bg:#fafaf9;--ink:#1c1917;--muted:#64748b;--line:#e7e5e4;--acc:#3730a3;--accink:#eef2ff;--panel:#fff}
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
:root[data-theme="dark"] .chip.done{background:#14532d;color:#bbf7d0}
:root[data-theme="dark"] .chip.run{background:#713f12;color:#fef08a}
.good{color:#15803d}.bad{color:#b91c1c}.missing{background:#fef9c3;color:#713f12;padding:6px 10px;border-radius:6px}
code{background:var(--accink);padding:1px 5px;border-radius:4px}
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
window.addEventListener('load',()=>showDate({len(dates) - 1}));
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
print(
    f"wrote master_report(.html/_artifact.html): {(C / 'master_report_artifact.html').stat().st_size // 1024} KB, "
    f"{len(dates)} dates / {len(ENTRIES)} experiments"
)
