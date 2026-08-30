"""Hub entry for the YAM dataset non-Markovianity measurement (eid nonmarkov-yam-meas).

Numbers recomputed from .scratch/nonmarkov_yam/results.json (job 36586); figure regenerated
by scripts/fig_nonmarkov_yam.py before publishing.
"""

import json
import pathlib
import subprocess

R = pathlib.Path(__file__).resolve().parents[1]
res = json.loads((R / ".scratch/nonmarkov_yam/results.json").read_text())
arms = res["arms"]
m0 = arms["0"]["val_mse"]
ks = sorted(int(k) for k in arms if k != "0")

stamp = subprocess.run(
    ["git", "-C", str(R), "log", "-1", "--format=fql-one-step-actor@%h"], capture_output=True, text=True, check=False
).stdout.strip()
if subprocess.run(
    ["git", "-C", str(R), "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False
).stdout.strip():
    stamp += "+dirty"

rows = f"<tr><td>0 (Markov)</td><td>{m0:.4f}</td><td>—</td><td>{arms['0']['val_mse_success']:.4f} / {arms['0']['val_mse_failure']:.4f}</td></tr>"
for k in ks:
    a = arms[str(k)]
    rows += (
        f"<tr><td>{k} ({k / 30:g}s)</td><td>{a['val_mse']:.4f}</td><td>−{100 * a['delta_rel']:.1f}%</td>"
        f"<td>{a['val_mse_success']:.4f} / {a['val_mse_failure']:.4f}</td></tr>"
    )
best = max(ks, key=lambda k: arms[str(k)]["delta_rel"])
best_pct = 100 * arms[str(best)]["delta_rel"]

head = (
    f"<table class='num'><tr><th>항목</th><th>내용</th></tr>"
    f"<tr><th>who</th><td>워커B(ACRFT-D) — 사용자 질문('YAM에도 non-Markov 측정 가능?')의 즉답 측정</td></tr>"
    f"<tr><th>where</th><td>pc_cache/yam_s347 (동결 DINOv2 pooled + proprio pos-14) · held-out 52ep(실패 7 층화) · L40S job 36586</td></tr>"
    f"<tr><th>what</th><td>ΔL_val 프로토콜: 용량-일치 history vs Markov 액션 예측 — 0.5초 창이 held-out MSE −{best_pct:.1f}%</td></tr>"
    f"<tr><th>how</th><td>measure_nonmarkov_yam.py (MLP 3×1024, 6프레임 입력 양팔 동일 차원), 표·그림 results.json 재계산</td></tr>"
    f"<tr><th>why</th><td>데이터셋의 non-Markov 함량 실증 — 크리틱측 belief-누출 이론의 정책측 쌍둥이 측정</td></tr>"
    f"<tr><th>코드</th><td><code>{stamp}</code></td></tr></table>"
)

KO = f"""{head}
<p><b>왜.</b> mimicgen/OGBench에서 하던 "history 있/없 액션 예측 격차" 측정을 YAM teleop에 적용할 수 있느냐는
사용자 질문. 이 격차(ΔL_val, Lazzati/Metelli의 정책측 척도)는 우리
<span class='xref' data-eid='rcv-value-of-information'>크리틱측 Q_reg−Q_syn</span>의 쌍둥이라, 두 지도의 대조가
belief-누출 이론의 상호검증이 된다.</p>

<p><b>어떻게.</b> π0.5 재학습 없이 patch-critic의 동결 DINOv2 캐시를 재사용: 평균풀 feature(384)+proprio(14)를
6프레임 블록으로 묶어, <b>용량 완전 일치</b> MLP 두 팔 — Markov(현재 프레임 6복제) vs history([t−k,t] 균등
6프레임) — 로 현재 액션(joint-delta z-score)을 회귀. held-out은 에피소드 단위 52개(성공/실패 층화).
아키텍처·입력차원이 동일하므로 격차는 전부 과거 프레임 속 정보다.</p>

<table class='num'><tr><th>창 k (초)</th><th>val MSE</th><th>Markov 대비</th><th>성공/실패</th></tr>{rows}</table>

<p><b>판정.</b> ① <b>YAM teleop의 non-Markov 함량은 실재하고 크다</b> — 0.5초 창만으로 held-out 예측 오차
−{best_pct:.1f}%. Markov 정책(현재 관측만 보는 π0.5)은 이 만큼의 행동 신호를 원리적으로 표현 못 한다는 뜻이고,
Park의 "Markovianized policy" 기제(<span class='xref' data-eid='jvp-offdiag'>실기 계보</span>의 이론적 배경)가
우리 데이터에서 정량 확인된 것. ② <b>정보는 근과거 0.5초에 집중</b> — 고정 6프레임으로 창을 넓히면(근과거
샘플링이 성겨지며) 단조 희석되어 5초 창에선 −4%만 남는다: teleop의 속도·관성 연속성 같은 짧은 시간 스케일
신호가 주성분. ③ 성공/실패 격차는 이 해상도에선 미미(실패가 일관되게 약간 높은 MSE).</p>

<p><b>귀결.</b> (a) <b>delayed-obs/history 크리틱 설계 입력</b>: 가치 있는 창은 ~15스텝 조밀 샘플 — 길게 줄
필요 없다. (b) per-frame Δ 지도를 저장해 두었다(.scratch/nonmarkov_yam/perframe_*.npy) — reactive-map
(Q_reg−Q_syn)·<span class='xref' data-eid='iql-pair-yam'>k* 지도</span>와의 상관 검정이 다음 단계.
(c) OGBench(상태기반)에서 같은 측정을 하면 <span class='xref' data-eid='rcv-honest-critic-recipe'>honest-critic
recipe</span>의 P1 기각조건 사전 점검이 된다 — 워커C에 릴레이됨.</p>

<p><b>한계.</b> k 스윕이 창길이×샘플밀도 confound(고정 6프레임): "0.5초 집중"은 확실하나 "원거리 무용"은 이
설계 안에서의 결론. 평균풀 feature라 절대 Δ는 하한. 액션 1-스텝 타깃 기준 — 청크 스케일 함량은 별도 측정
필요. Park caveat(history 정책의 배포 실패)은 무관 — 이 예측기들은 rollout하지 않는 데이터셋 성질 측정이다.</p>
<figure><img src='figures/nonmarkov-yam-meas/fig_nonmarkov_yam.png' alt='history window vs held-out improvement'>
<figcaption>held-out 액션 예측 개선율 vs history 창 길이. 0.5초에서 최대 +{best_pct:.1f}%, 창이 넓어질수록 희석.</figcaption></figure>"""

EN = f"""{head}
<p><b>Why.</b> The user asked whether the history-vs-no-history action-prediction gap measured on
mimicgen/OGBench can be measured on YAM teleop. That gap (ΔL_val, Lazzati/Metelli's policy-side metric) is the
twin of our critic-side <span class='xref' data-eid='rcv-value-of-information'>Q_reg−Q_syn</span>, so contrasting
the two maps cross-validates the belief-leakage theory.</p>

<p><b>How.</b> No pi0.5 retraining: reuse the patch-critic's frozen DINOv2 cache. Mean-pooled features (384) +
proprio (14) in 6-frame blocks feed two <b>capacity-matched</b> MLP arms — Markov (current frame repeated 6x) vs
history (6 frames evenly spaced over [t−k, t]) — regressing the current action (z-scored joint delta). Held-out =
52 episodes (outcome-stratified). Identical architecture and input dimension, so any gap is information in the
past frames.</p>

<table class='num'><tr><th>window k (s)</th><th>val MSE</th><th>vs Markov</th><th>success/failure</th></tr>{rows}</table>

<p><b>Verdict.</b> (1) <b>YAM teleop's non-Markovian content is real and large</b> — a 0.5s window alone cuts
held-out prediction error by {best_pct:.1f}%. A Markov policy (current-observation-only pi0.5) structurally cannot
express that share of the action signal — Park's "Markovianized policy" mechanism quantified on our own data.
(2) <b>The information concentrates in the recent 0.5s</b> — with 6 fixed frames, widening the window dilutes it
monotonically down to −4% at 5s: short-timescale teleop smoothness/momentum dominates. (3) Success/failure gaps
are minor at this resolution.</p>

<p><b>Consequences.</b> (a) Design input for the delayed-obs/history critic: the valuable window is ~15 steps,
densely sampled. (b) Per-frame Δ maps are saved for correlation against the reactive-map (Q_reg−Q_syn) and the
<span class='xref' data-eid='iql-pair-yam'>k* map</span>. (c) The same measurement on OGBench pre-checks the
<span class='xref' data-eid='rcv-honest-critic-recipe'>honest-critic recipe</span>'s P1 rejection condition —
relayed to worker C.</p>

<p><b>Limitations.</b> The k sweep confounds window length with sampling density (6 fixed frames): "0.5s
concentration" is firm, "far past useless" holds only within this design. Mean-pooled features make the absolute
Δ a lower bound. Single-step action target — chunk-scale content needs its own run. Park's caveat (history
policies failing at rollout) does not apply: these predictors are never rolled out.</p>
<figure><img src='figures/nonmarkov-yam-meas/fig_nonmarkov_yam.png' alt='history window vs held-out improvement'>
<figcaption>Held-out action-prediction improvement vs history window. Peak +{best_pct:.1f}% at 0.5s, diluting with width.</figcaption></figure>"""

entry = {
    "eid": "nonmarkov-yam-meas",
    "date": "2026-08-27 13:20",
    "worker": "B",
    "title": f"🧪 [워커B] YAM 데이터셋의 non-Markov 함량 실측 — 0.5초 history가 액션 예측 오차 −{best_pct:.0f}% (ΔL_val)",
    "summary": f"용량-일치 history vs Markov 액션 예측(동결 DINO 캐시, held-out 52ep): 0.5초 창 −{best_pct:.1f}%, 창 넓히면 단조 희석(5초 −4%). YAM teleop의 non-Markov 함량 실재·근과거 집중 확정 — 크리틱측 belief-누출의 정책측 쌍둥이 측정, per-frame 지도는 reactive-map/k* 상관 검정용으로 저장.",
    "tags": ["워커B", "non-markov", "dataset", "측정"],
    "status": "done",
    "phase": "진단·방법",
    "links": ["rcv-value-of-information", "rcv-honest-critic-recipe", "iql-pair-yam", "nonmarkov-longer"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/nonmarkov_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out, "stamp", stamp)
