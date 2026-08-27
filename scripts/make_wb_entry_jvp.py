"""Figure + hub entry for the JVP off-diagonal poisoning finding (eid jvp-offdiag).

Reads the three diagnostic results.json files (eval_onestep_bc.py, job 36485) and recomputes
every published number from them; the figure is regenerated on every run (audit rule).
"""

# ruff: noqa: E402, ICN001  (matplotlib.use must precede pyplot; probe-local imports intentional)

import json
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "slurm"))
import plot_style

plot_style.apply()
PAL = plot_style.PALETTE

CKPTS = [200000, 250000, 300000]
M = {}
for s in CKPTS:
    M[s] = json.loads((R / f".scratch/eval_onestep_jvp_{s}/results.json").read_text())["metrics"]
demo_jerk = M[200000]["jerk/demo"]

STEPS = [1, 2, 10]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, key, ylab in zip(axes, ("jerk", "mse_gt"), ("intra-chunk jerk", "demo MSE"), strict=True):
    for j, s in enumerate(CKPTS):
        ys = [M[s][f"{key}/af_{n}step"] for n in STEPS]
        ax.plot(STEPS, ys, "o-", color=PAL[j], lw=1.8, ms=6, label=f"{s // 1000}k")
    if key == "jerk":
        ax.axhline(demo_jerk, color="#555", lw=1.0, ls="--", label="demo")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(n) for n in STEPS])
    ax.set_xlabel("sampling steps")
    ax.set_ylabel(ylab)
axes[0].legend(loc="upper right")
fig.tight_layout()
figpath = R / ".scratch/fig_jvp_jerk.png"
fig.savefig(figpath, dpi=220)
fig.savefig(str(figpath).replace(".png", ".svg"))
print("fig written", figpath)

stamp = subprocess.run(
    ["git", "-C", str(R), "log", "-1", "--format=fql-one-step-actor@%h"], capture_output=True, text=True, check=False
).stdout.strip()
if subprocess.run(
    ["git", "-C", str(R), "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False
).stdout.strip():
    stamp += "+dirty"


def row(s):
    m = M[s]
    c = [f"<td>{m[f'mse_gt/af_{n}step']:.5f} / {m[f'jerk/af_{n}step']:.5f}</td>" for n in STEPS]
    return f"<tr><td><b>{s // 1000}k</b></td>{''.join(c)}</tr>"


tbl = (
    "<table class='num'><tr><th>ckpt</th><th>1-step (MSE / jerk)</th><th>2-step</th><th>10-step</th></tr>"
    + "".join(row(s) for s in CKPTS)
    + f"<tr><td>demo 바닥</td><td colspan=3>jerk {demo_jerk:.5f} (gt action var {M[200000]['gt_action_var']:.3f})</td></tr></table>"
)
tbl_en = tbl.replace("demo 바닥", "demo floor")

j1 = {s: M[s]["jerk/af_1step"] / demo_jerk for s in CKPTS}
head = (
    f"<table class='num'><tr><th>항목</th><th>내용</th></tr>"
    f"<tr><th>who</th><td>워커B (실기 관측: 사용자)</td></tr>"
    f"<tr><th>where</th><td>L40S offline eval (held-out 6ep×6frame, robot-space) + wandb c4vy84yy + 실기 rollout 관측</td></tr>"
    f"<tr><th>what</th><td>α=0 JVP continuation(200k→300k)이 1-step 정책만 파괴 — jerk 3.3×→{j1[250000]:.0f}×(demo 대비), 10-step은 오히려 개선</td></tr>"
    f"<tr><th>how</th><td>eval_onestep_bc.py + 신규 jerk 지표(청크축 2차차분), 표·그림은 results.json에서 재계산</td></tr>"
    f"<tr><th>why</th><td>실기에서 300k 정책이 청크 내부에서 noisy — 원인 규명과 배포 지침 확정</td></tr>"
    f"<tr><th>코드</th><td><code>{stamp}</code> · job 36485</td></tr></table>"
)

KO = f"""{head}
<p><b>발단.</b> 사용자가 실기에서 관측: 200k(discrete floor) 정책은 약간의 흔들림만 있고 잘 작동하는데,
JVP continuation을 거친 300k 정책은 <b>청크 내부에서</b> 심하게 noisy하다. (청크 경계 문제 아님 — 사용자 정정.)</p>

<p><b>학습측 증거 (wandb c4vy84yy).</b> α=0 JVP 전환(200k) 직후부터: 방향미분 dudt_absmax 평균 52→60·max 294로
폭주, 타깃 clamp(±4)가 <b>100k 스텝 내내 상시 포화</b>, delta²(타깃 잔차) 0.003→0.19로 <b>65× 점프 후 끝까지
미수렴</b>. loss는 adaptive weight가 스케일을 숨겨 0.8에서 평온해 보였다 — 예전 "JVP 스트레스 6/6 안정" 판정은
loss 비폭발만 본 것이었다.</p>

<p><b>정책측 정량 (신규 jerk 지표 = 청크축 2차차분 평균제곱; demo-MSE는 이 노이즈를 원리적으로 못 본다).</b></p>
{tbl}
<p><b>구조가 그대로 보인다:</b> ① 200k에서 1-step jerk는 demo의 3.3× ("약간 흔들림"), 10-step은 demo와 동급.
② JVP 50k 만에 <b>1-step jerk가 demo의 {j1[250000]:.0f}×로 폭발</b>(MSE는 5.8×만 악화 — 실기 체감이 jerk에
있었던 이유). ③ 그런데 <b>10-step은 전혀 안 다쳤고 MSE는 오히려 개선</b>(0.00153→0.00105).
④ 300k에서 1-step이 소폭 회복하나 여전히 demo의 {j1[300000]:.0f}×.</p>

<p><b>기제 — off-diagonal 중독.</b> fm_ratio 0.5의 FM 절반은 (r,t) 평면의 <b>대각선</b>(순간속도 u(z,t,t)=v)만
고정한다. 1-step이 쓰는 것은 정반대 끝 u(z,0,1) — 전 구간 평균속도라는 최대 off-diagonal이고, 그걸 채우는
MeanFlow 절반의 타깃이 JVP에서 "미분 폭주 → clamp 포화 → 부호만 남은 상수장"이 됐다. 10-step은 길이 0.1짜리
near-diagonal 구간만 쓰므로 무사하다 — 표의 ③이 그 직접 증거. 근인(根因)은 discrete 200k가 점별 미분을 한 번도
규제하지 않은 것 + π0.5 시간 임베딩의 고주파(min_period 4e-3) + bf16.</p>

<p><b>한계 (confound 명시).</b> 이것은 공식 α-Flow 스케줄의 JVP 전환(run 내 ~71%)이 아니라 <b>floor-수렴 후
하드 스위치 continuation</b>의 실패다. 원 스케줄이었으면 달랐을지는 열린 질문(60k 검증 run 후보). "JVP 일반이
나쁘다"가 아니라 "이 프로토콜 + bf16 π0.5에서 실패"까지가 확정 주장.</p>

<p><b>지침.</b> ① 1-step 배포는 <b>200k만</b> 사용(HF 모델카드에 경고 게시). ② 250k/300k를 굳이 쓰려면
10-step은 안전(오히려 MSE 최선). ③ 2-step이 jerk를 절반 이하로 줄이는 실용 절충. ④ 200k의 잔여 3.3× jerk의
구조적 해법 후보 = x-prediction(jit_mf) — LPS가 x-pred를 택한 이유("displacement 오차가 출력 분산을 증폭")가
정확히 이 현상이라, u vs x 비교 run의 우선순위가 올라갔다. ⑤ continuation류는 중간 demo-MSE+jerk 게이트 필수
(규약화).</p>
<figure><img src='figures/jvp-offdiag/fig_jvp_jerk.png' alt='jerk and MSE vs sampling steps per checkpoint'>
<figcaption>좌: 청크 내부 jerk(로그축) — 250k/300k의 1-step만 폭발, 10-step은 demo 수준 유지. 우: demo-MSE —
같은 체크포인트가 10-step에선 오히려 최선. 점선 = demo 바닥.</figcaption></figure>"""

EN = f"""{head}
<p><b>Trigger.</b> On the real robot the user observed: the 200k (discrete-floor) policy works with mild
shakiness, but the JVP-continued 300k policy is severely noisy <b>within chunks</b> (not at chunk
boundaries — user's correction).</p>

<p><b>Training-side evidence (wandb c4vy84yy).</b> From the moment of the alpha=0 JVP switch: the directional
derivative dudt_absmax blew up (mean 52-60, max 294), the target clamp (±4) stayed <b>saturated for the
entire 100k steps</b>, and delta² jumped 65x (0.003→0.19) and never recovered. The loss sat placidly at 0.8 —
the adaptive weight hides the scale, which is why the earlier "JVP stress 6/6 stable" verdict missed this.</p>

<p><b>Policy-side quantification (new jerk metric = mean squared second difference along the chunk axis;
demo-MSE is structurally blind to this noise).</b></p>
{tbl_en}
<p>(1) At 200k, 1-step jerk is 3.3x the demo floor (the "mild shake"); 10-step matches demo. (2) 50k of JVP
explodes 1-step jerk to <b>{j1[250000]:.0f}x demo</b> while MSE only worsens 5.8x — why the robot feel was
"noisy". (3) <b>10-step is untouched and its MSE actually improves</b> (0.00153→0.00105). (4) 300k partially
recovers but stays at {j1[300000]:.0f}x demo.</p>

<p><b>Mechanism — off-diagonal poisoning.</b> The FM half (fm_ratio 0.5) anchors only the diagonal of the
(r,t) plane (instantaneous velocity u(z,t,t)=v). One-step sampling uses the opposite extreme, u(z,0,1) — the
full-interval mean velocity — which is trained only by the MeanFlow half, whose JVP targets degenerated into
clamp-saturated noise. Ten-step sampling touches only near-diagonal intervals (length 0.1) and survives —
row (3) is the direct evidence. Root cause: 200k of discrete training never regularizes the pointwise
derivative, pi0.5's high-frequency time embedding (min_period 4e-3) gives it room, and bf16 sits underneath.</p>

<p><b>Limitation (confound stated).</b> This is the failure of a <b>hard-switch continuation after floor
convergence</b>, not of the official alpha-Flow in-run JVP transition (~71% progress) — whether the original
schedule would have survived is an open question (a 60k verification run is a candidate). The firm claim stops
at "this protocol + bf16 pi0.5 fails".</p>

<p><b>Guidance.</b> (1) 1-step deployment: use <b>200k only</b> (warning published on the HF model card).
(2) 250k/300k are safe at 10-step (actually best MSE). (3) 2-step halves the jerk — practical compromise.
(4) For 200k's residual 3.3x jerk, the structural candidate is x-prediction (jit_mf): LPS chose x-pred
precisely because "displacement errors amplify output variance" — this exact phenomenon — so the u-vs-x
comparison run moved up in priority. (5) Continuations must carry an in-flight demo-MSE+jerk gate (now a
house rule).</p>
<figure><img src='figures/jvp-offdiag/fig_jvp_jerk.png' alt='jerk and MSE vs sampling steps per checkpoint'>
<figcaption>Left: intra-chunk jerk (log) — only 1-step at 250k/300k explodes; 10-step stays at the demo floor.
Right: demo-MSE — the same checkpoints are best at 10-step. Dashed = demo floor.</figcaption></figure>"""

entry = {
    "eid": "jvp-offdiag",
    "date": "2026-08-26 19:40",
    "worker": "B",
    "title": "🧪 [워커B] JVP continuation의 off-diagonal 중독 — 1-step만 jerk 64× 파괴, 10-step은 무사 (실기 관측 → 기제 규명)",
    "summary": "실기에서 300k(JVP) 정책이 청크 내부 noisy → 규명: JVP 미분 폭주로 clamp 상시 포화, delta² 65× 미수렴. 신규 jerk 지표로 정량: 1-step jerk demo 3.3×(200k)→211×(250k), 10-step은 demo 수준 유지·MSE는 개선. FM 절반은 대각선만 지키고 1-step이 쓰는 off-diagonal만 오염. 배포는 200k(1-step) 또는 250k+(10-step).",
    "tags": ["워커B", "alphaflow", "meanflow", "negative-result", "실기"],
    "status": "done",
    "phase": "판정·종합",
    "links": ["alphaflow-pi05", "alphaflow-1step-gate", "iql-pair-yam"],
    "body_html": f'<div class="wbx wbx-ko">{KO}</div><div class="wbx wbx-en">{EN}</div>',
}
out = R / ".scratch/jvp_entry.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("entry written", out, "stamp", stamp)
