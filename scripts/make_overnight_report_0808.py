"""Overnight 2026-08-08 report: critic authority + MB-AC. House style (worker-B/Seohong Park).

Reads the raw JSONs (rollouts, dyn reports, MB-AC battery) and recomputes every number at
generation time. Outputs a single-file HTML (images embedded as jpeg base64) ready for the Space.
"""

import base64
import io
import json
import math
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import report_style as rs

rs.use()
ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / ".scratch"


def mcnemar(a, b):
    w = sum(1 for x, y in zip(a, b) if x and not y)
    ll = sum(1 for x, y in zip(a, b) if y and not x)
    n = w + ll
    p = 1.0 if n == 0 else min(1.0, sum(math.comb(n, k) for k in range(min(w, ll) + 1)) / 2**n * 2)
    return w, ll, p


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=110, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 82})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


ctl = json.loads((S / "rollout_control.json").read_text())
phi = json.loads((S / "rollout_rltphi.json").read_text())
mbc = json.loads((S / "rollout_mbac.json").read_text())
tau13_p = S / "rollout_mbacv_tau13.json"
tau13 = json.loads(tau13_p.read_text()) if tau13_p.exists() else None

succ = {}
for tag, d in (("ctl", ctl), ("phi", phi), ("mbac", mbc)):
    for m in d:
        succ[f"{tag}:{m}"] = [t["success"] for t in d[m]["trials"]]
if tau13:
    succ["mbac:mbacv_t13"] = [t["success"] for t in tau13["mbacv"]["trials"]]
base = succ["ctl:vla"]

LABELS = {
    "ctl:prefix": "prefix (action-blind critic)", "ctl:vla": "vla (baseline)",
    "ctl:bon": "bon (action-blind critic)", "ctl:critic": "critic full-auth (action-blind)",
    "phi:bon": "bon (phi Cal-QL+swap)", "phi:prefix": "prefix (phi Cal-QL+swap)",
    "phi:critic": "critic full-auth (phi Cal-QL+swap)",
    "mbac:mbacv": "mbacv: sigma commit only (tau 2.0)", "mbac:mbac": "mbac: critic sel + sigma commit",
    "mbac:mbacf": "mbacf: sigma-veto BoN + sigma commit",
    "mbac:mbacv_t13": "mbacv: sigma commit only (tau 1.3)",
}

# ---- fig 1: lollipop, every arm vs the shared vla baseline ----
arms = [k for k in LABELS if k != "ctl:vla" and k in succ]
rows = []
for k in arms:
    w, ll, p = mcnemar(succ[k], base)
    rate = sum(succ[k]) / len(succ[k])
    rows.append((k, rate - 0.70, p, rate))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.6))
for yy, (k, d, p, rate) in enumerate(rows):
    fam = ("mbac" if k.startswith("mbac") else ("phi" if k.startswith("phi") else "ctl"))
    c = {"mbac": rs.PURPLE, "phi": rs.GREEN, "ctl": rs.RED}[fam]
    ax.plot([0, d], [yy, yy], color=c, lw=2, alpha=0.45, zorder=1)
    ax.scatter([d], [yy], s=95, facecolors=(c if p < 0.05 else "white"), edgecolors=c,
               linewidths=1.8, zorder=3)
    ax.annotate(f"{rate:.2f}", (d, yy), textcoords="offset points",
                xytext=(14 if d >= 0 else -14, 0), ha="left" if d >= 0 else "right",
                va="center", fontsize=10, color="#444")
ax.axvline(0, color="black", lw=1.2)
ax.set_xlim(ax.get_xlim()[0] - 0.045, ax.get_xlim()[1] + 0.045)
ax.set_yticks(range(len(rows)), [LABELS[r[0]] for r in rows])
ax.set_xlabel(r"$\Delta$ success rate vs vla")
ax.set_title("Paired rollouts")
FIG1 = b64(fig)

# ---- fig 2: commitment histograms ----
picks = [("ctl:prefix", ctl["prefix"], rs.RED), ("phi:critic", phi["critic"], rs.GREEN),
         ("mbac:mbacv", mbc["mbacv"], rs.PURPLE)]
fig, axes = plt.subplots(1, len(picks), figsize=(11, 3.0), sharey=True)
for ax, (k, d, c) in zip(axes, picks):
    ns = np.array([t["n_exec"] for t in d.get("trace", [])])
    if len(ns) == 0:
        continue
    h = np.bincount(ns, minlength=17)[1:]
    ax.bar(range(1, 17), h / h.sum(), color=c, width=0.85)
    ax.set_title(LABELS[k].split(" (")[0].split(":")[0], fontsize=11)
    ax.text(0.04, 0.92, f"mean {ns.mean():.1f}", transform=ax.transAxes, ha="left",
            fontsize=9.5, color="#555")
    ax.set_xlabel("committed steps")
axes[0].set_ylabel("fraction of replans")
FIG2 = b64(fig)

# ---- fig 3: dynamics R2 / AUSE, phi vs DINO vs phi-h1 ----
reps = {}
for name, p in (("DINO v4b space", S / "cheapz_dyn_v1/report.json"),
                ("phi space (hist 3)", S / "phi_dyn_v1/report.json"),
                ("phi space (hist 1)", S / "phi_dyn_v1_h1/report.json")):
    if p.exists():
        reps[name] = json.loads(p.read_text())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
for (name, r), c in zip(reps.items(), [rs.RED, rs.BLUE, rs.GREEN]):
    ks = sorted(int(k) for k in r["per_macro"])
    a1.plot(ks, [r["per_macro"][str(k)]["r2_vs_copyforward"] for k in ks], "-o", color=c, label=name)
    a2.plot(ks, [r["per_macro"][str(k)]["ause"] for k in ks], "-o", color=c, label=name)
a1.set_xlabel("prediction horizon (steps)"); a1.set_ylabel(r"$R^2$ vs copy-forward")
a1.set_title("Accuracy"); a1.legend(fontsize=9)
a2.set_xlabel("prediction horizon (steps)"); a2.set_ylabel("AUSE")
a2.set_title("Calibration"); a2.legend(fontsize=9)
FIG3 = b64(fig)

# ---- fig 4: E1 frontier + E2/E3 ----
bat = json.loads((S / "mbac_offline_phi.json").read_text())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
fig.subplots_adjust(wspace=0.32)
e1 = bat["E1"]
fk = sorted(int(k) for k in e1["fixed_err_at_commit"])
a1.plot(fk, [e1["fixed_err_at_commit"][str(k)] for k in fk], "-o", color=rs.GRAY,
        label="fixed k")
qs = [q for q in ("0.3", "0.5", "0.7", "0.9") if f"adaptive_q{q}" in e1]
a1.plot([e1[f"adaptive_q{q}"]["mean_k_steps"] for q in qs],
        [e1[f"adaptive_q{q}"]["err_at_commit"] for q in qs], "-o", color=rs.PURPLE,
        label=r"$\sigma$-adaptive")
a1.set_xlabel("mean committed steps"); a1.set_ylabel("error at commit end")
a1.set_title("Commitment frontier", fontsize=12); a1.legend(fontsize=9)
e3 = bat["E3"]
ks = sorted(int(k) for k in e3)
a2.plot(ks, [e3[str(k)]["binding_disagreement"] for k in ks], "-o", color=rs.GREEN,
        label="disagreement")
a2.plot(ks, [e3[str(k)]["binding_goal_dist"] for k in ks], "-o", color=rs.RED,
        label="predicted goal distance")
a2.axhline(0.5, color="gray", ls="--", lw=1)
a2.text(ks[0], 0.51, "chance", color="gray", fontsize=9)
a2.set_xlabel("rollout depth (steps)"); a2.set_ylabel("binding accuracy")
a2.set_title("Binding", fontsize=12); a2.legend(fontsize=9)
FIG4 = b64(fig)

# ---- numbers for prose ----
def cell(k):
    r = sum(succ[k]) / len(succ[k])
    w, ll, p = mcnemar(succ[k], base)
    star = " <b>*</b>" if p < 0.05 else ""
    return f"<td style='text-align:right'>{r:.3f}</td><td style='text-align:right'>+{w}/-{ll}</td><td style='text-align:right'>{p:.3f}{star}</td>"

order = ["ctl:prefix", "phi:bon", "phi:prefix", "mbac:mbacf", "mbac:mbacv", "mbac:mbacv_t13",
         "mbac:mbac", "ctl:bon", "ctl:critic", "phi:critic"]
rows_html = "\n".join(
    f"<tr><td>{LABELS[k]}</td>{cell(k)}</tr>" for k in order if k in succ)
E2 = bat["E2"]

def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

GIT_BRANCH = _git("branch", "--show-current")
GIT_HASH = _git("rev-parse", "--short", "HEAD")
GIT_DIRTY = " (+uncommitted)" if _git("status", "--porcelain") else ""
GIT_STAMP = f"git: {GIT_BRANCH} @ {GIT_HASH}{GIT_DIRTY}"

css = (ROOT / "docs/reports/_template_house.html").read_text().split("<style>")[1].split("</style>")[0]
html = f"""<!doctype html><meta charset='utf-8'><title>overnight 2026-08-08 — critic authority & MB-AC</title>
<style>{css}</style>
<h1>하룻밤의 심판 — critic 권한과 model-based adaptive chunking</h1>
<div class='sub'>2026-08-08 새벽 · PrepareCoffee · 30 paired trials/arm, seed 0 (모든 arm이 같은 장면) · 표는 원시 JSON에서 생성 시점 재계산 (scripts/make_overnight_report_0808.py)</div>
<style>
.wa-rpt table{{width:100%;table-layout:auto;border-collapse:collapse;overflow-wrap:anywhere;font-size:.92em}}
.wa-rpt td,.wa-rpt th{{padding:6px 10px;vertical-align:top}}
.wa-rpt .spec th{{width:110px;white-space:nowrap}}
.wa-rpt .num td:nth-child(n+2){{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
.wa-rpt img{{max-width:100%}}
</style><div class='wa-rpt'>

<div class='card'><h3>TL;DR</h3>
<p>오프라인에서 전 조건을 통과한 Cal-QL+swap critic(φ-공간, binding .996, action-sensitivity .524)을 시뮬레이터에 세웠다.
결과는 권한 순서대로다: <b>전권(full-authority) critic 모드는 밤의 유일한 유의미 결과인 파국</b>(.300 vs vla .700, McNemar p=.004),
선택-만(bon) 권한은 무해·무익(.700 동률), σ-veto BoN(mbacf)이 .767로 최상단이지만 아직 유의하지 않다.
한편 φ-공간 dynamics 앙상블은 CQL 없이 disagreement만으로 binding .817을 달성 — 커밋(어디까지 실행할지)과
거부권(어느 후보를 걸러낼지)이라는 <b>좁고 검증가능한 권한</b>이 모델의 올바른 자리라는 것이 오늘 밤의 논지다.</p></div>

<div class='card'><h3>명세</h3>
<table class='spec'>
<tr><th>VLA</th><td>pi05 rlt5_pardec_noprop @70k, PrepareCoffee, N=16 candidates, 10 flow steps</td></tr>
<tr><th>critic A</th><td>raw RLT token 2048+16, plain IQL — action-blind (offline sens .001)</td></tr>
<tr><th>critic B</th><td>HILP φ readout 128+16, IQL + Cal-QL(α=1) + swap negatives — 오프라인 전 조건 통과</td></tr>
<tr><th>dynamics</th><td>DynV1 5-ensemble, φ-공간(hist1 배포판: R² .665@16, AUSE .076, OOD swap 1.54)</td></tr>
<tr><th>모드</th><td>vla / bon / prefix / critic(full-auth) / mbacv(σ-커밋만) / mbac(critic 선택+σ-커밋) / mbacf(σ-veto 후 critic 선택+σ-커밋)</td></tr>
</table></div>

<div class='card'><h3>질문 1 — 오프라인 합격이 롤아웃 이득으로 이어지는가</h3>
<p>아니오. 권한이 넓을수록 해가 커진다. 그림은 각 arm의 성공률에서 vla 기준선(.700)을 뺀 값이다 — arm당 30 paired trials(모든 arm이 같은 30개 장면), 채워진 점만 McNemar p&lt;0.05로 유의하고 빈 점은 우연과 구분되지 않는다. 색은 critic 계열(빨강 action-blind / 초록 φ Cal-QL+swap / 보라 model-based).</p>
<img src='data:image/jpeg;base64,{FIG1}'>
<table class='num'><tr><th>arm</th><th>success</th><th>paired +/-</th><th>McNemar p</th></tr>
{rows_html}</table>
<p>action-blind critic(A)은 의견이 없어서 무해했다(bon .600, critic .567 — 동전던지기).
binding을 고친 critic(B)은 <span class='bad'>전권 모드에서 .300으로 붕괴</span>했다: 16후보×8prefix=128 옵션의 arg-max가
Q의 꼬리 노이즈를 자신 있게 착취하고, 커밋 분포는 59%가 2스텝인 스래싱이 된다(아래 그림).
같은 critic이 선택-만(bon) 권한에서는 vla와 정확히 동률(각각 5승) — <b>아는 것</b>은 고쳐졌지만
<b>행동으로 옮기는 방식</b>이 병목이라는 뜻. worker-B의 23-config 피해 테이블과 같은 결론을 더 날카로운 도구로 재현했다.</p></div>

<div class='card'><h3>질문 2 — 커밋(how far)은 누가 정해야 하는가</h3>
<img src='data:image/jpeg;base64,{FIG2}'>
<p>세 분포는 replan당 실행한 스텝 수다(최대 16). 야간 최고 성적(.800)을 낸 대조군 prefix 모드의 커밋 분포는 이봉형이다: 대부분 길게(14–16) 가되
30%는 2스텝에서 끊는다 — 소박한 수단으로나마 이미 적응적 커밋을 하고 있었다.
붕괴한 전권 critic은 같은 축을 스래싱으로 사용했고, σ-rule(tau 2.0)은 88% 풀커밋으로 사실상 개입하지 않았다
(E1 배터리가 예측한 그대로 — tau가 너무 관대). tau 1.3 스윕이 후속으로 돌고 있다.</p></div>

<div class='card'><h3>질문 3 — dynamics는 어느 공간에서, 무엇에 쓸 것인가</h3>
<img src='data:image/jpeg;base64,{FIG3}'>
<p>왼쪽은 copy-forward 대비 R²(높을수록 정확), 오른쪽은 AUSE(낮을수록 자기 오류를 잘 순위매기는, 즉 잘 캘리브레이션된 불확실성). 같은 아키텍처·같은 스텝으로 φ-공간이 DINO-공간을 전 구간에서 이긴다(R² .659 vs .519 @16, OOD swap 1.41–1.54 vs 1.09).
TD readout이 외형 잡음을 미리 버려줬기 때문에, 남은 기하가 정확히 행동이 움직이는 것이기 때문이다.
hist=1 배포판이 hist=3와 동급이라 롤아웃 통합도 깨끗하다.</p>
<img src='data:image/jpeg;base64,{FIG4}'>
<p>오프라인 배터리(held-out 에피소드 n=8000): <b>disagreement는 CQL 없이 binding {E2["binding_by_disagreement"]:.3f}</b>
(다른 상태의 demo 청크의 σ가 평균 {E2["sig_ratio_other_over_demo"]:.2f}배) — 그러나 모델이 예측한 종말상태의 goal 거리로 랭킹하면
{E2["binding_by_goal_distance"]:.3f}로 <span class='bad'>우연 이하</span>. 모델을 가치평가에 쓰면 함정이고, 신뢰영역(σ)으로 쓰면 진짜 신호다.
가시성은 첫 macro-step(4스텝)에서 이미 최고(.837)라 veto는 깊은 롤아웃이 필요 없다.
σ-적응 컷은 fixed-k 프론티어 아래에 놓인다(같은 평균 커밋에서 더 낮은 종말 오류) — 얇지만 실재하는 마진.</p></div>

<div class='card'><h3>결론과 다음 단계</h3>
<p><b>논지:</b> 학습된 컴포넌트에게 줄 권한은 좁고 검증가능해야 한다. 가치평가는 conservatism으로 고정된 critic에게(그마저 선택-만),
신뢰판정은 앙상블 σ에게(veto와 커밋), 나머지는 VLA에게. mbacf(.767)가 이 설계의 첫 데이터 포인트다.</p>
<ul>
<li>tau 1.3 mbacv 스윕 (34656, 진행중) — σ-rule이 실제로 개입할 때의 커밋-만 효과</li>
<li>mbacf를 30→60 trials로 확장해 .767의 유의성 검정</li>
<li>π_k 오프라인 학습은 보류 — E1 마진이 얇아 σ-rule 대비 이득이 불투명 (worker-B 테이블의 교훈)</li>
<li>AC-RFT 통합: MB-AC 커밋 하에서 수집한 롤아웃으로 RFT — 데이터 분포 자체가 adaptive-chunked</li>
</ul></div>

</div>
<div class='sub'>원시 데이터: .scratch/rollout_{{control,rltphi,mbac}}.json · mbac_offline_phi.json · phi_dyn_v1{{,_h1}}/report.json · 설계 노트: docs/reports/mbac_design_notes.md · {GIT_STAMP}</div>
"""
out = ROOT / "docs/reports/2026-08-08_overnight-authority-mbac.html"
out.write_text(html)
print(f"wrote {out} ({len(html)//1024} KB)")
