"""Weekly presentation deck generator (user rule, 2026-08-23).

One self-contained HTML per week, slide-styled sections: every result plot is preceded by the
background visualization that makes it presentable (method schematic, architecture, pipeline).
Numbers in tables are recomputed from raw JSONs where they exist. Images are embedded base64.

    CACHE_DIR=/data5/jellyho/acrft_cache uv run python scripts/make_weekly_slides.py
"""

import base64
import io
import json
import os
import pathlib

from PIL import Image

P = pathlib.Path(os.environ.get("CACHE_DIR", "/data5/jellyho/acrft_cache")) / "plots"
REPO = pathlib.Path(__file__).parent.parent
OUT = REPO / ".scratch" / "weekly"
WEEK = "2026-08-23"


def img(name, alt=""):
    p = P / name
    if not p.exists():
        return f"<p class='missing'>figure missing: {name}</p>"
    im = Image.open(p).convert("RGB")
    if im.width > 1400:
        im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=87)
    return f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}' alt='{alt}'/>"


def gate_table():
    rows, data = [], {}
    for tag, d in (("160k", "eval_onestep_160k"), ("200k", "eval_onestep_200k")):
        f = REPO / ".scratch" / d / "results.json"
        if f.exists():
            data[tag] = json.loads(f.read_text())["metrics"]
    for k, label in [
        ("af_1step", "α-Flow 1-step"),
        ("af_2step", "α-Flow 2-step"),
        ("af_10step", "α-Flow 10-step"),
        ("bc_10step", "BC 기준 10-step"),
    ]:
        cells = "".join(
            f"<td>{data[t][f'mse_gt/{k}']:.5f}</td>" if t in data else "<td>—</td>" for t in ("160k", "200k")
        )
        hl = " class='hl'" if k == "af_1step" else ""
        rows.append(f"<tr{hl}><td>{label}</td>{cells}</tr>")
    return "".join(rows)


def p2_summary():
    f = REPO / ".scratch/p2_uncertainty/p2_split.json"
    if not f.exists():
        return ""
    import math

    import numpy as np

    d = json.loads(f.read_text())
    lnA, lnE = [], []
    for e in d["20k"]:
        a0 = np.median(np.asarray(d["20k"][e]["u_alea"])[:, -1])
        a1 = np.median(np.asarray(d["120k"][e]["u_alea"])[:, -1])
        e0 = np.median(np.asarray(d["20k"][e]["u_epis"])[:, -1])
        e1 = np.median(np.asarray(d["120k"][e]["u_epis"])[:, -1])
        lnA.append(math.log(a1 / a0))
        lnE.append(math.log(e1 / e0))
    return (
        f"기하평균 u_alea ×{math.exp(np.mean(lnA)):.2f} · u_epis ×{math.exp(np.mean(lnE)):.2f} "
        f"({len(lnA)} 에피소드)"
    )


CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;background:#e8e8e4;margin:0;padding:24px;color:#1a1a1a}
.slide{background:#fff;max-width:1060px;margin:0 auto 28px;padding:38px 48px;border-radius:14px;
  box-shadow:0 2px 10px rgba(0,0,0,.09);page-break-after:always}
.slide h1{font-size:30px;margin:0 0 6px} .slide h2{font-size:22px;margin:0 0 14px;border-bottom:2px solid #4c72b0;padding-bottom:8px}
.slide h3{font-size:15px;color:#4c72b0;margin:18px 0 6px}
.kicker{color:#888;font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;margin-bottom:4px}
img{max-width:100%;border:1px solid #eee;border-radius:8px;margin:8px 0}
table{border-collapse:collapse;margin:10px 0;font-size:13.5px;width:100%}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left} th{background:#f4f4f2}
tr.hl td{background:#fdeee6;font-weight:600}
.bg{background:#f2f6fc;border-left:4px solid #4c72b0;padding:10px 14px;border-radius:6px;font-size:13.5px;margin:10px 0}
.verdict{background:#eafaea;border-left:4px solid #55a868;padding:10px 14px;border-radius:6px;font-size:14px;margin:10px 0}
.warn{background:#fdf3e6;border-left:4px solid #dd8452;padding:10px 14px;border-radius:6px;font-size:13.5px;margin:10px 0}
.cols{display:flex;gap:18px} .cols>div{flex:1}
ul{margin:6px 0;padding-left:20px} li{margin:4px 0;font-size:14px}
.small{font-size:12.5px;color:#666}
"""


def build():
    s = [f"<style>{CSS}</style>"]

    s.append("""
<div class='slide'>
<div class='kicker'>ACRFT · 주간 실험 보고</div>
<h1>Week of 08-18 – 08-23 — α-Flow π0.5, 이론 프로그램, 크리틱·FQL</h1>
<p>한 주의 세 트랙과 결론:</p>
<ul>
<li><b>α-Flow π0.5</b> — 구현 → 커리큘럼 검증 → 200k 완주 → <b>1-step 게이트 통과</b> (원스텝 무손실, 오히려 이득)</li>
<li><b>밤샘 이론 프로그램</b> — "언제 replan하나"의 4힘 이론 + 계열 지도 + 예측 P1~P5, 허브 6편</li>
<li><b>크리틱·FQL</b> — g5_pi05 +100k 이어학습(실패 분리 강화), P2 첫 실측(미결), FQL 인계·수리·staged 레시피</li>
</ul>
<p class='small'>허브: jellyho/acrft-reports (이번 주 신규 9 entries) · wandb: yam-rlt/c4vy84yy · 모든 그림은 스크립트 재생성 가능
(scripts/make_weekly_figs.py, slurm/make_figures.py)</p>
</div>""")

    s.append(f"""
<div class='slide'>
<h2>1 · α-Flow π0.5 — 배경: 무엇을 왜 바꾸나</h2>
<div class='bg'><b>문제.</b> offline RL의 actor 업데이트는 매 스텝 정책 샘플이 필요한데, π0.5는 10-step ODE라
그 샘플링이 비용의 대부분이다. <b>α-Flow(ICLR'26)</b>는 순간속도 대신 구간 <b>평균속도</b> u(z,r,t)를
배워 한 번의 점프로 ODE를 대체한다 — distillation과 달리 데이터 위 회귀만으로 (teacher 샘플링 없음).</div>
{img('40_af_concept.png')}
<h3>설계 포인트 (우리 구현)</h3>
<ul>
<li>r-조건은 <b>zero-init</b> adaRMS — step 0에서 pretrained π0.5와 비트 일치 (재학습이 아니라 파인튜닝)</li>
<li>공식 스케줄(run 전체 sigmoid, γ=25, clamp 5e-3)을 <b>진행도 비율</b>로 — num_train_steps만 바꾸면 커리큘럼이 따라옴</li>
<li>JVP가 필요한 α=0 꼬리는 기본 OFF (floor 5e-3 discrete)</li>
</ul>
{img('30_af_sched.png')}
<p class='small'>사전 검증: 240-step 실학습에서 α 실측(점)이 이론 곡선(선)과 4자리 일치 — 커리큘럼이 스스로 도는 것의 실증.</p>
</div>""")

    s.append(f"""
<div class='slide'>
<h2>2 · α-Flow — 결과: 200k 완주와 JVP 안전성</h2>
{img('41_af_200k_curves.png')}
<div class='verdict'>커리큘럼 전 구간 무사고: α 이탈점 실측 = 이론(57.6k), 전환 충격 없음(grad_norm 평탄), delta² 20× 감소.</div>
<h3>JVP 폭발 우려 (과거 관측) — 스트레스 판정</h3>
<table><tr><th>regime</th><th>bf16</th><th>f32</th></tr>
<tr><td>discrete (대조)</td><td>stable</td><td>stable</td></tr>
<tr><td>순수 JVP (α=0)</td><td>stable</td><td>stable</td></tr>
<tr><td>전환 (discrete→JVP 경계 통과)</td><td>stable</td><td>stable</td></tr></table>
<p>6/6 stable — bf16 수치 가설 기각. 방지 장치 3겹(u_tgt clip 4.0 · adaptive weight · 타깃 stop-grad)이 유효한 것으로 추정.</p>
<div class='warn'><b>사고 기록.</b> /data5 100% 포화로 attempt1이 step-20k 저장에서 사망(orbax async future) →
체크포인트를 /data1로 이전해 재주행. 교훈: 대형 train_state는 처음부터 /data1.</div>
</div>""")

    s.append(f"""
<div class='slide'>
<h2>3 · 1-step 게이트 — 원스텝화는 무손실인가</h2>
<div class='bg'><b>측정 설계.</b> held-out 6 에피소드 × 6 프레임에서, 각 정책이 <b>자기 norm stats로
unnormalize한 로봇 공간</b> 30-step 청크의 demo-MSE. 프레임당 동일 노이즈로 분산 통제.</div>
<div class='cols'><div>{img('42_gate_bars.png')}</div><div>
<table><tr><th>변형</th><th>@160k</th><th>@200k</th></tr>{gate_table()}</table>
<p class='small'>self-gap(1↔10) = 액션 분산의 ~0.1%</p></div></div>
<div class='verdict'><b>판정: 통과.</b> 1-step(0.00096)이 같은 모델의 10-step(0.00153)보다 낫고 스텝 수에 단조.
floor 40k가 1-step을 더 조임(160k→200k 개선). <b>RL actor 자리에 그대로 투입 가능.</b></div>
<div class='warn'>BC 대비 2.8×는 참고치 — BC는 s300 성공-only 70k vs α-Flow s347 전체 200k (method-only-diff 아님),
demo-MSE는 성공률 프록시.</div>
</div>""")

    s.append(f"""
<div class='slide'>
<h2>4 · 이론 프로그램 — 배경: 왜 adaptive chunking인가</h2>
<div class='bg'><b>출발 질문(사용자).</b> 이론적 완벽(결정론·완전관측·최적정책)에서는 모든 커밋 길이 k가
동률(tie)이다 — 그런데 왜 실제로는 adaptive가 이기는가? 그 동률이 <b>칼날 특이점</b>임을 보이고,
동률을 깨는 힘들을 식별하는 것이 프로그램의 목표.</div>
{img('31_three_forces.png')}
<h3>결과: 4힘 모델 + 허브 6편</h3>
<table><tr><th>힘</th><th>방향</th><th>운명</th></tr>
<tr><td>① 분기 (aleatoric)</td><td>k ↓</td><td>회수 불가 — floor</td></tr>
<tr><td>② 정책 오차 (epistemic)</td><td>레짐 의존</td><td><b>개선이 흡수</b> → curriculum</td></tr>
<tr><td>③ 안정성 (Zhang: 잦은 replan=오차 재주입)</td><td>k ↑ (하계!)</td><td>Markov에서도 성립</td></tr>
<tr><td>④ 정보 (Blackwell 가블링/copycat)</td><td>k ↑</td><td>관측이 좋아지지 않는 한 잔존</td></tr></table>
<p class='small'>entries: adaptive-exec-map(계열 17+편 지도) · tie-knife-edge(K1~K5) · uncertainty-split ·
nonmarkov-longer · event-triggered-bridge(제어이론 40년 선배) · three-forces(예측 P1~P5). 인트로 반영 브랜치 intro-four-forces 푸시.</p>
</div>""")

    s.append(f"""
<div class='slide'>
<h2>5 · P2 첫 실측 — 학습은 epistemic만 줄이는가</h2>
<div class='bg'><b>배경.</b> 분포형 앙상블(K개 HL-Gauss)의 총분산 분해: u_alea=멤버 내 분산(학습해도 남음),
u_epis=멤버 간 불일치(학습이 줄임). 예측 P2: 20k→120k에서 u_epis만 감소해야 한다.
같은 크리틱의 20k / +100k(120k) 체크포인트가 마침 있어 직접 측정.</div>
{img('32_p2_split.png')}
<div class='warn'><b>판정: 미결·약지지.</b> {p2_summary()} — 방향은 맞으나(epis가 더 감소) 강한 형태는 미지지:
2/6 에피소드에서 epis 증가, alea도 동자릿수 이동. 지배 교란 = K=2 (두 멤버 불일치로 epistemic을 재는 것은
원리적 고분산) → head_ensemble K≥8 재측정이 전제.</div>
<p><b>부수 발견(다음 가설):</b> 성공 에피는 u_alea가 조여지고(×0.5~1.0) 실패 에피는 유지·상승(×1.1~1.4) —
행동정책 산포가 u_alea에 섞인다는 경고와 정합. 크리틱 이어학습의 실질 이득도 같은 그림: 실패 천장이
전부 하락(near-miss ep5 −948→−1542), 성공-실패 간극 ~780→~1200 (가치 비디오 12편, 허브 갤러리).</p>
</div>""")

    s.append(f"""
<div class='slide'>
<h2>6 · FQL — 인계·수리·staged 레시피 (다음 주의 본편)</h2>
<div class='bg'><b>배경.</b> 다른 세션의 스캐폴드가 import조차 안 되던 상태(frozen dataclass)를 인계.
frozen fix → 실백본 학습 스텝 검증(OOM 2회 수리: prefix-KV 공유 / donate_argnums) → 사용자 레시피 구현.</div>
{img('43_fql_arch.png')}
<table><tr><th>구간</th><th>상태</th></tr>
<tr><td>import / 4-expert 구성 / 세 forward</td><td>✅ 실백본</td></tr>
<tr><td>학습 스텝 (critic TD+mc_floor / distill / actor Q, disjoint 옵티마이저)</td><td>✅ 실백본 10 스텝</td></tr>
<tr><td>staged: critic MC-warmup → AC, backbone-grad 라우팅 {{never,warmup,always}}</td><td>✅ CPU 검증</td></tr>
<tr><td>pretrained 로딩(--init-base) / 실데이터 로더</td><td>⬜ 미검증 / 미구현</td></tr></table>
<div class='verdict'><b>다음 주 계획.</b> ① α-Flow 1-step 정책을 FQL actor로 결합 (distill teacher 비용 소멸)
② 실데이터 결정: YAM vs RoboCasa(DEAS 프로토콜) — <b>사용자 결정 대기</b> ③ K≥8 P2 재측정 ④ P1/P4 측정 착수.</div>
</div>""")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"weekly_{WEEK}.html"
    out.write_text(f"<!doctype html><meta charset='utf-8'><title>주간 보고 {WEEK}</title>" + "".join(s))
    print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    build()
