"""Publish the YAM pi05 real-robot checkpoint-scaling verdict as a hub entry.

Raw data lives in docs/reports/yam_pi05_progress_2026-08-10.json (user-reported milestone
progress, 10 trials per run); every table number is recomputed from it here.
"""

import datetime
import json
import pathlib
import re
import subprocess
import tempfile

from huggingface_hub import CommitOperationAdd
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import numpy as np

SPACE = "jellyho/acrft-reports"
FIG_LOCAL = pathlib.Path(".scratch/fig_yam_pi05_scaling.jpg")
FIG_REMOTE = "figs/wa_yam_pi05_scaling.jpg"
DATA = pathlib.Path("docs/reports/yam_pi05_progress_2026-08-10.json")

d = json.loads(DATA.read_text())
RUNS = d["runs"]
ORDER = ["rel_s200_50k", "rel_s200_100k", "rel_s200_150k", "rel_s200_200k", "rel_s200_100k_h60"]

branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=False).stdout.strip()
sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
).stdout.strip()
dirty = bool(
    subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()
)
stamp = f"{branch}@{sha}" + (" (+uncommitted)" if dirty else "")
now = (
    datetime.datetime.now(datetime.UTC)
    .astimezone(datetime.timezone(datetime.timedelta(hours=9)))
    .strftime("%Y-%m-%d %H:%M")
)


def stats(name):
    p = RUNS[name]["prog"]
    n = len(p)
    ge = [sum(x >= m for x in p) for m in (1, 2, 3, 4)]
    return {
        "n": n,
        "mean": float(np.mean(p)),
        "sem": float(np.std(p, ddof=1) / np.sqrt(n)),
        "ge": ge,
        "ckpt": RUNS[name]["ckpt"],
        "h": RUNS[name]["h"],
    }


S = {k: stats(k) for k in ORDER}


def table(lang):
    if lang == "ko":
        hdr = "<tr><th>run</th><th>ckpt</th><th>chunk H</th><th>mean progress</th><th>≥1</th><th>≥2</th><th>≥3</th><th>≥4 (완주)</th></tr>"
    else:
        hdr = "<tr><th>run</th><th>ckpt</th><th>chunk H</th><th>mean progress</th><th>≥1</th><th>≥2</th><th>≥3</th><th>≥4 (full)</th></tr>"
    rows = ""
    for k in ORDER:
        s = S[k]
        rows += (
            f"<tr><td>{k}</td><td>{s['ckpt'] // 1000}k</td><td>{s['h']}</td>"
            f"<td>{s['mean']:.1f} ± {s['sem']:.2f}</td>" + "".join(f"<td>{g}/{s['n']}</td>" for g in s["ge"]) + "</tr>"
        )
    return f"<div class='tblwrap'><table class='num'>{hdr}{rows}</table></div>"


def spec(rows):
    tr = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='tblwrap'><table class='spec w6'>{tr}</table></div>"


SPEC_KO = spec(
    [
        ("누가", "사용자(실기 롤아웃 수행) + 워커A(분석·게시)"),
        ("언제", now + " KST (평가 데이터 2026-08-10)"),
        ("어디서", "YAM 실기, lego_taxi 태스크 — π0.5 rel_s200 체크포인트 50k/100k/150k/200k + 100k H60 변형"),
        ("무엇을", "run당 10 trials, milestone progress 0–4 기록 → 체크포인트 스케일링과 milestone 분해"),
        ("어떻게", "milestone별 도달률(Wilson 95% 밴드) line plot + 평균 progress(SEM) — 원본 JSON에서 자동 재계산"),
        ("왜", "배포·critic 실험의 기준 체크포인트 선정 + s200 데이터 스케일의 실기 한계 확인"),
        ("코드", f"<code>{stamp}</code> · 원본 <code>docs/reports/yam_pi05_progress_2026-08-10.json</code>"),
    ]
)
SPEC_EN = spec(
    [
        ("Who", "user (real-robot rollouts) + worker-A (analysis/publication)"),
        ("When", now + " KST (evaluation data 2026-08-10)"),
        ("Where", "YAM real robot, lego_taxi — π0.5 rel_s200 checkpoints 50k/100k/150k/200k + a 100k H60 variant"),
        ("What", "10 trials per run, milestone progress 0–4 → checkpoint scaling and milestone decomposition"),
        ("How", "per-milestone attainment lines (Wilson 95% bands) + mean progress (SEM) — recomputed from raw JSON"),
        (
            "Why",
            "pick the reference checkpoint for deployment/critic work; probe the real-robot limit of s200 data scale",
        ),
        ("Code", f"<code>{stamp}</code> · raw <code>docs/reports/yam_pi05_progress_2026-08-10.json</code>"),
    ]
)

KO = f"""
{SPEC_KO}
<img src='{FIG_REMOTE}' alt='YAM pi05 checkpoint scaling' style='max-width:860px;width:100%'>
<p class='sub'>그림 읽는 법: 왼쪽 — milestone m 이상에 도달한 run의 비율을 체크포인트별로 이은 선, 음영은
Wilson 95% CI, 흰 다이아몬드 = 100k H60 변형. 오른쪽 — 평균 progress(0–4)와 SEM, 점선 = 완주(4).</p>
{table("ko")}
<p><b>판정 ① — 완주(milestone 4)는 전무.</b> 다섯 run 합쳐 50 trials에서 0회. 이 태스크에서 s200 데이터
스케일과 20만 스텝 학습으로는 마지막 단계 전환이 아직 안 열린다.</p>
<p><b>판정 ② — 스케일링은 평탄~하락 (잠정, n=10).</b> 평균 progress 50k 1.7 → 100k 1.7 → 150k 1.3 →
200k 1.4. n=10에서 SEM이 .15–.3이라 하락 자체는 유의하지 않지만, <b>학습을 4배 늘려도 개선이 전혀 없다</b>는
것은 명확하다.</p>
<p><b>판정 ③ — milestone 교차(crossover)가 진짜 정보다.</b> 200k는 시작 구간(≥1)을 10/10으로 가장 안정적으로
통과하지만 ≥3에 <b>한 번도</b> 도달하지 못한다(0/10). 반면 50k/100k는 ≥1이 각 10/10·9/10로 비슷하면서 ≥3을
2/10씩 만든다. 오래 학습할수록 "쉬운 구간은 더 확실히, 어려운 구간은 전멸" — 우리가 오프라인에서 잰
<b>200k 후보 스프레드 붕괴(.018, 16개 후보가 사실상 동일)</b>와 정합적인 그림이다: 다양성이 사라지면
초반 안정성은 얻지만 어려운 구간을 뚫을 변주가 없다. (인과는 미증명 — 정합 가설로 기록.)</p>
<p><b>판정 ④ — H60(긴 open-loop chunk)은 손해.</b> 같은 100k에서 1.7 → 1.2, ≥2가 6/10 → 2/10, ≥3은 0.
긴 chunk를 열린 루프로 밀어붙이면 어려운 구간에서 교정 기회를 잃는다 — <b>adaptive chunking(AC-RFT)의
동기 그 자체</b>가 실기에서 재현된 셈.</p>
<p><b>처방.</b> YAM 배포·critic 실험의 기준 체크포인트는 <b>50k–100k</b>로 고정한다(RoboCasa GP에서 90k가
표준이 된 것과 같은 패턴). 200k 계열은 후보 스프레드가 죽어 있어 BoN/critic 실험의 무대로 부적합. 다음 수:
100k에서 critic(iql_e90_g9998) 게이트 실기 테스트, 그리고 milestone 3→4 병목 구간의 영상 확인.</p>
<p class='sub'>단서: run 간 초기 배치는 통제되지 않은 실기 평가라 페어드 검정은 불가(비페어드 Wilson CI만).
n=10 — 모든 결론에 잠정(n=10) 라벨. git: <code>{stamp}</code></p>
"""

EN = f"""
{SPEC_EN}
<img src='{FIG_REMOTE}' alt='YAM pi05 checkpoint scaling' style='max-width:860px;width:100%'>
<p class='sub'>Reading the figure: left — fraction of runs reaching at least milestone m per checkpoint;
shading = Wilson 95% CI; open diamonds = the 100k H60 variant. Right — mean progress (0–4) with SEM;
dotted line = full completion (4).</p>
{table("en")}
<p><b>Verdict ① — zero full completions (milestone 4).</b> 0 of 50 trials across all five runs. At s200 data
scale and up to 200k steps, the final-stage transition has not opened on this task.</p>
<p><b>Verdict ② — scaling is flat-to-declining (provisional, n=10).</b> Mean progress 1.7 → 1.7 → 1.3 → 1.4
across 50k→200k. With SEM .15–.3 at n=10 the decline itself is not significant, but <b>4× more training buys
zero improvement</b> — that much is clear.</p>
<p><b>Verdict ③ — the milestone crossover is the real information.</b> 200k passes the opening segment (≥1)
most reliably (10/10) yet <b>never</b> reaches ≥3 (0/10), while 50k/100k are comparable at ≥1 (10/10, 9/10)
and each convert 2/10 at ≥3. Longer training: surer on the easy part, extinct on the hard part — consistent
with the <b>candidate-spread collapse we measured offline at 200k (.018; the 16 candidates are effectively
identical)</b>: once diversity dies, you gain early-stage stability but lose the variation needed to punch
through hard segments. (Causality unproven — recorded as a consistent hypothesis.)</p>
<p><b>Verdict ④ — H60 (long open-loop chunks) hurts.</b> At the same 100k: 1.7 → 1.2 mean, ≥2 drops 6/10 →
2/10, ≥3 zero. Pushing long chunks open-loop forfeits correction opportunities in hard segments — <b>the
motivation for adaptive chunking (AC-RFT), reproduced on the real robot.</b></p>
<p><b>Prescription.</b> Fix <b>50k–100k</b> as the reference checkpoints for YAM deployment and critic work
(the same pattern as 90k becoming the RoboCasa GP standard). The 200k family is unsuitable as a BoN/critic
stage — its candidate spread is dead. Next: a real-robot gate test of the critic (iql_e90_g9998) at 100k, and
video inspection of the milestone-3→4 bottleneck segment.</p>
<p class='sub'>Caveats: initial placements were not controlled across runs (real-robot evaluation), so no
paired tests — unpaired Wilson CIs only. n=10 — every conclusion carries a provisional (n=10) label.
git: <code>{stamp}</code></p>
"""

ENTRY = {
    "date": now,
    "title": "🤖 [워커A] YAM π0.5 실기 스케일링 — 완주 0/50, 200k는 쉬운 구간만 확실해지고 어려운 구간 전멸",
    "summary": (
        "실기 lego_taxi, 체크포인트 50k–200k × 10 trials, milestone 0–4. 완주 0/50. 평균 progress는 "
        "50k/100k 1.7에서 늘지 않음(잠정 n=10). 핵심은 교차: 200k는 ≥1을 10/10으로 통과하지만 ≥3은 0/10 — "
        "오프라인에서 잰 200k 후보 스프레드 붕괴(.018)와 정합. H60 변형은 1.2로 손해(open-loop 교정 상실 = "
        "adaptive chunking 동기). 처방: 기준 체크포인트 50k–100k."
    ),
    "tags": ["워커A", "YAM"],
    "status": "finding",
}


def resolve_links(reports, txt):
    def repl(m):
        key = m.group(1)
        for i, r in enumerate(reports):
            if key in r["title"]:
                return f"openReport({i})"
        return "goHome()"

    return re.sub(r"openReport\(#T:([^)]+)\)", repl, txt)


LINKS_KO = (
    "<p class='sub'>연결된 리포트: <span class='xref' onclick='openReport(#T:재실행 판정)'>재실행 판정(전권/BoN)</span> · "
    "<span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>관계 기하 판정</span></p>"
)
LINKS_EN = (
    "<p class='sub'>Links: <span class='xref' onclick='openReport(#T:재실행 판정)'>rerun verdict (authority/BoN)</span> · "
    "<span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>relational-geometry verdict</span></p>"
)

api = HfApi()
for attempt in range(6):
    head = api.repo_info(SPACE, repo_type="space").sha
    p = hf_hub_download(SPACE, "index.html", repo_type="space", revision=head, force_download=True)
    s = pathlib.Path(p).read_text()
    m = re.search(r"const REPORTS\s*=\s*(\[.*?\]);", s, re.DOTALL)
    reports = json.loads(m.group(1))
    if any(e.get("title") == ENTRY["title"] for e in reports):
        print("entry already present — nothing to do")
        break
    ko_l = resolve_links(reports, KO + LINKS_KO)
    en_l = resolve_links(reports, EN + LINKS_EN)
    idx = len(reports)
    reports.append(ENTRY)
    s = s[: m.start(1)] + json.dumps(reports, ensure_ascii=False) + s[m.end(1) :]
    section = (
        f'<section class="report" id="r{idx}" hidden>'
        f'<div class="wbx wbx-ko">{ko_l}</div><div class="wbx wbx-en">{en_l}</div></section>'
    )
    last = s.rindex("</section>") + len("</section>")
    s = s[:last] + "\n" + section + s[last:]
    assert len(s.encode()) < 9_500_000, "index.html would exceed the 9.5MB guard"
    tmp = pathlib.Path(tempfile.mkstemp(suffix=".html")[1])
    tmp.write_text(s)
    ops = [
        CommitOperationAdd(path_in_repo="index.html", path_or_fileobj=str(tmp)),
        CommitOperationAdd(path_in_repo=FIG_REMOTE, path_or_fileobj=str(FIG_LOCAL)),
    ]
    msg = f"worker-A r{idx}: YAM pi05 real-robot scaling verdict [{stamp}]"
    try:
        api.create_commit(SPACE, ops, repo_type="space", commit_message=msg, parent_commit=head)
        print(f"published r{idx}")
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {e}")
else:
    raise SystemExit("could not publish after retries")
