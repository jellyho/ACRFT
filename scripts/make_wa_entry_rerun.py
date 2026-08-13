"""Publish the phi-normalization rerun verdict as a hub entry (RULES.md-compliant).

Every number is recomputed from the two rollout JSONs at generation time; the BoN power-up
criterion is pre-registered here BEFORE seeds 30/60 land. Links are resolved by title search
at publish time so worker-B renumbering cannot break them silently.
"""

import datetime
import json
from math import comb
import pathlib
import re
import subprocess
import tempfile

from huggingface_hub import CommitOperationAdd
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

SPACE = "jellyho/acrft-reports"
FIG_LOCAL = pathlib.Path(".scratch/fig_rerun_verdict.jpg")
FIG_REMOTE = "figs/wa_rerun_verdict.jpg"

new = json.loads(pathlib.Path(".scratch/rollout_rltphi.json").read_text())
ctl = json.loads(pathlib.Path(".scratch/rollout_control.json").read_text())
vla = [bool(t["success"]) for t in ctl["vla"]["trials"]]


def mcnemar(a, b):
    x = sum(1 for i, j in zip(a, b, strict=True) if i and not j)
    y = sum(1 for i, j in zip(a, b, strict=True) if j and not i)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, sum(comb(n, k) for k in range(min(x, y) + 1)) / 2**n * 2)
    return x, y, p


ARMS = {}
for m in ("critic", "bon", "prefix"):
    s = [bool(t["success"]) for t in new[m]["trials"]]
    x, y, p = mcnemar(s, vla)
    ARMS[m] = {"k": sum(s), "n": len(s), "x": x, "y": y, "p": p}
V = {"k": sum(vla), "n": len(vla)}

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


def row(label, m):
    a = ARMS[m]
    sig = "<b>유의</b>" if a["p"] < 0.05 else "n.s."
    return (
        f"<tr><td>{label}</td><td>{a['k']}/{a['n']} = {a['k'] / a['n']:.3f}</td>"
        f"<td>+{a['x']}/−{a['y']}</td><td>{a['p']:.4f}</td><td>{sig}</td></tr>"
    )


def row_en(label, m):
    a = ARMS[m]
    sig = "<b>significant</b>" if a["p"] < 0.05 else "n.s."
    return (
        f"<tr><td>{label}</td><td>{a['k']}/{a['n']} = {a['k'] / a['n']:.3f}</td>"
        f"<td>+{a['x']}/−{a['y']}</td><td>{a['p']:.4f}</td><td>{sig}</td></tr>"
    )


def spec(rows):
    tr = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='tblwrap'><table class='spec w6'>{tr}</table></div>"


TB_KO = (
    "<div class='tblwrap'><table class='num'><tr><th>팔</th><th>성공률</th><th>페어 +/−</th>"
    "<th>McNemar p</th><th>판정</th></tr>"
    f"<tr><td>vla (대조)</td><td>{V['k']}/{V['n']} = {V['k'] / V['n']:.3f}</td><td>—</td><td>—</td><td>기준</td></tr>"
    + row("critic (전권: cand×prefix argmax)", "critic")
    + row("bon (선택만)", "bon")
    + row("prefix (커밋길이만)", "prefix")
    + "</table></div>"
)
TB_EN = (
    "<div class='tblwrap'><table class='num'><tr><th>arm</th><th>success</th><th>paired +/−</th>"
    "<th>McNemar p</th><th>verdict</th></tr>"
    f"<tr><td>vla (control)</td><td>{V['k']}/{V['n']} = {V['k'] / V['n']:.3f}</td><td>—</td><td>—</td><td>baseline</td></tr>"
    + row_en("critic (full authority: cand×prefix argmax)", "critic")
    + row_en("bon (selection only)", "bon")
    + row_en("prefix (commit length only)", "prefix")
    + "</table></div>"
)

SPEC_KO = spec(
    [
        ("누가", "워커A"),
        ("언제", now + " KST"),
        ("어디서", "PrepareCoffee @70k, 페어드 롤아웃 30 trials (scene은 trial별 고정, vla 대조군과 동일 시드)"),
        ("무엇을", "φ 정규화 버그(무효 공지했던 p=.004 건)의 동일-시드 재실행 — φ+calswap critic 3팔"),
        ("어떻게", "eval_critic.py — 수정된 φ 어댑터(cache 통계로 토큰 표준화 후 φ 적용), --modes critic bon prefix"),
        ("왜", "retraction의 결말 확정: 파국이 버그 산물이었는지, 유효한 φ에서도 재현되는지"),
        ("코드", f"<code>{stamp}</code>"),
    ]
)
SPEC_EN = spec(
    [
        ("Who", "worker-A"),
        ("When", now + " KST"),
        (
            "Where",
            "PrepareCoffee @70k, paired rollouts, 30 trials (scene pinned per trial, seeds shared with the vla control)",
        ),
        (
            "What",
            "identical-seed rerun of the φ-normalization bug (the retracted p=.004 case) — 3 φ+calswap critic arms",
        ),
        (
            "How",
            "eval_critic.py with the fixed φ adapter (tokens standardized with cache stats before φ), --modes critic bon prefix",
        ),
        ("Why", "settle the retraction: was the catastrophe a bug artifact, or does it reproduce with a valid φ"),
        ("Code", f"<code>{stamp}</code>"),
    ]
)

KO = f"""
{SPEC_KO}
<p class='sub'>배경. 08-10 오전 정정 공지: φ-소비 롤아웃 팔 전체가 정규화 버그(표준화 토큰으로 학습된 phi.pt에
raw 토큰 주입, 상대 출력 오차 45%)로 무효였다 — "전권 파국 .300, McNemar p=.004" 포함. 이 엔트리는 그
retraction의 결말이다: 버그를 고친 φ로 동일 scene 시드에서 3팔을 재실행했다. 워커B가 인용 7곳에 달아 둔
주의 문구는 이 결과로 해소된다.</p>
<img src='{FIG_REMOTE}' alt='rerun paired rollouts' style='max-width:560px;width:100%'>
<p class='sub'>그림 읽는 법: 점 = 팔별 성공률, 채운 점 = vla 대비 McNemar p&lt;0.05, 점선 = vla 대조군.</p>
{TB_KO}
<p><b>판정 ① — 전권 파국은 실재한다.</b> 유효한 φ에서 .133 (+2/−19, p=.0002) — 무효였던 .300보다 오히려
심하다. 즉 정정된 것은 숫자이지 결론이 아니다: <b>authority 명제(선택권의 폭이 위험 축)는 이제 유효한
증거 위에 선다.</b> 버그는 critic 입력을 훼손했는데도 파국 방향이 같았다는 점은, 전권 팔의 해악이 critic의
정확도와 거의 무관하게 구조적임을 재확인한다 (워커B conservatism 축2와 합치).</p>
<p><b>판정 ② — BoN 첫 양(+) 신호, 그러나 미확정.</b> .800 vs .700 (+6/−3, p=.51). 30-trial ±.1은 노이즈
범위라는 우리 규칙 그대로, 이것은 주장이 아니라 가설이다. <b>사전등록</b> (seed 30/60 잡이 도는 지금, 결과
도착 전에 명시): seed 30–59·60–89에서 bon vs vla 각 30쌍을 추가해 총 90쌍을 모으고, <b>합산 McNemar
p&lt;0.05일 때만</b> "BoN이 vla를 이긴다"로 승격한다. 아니면 null로 기록한다. 단서: 이 critic(calswap)의
오프라인 sensitivity는 γ-천장 초과(인공 마진) 판정을 받았으므로, 이득이 실재하더라도 그 근원은 보정된
마진이 아니라 순서(ranking)일 수 있다 — 승격 시 순서-기반 해석을 기본으로 한다.</p>
<p><b>판정 ③ — prefix(커밋길이만)는 null</b> (.667, +6/−7, p=1.0).</p>
<p class='sub'>git: <code>{stamp}</code> · 원본: <code>.scratch/rollout_rltphi.json</code> +
<code>.scratch/rollout_control.json</code> (표·그림은 생성 시 JSON에서 재계산) · 재실행 잡 34923,
파워 확장 잡 34925/34926 진행 중.</p>
"""

EN = f"""
{SPEC_EN}
<p class='sub'>Background. The 08-10 morning retraction: every φ-consuming rollout arm was voided by a
normalization bug (raw tokens fed to a phi.pt trained on standardized ones; 45% relative output error) —
including the ".300 full-authority catastrophe, McNemar p=.004". This entry closes that retraction: all three
arms were rerun on identical scene seeds with the fixed φ. The caveats worker-B attached at 7 citation sites
are resolved by this result.</p>
<img src='{FIG_REMOTE}' alt='rerun paired rollouts' style='max-width:560px;width:100%'>
<p class='sub'>Reading the figure: dots = per-arm success rate, filled = McNemar p&lt;0.05 vs vla,
dashed line = the vla control.</p>
{TB_EN}
<p><b>Verdict ① — the full-authority catastrophe is real.</b> With a valid φ: .133 (+2/−19, p=.0002) — worse
than the voided .300. What the retraction corrected was the number, not the conclusion: <b>the authority
thesis (breadth of authority as the risk axis) now stands on valid evidence.</b> That the bug corrupted the
critic's input yet the catastrophe pointed the same way reconfirms the harm as structural, nearly independent
of critic accuracy (consistent with worker-B's conservatism axis 2).</p>
<p><b>Verdict ② — first positive BoN tilt, unconfirmed.</b> .800 vs .700 (+6/−3, p=.51). By our own rule
(±.1 at 30 trials is noise) this is a hypothesis, not a claim. <b>Pre-registration</b> (stated now, while the
seed-30/60 jobs run and before their results arrive): add 30 bon-vs-vla pairs each at seeds 30–59 and 60–89
for 90 pairs total; promote to "BoN beats vla" <b>only if the pooled McNemar p&lt;0.05</b>; otherwise record
a null. Caveat: this critic's (calswap) offline sensitivity was judged an over-γ-ceiling artificial margin,
so even a real gain would be attributed to ranking order, not calibrated margins.</p>
<p><b>Verdict ③ — prefix (commit-length only) is null</b> (.667, +6/−7, p=1.0).</p>
<p class='sub'>git: <code>{stamp}</code> · raw: <code>.scratch/rollout_rltphi.json</code> +
<code>.scratch/rollout_control.json</code> (tables/figure recomputed from JSON at generation) · rerun job
34923; power-up jobs 34925/34926 in flight.</p>
"""

ENTRY = {
    "date": now,
    "title": "🔬 [워커A] 재실행 판정 — 전권 파국은 실재(.133, p=.0002), BoN 첫 양(+) 신호는 사전등록 후 검증",
    "summary": (
        "φ 정규화 버그 retraction의 결말: 동일 시드 재실행에서 전권(full-authority) 파국이 유효한 φ로 재현 "
        "(.133, +2/−19, p=.0002 — 무효였던 .300보다 심함). 정정된 것은 숫자이지 결론이 아니다. "
        "BoN은 .800 vs .700 (+6/−3, p=.51)로 첫 양(+) 신호 — 90쌍 합산 McNemar p<.05를 승격 기준으로 "
        "사전등록하고 seed 30/60 잡을 돌리는 중. prefix는 null."
    ),
    "tags": ["워커A", "RoboCasa"],
    "status": "finding",
}


def resolve_links(reports, ko, en):
    """Rewrite openReport(#T:...) placeholders to indices found by title substring."""

    def sub(txt):
        def repl(m):
            key = m.group(1)
            for i, r in enumerate(reports):
                if key in r["title"]:
                    return f"openReport({i})"
            return "goHome()"

        return re.sub(r"openReport\(#T:([^)]+)\)", repl, txt)

    return sub(ko), sub(en)


LINKS_KO = (
    "<p class='sub'>연결된 리포트: <span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>관계 기하 판정(r49)</span> · "
    "<span class='xref' onclick='openReport(#T:TD-SF-ARQ)'>워커B TD-SF-ARQ 사전등록</span> · "
    "<span class='xref' onclick='openReport(#T:Conservatism)'>워커B conservatism 스펙트럼</span></p>"
)
LINKS_EN = (
    "<p class='sub'>Links: <span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>relational-geometry verdict (r49)</span> · "
    "<span class='xref' onclick='openReport(#T:TD-SF-ARQ)'>worker-B TD-SF-ARQ pre-registration</span> · "
    "<span class='xref' onclick='openReport(#T:Conservatism)'>worker-B conservatism spectrum</span></p>"
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
    ko_l, en_l = resolve_links(reports, KO + LINKS_KO, EN + LINKS_EN)
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
    msg = f"worker-A r{idx}: rerun verdict — authority catastrophe reproduces, BoN pre-registered [{stamp}]"
    try:
        api.create_commit(SPACE, ops, repo_type="space", commit_message=msg, parent_commit=head)
        print(f"published r{idx}")
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {e}")
else:
    raise SystemExit("could not publish after retries")
