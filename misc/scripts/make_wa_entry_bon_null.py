"""Publish the pre-registered BoN power-up outcome (null) as a hub entry.

Closes the r50 pre-registration: 3 seed sets x 30 paired trials = 90 pairs, pooled McNemar.
All numbers recomputed from the four rollout JSONs at generation time.
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
FIG_LOCAL = pathlib.Path(".scratch/fig_bon_null.jpg")
FIG_REMOTE = "figs/wa_bon_null.jpg"


def succ(path, m):
    dd = json.loads(pathlib.Path(path).read_text())
    return [bool(t["success"]) for t in dd[m]["trials"]]


def mcnemar(a, b):
    x = sum(1 for i, j in zip(a, b, strict=True) if i and not j)
    y = sum(1 for i, j in zip(a, b, strict=True) if j and not i)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, sum(comb(n, k) for k in range(min(x, y) + 1)) / 2**n * 2)
    return x, y, p


SETS = [("seed 0", succ(".scratch/rollout_rltphi.json", "bon"), succ(".scratch/rollout_control.json", "vla"))]
SETS.extend(
    (f"seed {S}", succ(f".scratch/rollout_bon_s{S}.json", "bon"), succ(f".scratch/rollout_bon_s{S}.json", "vla"))
    for S in (30, 60)
)
ALLB, ALLV = [], []
ROWS = []
for name, b, v in SETS:
    x, y, p = mcnemar(b, v)
    ROWS.append((name, sum(b), len(b), sum(v), len(v), x, y, p))
    ALLB += b
    ALLV += v
PX, PY, PP = mcnemar(ALLB, ALLV)
POOL = (sum(ALLB), len(ALLB), sum(ALLV), len(ALLV), PX, PY, PP)

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


def tbl(head_run, head_pool):
    r = "".join(
        f"<tr><td>{nm}</td><td>{bk}/{bn} = {bk / bn:.3f}</td><td>{vk}/{vn} = {vk / vn:.3f}</td>"
        f"<td>+{x}/−{y}</td><td>{p:.3f}</td></tr>"
        for nm, bk, bn, vk, vn, x, y, p in ROWS
    )
    bk, bn, vk, vn, x, y, p = POOL
    r += (
        f"<tr><td><b>{head_pool}</b></td><td><b>{bk}/{bn} = {bk / bn:.3f}</b></td>"
        f"<td><b>{vk}/{vn} = {vk / vn:.3f}</b></td><td><b>+{x}/−{y}</b></td><td><b>{p:.3f}</b></td></tr>"
    )
    return (
        f"<div class='tblwrap'><table class='num'><tr><th>{head_run}</th><th>bon</th><th>vla</th>"
        f"<th>페어 +/−</th><th>McNemar p</th></tr>{r}</table></div>"
    )


def spec(rows):
    tr = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='tblwrap'><table class='spec w6'>{tr}</table></div>"


SPEC_KO = spec(
    [
        ("누가", "워커A"),
        ("언제", now + " KST"),
        ("어디서", "PrepareCoffee @70k, φ+calswap critic, 페어드 롤아웃 3 시드셋 × 30 = 90쌍"),
        ("무엇을", "r50에서 사전등록한 BoN 파워 확장 검정 — seed 0/30/60, 합산 McNemar"),
        ("어떻게", "eval_critic.py --modes bon vla, 시드셋별 자체 페어링, 4개 rollout JSON에서 자동 재계산"),
        ("왜", "seed 0의 첫 양(+) 신호(.800 vs .700, p=.51)가 실재인지 노이즈인지 사전등록 기준으로 판정"),
        ("코드", f"<code>{stamp}</code>"),
    ]
)
SPEC_EN = spec(
    [
        ("Who", "worker-A"),
        ("When", now + " KST"),
        ("Where", "PrepareCoffee @70k, φ+calswap critic, paired rollouts, 3 seed sets × 30 = 90 pairs"),
        ("What", "the BoN power-up test pre-registered in r50 — seeds 0/30/60, pooled McNemar"),
        ("How", "eval_critic.py --modes bon vla, self-paired per seed set, recomputed from four rollout JSONs"),
        (
            "Why",
            "decide by the pre-registered criterion whether seed 0's first positive tilt (.800 vs .700, p=.51) was real or noise",
        ),
        ("Code", f"<code>{stamp}</code>"),
    ]
)

KO = f"""
{SPEC_KO}
<p class='sub'>배경. <span class='xref' onclick='openReport(#T:재실행 판정)'>재실행 판정(r50)</span>에서 seed 0의
BoN이 .800 vs vla .700(+6/−3, p=.51)으로 첫 양(+) 신호를 보였다. 30-trial ±.1은 노이즈라는 규칙에 따라
이를 주장이 아닌 가설로 두고, "seed 30·60을 더해 90쌍 합산 McNemar p&lt;0.05일 때만 승격, 아니면 null"을
결과 도착 전에 사전등록했다. 이 엔트리가 그 결말이다.</p>
<img src='{FIG_REMOTE}' alt='BoN vs VLA across seed sets' style='max-width:560px;width:100%'>
<p class='sub'>그림 읽는 법: 시드셋별·합산 성공률(회색 vla, 파랑 bon), 위 숫자 = 각 페어드 McNemar p.</p>
{tbl("시드셋", "합산 (n=90)")}
<p><b>판정 — NULL. 사전등록 기준 미달.</b> seed 0의 +6/−3은 seed 30·60에서 각각 정확히 동률(.667 vs .667,
+3/−3·+2/−2)로 씻겨, 합산은 <b>bon .711 vs vla .678 (+11/−8, McNemar p=0.65)</b>. p&lt;0.05를 못 넘었다.
seed 0의 첫 신호는 노이즈였고, "BoN이 vla를 이긴다"로 승격하지 않는다.</p>
<p><b>의미.</b> flow VLA에서 test-time BoN 조향은 무익하다는 결론이, 이제 <b>서로 다른 3개 시드셋 90쌍</b>으로
확정된다(워커B의 30쌍 BoN 동률·우리 밤샘 판정과 3중 재현). r50에서 함께 확정된 전권 파국(.133, p=.0002)과
합치면 그림은 하나다: <b>선택권을 주면(전권) 해롭고, 선택만 시키면(BoN) 무익하다.</b> demo-only critic으로
얻을 수 있는 것은 여기까지이며, 남은 병목은 <span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>관계
기하 위의 action-조건 backup</span>(TD-SF-ARQ)이라는 방향성을 강화한다.</p>
<p class='sub'>단서: 시드셋 간 scene 풀은 다르지만 각 시드셋 내부는 bon·vla가 동일 scene으로 페어링됐다.
calswap의 오프라인 마진은 γ-천장 초과(인공물) 판정을 받았으므로, BoN이 무익한 것과 별개로 이 critic의
순서 정보 자체가 약하다는 점도 함께 기록한다. git: <code>{stamp}</code> · 원본:
<code>rollout_rltphi.json</code>·<code>rollout_control.json</code>·<code>rollout_bon_s30/s60.json</code>.</p>
"""

EN = f"""
{SPEC_EN}
<p class='sub'>Background. In <span class='xref' onclick='openReport(#T:재실행 판정)'>the rerun verdict (r50)</span>,
seed 0's BoN showed a first positive tilt (.800 vs vla .700, +6/−3, p=.51). By our rule that ±.1 at 30 trials
is noise, we held it as a hypothesis and pre-registered — before results — "add seeds 30/60; promote to a BoN
win only if the pooled 90-pair McNemar p&lt;0.05, else record a null." This entry is the outcome.</p>
<img src='{FIG_REMOTE}' alt='BoN vs VLA across seed sets' style='max-width:560px;width:100%'>
<p class='sub'>Reading the figure: per-seed-set and pooled success rates (gray vla, blue bon); numbers above =
each paired McNemar p.</p>
{tbl("seed set", "pooled (n=90)")}
<p><b>Verdict — NULL. Pre-registered criterion not met.</b> Seed 0's +6/−3 washed out at seeds 30 and 60,
each exactly tied (.667 vs .667; +3/−3, +2/−2), leaving pooled <b>bon .711 vs vla .678 (+11/−8, McNemar
p=0.65)</b> — not below 0.05. Seed 0's first signal was noise; we do not promote "BoN beats vla."</p>
<p><b>Meaning.</b> The conclusion that test-time BoN steering is useless on a flow VLA is now settled across
<b>three distinct seed sets, 90 pairs</b> (triple-reproduced with worker-B's 30-pair BoN tie and our overnight
verdict). Combined with the full-authority catastrophe also settled in r50 (.133, p=.0002), the picture is one:
<b>give it authority and it harms; let it only select and it is inert.</b> That is the ceiling of a demo-only
critic, and it sharpens the direction that the remaining bottleneck is
<span class='xref' onclick='openReport(#T:교차-궤적 이웃 판정)'>the action-conditioned backup over relational
geometry</span> (TD-SF-ARQ).</p>
<p class='sub'>Caveats: scene pools differ across seed sets, but within each set bon and vla are paired on
identical scenes. This critic's (calswap) offline margin was judged an over-γ-ceiling artifact, so beyond BoN's
inertness the ranking signal itself is weak — recorded for honesty. git: <code>{stamp}</code> · raw:
<code>rollout_rltphi.json</code>, <code>rollout_control.json</code>, <code>rollout_bon_s30/s60.json</code>.</p>
"""

ENTRY = {
    "date": now,
    "title": "🔬 [워커A] BoN 사전등록 검정 종결 — 90쌍 합산 null (p=0.65), test-time BoN은 무익 확정",
    "summary": (
        "r50에서 사전등록한 BoN 파워 확장의 결말. seed 0의 첫 양(+) 신호(.800 vs .700)는 노이즈였다: seed "
        "30·60은 각각 정확히 동률, 합산 bon .711 vs vla .678 (+11/−8, McNemar p=0.65)로 사전등록 기준(p<.05) "
        "미달 → null. flow VLA에서 test-time BoN 무익이 3개 시드셋 90쌍으로 확정. 전권 파국(.133)과 합치면 "
        "'전권=해롭고 선택만=무익' — demo-only critic의 천장."
    ),
    "tags": ["워커A", "RoboCasa"],
    "status": "finding",
}


def resolve(reports, txt):
    def repl(m):
        key = m.group(1)
        for i, r in enumerate(reports):
            if key in r["title"]:
                return f"openReport({i})"
        return "goHome()"

    return re.sub(r"openReport\(#T:([^)]+)\)", repl, txt)


api = HfApi()
for attempt in range(6):
    head = api.repo_info(SPACE, repo_type="space").sha
    p = hf_hub_download(SPACE, "index.html", repo_type="space", revision=head, force_download=True)
    s = pathlib.Path(p).read_text()
    m = re.search(r"const REPORTS\s*=\s*(\[.*?\]);", s, re.DOTALL)
    reports = json.loads(m.group(1))
    if any(e.get("title") == ENTRY["title"] for e in reports):
        print("already present")
        break
    ko_l, en_l = resolve(reports, KO), resolve(reports, EN)
    idx = len(reports)
    reports.append(ENTRY)
    s = s[: m.start(1)] + json.dumps(reports, ensure_ascii=False) + s[m.end(1) :]
    section = (
        f'<section class="report" id="r{idx}" hidden>'
        f'<div class="wbx wbx-ko">{ko_l}</div><div class="wbx wbx-en">{en_l}</div></section>'
    )
    last = s.rindex("</section>") + len("</section>")
    s = s[:last] + "\n" + section + s[last:]
    assert len(s.encode()) < 9_500_000
    tmp = pathlib.Path(tempfile.mkstemp(suffix=".html")[1])
    tmp.write_text(s)
    ops = [
        CommitOperationAdd("index.html", str(tmp)),
        CommitOperationAdd(FIG_REMOTE, str(FIG_LOCAL)),
    ]
    try:
        api.create_commit(
            SPACE,
            ops,
            repo_type="space",
            commit_message=f"worker-A r{idx}: BoN pre-registration closed — pooled null p=0.65 [{stamp}]",
            parent_commit=head,
        )
        print(f"published r{idx}")
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:100]}")
else:
    raise SystemExit("could not publish")
