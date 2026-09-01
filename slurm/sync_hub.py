"""Sync worker-B experiments into the shared HF Space hub — one blog entry PER experiment.

Reads the experiment ENTRIES + META (5W1H, real dates, cross-links) from make_master_report
(single source of truth) and writes them into the hub's `entries.json`.

    uv run --no-sync python slurm/sync_hub.py                 # every entry in ENTRIES
    uv run --no-sync python slurm/sync_hub.py --only <eid>    # just one

THE HUB CHANGED FORMAT. It used to be a single index.html carrying `const REPORTS = [...]` inline
with a <section> per entry, and this script spliced that array. It is now data-driven: index.html
is static and fetches `entries.json` at runtime, so publishing is a read-modify-write of that one
file plus the figures it references. The thread digest and the mindmap are no longer entries this
script synthesises -- the page builds them from the data itself.

SAFETY. It replaces ONLY entries whose eid appears in ENTRIES, and leaves every other entry byte
for byte. The old version replaced every worker-B entry, which was correct when this file was the
single source for all of them; it no longer is -- other sessions publish worker-B entries that
were never in make_master_report, and a blanket replace would delete them.

Figures ride as base64 in the body and are extracted to `figures/<eid>/<sha>.png`, matching what
the hub's existing entries reference: HF tracks files over 10 MB as LFS, and a static Space serves
LFS files as downloads rather than rendering them.
"""

import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import make_master_report as mm

# same cache root as make_master_report (CACHE_DIR), so this runs on hosts without /scratch
CACHE = pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))

SPACE = "jellyho/acrft-reports"
MARK = "워커B"
DATE_MAP = {
    "~08-01": "2026-08-01",
    "08-02~03": "2026-08-03",
    "08-05": "2026-08-05",
    "08-06": "2026-08-06",
    "08-07": "2026-08-07",
    "타임라인": "2026-08-08",
}
WBX_STYLE = """<style id="wbx-style">
/* ===== 허브 전면 백색 테마 (2026-08-08 지시: 메인 페이지·버튼까지 전부 라이트) =====
   허브 공통 CSS 변수를 라이트 팔레트로 오버라이드 — 구조·콘텐츠는 불변, 색만 교체 */
:root{--bg:#ffffff !important;--card:#f4f5f8 !important;--card2:#fafafa !important;
  --text:#1a1a1a !important;--muted:#5f6b7a !important;--line:#e2e2e2 !important;
  --green:#15803d !important;--red:#b91c1c !important;--yellow:#a16207 !important;--blue:#3730a3 !important}
body{background:#ffffff}
/* 워커B 리포트 = 흰 종이 카드 (프로페셔널·깔끔) */
.wbx{line-height:1.68;background:#ffffff;color:#1a1a1a;border:1px solid #e2e2e2;border-radius:12px;
  padding:26px 30px;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.wbx a{color:#3730a3}
.wbx h2,.wbx h3{margin:.7em 0 .4em;letter-spacing:-.01em;color:#111;font-family:Georgia,'Times New Roman',serif}
.wbx table{border-collapse:collapse;font-size:.9em;margin:10px 0;background:#fff;color:#1a1a1a}
/* 호스트 페이지가 .num을 26px 원형 배지(inline-flex)로 정의 — 우리 <table class='num'>이 그 규칙에
   잡히면 표 전체가 26×26 상자로 붕괴해 산문과 겹친다. 표 display를 강제 복원한다. */
.wbx table.num,.wbx table.spec{display:table !important;width:auto !important;height:auto !important;
  border-radius:0 !important;flex:none !important}
.wbx .tblwrap{overflow-x:auto !important;max-width:100%}.wbx .tblwrap table{margin:10px 2px}
.wbx td,.wbx th{border:1px solid #e2e2e2;padding:5px 10px;text-align:left;vertical-align:top}
.wbx th{background:#f3f4f8}.wbx img{max-width:100%;border:1px solid #e2e2e2;border-radius:8px;margin:8px 0;background:#fff}
.wbx .missing{background:#fef9c3;color:#713f12;padding:6px 10px;border-radius:6px}
.wbx code{background:#f3f4f8;padding:1px 5px;border-radius:4px;color:#1a1a1a}
/* 떠 있는 내비 버튼: 리포트를 한참 스크롤한 뒤 목록/맨위로 바로 이동 */
#wb-float{position:fixed;right:18px;bottom:18px;z-index:80;display:none;flex-direction:column;gap:8px}
#wb-float button{width:46px;height:46px;border-radius:50%;border:1px solid #cfd4dd;background:#fff;
  color:#1f2430;font-size:18px;cursor:pointer;box-shadow:0 3px 12px rgba(0,0,0,.18);line-height:1}
#wb-float button:hover{border-color:#3730a3;color:#3730a3}
.wbx .tl{position:relative;margin:14px 0 14px 8px;border-left:3px solid #3730a3;padding-left:22px}
.wbx .node{position:relative;margin-bottom:18px}
.wbx .node:before{content:'';position:absolute;left:-31px;top:6px;width:12px;height:12px;border-radius:50%;background:#3730a3}
.wbx .when{font-size:.82em;color:#5f6b7a}.wbx .card{background:#fafafa;border:1px solid #e2e2e2;border-radius:10px;padding:10px 14px}
.wbx .next{margin-top:6px;font-size:.88em;color:#3730a3;font-weight:600}
.wbx .now{border:2px solid #3730a3;border-radius:10px;padding:12px 16px;margin-top:8px;background:#eef2ff}
.wbx .good{color:#15803d}.wbx .bad{color:#b91c1c}.wbx .chip{display:none}
.wbx .sub{color:#5f6b7a;font-size:.92em}
.wbx .w6{border-left:3px solid #3730a3}.wbx .w6 th{width:64px}
.wbx td.pending{color:#5f6b7a;font-style:italic}
.wbx .xrefs{margin-top:14px;border-top:1px dashed #e2e2e2;padding-top:8px}
.wbx .xref{display:inline-block;border:1px solid #3730a3;color:#3730a3;border-radius:99px;padding:2px 11px;margin:2px 4px 2px 0;font-size:.85em;cursor:pointer;background:#fff}
.wbx .xref:hover{background:#eef2ff}
.wbx .day{margin-bottom:18px}.wbx .day h3{border-bottom:1px solid #e2e2e2;padding-bottom:4px}
.wbx .day li{margin:6px 0}.wbx .day .sm{color:#5f6b7a;font-size:.88em}
.wbx .day a{color:#3730a3;cursor:pointer;text-decoration:none}.wbx .day a:hover{text-decoration:underline}
.wbx svg text{fill:#1a1a1a}
/* 와이드 레이아웃 + 상단 탭 (2026-08-08 지시: 가로로 넓게, 스레드·관계도는 별도 탭) */
.wrap{max-width:1560px !important;padding-left:28px;padding-right:28px}
.wb-tab{border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:10px;
  padding:9px 18px;font-size:.95em;cursor:pointer}
.wb-tab.on{background:var(--blue);border-color:var(--blue);color:#fff;font-weight:600}
#wb-thread .wbx,#wb-map .wbx{margin-top:4px}
body:not([data-wblang=en]) .wbx-en{display:none}
body[data-wblang=en] .wbx-ko{display:none}
</style>"""

# 마인드맵 컬럼(연구 국면) 배치와 컬럼 색 (seaborn deep — 논문 팔레트)
MM_COLS = [
    ("기반 탐색", ["genesis", "vbias", "families", "wcurse", "duel"]),
    ("정합성 검증", ["singlefit", "ladders", "fullfit", "highpower"]),
    ("진단·방법", ["randh", "aqc", "autopsy", "pools", "failpipe", "calql", "v14"]),
    (
        "판정·종합",
        ["v11", "v12", "final", "conservatism", "morning-0808", "morning-0809", "critic-heads", "critic-pfx", "deas"],
    ),
    ("표현·설계", ["phi-ladder", "model-based", "embed-compare", "tdsf-arq"]),
    ("논문·교차", ["papers-value-steering", "papers-tdjepa", "papers-byolg", "papers-dbc", "xworker-0808", "floq"]),
    ("이식·인프라", ["kper", "td-segv", "video-gallery", "flow", "gr1-port", "horizon-probe"]),
]
# seaborn deep — 7 국면 + '신규'(마지막)
MM_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]


def summary_of(body: str) -> str:
    m = re.search(r"<p><b>(?:해석|결과|관측|잠정|근본|Takeaway)[^<]*</b>\s*([\s\S]{0,600}?)</p>", body) or re.search(
        r"<p>([\s\S]{0,600}?)</p>", body
    )
    txt = re.sub(r"<[^>]+>", "", m.group(1) if m else body)
    txt = re.sub(r"\s+", " ", txt).strip()
    return (txt[:180] + "…") if len(txt) > 180 else txt


def build_thread(merged_reports):
    """일자별 포스트 다이제스트 — 어떤 날 무엇이 올라갔는지 한눈에."""
    by_day = {}
    for i, r in enumerate(merged_reports):
        by_day.setdefault(r["date"].split()[0], []).append((i, r))
    days = []
    for d in sorted(by_day, reverse=True):
        entries = sorted(by_day[d], key=lambda x: x[1]["date"], reverse=True)
        items = "".join(
            f"<li><span class='sm'>{([*r['date'].split(), ''])[1]}</span> "
            f"<a onclick='openReport({i});return false'>{r['title']}</a>"
            f"<div class='sm'>{r['summary'][:110]}</div></li>"
            for i, r in entries
        )
        days.append(f"<div class='day'><h3>{d} <span class='sm'>({len(by_day[d])}건)</span></h3><ul>{items}</ul></div>")
    return "<p>일자별로 올라온 리포트 전부(양쪽 워커 포함). 제목을 누르면 해당 리포트로 이동한다.</p>" + "".join(days)


def build_mindmap(eid_idx, summaries=None):
    """인터랙티브 force-directed 관계도 — META의 links에서 자동 생성, 드래그·호버·클릭.

    새 포스트가 META에 links만 채우면 노드·간선이 스스로 자란다. 국면(컬럼) 배정은 초기 x 편향과
    색으로만 쓰이고, 배정 없는 새 eid는 자동으로 '신규' 색을 받는다."""
    cat = {}
    for ci, (_, eids) in enumerate(MM_COLS):
        for e in eids:
            cat[e] = ci
    phases = [c[0] for c in MM_COLS] + ["신규"]
    nodes = []
    for eid, i in eid_idx.items():
        m = mm.META.get(eid, {})
        nodes.append(
            {
                "id": eid,
                "idx": i,
                "cat": cat.get(eid, len(MM_COLS)),
                "what": m.get("what", eid),
                "date": m.get("date", ""),
                "sum": (summaries or {}).get(eid, ""),
            }
        )
    links, seen = [], set()
    for eid, m in mm.META.items():
        for tgt in m.get("links", []):
            if eid in eid_idx and tgt in eid_idx and (tgt, eid) not in seen and (eid, tgt) not in seen:
                seen.add((eid, tgt))
                links.append([eid, tgt])
    data = json.dumps(
        {"nodes": nodes, "links": links, "colors": [*MM_COLORS, "#64b5cd"], "phases": phases}, ensure_ascii=False
    )
    legend = "".join(
        f"<span style='display:inline-block;margin-right:14px'><span style='display:inline-block;width:11px;height:11px;border-radius:50%;background:{([*MM_COLORS, '#64b5cd'])[i]};margin-right:5px'></span>{p}</span>"
        for i, p in enumerate(phases)
    )
    return (
        "<p>실험 리포트의 관계도 — 간선은 각 리포트의 '연결된 리포트'에서 자동 생성된다. "
        "<b>노드를 드래그</b>해 재배치하고, <b>호버</b>로 연결을 강조하고, <b>클릭</b>하면 리포트가 열린다.</p>"
        f"<p class='sub'>{legend}</p>"
        f"<div id='wb-graph' style='width:100%;height:640px;border:1px solid #e2e2e2;border-radius:10px;background:#fff'></div>"
        f"<script id='wb-graph-data' type='application/json'>{data}</script>"
    )


def _entry_payload(coarse_date, eid, title, status, body, en_body):
    """One hub entry. KO and EN are both carried; the page toggles between the two wrappers."""
    meta = mm.META.get(eid, {})
    dual = f'<div class="wbx wbx-ko">{body}</div>'
    if en_body:
        dual += f'<div class="wbx wbx-en">{en_body}</div>'
    else:
        dual += (
            f'<div class="wbx wbx-en"><p class="sub">English version pending — Korean original '
            f"below.</p>{body}</div>"
        )
    return {
        "eid": eid,
        # META carries the real "YYYY-MM-DD HH:MM"; DATE_MAP maps the coarse label
        # ("~08-01") that a few early entries still use as their only date.
        "date": meta.get("date") or DATE_MAP.get(coarse_date, "2026-08-08"),
        "worker": "B",
        "title": f"🤖 [{MARK}] {title}",
        "summary": summary_of(body),
        "tags": [MARK, *meta.get("tags", [])],
        "links": meta.get("links", []),
        "phase": meta.get("phase", "실험"),
        # 완결 -> a result that stands; 살아있음 -> a page that keeps being updated.
        "status": {"완결": "finding", "진행 중": "ongoing", "살아있음": "living"}.get(status, "done"),
        "body_html": wrap_tables(dual),
    }


def wrap_tables(body: str) -> str:
    """Tables go inside a scrollable wrapper — display:block on the table itself folds the rows."""
    body = re.sub(r"<table(?![^>]*tblwrapped)", "<div class='tblwrap'><table", body)
    return body.replace("</table>", "</table></div>")


def main():
    import argparse
    import base64
    import hashlib

    from huggingface_hub import CommitOperationAdd
    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="publish just these eids")
    args = ap.parse_args()

    wanted = set(args.only) if args.only else None
    ours = {}
    for date, eid, title, status, body in mm.ENTRIES:
        if wanted and eid not in wanted:
            continue
        ours[eid] = _entry_payload(date, eid, title, status, body, mm.EN_BODIES.get(eid))
    if wanted and set(wanted) - set(ours):
        raise SystemExit(f"not in ENTRIES: {sorted(set(wanted) - set(ours))}")
    if not ours:
        raise SystemExit("nothing to publish")

    api = HfApi()
    live = json.loads(
        pathlib.Path(hf_hub_download(SPACE, "entries.json", repo_type="space", force_download=True)).read_text()
    )

    # Figures out of the base64 bodies, one directory per entry so a re-publish overwrites its own
    # and nothing else.
    fig_dir = CACHE / "hub_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ops, seen = [], set()

    def extract(eid):
        def sub(m):
            ext, b64 = m.group(1), m.group(2)
            raw = base64.b64decode(b64)
            name = f"figures/{eid}/{hashlib.sha1(raw).hexdigest()[:16]}.{ext}"
            if name not in seen:
                seen.add(name)
                fp = fig_dir / name.replace("/", "_")
                fp.write_bytes(raw)
                ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(fp)))
            return f'src="{name}"'

        return sub

    for eid, e in ours.items():
        e["body_html"] = re.sub(
            r'src=["\']data:image/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/=]+)["\']',
            extract(eid),
            e["body_html"],
        )

    # Replace ours in place (keeping position), append the rest. Everything not ours is untouched.
    merged, replaced = [], set()
    for row in live:
        eid = row.get("eid")
        if eid in ours:
            merged.append(ours[eid])
            replaced.add(eid)
        else:
            merged.append(row)
    added = [e for eid, e in ours.items() if eid not in replaced]
    merged.extend(added)
    merged.sort(key=lambda r: str(r.get("date", "")), reverse=True)

    tmp = CACHE / "entries_new.json"
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    size_mb = tmp.stat().st_size / 1e6
    if size_mb > 9.5:
        raise RuntimeError(f"entries.json {size_mb:.1f}MB — 10MB LFS 경계 초과 위험, 본문을 더 줄여야 함")

    res = api.create_commit(
        repo_id=SPACE,
        repo_type="space",
        operations=[CommitOperationAdd(path_in_repo="entries.json", path_or_fileobj=str(tmp)), *ops],
        commit_message=f"worker-B: {len(ours)} entries [{mm.GIT_STAMP}]",
        create_pr=True,
    )
    if res.pr_url is None:
        print("허브 동기화: 변경 없음 — PR 생략")
        return
    num = int(res.pr_url.rstrip("/").split("/")[-1])
    api.merge_pull_request(SPACE, num, repo_type="space")
    print(
        f"허브 동기화: 교체 {len(replaced)} + 신규 {len(added)} = {len(ours)}, "
        f"그림 {len(ops)}장, 전체 {len(merged)} 엔트리, PR#{num} 머지"
    )


if __name__ == "__main__":
    main()
