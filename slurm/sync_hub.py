"""Sync worker-B experiments into the shared HF Space hub — one blog entry PER experiment.

Reads the experiment ENTRIES + META (5W1H, real dates, cross-links) from make_master_report
(single source of truth), converts each into the hub's native format, and adds two pinned
overview entries: a daily THREAD digest (what was posted each day, at a glance) and a MINDMAP
(the experiments as a linked graph). Replaces previous worker-B entries, keeps the other
worker's entries untouched, and opens+merges a PR.

    uv run --no-sync python slurm/sync_hub.py
"""

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import make_master_report as mm  # noqa: E402  (importing builds ENTRIES and regenerates local reports)

SPACE = "jellyho/acrft-reports"
MARK = "워커B"
DATE_MAP = {"~08-01": "2026-08-01", "08-02~03": "2026-08-03", "08-05": "2026-08-05",
            "08-06": "2026-08-06", "08-07": "2026-08-07", "타임라인": "2026-08-08"}
WBX_STYLE = """<style id="wbx-style">
/* 워커B 리포트 = 흰 종이 카드 (프로페셔널·깔끔, 허브 테마와 무관하게 백색 지면) */
.wbx{line-height:1.68;background:#ffffff;color:#1a1a1a;border:1px solid #e2e2e2;border-radius:12px;
  padding:26px 30px;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.wbx a{color:#3730a3}
.wbx h2,.wbx h3{margin:.7em 0 .4em;letter-spacing:-.01em;color:#111;font-family:Georgia,'Times New Roman',serif}
.wbx table{border-collapse:collapse;font-size:.9em;margin:10px 0;background:#fff;color:#1a1a1a;
  display:block;width:max-content;max-width:100%;overflow-x:auto}
.wbx td,.wbx th{border:1px solid #e2e2e2;padding:5px 10px;text-align:left;vertical-align:top}
.wbx th{background:#f3f4f8}.wbx img{max-width:100%;border:1px solid #e2e2e2;border-radius:8px;margin:8px 0;background:#fff}
.wbx .missing{background:#fef9c3;color:#713f12;padding:6px 10px;border-radius:6px}
.wbx code{background:#f3f4f8;padding:1px 5px;border-radius:4px;color:#1a1a1a}
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
</style>"""

# 마인드맵 컬럼(연구 국면) 배치와 컬럼 색 (seaborn deep — 논문 팔레트)
MM_COLS = [
    ("기반 탐색", ["genesis", "vbias", "families", "wcurse", "duel"]),
    ("정합성 검증", ["singlefit", "ladders", "fullfit", "highpower"]),
    ("진단·방법", ["randh", "aqc", "autopsy", "pools", "failpipe"]),
    ("판정 캠페인", ["v11", "v12", "final"]),
    ("확장·인프라", ["kper", "td-segv", "video-gallery", "flow"]),
]
MM_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def summary_of(body: str) -> str:
    m = re.search(r"<p><b>(?:해석|결과|관측|잠정|근본|Takeaway)[^<]*</b>\s*([\s\S]{0,600}?)</p>", body) or re.search(
        r"<p>([\s\S]{0,600}?)</p>", body)
    txt = re.sub(r"<[^>]+>", "", m.group(1) if m else body)
    txt = re.sub(r"\s+", " ", txt).strip()
    return (txt[:180] + "…") if len(txt) > 180 else txt


def build_thread(merged_reports):
    """일자별 포스트 다이제스트 — 어떤 날 무엇이 올라갔는지 한눈에."""
    by_day = {}
    for i, r in enumerate(merged_reports):
        by_day.setdefault(r["date"], []).append((i, r))
    days = []
    for d in sorted(by_day, reverse=True):
        items = "".join(
            f"<li><a onclick='openReport({i});return false'>{r['title']}</a>"
            f"<div class='sm'>{r['summary'][:110]}</div></li>"
            for i, r in by_day[d]
        )
        days.append(f"<div class='day'><h3>{d} <span class='sm'>({len(by_day[d])}건)</span></h3><ul>{items}</ul></div>")
    return (
        "<p>일자별로 올라온 리포트 전부(양쪽 워커 포함). 제목을 누르면 해당 리포트로 이동한다.</p>"
        + "".join(days)
    )


def build_mindmap(eid_idx):
    """실험 리포트를 국면 컬럼 × 연결 간선의 그래프로 — 클릭하면 해당 리포트로 이동."""
    W, COLW, NH, TOP = 1180, 236, 44, 64
    pos = {}
    height = TOP + 30 + max(len(c[1]) for c in MM_COLS) * (NH + 22)
    nodes, edges = [], []
    for ci, (label, eids) in enumerate(MM_COLS):
        x = 14 + ci * COLW
        nodes.append(f"<text x='{x + 100}' y='34' text-anchor='middle' font-size='15' font-weight='700'>{label}</text>")
        nodes.append(f"<line x1='{x}' y1='46' x2='{x + 200}' y2='46' stroke='{MM_COLORS[ci]}' stroke-width='2.5'/>")
        for ni, eid in enumerate(eids):
            if eid not in eid_idx:
                continue
            y = TOP + ni * (NH + 22)
            pos[eid] = (x, y, ci)
            title = mm.META[eid]["what"] if eid in mm.META else eid
            short = title if len(title) <= 17 else title[:16] + "…"
            nodes.append(
                f"<g style='cursor:pointer' onclick='openReport({eid_idx[eid]})'>"
                f"<rect x='{x}' y='{y}' width='200' height='{NH}' rx='9' fill='none' stroke='{MM_COLORS[ci]}' stroke-width='1.6'/>"
                f"<text x='{x + 100}' y='{y + 19}' text-anchor='middle' font-size='12.5' font-weight='600'>{eid}</text>"
                f"<text x='{x + 100}' y='{y + 35}' text-anchor='middle' font-size='10' opacity='.75'>{short}</text></g>"
            )
    seen = set()
    for eid, m in mm.META.items():
        for tgt in m.get("links", []):
            if eid not in pos or tgt not in pos or (tgt, eid) in seen:
                continue
            seen.add((eid, tgt))
            (x1, y1, c1), (x2, y2, c2) = pos[eid], pos[tgt]
            if c1 == c2:
                xa, ya, xb, yb = x1 + 200, y1 + NH / 2, x2 + 200, y2 + NH / 2
                path = f"M{xa},{ya} C{xa + 26},{ya} {xb + 26},{yb} {xb},{yb}"
            else:
                if c1 > c2:
                    (x1, y1, c1), (x2, y2, c2) = (x2, y2, c2), (x1, y1, c1)
                xa, ya, xb, yb = x1 + 200, y1 + NH / 2, x2, y2 + NH / 2
                path = f"M{xa},{ya} C{(xa + xb) / 2},{ya} {(xa + xb) / 2},{yb} {xb},{yb}"
            edges.append(f"<path d='{path}' fill='none' stroke='{MM_COLORS[c1]}' stroke-width='1.1' opacity='.45'/>")
    svg = (
        f"<div style='overflow-x:auto'><svg viewBox='0 0 {W} {height}' width='100%' "
        f"style='min-width:980px;color:inherit'>{''.join(edges)}{''.join(nodes)}</svg></div>"
    )
    return (
        "<p>실험 리포트의 관계도. 컬럼 = 연구 국면(시간순), 간선 = '연결된 리포트' 관계, 노드 클릭 = 해당 리포트로 이동.</p>"
        + svg
    )


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    idx = pathlib.Path(hf_hub_download(SPACE, "index.html", repo_type="space", force_download=True)).read_text()

    # ---- parse REPORTS array
    i0 = idx.find("const REPORTS = ")
    arr_start = idx.find("[", i0)
    reports, arr_end_rel = json.JSONDecoder().raw_decode(idx[arr_start:])
    arr_end = arr_start + arr_end_rel
    sec_pat = re.compile(r'<section class="report" id="r(\d+)"[\s\S]*?</section>\n?')
    blocks = [m.group(0) for m in sec_pat.finditer(idx)]
    assert len(blocks) == len(reports), f"섹션 {len(blocks)} vs 엔트리 {len(reports)} 불일치"
    keep = [(r, b) for r, b in zip(reports, blocks) if MARK not in json.dumps(r, ensure_ascii=False)]

    # ---- ours from ENTRIES (real per-eid dates from META)
    ours = []
    for date, eid, title, status, body in mm.ENTRIES:
        iso = mm.META.get(eid, {}).get("date") or DATE_MAP.get(date, "2026-08-08")
        ours.append((eid, {
            "date": iso, "title": f"🧪 [{MARK}] {title}", "summary": summary_of(body),
            "tags": [MARK, "RoboCasa"], "status": "living" if status != "완결" else "finding",
        }, f'<div class="wbx">{body}</div>'))
    ours.sort(key=lambda x: x[1]["date"], reverse=True)

    latest = max(o[1]["date"] for o in ours)
    # 자리 확보: 0=스레드, 1=마인드맵, 2.. = 우리+기존
    thread_r = {"date": latest, "title": f"🧵 [{MARK}] 데일리 스레드 — 일자별 포스트 한눈에",
                "summary": "매일 어떤 리포트가 올라갔는지 날짜별 다이제스트. 제목 클릭으로 이동.",
                "tags": [MARK, "인덱스"], "status": "living"}
    mindmap_r = {"date": latest, "title": f"🗺️ [{MARK}] 실험 마인드맵 — 리포트 관계도",
                 "summary": "연구 국면 5컬럼 × 상호 연결 간선의 그래프 뷰. 노드 클릭으로 이동.",
                 "tags": [MARK, "인덱스"], "status": "living"}

    merged = [(thread_r, None), (mindmap_r, None)] + [(r, b) for _, r, b in ours] + keep
    eid_idx = {eid: 2 + i for i, (eid, _, _) in enumerate(ours)}

    # ---- xref 활성화: data-eid → openReport(idx)
    def activate(body):
        def sub(m):
            e = m.group(1)
            if e in eid_idx:
                return f"<span class='xref' onclick='openReport({eid_idx[e]})'"
            return "<span class='xref' style='opacity:.5'"
        return re.sub(r"<span class='xref' data-eid='([^']+)'", sub, body)

    bodies = [f'<div class="wbx">{build_thread([r for r, _ in merged])}</div>',
              f'<div class="wbx">{build_mindmap(eid_idx)}</div>']
    bodies += [activate(b) for _, _, b in ours]
    bodies += [b for _, b in keep]

    new_reports = [r for r, _ in merged]

    def rebuild(i, b):
        inner = re.sub(r"^<section[^>]*>", "", b)
        inner = re.sub(r"</section>\n?$", "", inner)
        return f'<section class="report" id="r{i}" hidden>{inner}</section>\n'

    new_blocks = []
    for i, b in enumerate(bodies):
        new_blocks.append(rebuild(i, b) if b.startswith("<section") else f'<section class="report" id="r{i}" hidden>{b}</section>\n')

    # ---- assemble
    out = idx[:arr_start] + json.dumps(new_reports, ensure_ascii=False) + idx[arr_end:]
    first = out.find('<section class="report"')
    last_m = None
    for last_m in re.finditer(r"</section>\n?", out):
        pass
    out = out[:first] + "".join(new_blocks) + out[last_m.end():]
    out = re.sub(r'<style id="wbx-style">[\s\S]*?</style>', "", out)
    out = out.replace("</head>", WBX_STYLE + "</head>", 1)

    # 인덱스(스레드·마인드맵) 엔트리를 홈 리스트 최상단에 고정 — 날짜 정렬보다 우선.
    old_sort = ".sort((a,b)=>b.date.localeCompare(a.date));"
    new_sort = (".sort((a,b)=>{const p=t=>t.tags&&t.tags.includes(\"인덱스\")?1:0;"
                "if(p(a)!==p(b))return p(b)-p(a);return b.date.localeCompare(a.date);});")
    if new_sort not in out:
        out = out.replace(old_sort, new_sort, 1)

    tmp = pathlib.Path("/scratch/jellyho/acrft/hub_index_new.html")
    tmp.write_text(out)
    res = api.create_commit(repo_id=SPACE, repo_type="space",
        operations=[CommitOperationAdd(path_in_repo="index.html", path_or_fileobj=str(tmp))],
        commit_message=f"worker-B: {len(ours)} entries + thread digest + mindmap (5W1H headers, cross-links, real dates)",
        create_pr=True)
    num = int(res.pr_url.rstrip("/").split("/")[-1])
    api.merge_pull_request(SPACE, num, repo_type="space")
    print(f"허브 동기화: 스레드+마인드맵+우리 {len(ours)} + 기존 {len(keep)} = {len(merged)} 엔트리, PR#{num} 머지")


if __name__ == "__main__":
    main()
