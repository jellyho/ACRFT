"""Sync worker-B experiments into the shared HF Space hub — one blog entry PER experiment.

Reads the experiment ENTRIES from make_master_report (single source of truth), converts each into
the hub's native format (REPORTS array entry + inline <section> with the full body), replaces any
previous worker-B entries, keeps the other worker's entries untouched, and opens+merges a PR.

    uv run --no-sync python slurm/sync_hub.py
"""

import html
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import make_master_report as mm  # noqa: E402  (importing builds ENTRIES and regenerates local reports)

SPACE = "jellyho/acrft-reports"
MARK = "워커B"
DATE_MAP = {"~08-01": "2026-08-01", "08-02~03": "2026-08-03", "08-05": "2026-08-05",
            "08-06": "2026-08-06", "08-07": "2026-08-07", "타임라인": "2026-08-07"}
WBX_STYLE = """<style id="wbx-style">
.wbx{line-height:1.65}.wbx table{border-collapse:collapse;font-size:.9em;margin:10px 0;max-width:100%;background:var(--card2)}
.wbx td,.wbx th{border:1px solid var(--line);padding:5px 10px;text-align:left;vertical-align:top}
.wbx th{background:var(--card)}.wbx img{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:8px 0}
.wbx h2,.wbx h3{margin:.6em 0 .4em}.wbx .missing{background:#3a3320;padding:6px 10px;border-radius:6px}
.wbx code{background:var(--card);padding:1px 5px;border-radius:4px}
.wbx .tl{position:relative;margin:14px 0 14px 8px;border-left:3px solid var(--blue);padding-left:22px}
.wbx .node{position:relative;margin-bottom:18px}
.wbx .node:before{content:'';position:absolute;left:-31px;top:6px;width:12px;height:12px;border-radius:50%;background:var(--blue)}
.wbx .when{font-size:.82em;color:var(--muted)}.wbx .card{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.wbx .next{margin-top:6px;font-size:.88em;color:var(--blue);font-weight:600}
.wbx .now{border:2px solid var(--blue);border-radius:10px;padding:12px 16px;margin-top:8px}
.wbx .good{color:var(--green)}.wbx .bad{color:var(--red)}.wbx .chip{display:none}
</style>"""


def summary_of(body: str) -> str:
    m = re.search(r"<p><b>(?:해석|결과|관측|Takeaway)[^<]*</b>\s*([\s\S]{0,600}?)</p>", body) or re.search(
        r"<p>([\s\S]{0,600}?)</p>", body)
    txt = re.sub(r"<[^>]+>", "", m.group(1) if m else body)
    txt = re.sub(r"\s+", " ", txt).strip()
    return (txt[:180] + "…") if len(txt) > 180 else txt


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    idx = pathlib.Path(hf_hub_download(SPACE, "index.html", repo_type="space", force_download=True)).read_text()

    # ---- parse REPORTS array
    i0 = idx.find("const REPORTS = ")
    arr_start = idx.find("[", i0)
    reports, arr_end_rel = json.JSONDecoder().raw_decode(idx[arr_start:])
    arr_end = arr_start + arr_end_rel
    # ---- collect existing sections in order
    sec_pat = re.compile(r'<section class="report" id="r(\d+)"[\s\S]*?</section>\n?')
    sections = sec_pat.findall(idx)
    sec_blocks = sec_pat.finditer(idx)
    blocks = [m.group(0) for m in sec_blocks]
    assert len(blocks) == len(reports), f"섹션 {len(blocks)} vs 엔트리 {len(reports)} 불일치"
    # ---- drop previous worker-B pairs
    keep = [(r, b) for r, b in zip(reports, blocks) if MARK not in json.dumps(r, ensure_ascii=False)]

    # ---- build ours from ENTRIES
    ours = []
    for date, eid, title, status, body in mm.ENTRIES:
        iso = DATE_MAP.get(date, "2026-08-07")
        ours.append((
            {"date": iso, "title": f"🧪 [{MARK}] {title}", "summary": summary_of(body),
             "tags": [MARK, date, "RoboCasa"], "status": "living" if status != "완결" else "finding"},
            f'<div class="wbx">{body}</div>',
        ))
    ours.sort(key=lambda x: x[0]["date"], reverse=True)

    merged = ours + keep  # 우리(날짜 역순) 먼저, 그다음 기존 유지
    new_reports = [r for r, _ in merged]
    new_blocks = [f'<section class="report" id="r{i}" hidden>{re.sub(chr(60) + "section[^>]*>|</section>", "", b) if MARK not in json.dumps(r, ensure_ascii=False) else b}</section>'
                  for i, (r, b) in enumerate(merged)]
    # 위 한 줄이 기존 블록의 래퍼를 벗겨 재래핑: 기존 블록은 <section ...>내용</section> 형태
    def rebuild(i, r, b):
        inner = re.sub(r'^<section[^>]*>', "", b)
        inner = re.sub(r"</section>\n?$", "", inner)
        return f'<section class="report" id="r{i}" hidden>{inner}</section>\n'
    new_blocks = [rebuild(i, r, b) for i, (r, b) in enumerate(merged)]

    # ---- assemble new index
    out = idx[:arr_start] + json.dumps(new_reports, ensure_ascii=False) + idx[arr_end:]
    # 섹션 전부 교체: 첫 섹션 시작부터 마지막 섹션 끝까지를 새 블록으로
    first = out.find('<section class="report"')
    last_m = None
    for last_m in re.finditer(r'</section>\n?', out):
        pass
    out = out[:first] + "".join(new_blocks) + out[last_m.end():]
    if 'id="wbx-style"' not in out:
        out = out.replace("</head>", WBX_STYLE + "</head>", 1)

    tmp = pathlib.Path("/scratch/jellyho/acrft/hub_index_new.html")
    tmp.write_text(out)
    res = api.create_commit(repo_id=SPACE, repo_type="space",
        operations=[CommitOperationAdd(path_in_repo="index.html", path_or_fileobj=str(tmp))],
        commit_message=f"worker-B: {len(ours)} per-experiment blog entries (replaces single embed)",
        create_pr=True)
    num = int(res.pr_url.rstrip("/").split("/")[-1])
    api.merge_pull_request(SPACE, num, repo_type="space")
    print(f"허브 동기화: 우리 {len(ours)}개 + 기존 {len(keep)}개 = {len(merged)} 엔트리, PR#{num} 머지")


if __name__ == "__main__":
    main()
