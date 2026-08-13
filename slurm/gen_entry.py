import json
import pathlib
import re
import sys

sys.path.insert(0, "slurm")
import make_master_report as mm

EID = sys.argv[1]
STATUS = {"완결": "finding", "진행 중": "ongoing", "살아있음": "living"}
PHASE = {"mb-arq": ["표현", "MB-AC", "설명"], "exp-board": ["실험", "보드"]}


def wrap(b):
    b = re.sub(r"<table(?![^>]*tblwrapped)", "<div class='tblwrap'><table", b)
    return b.replace("</table>", "</table></div>")


def summ(b):
    m = re.search(r"<p class='sub'>([\s\S]{0,400}?)</p>", b) or re.search(r"<p>([\s\S]{0,400}?)</p>", b)
    t = re.sub(r"<[^>]+>", "", m.group(1) if m else b)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:180] + "…") if len(t) > 180 else t


row = next(e for e in mm.ENTRIES if e[1] == EID)
date, eid, title, status, body = row
en = mm.EN_BODIES.get(eid)
dual = f'<div class="wbx wbx-ko">{body}</div>' + (f'<div class="wbx wbx-en">{en}</div>' if en else "")
entry = {
    "eid": eid,
    "worker": "B",
    "date": mm.META.get(eid, {}).get("date", date),
    "title": f"🧪 [워커B] {title}",
    "status": STATUS.get(status, "finding"),
    "summary": summ(body),
    "tags": ["워커B", "RoboCasa", *PHASE.get(eid, [])],
    "body_html": wrap(dual),
}
out = pathlib.Path(f"/scratch/jellyho/acrft/entry_{eid}.json")
out.write_text(json.dumps(entry, ensure_ascii=False))
print(
    "wrote",
    out,
    "| status",
    entry["status"],
    "| tags",
    entry["tags"],
    "| body",
    len(entry["body_html"]),
    "| summary:",
    entry["summary"][:80],
)
