import json
import pathlib
import re
import sys

sys.path.insert(0, "slurm")
import make_master_report as mm

EID = sys.argv[1]
STATUS = {"완결": "finding", "진행 중": "ongoing", "살아있음": "living"}
# extra tag chips (for the tag bar) per eid
TAGS = {
    "mb-arq": ["표현", "MB-AC", "설명"],
    "exp-board": ["실험", "보드"],
    "aqc-ablation": ["ablation", "OGBench", "AQC"],
    "theory-preexp": ["이론", "사전등록", "논문"],
}
# mindmap column (worker A's phase taxonomy) per eid — the client-side mindmap groups by this
PHASE_MAP = {
    "genesis": "기반 탐색",
    "vbias": "기반 탐색",
    "families": "기반 탐색",
    "wcurse": "기반 탐색",
    "duel": "기반 탐색",
    "singlefit": "정합성 검증",
    "ladders": "정합성 검증",
    "fullfit": "정합성 검증",
    "highpower": "정합성 검증",
    "randh": "진단·방법",
    "aqc": "진단·방법",
    "autopsy": "진단·방법",
    "pools": "진단·방법",
    "failpipe": "진단·방법",
    "calql": "진단·방법",
    "v14": "진단·방법",
    "v11": "판정·종합",
    "v12": "판정·종합",
    "final": "판정·종합",
    "conservatism": "판정·종합",
    "morning-0808": "판정·종합",
    "morning-0809": "판정·종합",
    "critic-heads": "판정·종합",
    "critic-pfx": "판정·종합",
    "deas": "판정·종합",
    "phi-ladder": "표현·설계",
    "model-based": "표현·설계",
    "embed-compare": "표현·설계",
    "tdsf-arq": "표현·설계",
    "mb-arq": "표현·설계",
    "papers-value-steering": "논문·교차",
    "papers-tdjepa": "논문·교차",
    "papers-byolg": "논문·교차",
    "papers-dbc": "논문·교차",
    "xworker-0808": "논문·교차",
    "floq": "논문·교차",
    "kper": "이식·인프라",
    "td-segv": "이식·인프라",
    "video-gallery": "이식·인프라",
    "flow": "이식·인프라",
    "gr1-port": "이식·인프라",
    "horizon-probe": "이식·인프라",
    "exp-board": "이식·인프라",
    "aqc-ablation": "판정·종합",
    "paper-intro": "논문·교차",
    "papers-tier1": "논문·교차",
    "tier1-intros": "논문·교차",
    "task-scan": "판정·종합",
    "theory-preexp": "논문·교차",
}


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
    "tags": ["워커B", "RoboCasa", *TAGS.get(eid, [])],
    "phase": PHASE_MAP.get(eid, "신규"),  # mindmap column
    "links": mm.META.get(eid, {}).get("links", []),  # mindmap edges (semantic connections)
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
