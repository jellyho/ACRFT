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
import make_master_report as mm

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
    ("판정·종합", ["v11", "v12", "final", "conservatism", "morning-0808", "morning-0809", "critic-heads"]),
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


def main():
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download

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
    keep = [(r, b) for r, b in zip(reports, blocks, strict=True) if MARK not in json.dumps(r, ensure_ascii=False)]

    # ---- ours from ENTRIES (real per-eid dates from META)
    ours = []
    for date, eid, title, status, body in mm.ENTRIES:
        iso = mm.META.get(eid, {}).get("date") or DATE_MAP.get(date, "2026-08-08")
        en_body = mm.EN_BODIES.get(eid)
        if en_body:
            dual = f'<div class="wbx wbx-ko">{body}</div><div class="wbx wbx-en">{en_body}</div>'
        else:
            dual = (
                f'<div class="wbx wbx-ko">{body}</div>'
                f'<div class="wbx wbx-en"><p class="sub">English version pending — Korean original below.</p>{body}</div>'
            )
        ours.append(
            (
                eid,
                {
                    "date": iso,
                    "title": f"🧪 [{MARK}] {title}",
                    "summary": summary_of(body),
                    "tags": [MARK, "RoboCasa"],
                    "status": "living" if status != "완결" else "finding",
                },
                dual,
            )
        )
    ours.sort(key=lambda x: x[1]["date"], reverse=True)

    # 스레드·마인드맵은 리스트 항목이 아니라 별도 탭(#wb-thread/#wb-map)으로 상주한다.
    merged = [(r, b) for _, r, b in ours] + keep
    eid_idx = {eid: i for i, (eid, _, _) in enumerate(ours)}

    # ---- xref 활성화: data-eid → openReport(idx)
    def activate(body):
        def sub(m):
            e = m.group(1)
            if e in eid_idx:
                return f"<span class='xref' onclick='openReport({eid_idx[e]})'"
            return "<span class='xref' style='opacity:.5'"

        return re.sub(r"<span class='xref' data-eid='([^']+)'", sub, body)

    def wrap_tables(body):
        """표는 스크롤 가능한 래퍼 DIV 안에 — table에 display:block을 주면 행이 접혀버린다."""
        body = re.sub(r"<table(?![^>]*tblwrapped)", "<div class='tblwrap'><table", body)
        return body.replace("</table>", "</table></div>")

    bodies = [wrap_tables(activate(b)) for _, _, b in ours]
    bodies += [b for _, b in keep]

    new_reports = [r for r, _ in merged]

    def rebuild(i, b):
        inner = re.sub(r"^<section[^>]*>", "", b)
        inner = re.sub(r"</section>\n?$", "", inner)
        return f'<section class="report" id="r{i}" hidden>{inner}</section>\n'

    new_blocks = []
    for i, b in enumerate(bodies):
        new_blocks.append(
            rebuild(i, b) if b.startswith("<section") else f'<section class="report" id="r{i}" hidden>{b}</section>\n'
        )

    # ---- assemble
    out = idx[:arr_start] + json.dumps(new_reports, ensure_ascii=False) + idx[arr_end:]
    first = out.find('<section class="report"')
    last_end = max(mm_.end() for mm_ in re.finditer(r"</section>\n?", out))
    out = out[:first] + "".join(new_blocks) + out[last_end:]
    out = re.sub(r'<style id="wbx-style">[\s\S]*?</style>', "", out)
    out = out.replace("</head>", WBX_STYLE + "</head>", 1)
    # 이전 정렬 패치 원복(스레드·마인드맵이 리스트에서 빠졌으므로 불필요)
    out = out.replace(
        '.sort((a,b)=>{const p=t=>t.tags&&t.tags.includes("인덱스")?1:0;'
        "if(p(a)!==p(b))return p(b)-p(a);return b.date.localeCompare(a.date);});",
        ".sort((a,b)=>b.date.localeCompare(a.date));",
        1,
    )

    # ---- 상단 탭 + 상주 뷰(#wb-thread/#wb-map) 주입 (이전 주입분은 마커로 제거 후 재주입)
    out = re.sub(r"<!--wb-tabs-->[\s\S]*?<!--/wb-tabs-->", "", out)
    out = re.sub(r"<!--wb-views-->[\s\S]*?<!--/wb-views-->", "", out)
    out = re.sub(r'<script id="wb-tabs-js">[\s\S]*?</script>', "", out)
    tabs = (
        '<!--wb-tabs--><div id="wb-tabbar" style="display:flex;gap:8px;margin:14px 0">'
        '<button class="wb-tab on" data-v="list" onclick="wbView(\'list\')">📋 리포트 목록</button>'
        '<button class="wb-tab" data-v="thread" onclick="wbView(\'thread\')">🧵 데일리 스레드</button>'
        '<button class="wb-tab" data-v="map" onclick="wbView(\'map\')">🗺️ 관계도</button>'
        '<button id="wb-lang" class="wb-tab" style="margin-left:auto" onclick="wbLang()">EN</button>'
        "</div><!--/wb-tabs-->"
    )
    views = (
        "<!--wb-views-->"
        f'<div id="wb-thread" hidden><div class="wbx">{build_thread([r for r, _ in merged])}</div></div>'
        f'<div id="wb-map" hidden><div class="wbx">{build_mindmap(eid_idx, {e: r["summary"] for e, r, _ in ours})}</div></div>'
        "<!--/wb-views-->"
    )
    js = """<script id="wb-tabs-js">
function wbLang(){
  const b=document.body, to=(b.dataset.wblang==='en')?'ko':'en';
  b.dataset.wblang=to; localStorage.setItem('wblang',to);
  document.getElementById('wb-lang').textContent=(to==='en')?'한국어':'EN';
}
(function(){const s=localStorage.getItem('wblang'); if(s==='en'){document.body.dataset.wblang='en';
  addEventListener('DOMContentLoaded',()=>{const e=document.getElementById('wb-lang'); if(e) e.textContent='한국어';});}})();
function wbView(v){
  document.querySelectorAll('.wb-tab').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
  const listy=['list','count','chips'];
  listy.forEach(id=>{const e=document.getElementById(id); if(e) e.hidden=(v!=='list');});
  document.querySelectorAll('#home .controls').forEach(e=>e.hidden=(v!=='list'));
  document.getElementById('wb-thread').hidden=(v!=='thread');
  document.getElementById('wb-map').hidden=(v!=='map');
  if(v==='map') wbGraphInit();
}
// ---- 인터랙티브 관계도: 자체 force-directed 시뮬레이션 (드래그·호버·클릭)
let _wbG=null;
function wbGraphInit(){
  if(_wbG) return; const box=document.getElementById('wb-graph'); if(!box) return;
  const D=JSON.parse(document.getElementById('wb-graph-data').textContent);
  const Wd=box.clientWidth||1200, H=box.clientHeight||640, NS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(NS,'svg'); svg.setAttribute('width','100%'); svg.setAttribute('height','100%');
  box.style.position='relative'; box.appendChild(svg);
  const tip=document.createElement('div'); tip.id='wb-tip';
  tip.style.cssText='position:absolute;display:none;max-width:300px;background:#fff;border:1px solid #cfd4dd;'+
    'border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.14);padding:10px 13px;font-size:.85em;'+
    'line-height:1.5;color:#1a1a1a;pointer-events:none;z-index:10';
  box.appendChild(tip);
  const ncat=D.phases.length;
  const N=D.nodes.map((n,i)=>({...n,
    x:(Wd*0.12)+(n.cat+0.5)*(Wd*0.76)/ncat+40*(Math.random()-0.5),
    y:H*0.18+H*0.64*Math.random(), vx:0, vy:0}));
  const byId={}; N.forEach(n=>byId[n.id]=n);
  const L=D.links.map(([a,b])=>({a:byId[a],b:byId[b]})).filter(l=>l.a&&l.b);
  const adj={}; L.forEach(l=>{(adj[l.a.id]=adj[l.a.id]||new Set()).add(l.b.id);(adj[l.b.id]=adj[l.b.id]||new Set()).add(l.a.id);});
  const eEls=L.map(l=>{const p=document.createElementNS(NS,'line');
    p.setAttribute('stroke',D.colors[l.a.cat]); p.setAttribute('stroke-width','1.3'); p.setAttribute('opacity','.38');
    svg.appendChild(p); return p;});
  const nEls=N.map(n=>{
    const g=document.createElementNS(NS,'g'); g.style.cursor='grab';
    const c=document.createElementNS(NS,'circle');
    c.setAttribute('r', 14+3*((adj[n.id]&&adj[n.id].size)||0));
    c.setAttribute('fill','#fff'); c.setAttribute('stroke',D.colors[n.cat]); c.setAttribute('stroke-width','2.4');
    const t=document.createElementNS(NS,'text'); t.textContent=n.id;
    t.setAttribute('text-anchor','middle'); t.setAttribute('dy','-1.4em');
    t.setAttribute('font-size','12'); t.setAttribute('font-weight','600'); t.setAttribute('fill','#1a1a1a');
    g.appendChild(c); g.appendChild(t); svg.appendChild(g);
    g.addEventListener('mouseenter',()=>{
      eEls.forEach((e,i)=>e.setAttribute('opacity',(L[i].a===n||L[i].b===n)?'0.95':'0.08'));
      nEls.forEach((m,i)=>m.g.setAttribute('opacity',(N[i]===n||(adj[n.id]&&adj[n.id].has(N[i].id)))?'1':'0.25'));
      tip.innerHTML='<b>'+n.id+'</b> <span style="color:#5f6b7a">'+n.date+'</span>'+
        '<div style="margin-top:3px">'+n.what+'</div>'+
        (n.sum?'<div style="margin-top:5px;color:#5f6b7a">'+n.sum+'</div>':'');
      tip.style.display='block';});
    g.addEventListener('mousemove',ev=>{const r=box.getBoundingClientRect();
      let tx=ev.clientX-r.left+16, ty=ev.clientY-r.top+14;
      if(tx+310>r.width) tx=ev.clientX-r.left-316; if(ty+140>r.height) ty=ev.clientY-r.top-120;
      tip.style.left=tx+'px'; tip.style.top=ty+'px';});
    g.addEventListener('mouseleave',()=>{
      eEls.forEach(e=>e.setAttribute('opacity','.38')); nEls.forEach(m=>m.g.setAttribute('opacity','1'));
      tip.style.display='none';});
    return {g,c,t,n};});
  let drag=null, moved=0, alpha=1;
  nEls.forEach(({g,n})=>{
    g.addEventListener('pointerdown',ev=>{drag={n,dx:n.x-ev.clientX,dy:n.y-ev.clientY}; moved=0; alpha=Math.max(alpha,.35); g.setPointerCapture(ev.pointerId);});
    g.addEventListener('pointermove',ev=>{if(!drag||drag.n!==n)return; n.x=ev.clientX+drag.dx; n.y=ev.clientY+drag.dy; n.vx=n.vy=0; moved++; alpha=Math.max(alpha,.3);});
    g.addEventListener('pointerup',()=>{if(moved<4) openReport(n.idx); drag=null;});});
  function tick(){
    if(alpha>0.005){
      for(let i=0;i<N.length;i++) for(let j=i+1;j<N.length;j++){
        const a=N[i],b=N[j]; let dx=b.x-a.x,dy=b.y-a.y; let d2=dx*dx+dy*dy||1; if(d2<40000){
          const f=1800/d2; const d=Math.sqrt(d2); dx/=d; dy/=d;
          a.vx-=f*dx; a.vy-=f*dy; b.vx+=f*dx; b.vy+=f*dy;}}
      L.forEach(({a,b})=>{const dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)||1,f=0.012*(d-130);
        a.vx+=f*dx/d; a.vy+=f*dy/d; b.vx-=f*dx/d; b.vy-=f*dy/d;});
      N.forEach(n=>{ n.vx+=0.004*((Wd*0.12)+(n.cat+0.5)*(Wd*0.76)/ncat-n.x); n.vy+=0.002*(H/2-n.y);
        if(drag&&drag.n===n) return;
        n.x+=n.vx*=0.86; n.y+=n.vy*=0.86;
        n.x=Math.max(30,Math.min(Wd-30,n.x)); n.y=Math.max(34,Math.min(H-24,n.y));});
      alpha*=0.985;}
    eEls.forEach((e,i)=>{e.setAttribute('x1',L[i].a.x);e.setAttribute('y1',L[i].a.y);e.setAttribute('x2',L[i].b.x);e.setAttribute('y2',L[i].b.y);});
    nEls.forEach(({g,n})=>g.setAttribute('transform','translate('+n.x+','+n.y+')'));
    requestAnimationFrame(tick);}
  _wbG=true; tick();
}
// ---- 떠 있는 내비: 리포트 열람 중 목록으로/맨위로 즉시 이동
(function(){
  function ensure(){
    if(document.getElementById('wb-float')) return;
    const box=document.createElement('div'); box.id='wb-float';
    const home=document.createElement('button'); home.textContent='☰'; home.title='목록으로';
    home.onclick=function(){ if(typeof goHome==='function') goHome(); else location.hash=''; window.scrollTo(0,0); };
    const top=document.createElement('button'); top.textContent='↑'; top.title='맨 위로';
    top.onclick=function(){ window.scrollTo({top:0,behavior:'smooth'}); };
    box.appendChild(home); box.appendChild(top); document.body.appendChild(box);
  }
  function inReader(){ const r=document.getElementById('reader'); return r && !r.hidden; }
  function upd(){ ensure(); const f=document.getElementById('wb-float');
    f.style.display=(inReader() && window.scrollY>320)?'flex':'none'; }
  addEventListener('scroll',upd,{passive:true});
  addEventListener('click',()=>setTimeout(upd,60));
  addEventListener('DOMContentLoaded',()=>{ensure();upd();});
  ensure();
})();
</script>"""
    m = re.search(r'(<div class="sub">[^<]*</div>)', out)
    out = out[: m.end()] + tabs + out[m.end() :]
    i_list = out.find('<div id="list"></div>')
    out = out[: i_list + len('<div id="list"></div>')] + views + out[i_list + len('<div id="list"></div>') :]
    out = out.replace("</body>", js + "</body>", 1)

    # HF auto-tracks files >10MB as LFS, and static Spaces serve LFS files as downloads instead of
    # rendering them — so inline base64 figures are extracted to figs/<sha>.png and referenced by
    # relative path, keeping index.html a small regular-git blob.
    import base64
    import hashlib

    fig_dir = pathlib.Path("/scratch/jellyho/acrft/hub_figs")
    fig_dir.mkdir(exist_ok=True)
    fig_ops = []
    seen_figs = set()

    def extract_fig(m):
        ext, b64 = m.group(1), m.group(2)
        raw = base64.b64decode(b64)
        name = f"figs/{hashlib.sha1(raw).hexdigest()[:16]}.{ext}"
        if name not in seen_figs:
            seen_figs.add(name)
            fp = fig_dir / name.split("/")[1]
            fp.write_bytes(raw)
            fig_ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(fp)))
        return f'src="{name}"'

    out = re.sub(r'src="data:image/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/=]+)"', extract_fig, out)

    tmp = pathlib.Path("/scratch/jellyho/acrft/hub_index_new.html")
    tmp.write_text(out)
    size_mb = tmp.stat().st_size / 1e6
    if size_mb > 9.5:
        raise RuntimeError(f"index.html {size_mb:.1f}MB — 10MB LFS 경계 초과 위험, 본문을 더 줄여야 함")
    # drop the stale LFS rule for index.html so re-uploads return to regular git storage
    attr_path = hf_hub_download(SPACE, ".gitattributes", repo_type="space")
    attrs = pathlib.Path(attr_path).read_text()
    ops_extra = []
    if "index.html filter=lfs" in attrs:
        attrs = "\n".join(line for line in attrs.splitlines() if not line.startswith("index.html ")) + "\n"
        atmp = pathlib.Path("/scratch/jellyho/acrft/hub_gitattributes")
        atmp.write_text(attrs)
        ops_extra.append(CommitOperationAdd(path_in_repo=".gitattributes", path_or_fileobj=str(atmp)))
    res = api.create_commit(
        repo_id=SPACE,
        repo_type="space",
        operations=[CommitOperationAdd(path_in_repo="index.html", path_or_fileobj=str(tmp)), *ops_extra, *fig_ops],
        commit_message=f"worker-B: {len(ours)} entries [{mm.GIT_STAMP}]",
        create_pr=True,
    )
    if res.pr_url is None:
        print("허브 동기화: 변경 없음 — PR 생략")
        return
    num = int(res.pr_url.rstrip("/").split("/")[-1])
    api.merge_pull_request(SPACE, num, repo_type="space")
    print(f"허브 동기화: 스레드+마인드맵+우리 {len(ours)} + 기존 {len(keep)} = {len(merged)} 엔트리, PR#{num} 머지")


if __name__ == "__main__":
    main()
