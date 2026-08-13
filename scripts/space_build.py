"""Build the fixed, data-driven index.html for the AC-RFT reports Space — by MINIMAL SURGERY on the
existing hub, so every feature (chronological feed, daily grouping, tag chips + search, KO/EN toggle,
the xref mind-map / connections graph, and all body styling) is preserved exactly.

The only change vs the live hub: the report DATA (the ``REPORTS`` metadata array and the inline
``<section class="report" id="rN">`` bodies) is loaded from an external ``entries.json`` at runtime
instead of being baked into the HTML. Plus a white/light palette override. Workers then only ever
append to entries.json; this template never changes.

    uv run python scripts/space_build.py --out space_v2 [--from index_legacy_backup.html] [--preview]
"""

import argparse
import json
import pathlib
import re

from huggingface_hub import hf_hub_download

SPACE = "jellyho/acrft-reports"

WHITE_OVERRIDE = """
/* --- white/light palette override (worker-A) --- */
:root{--bg:#ffffff;--card:#f9f9f7;--card2:#f2f1ec;--text:#0b0b0b;--muted:#6b6560;--line:#e1e0d9;
  --green:#008300;--red:#c0392b;--yellow:#c98a00;--blue:#2a78d6;--purple:#7c3aed}
body{background:#ffffff}
/* MathJax display equations: journal-style, centered, black, breathing room */
mjx-container[display="true"]{margin:1.05em 0!important;overflow-x:auto;overflow-y:hidden}
mjx-container{color:var(--text)}
"""

# Real math typesetting (MathJax v3, SVG so glyphs are self-contained & inherit text colour).
# Only \(...\) and \[...\] are active delimiters — never $...$, so existing worker bodies are untouched.
MATHJAX_HEAD = (
    "<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],"
    "tags:'none'},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code'],"
    "ignoreHtmlClass:'no-mathjax'},svg:{fontCache:'global'}};</script>"
    '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>'
)

_INJECT = (
    "D.sort(function(a,b){return (b.date||'').localeCompare(a.date||'');});"
    # keep each entry's stable eid alongside the feed metadata (used for eid-based navigation)
    "REPORTS=D.map(function(e){return{eid:e.eid,date:e.date,title:e.title,summary:e.summary,tags:e.tags,status:e.status};});"
    "window.EID2I={};REPORTS.forEach(function(r,i){if(r.eid)window.EID2I[r.eid]=i;});"
    "var rd=document.getElementById('reader');"
    "D.forEach(function(e,i){var s=document.createElement('section');s.className='report';"
    # DOM id stays r{index} for the reader, but the stable eid rides along as data-eid so
    # cross-links resolve by eid regardless of how the merged feed is re-sorted.
    "s.id='r'+i;s.setAttribute('data-eid',e.eid||('r'+i));s.hidden=true;s.innerHTML=e.body_html||'';rd.appendChild(s);});"
    "render();"
    "if(window.buildThread){window.buildThread();}"
    # feed the mindmap live graph data (nodes/links/phases) before it is first drawn
    "if(window.buildGraphData){var _gd=document.getElementById('wb-graph-data');"
    "if(_gd)_gd.textContent=JSON.stringify(window.buildGraphData());}"
    # delegated navigation: any [data-eid] link (xref, thread item) opens that entry by eid
    "if(!window.__eidnav){window.__eidnav=1;document.addEventListener('click',function(ev){"
    "var el=ev.target.closest('[data-eid]');if(!el||el.classList.contains('report'))return;"
    "var eid=el.getAttribute('data-eid');if(!eid)return;ev.preventDefault();openReport(eid);});}"
    # typeset math once bodies are in the DOM (MathJax loads async → poll until ready)
    "(function tj(){if(window.MathJax&&MathJax.typesetPromise){MathJax.typesetPromise();}"
    "else{setTimeout(tj,150);}})();"
)

# eid-first openReport: accepts a stable eid string OR a numeric array index (back-compat).
# This is what makes two-worker cross-links survive re-sorting of the merged feed.
OPENREPORT_EID = (
    "function openReport(ref){"
    "var i=(typeof ref==='number')?ref:((window.EID2I&&(ref in window.EID2I))?window.EID2I[ref]"
    ":(/^\\d+$/.test(ref)?+ref:-1));"
    "if(i<0||!REPORTS[i])return;"
    'document.getElementById("home").hidden=true;'
    'document.getElementById("reader").hidden=false;'
    'document.querySelectorAll(".report").forEach(function(s){s.hidden=true;});'
    'var sec=document.getElementById("r"+i);if(sec)sec.hidden=false;'
    'document.getElementById("rtitle").textContent=REPORTS[i].title;'
    'document.getElementById("rdate").textContent=REPORTS[i].date;'
    "window.scrollTo(0,0);}"
)

# The 🧵 daily thread is regenerated client-side from the live feed (it used to be a frozen
# snapshot that went stale + carried stale index links). Links target eids → never break.
THREAD_JS = (
    "<script>window.buildThread=function(){"
    "var el=document.getElementById('wb-thread');if(!el)return;"
    "var esc=function(x){return (x||'').replace(/</g,'&lt;');};"
    "var byDay={};REPORTS.forEach(function(r){var d=(r.date||'').slice(0,10);(byDay[d]=byDay[d]||[]).push(r);});"
    "var days=Object.keys(byDay).sort(function(a,b){return b.localeCompare(a);});"
    "var html=\"<div class='wbx'><p>일자별로 올라온 리포트 전부(양쪽 워커 포함). 제목을 누르면 해당 리포트로 이동한다.</p>\";"
    "days.forEach(function(d){"
    "var items=byDay[d].slice().sort(function(a,b){return (b.date||'').localeCompare(a.date||'');});"
    'html+="<div class=\'day\'><h3>"+d+" <span class=\'sm\'>("+items.length+"건)</span></h3><ul>";'
    "items.forEach(function(r){var t=(r.date||'').slice(11);"
    'html+="<li><span class=\'sm\'>"+t+"</span> <a data-eid=\'"+r.eid+"\' style=\'cursor:pointer\'>"+esc(r.title)+"</a>"'
    "+\"<div class='sm'>\"+esc((r.summary||'').slice(0,120))+\"</div></li>\";});"
    'html+="</ul></div>";});'
    'html+="</div>";el.innerHTML=html;};</script>'
)

# The 🗺️ mindmap graph is regenerated client-side from the live feed too (it used to read a
# frozen JSON snapshot with stale array-index links). Nodes are keyed by eid; edges come from
# each entry's `links: [eid,...]`; the column/colour comes from its `phase`. New entries with
# no phase land in the '신규' column automatically.
GRAPH_JS = (
    "<script>window.buildGraphData=function(){"
    'var PHASES=["기반 탐색","정합성 검증","진단·방법","판정·종합","표현·설계","논문·교차","이식·인프라","신규"];'
    'var COLORS=["#4c72b0","#dd8452","#55a868","#c44e52","#8172b3","#937860","#da8bc3","#64b5cd"];'
    "var known={};REPORTS.forEach(function(r){known[r.eid]=1;});"
    "var nodes=REPORTS.map(function(r){var ci=PHASES.indexOf(r.phase||'신규');if(ci<0)ci=PHASES.length-1;"
    "return{id:r.eid,cat:ci,date:r.date,what:r.title,sum:r.summary};});"
    "var seen={},links=[];"
    "REPORTS.forEach(function(r){(r.links||[]).forEach(function(t){if(!known[t])return;"
    "var k=r.eid<t?r.eid+'|'+t:t+'|'+r.eid;if(seen[k])return;seen[k]=1;links.push([r.eid,t]);});});"
    "return{nodes:nodes,links:links,colors:COLORS,phases:PHASES};};</script>"
)
BOOTSTRAP = (
    "fetch('entries.json',{cache:'no-store'}).then(function(r){return r.json();}).then(function(D){"
    + _INJECT
    + "}).catch(function(err){document.getElementById('list').innerHTML="
    "'<p style=\"color:var(--red)\">entries.json load failed: '+err+'</p>';});"
)


def build(src_html: str, inline_entries=None) -> str:
    h = src_html
    # 1) strip the inline report bodies (all <section class="report" id="rN"> up to the block end)
    first = h.find('<section class="report" id="r0"')
    if first < 0:
        raise SystemExit("no inline report sections found")
    tail = h.find("<script>", first)
    wrapper_close = h[h.rfind("</section>", first, tail) + len("</section>") : tail]  # keep closing </div>s
    h = h[:first] + wrapper_close + h[tail:]
    # 2) REPORTS array -> empty (filled at runtime)
    h = re.sub(r"const REPORTS\s*=\s*\[.*?\];", "let REPORTS = [];", h, count=1, flags=re.DOTALL)
    # 3) the load-time render() call -> data bootstrap (or inline data for a fetch-free preview)
    boot = (
        ("var D=" + json.dumps(inline_entries, ensure_ascii=False) + ";" + _INJECT)
        if inline_entries is not None
        else BOOTSTRAP
    )
    last = h.rfind("render();")
    h = h[:last] + boot + h[last + len("render();") :]
    # 4) eid-based navigation: stable cross-links that survive re-sorting the merged feed
    # replacement passed as a callable so backslashes (e.g. \d) aren't treated as regex escapes
    h, n_or = re.subn(r"function openReport\(i\)\{.*?\n\}", lambda _m: OPENREPORT_EID, h, count=1, flags=re.DOTALL)
    if n_or != 1:
        raise SystemExit("could not patch openReport for eid navigation")
    h = h.replace("b.onclick=()=>openReport(r.idx);", "b.onclick=()=>openReport(r.eid||r.idx);", 1)
    # mindmap: click a node by eid (was a stale array index), and empty the frozen graph snapshot
    # so the client-side buildGraphData() repopulates it from the live feed.
    h = h.replace("openReport(n.idx)", "openReport(n.id)", 1)
    h, n_gd = re.subn(
        r"(<script id='wb-graph-data'[^>]*>).*?(</script>)",
        lambda _m: _m.group(1) + '{"nodes":[],"links":[],"colors":[],"phases":[]}' + _m.group(2),
        h,
        count=1,
        flags=re.DOTALL,
    )
    if n_gd != 1:
        raise SystemExit("could not empty the baked wb-graph-data snapshot")
    # 5) white/light override (append inside <style> so it wins the cascade)
    h = h.replace("</style>", WHITE_OVERRIDE + "</style>", 1)
    # 6) real math typesetting (MathJax) + client-side daily-thread & mindmap rebuild
    h = h.replace("</head>", MATHJAX_HEAD + "</head>", 1)
    return h.replace("</body>", THREAD_JS + GRAPH_JS + "</body>", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2"))
    ap.add_argument("--from", dest="src", default="index_legacy_backup.html")
    ap.add_argument("--preview", action="store_true", help="inline entries.json for a fetch-free preview")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    src = pathlib.Path(hf_hub_download(SPACE, a.src, repo_type="space", force_download=True)).read_text()
    inline = None
    if a.preview:
        ent = json.loads((a.out / "entries.json").read_text())
        ent.sort(key=lambda e: e.get("date", ""), reverse=True)
        inline = ent[:10]
    html = build(src, inline_entries=inline)
    name = "preview.html" if a.preview else "index.html"
    (a.out / name).write_text(html)
    kind = " · inline preview" if a.preview else " · fetches entries.json"
    print(f"wrote {a.out / name}  ({len(html) / 1000:.0f} KB{kind})")


if __name__ == "__main__":
    main()
