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
"""

_INJECT = (
    "D.sort(function(a,b){return (b.date||'').localeCompare(a.date||'');});"
    "REPORTS=D.map(function(e){return{date:e.date,title:e.title,summary:e.summary,tags:e.tags,status:e.status};});"
    "var rd=document.getElementById('reader');"
    "D.forEach(function(e,i){var s=document.createElement('section');s.className='report';"
    "s.id='r'+i;s.hidden=true;s.innerHTML=e.body_html||'';rd.appendChild(s);});"
    "render();"
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
    # 4) white/light override (append inside <style> so it wins the cascade)
    return h.replace("</style>", WHITE_OVERRIDE + "</style>", 1)


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
