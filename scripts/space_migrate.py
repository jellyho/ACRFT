"""Migrate the AC-RFT reports Space from an inline-HTML SPA to a data/presentation split.

The live index.html embeds `const REPORTS = [{date,title,summary,tags,status},...]` (feed metadata)
plus 62 `<section class="report" id="rN">…</section>` bodies (the 4.5 MB bulk, KO/EN inside each).
This pulls both apart into a single `entries.json` "database" — one record per report, body HTML
preserved VERBATIM so no worker's content is reformatted or lost. A fixed template then renders it
client-side; workers only ever append to the data.

    uv run python scripts/space_migrate.py --out space_v2
"""

import argparse
import json
import pathlib
import re

from huggingface_hub import hf_hub_download
import lxml.html

SPACE = "jellyho/acrft-reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    p = hf_hub_download(SPACE, "index.html", repo_type="space", force_download=True)
    h = pathlib.Path(p).read_text()

    m = re.search(r"const REPORTS\s*=\s*(\[.*?\]);", h, re.DOTALL)
    if not m:
        raise SystemExit("could not locate REPORTS array")
    reports = json.loads(m.group(1))
    print(f"REPORTS metadata: {len(reports)} records")

    doc = lxml.html.fromstring(h)
    entries = []
    for i, meta in enumerate(reports):
        sec = doc.get_element_by_id(f"r{i}", None)
        if sec is None:
            print(f"  ! r{i} body missing")
            body = ""
        else:
            # inner HTML of the <section> verbatim (children only, not the <section> wrapper)
            body = (sec.text or "") + "".join(lxml.html.tostring(c, encoding="unicode") for c in sec)
        entries.append(
            {
                "eid": f"r{i}",
                "date": meta.get("date", ""),
                "title": meta.get("title", ""),
                "summary": meta.get("summary", ""),
                "tags": meta.get("tags", []),
                "status": meta.get("status", ""),
                "body_html": body,
            }
        )

    (a.out / "entries.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    total = sum(len(e["body_html"]) for e in entries)
    print(f"wrote {len(entries)} entries -> {a.out/'entries.json'}  (body bytes {total/1e6:.2f} MB)")
    # sanity: every entry has a date + non-trivial body (except intentionally-empty summaries)
    empty = [e["eid"] for e in entries if not e["body_html"]]
    print(f"  empty bodies: {empty or 'none'}")


if __name__ == "__main__":
    main()
