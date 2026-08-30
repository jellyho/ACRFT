"""Append (or replace) one report entry in the data-driven AC-RFT Space.

The Space is now data/presentation split: a fixed ``index.html`` renders ``entries.json`` client-side.
Publishing = appending your entry to ``entries.json`` and committing it — you never touch index.html
or another worker's entries. This helper does exactly that, race-safely (fetch latest -> append your
eid only -> commit with parent_commit).

    uv run python scripts/space_add_entry.py --json my_entry.json [--figures fig1.svg ...]

`my_entry.json` schema (5W1H + links live inside body_html per the hub convention):
    {"eid":"wa-...", "worker":"A", "date":"YYYY-MM-DD HH:MM", "status":"ongoing|finding|done",
     "title":"...", "summary":"...", "tags":["워커A",...],
     "body_html":"<div class='wbx wbx-ko'>…</div><div class='wbx wbx-en'>…</div>"}
An entry with an existing eid REPLACES it (same-thread update); a new eid is appended.
"""

import argparse
import json
import pathlib
import tempfile

from huggingface_hub import CommitOperationAdd
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

SPACE = "jellyho/acrft-reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, required=True, help="the entry JSON to add/replace")
    ap.add_argument("--figures", nargs="*", default=[], help="figure files to upload under figures/<eid>/")
    ap.add_argument("--message", default=None)
    a = ap.parse_args()

    entry = json.loads(a.json.read_text())
    assert entry.get("eid"), "entry needs eid"
    assert entry.get("worker"), "entry needs worker"

    api = HfApi()
    head = api.repo_info(SPACE, repo_type="space").sha
    cur = json.loads(
        pathlib.Path(
            hf_hub_download(SPACE, "entries.json", repo_type="space", revision=head, force_download=True)
        ).read_text()
    )
    # replace same eid (yours only) or append; never reorder others
    cur = [e for e in cur if e.get("eid") != entry["eid"]]
    cur.append(entry)
    tmp = pathlib.Path(tempfile.mkdtemp()) / "entries.json"
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
    assert tmp.stat().st_size < 9_500_000, "entries.json would exceed the 9.5MB static-Space guard"

    ops = [CommitOperationAdd(path_in_repo="entries.json", path_or_fileobj=str(tmp))]
    for fig in a.figures:
        fp = pathlib.Path(fig)
        ops.append(CommitOperationAdd(path_in_repo=f"figures/{entry['eid']}/{fp.name}", path_or_fileobj=str(fp)))
    msg = a.message or f"[worker {entry['worker']}] entry {entry['eid']} ({entry.get('date','')})"
    api.create_commit(SPACE, ops, repo_type="space", commit_message=msg, parent_commit=head)
    print(f"published {entry['eid']} -> {SPACE}  ({len(cur)} entries)")


if __name__ == "__main__":
    main()
