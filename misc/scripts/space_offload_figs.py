"""Offload base64-inlined figures from worker-B entries to Space files (figures/<eid>/embN.*).

entries.json hit the 9.5MB static-Space guard; 7.3MB of it was worker-B body_html base64.
This rewrites ONLY worker-B entries: each data:image URI becomes a figures/<eid>/embN.<ext>
file (same rendering path as the newer entries) and the src points at it. Other workers'
entries are byte-identical. Race-safe single commit with parent_commit.
"""

import base64
import json
import pathlib
import re
import tempfile

from huggingface_hub import CommitOperationAdd
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

SPACE = "jellyho/acrft-reports"
PAT = re.compile(r"data:image/(png|jpeg|jpg|gif|svg\+xml);base64,([A-Za-z0-9+/=\s]+?)(?=['\"])")
EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "gif": "gif", "svg+xml": "svg"}


def main():
    api = HfApi()
    head = api.repo_info(SPACE, repo_type="space").sha
    p = hf_hub_download(SPACE, "entries.json", repo_type="space", revision=head, force_download=True)
    cur = json.loads(pathlib.Path(p).read_text())

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    ops = []
    saved = 0
    for e in cur:
        if e.get("worker") != "B":
            continue
        body = e.get("body_html", "")
        if "base64," not in body:
            continue
        idx = 0

        def repl(m, e=e):
            nonlocal idx, saved
            ext = EXT[m.group(1)]
            raw = base64.b64decode(m.group(2))
            fp = tmpdir / f"{e['eid']}_emb{idx}.{ext}"
            fp.write_bytes(raw)
            rel = f"figures/{e['eid']}/emb{idx}.{ext}"
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(fp)))
            saved += len(m.group(0))
            idx += 1
            return rel

        e["body_html"] = PAT.sub(repl, body)
        if idx:
            print(f"{e['eid']}: {idx} figures offloaded")

    out = tmpdir / "entries.json"
    out.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
    print(f"entries.json: {out.stat().st_size / 1e6:.2f}MB (freed ~{saved / 1e6:.2f}MB), {len(ops)} figure files")
    assert out.stat().st_size < 9_500_000
    ops.append(CommitOperationAdd(path_in_repo="entries.json", path_or_fileobj=str(out)))
    api.create_commit(
        SPACE,
        ops,
        repo_type="space",
        commit_message="[worker B] offload inline base64 figures to figures/<eid>/ (9.5MB guard)",
        parent_commit=head,
    )
    print("committed")


if __name__ == "__main__":
    main()
