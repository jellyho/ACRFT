"""Move a LeRobot dataset's `v3.0` tag onto `main`.

Why this is needed. lerobot pins `CODEBASE_VERSION = "v3.0"` and resolves that TAG for every read
(lerobot_dataset.py:83, :96), so a dataset whose tag lags `main` serves stale content to every
consumer -- training included -- while the Hub page shows the new data. Measured on this project:
`jellyho/yam_lego_taxi` and `jellyho/yam_cable_tie` both gained `next.success` / `next.done` on main
on 2026-09-02, and both v3.0 tags still pointed before that, so `--data.success-only` could not see
a verdict on either. The lego run that DID work was reading a local cache whose meta/info.json
predated the tag -- i.e. it worked by accident and would not reproduce on a clean machine.

This is deliberately a separate, explicit script and not part of any training path: it deletes and
recreates a tag on a public repository, which is outward-facing and not trivially reversible. It
prints the rollback commit before doing anything and refuses to run without --yes.

    uv run scripts/move_lerobot_v3_tag.py --repos jellyho/yam_lego_taxi jellyho/yam_cable_tie
    uv run scripts/move_lerobot_v3_tag.py --repos ... --yes
"""

import argparse
import json
import pathlib

TAG = "v3.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--yes", action="store_true", help="actually move the tags; without it this only reports")
    ap.add_argument("--record", type=pathlib.Path, default=pathlib.Path(".scratch/hf_tag_move_record.json"))
    a = ap.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download

    api = HfApi()
    plan = {}
    for r in a.repos:
        refs = api.list_repo_refs(r, repo_type="dataset")
        main = next(b.target_commit for b in refs.branches if b.name == "main")
        # An annotated tag's target_commit is the tag object, not the commit, so resolve through a
        # download rather than trusting it as a rollback anchor.
        cur = None
        for c in api.list_repo_commits(r, repo_type="dataset"):
            try:
                hf_hub_download(r, "meta/info.json", repo_type="dataset", revision=TAG)
            except Exception:
                break
            cur = c.commit_id
            break

        def feats(rev, repo=r):
            path = hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision=rev, force_download=True)
            return set(json.loads(pathlib.Path(path).read_text()).get("features", {}))

        f_tag, f_main = feats(TAG), feats("main")
        plan[r] = {"main": main, "tag_features": len(f_tag), "main_features": len(f_main)}
        print(
            f"{r}\n   {TAG}: {len(f_tag)} features   main: {len(f_main)} features   missing from tag: {sorted(f_main - f_tag)}"
        )
        print(f"   rollback commit for {TAG}: {cur}")
        plan[r]["rollback_commit"] = cur

    if not a.yes:
        print("\n--yes not given; nothing changed.")
        return
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(plan, indent=1))
    for r in a.repos:
        api.delete_tag(r, tag=TAG, repo_type="dataset")
        api.create_tag(r, tag=TAG, repo_type="dataset", revision="main")
        f = json.loads(
            pathlib.Path(
                hf_hub_download(r, "meta/info.json", repo_type="dataset", revision=TAG, force_download=True)
            ).read_text()
        ).get("features", {})
        ok = "next.success" in f and "next.done" in f
        print(f"{r}: {TAG} -> main   verdicts now visible: {ok}")
    print(f"\nrollback record: {a.record}")


if __name__ == "__main__":
    main()
