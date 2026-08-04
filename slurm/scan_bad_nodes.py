"""Find nodes that broke our jobs, and record only those — never nodes we broke ourselves.

An excluded node is a node the cluster loses for every future submission, so the bar for adding one
has to be evidence of a NODE fault, not merely a job that died there. The two are easy to confuse:
over one night on this cluster, five different nodes showed failures and only one of them was the
node's fault — the rest were our own shape bug, our own batch size, and a dataset we had not
downloaded. Excluding on "a job failed here" would have retired four healthy machines.

So the classification is explicit:

  NODE fault    slurm allocated a GPU the node could not hand over — jax reports no visible device,
                cuInit fails, the driver is mismatched. Nothing we can change in our code fixes it.
  OUR fault     OOM (our batch size), shape/type errors (our code), missing files (our setup),
                import errors (our venv). These say nothing about the node.

A log showing OUR fault is never enough to record a node, even if that is the only failure there.
A log must show a NODE signature and no plausible reading as ours.

    uv run slurm/scan_bad_nodes.py                 # report only
    uv run slurm/scan_bad_nodes.py --write         # append newly-implicated nodes to the list
"""

import argparse
import collections
import os
import pathlib
import re

# Signatures that can only be the node. Kept narrow on purpose: anything ambiguous belongs below.
NODE_FAULT = {
    "jax sees no GPU": r"No visible GPU devices|NO GPU visible to JAX",
    "CUDA init failed": r"failed call to cuInit|CUDA_ERROR_NO_DEVICE|CUDA_ERROR_DEVICE_UNAVAILABLE",
    "driver mismatch": r"driver version is insufficient|forward compatibility was attempted",
    "GPU fell off the bus": r"GPU is lost|Xid \d+|ECC error|GPU has fallen off the bus",
}
# Signatures that are ours. Their presence does NOT clear a node, but their presence ALONE blocks it.
OUR_FAULT = {
    "OOM": r"RESOURCE_EXHAUSTED|Out of memory while trying to allocate",
    "shape/type": r"incompatible shapes|ScopeParamShapeError|TypeError|AssertionError|ValueError",
    "missing input": r"FileNotFoundError|no meta\.json|is not a directory|missing \(incomplete",
    "import/env": r"ModuleNotFoundError|ImportError|OPENSSL_",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=pathlib.Path, default=None, help="default: $SLURM_LOGS")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="default: $ACRFT_BAD_NODES")
    ap.add_argument("--write", action="store_true", help="append newly-implicated nodes to the list")
    args = ap.parse_args()

    cache = os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft")
    logs = args.logs or pathlib.Path(os.environ.get("SLURM_LOGS", f"{cache}/logs"))
    out = args.out or pathlib.Path(os.environ.get("ACRFT_BAD_NODES", f"{cache}/bad_nodes.txt"))

    node_hits = collections.defaultdict(collections.Counter)
    our_hits = collections.defaultdict(collections.Counter)
    seen = collections.Counter()
    for f in sorted(logs.glob("*.out")):
        try:
            t = f.read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"^node +: (\S+)", t, re.MULTILINE)
        if not m:
            continue
        n = m.group(1)
        seen[n] += 1
        for lbl, pat in NODE_FAULT.items():
            if re.search(pat, t):
                node_hits[n][lbl] += 1
        for lbl, pat in OUR_FAULT.items():
            if re.search(pat, t):
                our_hits[n][lbl] += 1

    print(f"{len(seen)} nodes seen across {sum(seen.values())} logs in {logs}\n")
    print(f"{'node':<12} {'logs':>5}  {'verdict':<12} evidence")
    print("-" * 78)
    implicated = []
    for n in sorted(seen):
        nf, of = node_hits.get(n), our_hits.get(n)
        if nf:
            implicated.append((n, ", ".join(f"{k} x{v}" for k, v in nf.items())))
            verdict, ev = "NODE FAULT", implicated[-1][1]
        elif of:
            verdict, ev = "ours", ", ".join(f"{k} x{v}" for k, v in of.items())
        else:
            verdict, ev = "clean", ""
        print(f"{n:<12} {seen[n]:>5}  {verdict:<12} {ev}")

    existing = set()
    if out.exists():
        existing = {ln.split("#")[0].strip() for ln in out.read_text().splitlines() if ln.split("#")[0].strip()}
    new = [(n, ev) for n, ev in implicated if n not in existing]

    print()
    if not implicated:
        print("no node faults found — nothing to exclude")
    else:
        print(f"implicated: {', '.join(n for n, _ in implicated)}")
        print(f"already listed: {', '.join(sorted(existing)) or '(none)'}")
    if new:
        print(f"NEW: {', '.join(n for n, _ in new)}")
        if args.write:
            with out.open("a") as fh:
                for n, ev in new:
                    fh.write(f"{n}  # {ev}  (scan_bad_nodes)\n")
            print(f"appended to {out}")
        else:
            print("(re-run with --write to record them)")
    elif implicated:
        print("nothing new")

    # Nodes that only ever showed OUR failures are worth naming explicitly: they are exactly the ones
    # a careless "it failed there once" rule would have excluded.
    spared = [n for n in sorted(our_hits) if n not in node_hits]
    if spared:
        print(f"\nfailed here but NOT the node's fault (kept in service): {', '.join(spared)}")


if __name__ == "__main__":
    main()
