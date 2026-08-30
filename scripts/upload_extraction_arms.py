"""Upload finished policy-extraction arms to a public HF model repo.

One repo, one folder per arm (``<arm>_run1/...``), plus a README that says what each arm is,
where its provenance comes from, and how to serve it. Only arms whose training run has FINISHED
are uploaded — a still-running arm's latest checkpoint is a moving target, so we skip it unless
--include-running is passed.

    uv run python scripts/upload_extraction_arms.py --arms dql qam lps lpsd flowdagger
"""

# ruff: noqa: PLC0415

import argparse
import pathlib
import subprocess

REPO = "jellyho/acrft-yam-extraction"
ROOT = pathlib.Path("/data1/jellyho/acrft_ckpts/extraction")

ABOUT = {
    "awr": ("AWR", "xbpeng/awr awr_agent.py:403,407,41", "expert overlay; advantage-weighted flow-BC"),
    "cfgrl": ("CFGRL", "kvfrans/cfgrl iql_diffusion.py:157,170-179,213", "expert+opt_embed; CFG sampling at w"),
    "flowdpg": ("FlowDPG", "arXiv 2606.22303 Eq. 4-9 (no official code)", "expert overlay; Tweedie + twin-min grad-Q"),
    "qam": ("QAM", "ColinQiyangLi/qam agents/qam.py:49-145", "expert overlay; adjoint-matched fast field"),
    "dql": ("DQL", "Zhendong-Wang .../ql_diffusion.py:140-148", "expert overlay; BC + eta*(-Q) with BPTT"),
    "lps": ("LPS", "author's lps.py:185-199 (ddpg)", "latent actor MLP over the frozen alpha-Flow one-step base"),
    "lpsd": ("LPSD", "author's lps.py:201-224 (onestep_ddpg)", "latent actor + anchor MSE"),
    "flowdagger": ("FlowDAgger", "microsoft/FlowDAgger", "DCT steering head predicting the sampler's seed"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--private", action="store_true", help="default is PUBLIC (user-authorized)")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)

    stamp = subprocess.run(
        ["git", "-C", str(pathlib.Path(__file__).resolve().parents[1]), "log", "-1", "--format=%H"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    uploaded = []
    for arm in a.arms:
        src = ROOT / f"{arm}_run1"
        if not src.exists():
            print(f"skip {arm}: {src} missing")
            continue
        print(f"uploading {arm} from {src} ...", flush=True)
        api.upload_folder(
            repo_id=a.repo,
            repo_type="model",
            folder_path=str(src),
            path_in_repo=f"{arm}_run1",
            commit_message=f"{arm} extraction arm ({stamp[:8]})",
        )
        uploaded.append(arm)

    rows = "\n".join(
        f"| `{arm}_run1` | {ABOUT[arm][0]} | {ABOUT[arm][1]} | {ABOUT[arm][2]} |" for arm in uploaded if arm in ABOUT
    )
    readme = f"""---
license: apache-2.0
tags: [robotics, offline-rl, vla, pi0.5, policy-extraction]
---

# ACRFT — YAM lego-taxi policy-extraction arms

Policy-extraction methods applied to the same frozen pi0.5 base and the same frozen patch
critic, as a **method-only-diff** comparison ring: identical BC init
(`yam_bc_s300_h30_successonly/100000`), backbone frozen, **action expert only** trained, critic
`patch_critic_yam_s347_fixed_tau9_min_200k` fixed.

| folder | method | provenance | what it swaps |
|---|---|---|---|
{rows}

Each implementation carries file/line-level provenance comments from the official code (or the
paper + appendices where no code exists). Two further arms need **no weights** — QPILOTS-U
(test-time critic steering, arXiv 2606.14801) and IDQL/BoN (N-sample argmax of min-ensemble Q,
philippe-eecs/IDQL) — they run from the BC checkpoint plus the critic.

## Serving

```bash
# from the openpi repo, branch integration
uv run python scripts/serve_extraction_arm.py --arm dql --port 8000
uv run python scripts/serve_extraction_arm.py --arm qpilots --alpha 0.2   # needs no weights
```

or in Python:

```python
from openpi.extraction import serving
policy = serving.load_arm("lpsd")           # or dql / qam / lps / flowdagger / idql / bon
chunk = policy.infer(obs)["actions"]
```

The expert-overlay arms (`awr`, `cfgrl`, `flowdpg`, `qam`, `dql`) store an orbax
`{{"expert": ...}}` subtree that is overlaid on the BC parameters; `lps`/`lpsd` store a small
latent-actor MLP (msgpack); `flowdagger` stores the DCT steering head plus its basis.

## Caveats

- Offline metrics only so far (critic-Q, held-out demo-MSE, chunk jerk); on-robot success rates
  are pending. Critic-Q is self-refereed for critic-ascending arms — read it with that in mind.
- Batch sizes differ per arm (4-32) because of VLA-scale memory limits, so at equal step counts
  the arms have consumed different sample counts.
- Code: `{stamp}`
"""
    tmp = pathlib.Path("/tmp/extraction_readme.md")
    tmp.write_text(readme)
    api.upload_file(
        path_or_fileobj=str(tmp), path_in_repo="README.md", repo_id=a.repo, repo_type="model", commit_message="README"
    )
    print(f"\ndone: https://huggingface.co/{a.repo}  ({len(uploaded)} arms: {', '.join(uploaded)})")


if __name__ == "__main__":
    main()
