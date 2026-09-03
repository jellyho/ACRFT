"""Upload finished policy-extraction arms to a public HF model repo.

One repo, one folder per run (``<arm><suffix>/...``), plus a README that says what each arm is,
where its provenance comes from, and how to serve it. Only arms whose training run has FINISHED
are uploaded — a still-running arm's latest checkpoint is a moving target.

By default only the FINAL step is uploaded, as a whole openpi checkpoint (user's convention:
"전체 체크포인트 통채로", "최종 스텝만"). Intermediate steps stay on disk; they are useful locally
and would multiply the repo size by the save-every count for nothing.

    uv run python scripts/upload_extraction_arms.py --arms awr --suffix _bb
"""

# ruff: noqa: PLC0415

import argparse
import pathlib
import subprocess

REPO = "jellyho/acrft-yam-extraction"
ROOT = pathlib.Path("/data1/jellyho/acrft_ckpts/extraction")

ABOUT = {
    "awr": ("AWR", "xbpeng/awr awr_agent.py:403,407,41", "advantage-weighted flow-BC"),
    "cfgrl": ("CFGRL", "kvfrans/cfgrl iql_diffusion.py:157,170-179,213", "CFG sampling at w"),
    "flowdpg": ("FlowDPG", "arXiv 2606.22303 Eq. 4-9 (no official code)", "Tweedie + twin-min grad-Q"),
    "qam": ("QAM", "ColinQiyangLi/qam agents/qam.py:49-145", "adjoint-matched fast field"),
    "dql": ("DQL", "Zhendong-Wang .../ql_diffusion.py:140-148", "BC + eta*(-Q) with BPTT"),
    "fqlx": ("QC-FQL one-step", "seohong/fql actor loss, frozen critic", "distill teacher + (-Q)"),
    "lps": ("LPS", "author's lps.py:185-199 (ddpg)", "latent actor MLP over the frozen alpha-Flow one-step base"),
    "lpsd": ("LPSD", "author's lps.py:201-224 (onestep_ddpg)", "latent actor + anchor MSE"),
    "flowdagger": ("FlowDAgger", "microsoft/FlowDAgger", "DCT steering head predicting the sampler's seed"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--private", action="store_true", help="default is PUBLIC (user-authorized)")
    ap.add_argument(
        "--suffix",
        default="_bb",
        help="run-folder suffix under the checkpoint root. _bb = the BC-budget runs (whole model "
        "trained, batch 32, lr 5e-5, 30k steps); _run1 = the earlier expert-only runs.",
    )
    ap.add_argument(
        "--all-steps",
        action="store_true",
        help="upload every saved step instead of only the last one (the default is last-only)",
    )
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
        run = f"{arm}{a.suffix}"
        src = ROOT / run
        if not src.exists():
            print(f"skip {arm}: {src} missing")
            continue
        steps = sorted((d for d in src.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
        if not steps:
            print(f"skip {arm}: no numbered step directories under {src}")
            continue
        if a.all_steps:
            folder, in_repo, what = src, run, f"all {len(steps)} steps"
        else:
            folder, in_repo, what = steps[-1], f"{run}/{steps[-1].name}", f"step {steps[-1].name} only"
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e9
        print(f"uploading {arm}: {what} from {folder} ({size:.1f} GB) ...", flush=True)
        api.upload_folder(
            repo_id=a.repo,
            repo_type="model",
            folder_path=str(folder),
            path_in_repo=in_repo,
            commit_message=f"{arm} extraction arm, {what} ({stamp[:8]})",
        )
        uploaded.append((arm, in_repo, steps[-1].name, size))

    rows = "\n".join(
        f"| `{in_repo}` | {ABOUT[arm][0]} | {step} | {size:.0f} GB | {ABOUT[arm][1]} | {ABOUT[arm][2]} |"
        for arm, in_repo, step, size in uploaded
        if arm in ABOUT
    )
    readme = f"""---
license: apache-2.0
tags: [robotics, offline-rl, vla, pi0.5, policy-extraction]
---

# ACRFT — YAM lego-taxi policy-extraction arms

Policy-extraction methods applied to the same pi0.5 base and the same frozen patch critic, as a
**method-only-diff** comparison ring: identical BC init
(`yam_bc_s300_h30_successonly/100000`), critic `patch_critic_yam_s347_fixed_tau9_min_200k` frozen,
and only the extraction objective differs between arms.

Runs suffixed **`_bb`** match the BC fine-tune's own training budget: the **whole model** is
trainable (the BC config sets no freeze filter, so matching its budget means matching what it was
allowed to move, not only steps and batch), batch 32, constant lr 5e-5, 30k steps. These are whole
openpi checkpoints, **final step only** — load them the way you load any pi0.5 checkpoint.

Runs suffixed **`_run1`** are the earlier pass with the backbone frozen and the **action expert
only** trainable. They are a smaller budget than BC's and are kept for comparison, not as the
headline result.

| folder | method | step | size | provenance | what the objective swaps |
|---|---|---|---|---|---|
{rows}

How the weights are delivered follows the suffix, not the arm: a `_bb` folder is a **whole openpi
checkpoint** (load it directly); a `_run1` folder is an **action-expert subtree** that is overlaid
on the BC checkpoint at serving time. CFGRL additionally carries an optimality embedding and is
served through its own config.

Each implementation carries file/line-level provenance comments from the official code (or the
paper + appendices where no code exists). Two further arms need **no weights** — QPILOTS-U
(test-time critic steering, arXiv 2606.14801) and IDQL/BoN (N-sample argmax of min-ensemble Q,
philippe-eecs/IDQL) — they run from the BC checkpoint plus the critic.

## Serving

There is ONE serving entry point, `scripts/serve_policy.py`. Arms reach it two ways, and which
way depends on whether the arm changed the policy's weights or only how a chunk is chosen.

**Weight-only arms** (`awr`, `flowdpg`, `qam`, `dql`, `fqlx`) fine-tune the pi0.5 action expert,
so they are exported to ordinary openpi checkpoints and served like any checkpoint:

```bash
uv run python scripts/export_extraction_checkpoint.py --arm dql          # -> exported/dql_30000
uv run python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_yam_lego_taxi --policy.dir <exported>/dql_30000
```

**CFGRL** additionally carries the optimality embedding and samples with classifier-free
guidance, which are model properties, so it travels in its own config (`with_cfgrl` builds the
variant of any pi0.5 task config; the guidance weight is `cfg_w`):

```bash
uv run python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_yam_lego_taxi_cfgrl --policy.dir <exported>/cfgrl_30000
```

**Critic-consuming arms** need no policy of their own; they are modes of the critic wrapper:

```bash
# selection: bon executes the argmax of N draws (this is also IDQL's argmax rule -- label it by N)
uv run python scripts/serve_policy.py --port 8000 --critic <critic_dir> --critic-mode bon --num-samples 8 \
    policy:checkpoint --policy.config pi05_yam_lego_taxi --policy.dir <BC checkpoint>

# implicit: IDQL's implicit policy -- one draw sampled with expectile weights on the advantage
#   --critic-mode implicit --num-samples 64
# qpilots: test-time Q-steering of the sampler, no weights at all
#   --critic-mode qpilots --alpha 0.2
# lps / lpsd / flowdagger: pass their small head
#   --critic-mode lpsd --extraction-head <latent_actor_*.msgpack>
#   --critic-mode flowdagger --extraction-head <flowdagger_run1 dir>
```

`adaptive` (execute only the best commitment prefix, then replan) needs a critic trained with
several commitment groups, i.e. `macro_group_size < horizon`; the fixed-chunk critic this ring
trains against has a single group, where adaptive is bon under another name.

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
    print(f"\ndone: https://huggingface.co/{a.repo}  ({len(uploaded)} runs: {chr(44).join(r[1] for r in uploaded)})")


if __name__ == "__main__":
    main()
