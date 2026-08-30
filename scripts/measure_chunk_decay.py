"""Measure how a chunk policy's accuracy decays along the chunk -- the q(j) curve.

The commitment model (scripts/fig_four_forces_kstar.py) has a tail-accuracy parameter q: the chunk's
first slot is chosen from the current observation and is accurate, later slots predict further ahead and
degrade. q is what policy improvement raises, and it is the epistemic force that pushes commitments
short. This script measures the decay directly:

    e(j) = || a_hat_j(o_t) - a_{t+j} ||^2      averaged over decision points

reported per slot j, so the shape of the decay is visible rather than a single number.

The trap, and how this handles it. e(j) grows for TWO reasons that must not be conflated:
  (1) the policy cannot predict that far ahead        -- epistemic, the q we want;
  (2) the future genuinely is not determined yet      -- aleatoric, which no policy removes.
Sampling the policy several times at the same observation separates part of this: the spread ACROSS
samples is the policy's own uncertainty, while the residual of the sample mean against the recorded
action mixes in whatever the demonstrator did that the observation never determined. We report both,
and we split by episode outcome and by phase (free-space vs contact, proxied by gripper motion), since
aleatoric mass concentrates at contact. Nothing here is a q estimate on its own; the decomposition is.

    uv run python scripts/measure_chunk_decay.py --config pi05_yam_lego_taxi \
        --checkpoint checkpoints/.../150000 --episodes 6 --stride 40
"""

import argparse
import json
import pathlib

import numpy as np

CAMS = ("observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right")
OBS_KEYS = ("observation/image", "observation/wrist_image", "observation/image_right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_yam_lego_taxi")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--outcomes", default=".scratch/yam_outcomes_347.jsonl")
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=pathlib.Path(".scratch/yam_homing_onsets.json"))
    ap.add_argument("--episodes", type=int, default=6, help="episodes per outcome class")
    ap.add_argument("--stride", type=int, default=40, help="frames between decision points")
    ap.add_argument("--samples", type=int, default=4, help="policy samples per observation (for the spread)")
    ap.add_argument("--prompt", default="assemble lego blocks to make yellow taxi")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/chunk_decay.json"))
    a = ap.parse_args()

    import os

    os.environ.setdefault("LEROBOT_VIDEO_BACKEND", "pyav")
    from lerobot.datasets import lerobot_dataset

    from openpi.policies import policy_config
    from openpi.training import config as _config

    outc = {
        int(json.loads(x)["episode"]): json.loads(x)["outcome"]
        for x in pathlib.Path(a.outcomes).read_text().splitlines()
        if x.strip()
    }
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets.exists() else {}
    succ = [e for e in sorted(outc) if outc[e] == "success"][: a.episodes]
    fail = [e for e in sorted(outc) if outc[e] != "success"][: a.episodes]
    eps = succ + fail
    print(f"episodes: {len(succ)} success + {len(fail)} failure", flush=True)

    cfg = _config.get_config(a.config)
    pol = policy_config.create_trained_policy(cfg, a.checkpoint)
    H = int(pol._model.action_horizon)
    print(f"policy loaded, action_horizon H={H}", flush=True)

    recs = []
    for e in eps:
        ds = lerobot_dataset.LeRobotDataset(a.repo_id, root=a.root, episodes=[int(e)], video_backend="pyav")
        n = len(ds)
        eff = int(homing[str(e)]["homing_onset"]) if str(e) in homing else n
        eff = min(eff, n)
        is_succ = outc[e] == "success"
        # recorded actions for the whole episode (the ground truth chunk at every t)
        acts = np.stack([np.asarray(ds[i]["action"], np.float32) for i in range(n)])
        for t in range(0, max(1, eff - H), a.stride):
            s = ds[t]
            obs = {k: np.asarray(s[c], np.float32) for k, c in zip(OBS_KEYS, CAMS, strict=True)}
            obs = {k: (np.clip(np.transpose(v, (1, 2, 0)), 0, 1) * 255).astype(np.uint8) for k, v in obs.items()}
            obs["observation/state"] = np.asarray(s["observation.state"], np.float32)
            obs["prompt"] = a.prompt
            chunks = np.stack([np.asarray(pol.infer(dict(obs))["actions"], np.float32) for _ in range(a.samples)])
            truth = acts[t : t + H]  # [H, A]
            mean_chunk = chunks.mean(0)
            # per-slot: error of the sample mean, and the spread across samples
            err = ((mean_chunk - truth) ** 2).mean(-1)  # [H]
            spread = chunks.var(0).mean(-1)  # [H] policy's own uncertainty
            # phase proxy: gripper channels (6, 13) moving -> contact/manipulation rather than transport
            grip = np.abs(np.diff(acts[max(0, t - 5) : t + 6][:, [6, 13]], axis=0)).mean() if t + 6 <= n else 0.0
            recs.append(
                {
                    "ep": int(e),
                    "outcome": "success" if is_succ else "failure",
                    "t": int(t),
                    "frac": float(t / max(1, eff)),
                    "err": err.tolist(),
                    "spread": spread.tolist(),
                    "grip_motion": float(grip),
                }
            )
        print(f"  ep {e} ({'succ' if is_succ else 'fail'}): {sum(1 for r in recs if r['ep'] == e)} points", flush=True)

    def curve(rs, key):
        return np.stack([r[key] for r in rs]).mean(0) if rs else np.zeros(H)

    allr = recs
    s_r = [r for r in recs if r["outcome"] == "success"]
    f_r = [r for r in recs if r["outcome"] == "failure"]
    gm = np.array([r["grip_motion"] for r in recs])
    contact = [r for r, g in zip(recs, gm, strict=True) if g > np.median(gm)]
    free = [r for r, g in zip(recs, gm, strict=True) if g <= np.median(gm)]

    e_all, sp_all = curve(allr, "err"), curve(allr, "spread")
    print(f"\nchunk decay over {len(recs)} decision points, H={H}")
    print(f"{'slot j':>7} {'err e(j)':>10} {'e(j)/e(0)':>10} {'spread':>10} {'q~e(0)/e(j)':>12}")
    for j in range(H):
        if j % max(1, H // 10) == 0 or j == H - 1:
            print(
                f"{j:7d} {e_all[j]:10.5f} {e_all[j] / max(e_all[0], 1e-9):10.2f} {sp_all[j]:10.5f} {min(1.0, e_all[0] / max(e_all[j], 1e-9)):12.3f}"
            )

    print("\nsplits (error at the last slot / first slot -- how fast the tail degrades):")
    for name, rs in (("success", s_r), ("failure", f_r), ("contact", contact), ("free-space", free)):
        if not rs:
            continue
        c_ = curve(rs, "err")
        s_ = curve(rs, "spread")
        print(
            f"  {name:11s} e(0)={c_[0]:.5f}  e(H-1)={c_[-1]:.5f}  ratio={c_[-1] / max(c_[0], 1e-9):6.2f}   "
            f"spread(H-1)/err(H-1)={s_[-1] / max(c_[-1], 1e-9):.3f}"
        )
    print("\n  spread/err at the tail = the fraction of the tail error that is the policy's own")
    print("  uncertainty (epistemic-ish) rather than unexplained residual (aleatoric + bias).")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "checkpoint": a.checkpoint,
                "config": a.config,
                "H": H,
                "samples": a.samples,
                "curves": {
                    "all_err": e_all.tolist(),
                    "all_spread": sp_all.tolist(),
                    "success_err": curve(s_r, "err").tolist(),
                    "failure_err": curve(f_r, "err").tolist(),
                    "contact_err": curve(contact, "err").tolist(),
                    "free_err": curve(free, "err").tolist(),
                },
                "points": recs,
            },
            indent=1,
        )
    )
    print(f"\nwrote {a.out}  ({len(recs)} points)")


if __name__ == "__main__":
    main()
