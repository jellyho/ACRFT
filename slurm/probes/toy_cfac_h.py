"""Can data provenance replace the interventional pairing? (CFAC-H design probe)

The interventional resampling that made the first design work does not transfer: swapping successors
is only valid when the variation being swapped is exogenous, which holds in the toy but not on a
robot, where the successor depends on the action taken. So this probe asks whether the same defect
can be removed by WHERE THE CRITIC'S DATA COMES FROM instead of by a trick.

The confound exists because a person choosing an action already saw the event that would arrive
inside the window. A robot's own rollout has no such property: the policy acted blind. So a critic
fitted on self-rollouts should price commitment honestly with the plain data-paired backup.

Two axes, interventional pairing OFF everywhere:
    critic data : demos (closed-loop human) | rollouts (the policy's own, blind) | mixed
    history     : critic sees executed actions | observation only

Pre-registered predictions (fixed before running):
  H1 With rollout-trained critics, the junction is handled without any intervention: reaction rate
     at the junction is high and return approaches the interventional design's.
  H2 History remains necessary in both data regimes: removing it collapses corridor commitment,
     because blindness after a re-query is only visible through what was already executed.
  H3 Demo-trained critics without intervention over-commit at the junction (the failure the first
     design fixed), reproducing the earlier ablation.
Rejected if: rollout-trained critics do no better than demo-trained ones at the junction (then
provenance is not the lever and the intervention was doing something else).

Run: python slurm/probes/toy_cfac_h.py --seeds 6 --out /scratch/jellyho/acrft/probes/toy_cfac_h
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "slurm/probes")

import numpy as np
import torch
import toy_cfac_nn as T  # noqa: N812


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--rollouts", type=int, default=800)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eval-eps", type=int, default=300)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/probes/toy_cfac_h"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    cells = {
        "demo_hist": ("demos", True),
        "demo_nohist": ("demos", False),
        "roll_hist": ("rollouts", True),
        "roll_nohist": ("rollouts", False),
        "mixed_hist": ("mixed", True),
    }
    per_seed = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        demos = T.gen_demos(rng, a.episodes)
        pi = T.train_bc(demos, a.steps, seed)
        rolls = T.gen_self_rollouts(20_000 + seed, pi, a.rollouts)
        mixed = {k: torch.cat([demos[k], rolls[k]], 0) for k in demos}

        pool = {"demos": demos, "rollouts": rolls, "mixed": mixed}
        res = {}
        for name, (src, hist) in cells.items():
            q = T.train_cfac_critic(pool[src], pi, a.steps, seed, use_hist=hist, interventional=False)
            res[name] = T.rollout(9000 + seed, pi, q=q, n_ep=a.eval_eps)
        # the reference points: the interventional design, and the hand-written rule
        q_ref = T.train_cfac_critic(demos, pi, a.steps, seed, use_hist=True, interventional=True)
        res["demo_hist_interv"] = T.rollout(9000 + seed, pi, q=q_ref, n_ep=a.eval_eps)
        res["oracle"] = T.rollout(9000 + seed, pi, oracle=True, n_ep=a.eval_eps)
        for k in (1, 4):
            res[f"bc_k{k}"] = T.rollout(9000 + seed, pi, fixed_k=k, n_ep=a.eval_eps)
        per_seed.append(res)
        print(
            f"seed {seed}: demo+hist {res['demo_hist']['ret']:.2f}/react {res['demo_hist']['react_rate']:.2f} | "
            f"roll+hist {res['roll_hist']['ret']:.2f}/react {res['roll_hist']['react_rate']:.2f}/kC {res['roll_hist']['k_corridor']:.2f} | "
            f"roll-nohist {res['roll_nohist']['ret']:.2f}/kC {res['roll_nohist']['k_corridor']:.2f} | "
            f"interv-ref {res['demo_hist_interv']['ret']:.2f}",
            flush=True,
        )

    arms = list(per_seed[0])
    summary = {
        arm: {
            m: [
                float(np.nanmean([s[arm][m] for s in per_seed])),
                float(np.nanstd([s[arm][m] for s in per_seed])),
            ]
            for m in ("ret", "k_corridor", "k_junction", "react_rate")
        }
        for arm in arms
    }

    def paired(x, y):
        d = [s[x]["ret"] - s[y]["ret"] for s in per_seed]
        return [float(np.mean(d)), float(np.std(d)), int(sum(v > 0 for v in d)), len(d)]

    summary["_paired"] = {
        "roll_hist-demo_hist": paired("roll_hist", "demo_hist"),
        "roll_hist-demo_hist_interv": paired("roll_hist", "demo_hist_interv"),
        "roll_hist-roll_nohist": paired("roll_hist", "roll_nohist"),
        "mixed_hist-demo_hist": paired("mixed_hist", "demo_hist"),
    }
    (a.out / "results.json").write_text(json.dumps({"summary": summary, "per_seed": per_seed}, indent=1))
    print(json.dumps(summary["_paired"], indent=1))


if __name__ == "__main__":
    main()
