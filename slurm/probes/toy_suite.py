"""A family of harder toy tasks for commitment learning.

The first toy had one past-latent segment and one future-latent segment, which was enough to show
the mechanism and not enough to break it. These variants each stress something the design has to
survive, and they share one interface so the same critic, policy and evaluation code run on all of
them (the training module's globals are rebound per task; see `activate`).

    plan_branch      the original: a plan visible only at entry, then a branch revealed one step in
    varlen           corridors of different lengths, so the right commitment differs WITHIN a type
    noisy_reveal     the branch signal arrives corrupted, so the best commitment is interior
    drift            actions execute with noise, so committing accumulates error (a real floor)
    decoy            a second, irrelevant signal fires mid-window: reacting to it is a mistake
    longmemory       the plan is shown once at episode start and never again
    multimodal       two plans are equally good, but mixing them mid-way is not

Each task defines, per segment, when the target is visible, when it is decided, and how actions are
scored, so the optimal commitment pattern is known by construction and can be compared against.

    python slurm/probes/toy_suite.py --list
    python slurm/probes/toy_suite.py --task varlen --seeds 4
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "slurm/probes")

import numpy as np
import torch
import toy_cfac_nn as T  # noqa: N812

ADIM = 2


class SegSpec:
    """One stretch of an episode.

    kind:
        plan   the target is visible at the first step, then hidden (a PAST latent)
        branch the target is undecided at entry and revealed after `reveal` steps (a FUTURE latent)
        decoy  like plan, but a second cue fires mid-segment that carries no reward information
        free   unscored filler
    """

    def __init__(self, kind, length, *, reveal=1, cue_noise=0.0, act_noise=0.0, modes=1):
        self.kind, self.length = kind, length
        self.reveal, self.cue_noise, self.act_noise, self.modes = reveal, cue_noise, act_noise, modes


TASKS = {
    "plan_branch": [SegSpec("plan", 4), SegSpec("branch", 4), SegSpec("plan", 4)],
    "varlen": [SegSpec("plan", 2), SegSpec("plan", 6), SegSpec("branch", 4)],
    "noisy_reveal": [SegSpec("plan", 4), SegSpec("branch", 4, cue_noise=0.8), SegSpec("plan", 4)],
    "drift": [
        SegSpec("plan", 4, act_noise=0.35),
        SegSpec("branch", 4, act_noise=0.35),
        SegSpec("plan", 4, act_noise=0.35),
    ],
    "decoy": [SegSpec("plan", 4), SegSpec("decoy", 4), SegSpec("branch", 4)],
    "longmemory": [SegSpec("plan", 4), SegSpec("free", 2), SegSpec("plan", 4)],
    "multimodal": [SegSpec("plan", 4, modes=2), SegSpec("branch", 4), SegSpec("plan", 4, modes=2)],
}

# observation layout: segment one-hot (max 3), step one-hot (max 6), cue (2), decoy cue (2)
MAX_SEGS, MAX_LEN = 3, 6
OBS_DIM = MAX_SEGS + MAX_LEN + 2 + 2


class SuiteEnv:
    """Generic episode over a list of SegSpec, with the training module's interface."""

    SPEC: list = TASKS["plan_branch"]

    def __init__(self, rng):
        self.rng = rng
        self.bounds = np.cumsum([0] + [s.length for s in self.SPEC])

    # --- geometry -----------------------------------------------------------
    def _seg_step(self):
        seg = int(np.searchsorted(self.bounds, self.t, side="right") - 1)
        seg = min(seg, len(self.SPEC) - 1)
        return seg, self.t - self.bounds[seg]

    def reset(self):
        ang = self.rng.uniform(0, 2 * np.pi, size=len(self.SPEC))
        self.g = np.stack([np.cos(ang), np.sin(ang)], 1)
        dang = self.rng.uniform(0, 2 * np.pi, size=len(self.SPEC))
        self.decoy = np.stack([np.cos(dang), np.sin(dang)], 1)
        # a multimodal segment keeps a second, equally good target; mixing them is what fails
        self.g2 = np.stack([np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)], 1)
        if self.SPEC is TASKS["longmemory"]:
            # the last segment is scored against the plan shown at the very start, and nothing is
            # shown again; writing it into g keeps one target definition for scoring and for demos
            self.g[2] = self.g[0].copy()
        self.t = 0
        return self.obs()

    # --- what the world shows ----------------------------------------------
    def obs(self):
        seg, step = self._seg_step()
        spec = self.SPEC[seg]
        o = np.zeros(OBS_DIM, np.float32)
        o[min(seg, MAX_SEGS - 1)] = 1.0
        o[MAX_SEGS + min(step, MAX_LEN - 1)] = 1.0
        cue = None
        if spec.kind in ("plan", "decoy") and step == 0:
            hidden = self.SPEC is TASKS["longmemory"] and seg == 2  # shown once, at the start only
            cue = None if hidden else self.g[seg]
        elif spec.kind == "branch" and step >= spec.reveal:
            cue = self.g[seg]
            if spec.cue_noise:
                cue = cue + self.rng.normal(0, spec.cue_noise, 2)
        if cue is not None:
            o[-4:-2] = cue
        if spec.kind == "decoy" and step == max(1, spec.length // 2):
            o[-2:] = self.decoy[seg]  # fires, means nothing
        return o

    def scored(self, seg=None, step=None):
        seg, step = self._seg_step() if seg is None else (seg, step)
        spec = self.SPEC[seg]
        if spec.kind == "free":
            return False
        if spec.kind == "branch":
            return step >= spec.reveal
        return True

    def step(self, a):
        seg, step = self._seg_step()
        spec = self.SPEC[seg]
        a = np.clip(a, -1, 1)
        if spec.act_noise:
            a = np.clip(a + self.rng.normal(0, spec.act_noise, 2), -1, 1)
        if self.scored(seg, step):
            # longmemory scores the last segment against the target shown at the episode start
            r = float(np.exp(-2.0 * np.sum((a - self.g[seg]) ** 2)))
            if spec.modes > 1:  # either target is fine; the max keeps both modes valid
                r = max(r, float(np.exp(-2.0 * np.sum((a - self.g2[seg]) ** 2))))
        else:
            r = 0.0
        self.t += 1
        done = self.t >= self.bounds[-1]
        return (None if done else self.obs()), r, done


def activate(task: str, horizon: int = 4):
    """Point the training module at this task. The functions there read module-level globals for
    the observation size and episode geometry, so binding them here lets one implementation of the
    critic, the policy and the evaluation loop serve every task."""
    spec = TASKS[task]
    SuiteEnv.SPEC = spec
    T.PlanReach = SuiteEnv
    T.OBS_DIM = OBS_DIM
    T.H = horizon
    T.T = int(sum(s.length for s in spec))
    T.SEGS = tuple("C" if s.kind in ("plan", "decoy", "free") else "J" for s in spec)
    T.HIST_DIM = horizon * (ADIM + 1)
    return spec


def run(task, seeds, episodes, steps, eval_eps, rollouts, out):
    activate(task)
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        demos = T.gen_demos(rng, episodes)
        pi = T.train_bc(demos, steps, seed)
        rolls = T.gen_self_rollouts(20_000 + seed, pi, rollouts)
        mixed = {k: torch.cat([demos[k], rolls[k]], 0) for k in demos}

        res = {}
        for name, pool, hist in [
            ("demo_hist", demos, True),
            ("roll_hist", rolls, True),
            ("mixed_hist", mixed, True),
            ("mixed_nohist", mixed, False),
        ]:
            q = T.train_cfac_critic(pool, pi, steps, seed, use_hist=hist, interventional=False)
            res[name] = T.rollout(9000 + seed, pi, q=q, n_ep=eval_eps)
        for k in (1, 2, T.H):
            res[f"fixed_k{k}"] = T.rollout(9000 + seed, pi, fixed_k=k, n_ep=eval_eps)
        rows.append(res)
        print(
            f"  [{task}] seed {seed}: mixed {res['mixed_hist']['ret']:.2f} | "
            f"roll {res['roll_hist']['ret']:.2f} | demo {res['demo_hist']['ret']:.2f} | "
            f"best fixed {max(res[f'fixed_k{k}']['ret'] for k in (1, 2, T.H)):.2f}",
            flush=True,
        )

    arms = list(rows[0])
    summary = {
        arm: {
            m: [float(np.nanmean([r[arm][m] for r in rows])), float(np.nanstd([r[arm][m] for r in rows]))]
            for m in ("ret", "k_corridor", "k_junction", "react_rate")
        }
        for arm in arms
    }
    best_fixed = [max(r[f"fixed_k{k}"]["ret"] for k in (1, 2, T.H)) for r in rows]
    for arm in ("mixed_hist", "roll_hist", "demo_hist", "mixed_nohist"):
        d = [r[arm]["ret"] - b for r, b in zip(rows, best_fixed, strict=True)]
        summary.setdefault("_vs_best_fixed", {})[arm] = [
            float(np.mean(d)),
            float(np.std(d)),
            int(sum(x > 0 for x in d)),
            len(d),
        ]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task}.json").write_text(json.dumps({"summary": summary, "per_seed": rows}, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="one task name, or all of them when omitted")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--rollouts", type=int, default=600)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-eps", type=int, default=250)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/probes/toy_suite"))
    a = ap.parse_args()
    if a.list:
        for name, spec in TASKS.items():
            print(f"{name:<14} " + " · ".join(f"{s.kind}({s.length})" for s in spec))
        return
    torch.set_num_threads(4)
    tasks = [a.task] if a.task else list(TASKS)
    for task in tasks:
        print(f"=== {task} ===", flush=True)
        s = run(task, a.seeds, a.episodes, a.steps, a.eval_eps, a.rollouts, a.out)
        print(
            "  vs best fixed: " + ", ".join(f"{k} {v[0]:+.3f}({v[2]}/{v[3]})" for k, v in s["_vs_best_fixed"].items()),
            flush=True,
        )


if __name__ == "__main__":
    main()
