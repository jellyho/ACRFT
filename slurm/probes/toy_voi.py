"""Commit long unless the environment is about to tell you something.

The value-based selector alone inherits whatever is wrong with the value estimate, and the data
problem that biases it (a demonstrator who already saw the event) is not always fixable. This probe
adds a second signal that is immune to that problem because it never conditions on demonstrator
actions at all: how the POLICY'S OWN UNCERTAINTY changes if it waits.

    g(x, k) = H[pi(.|x)] - E[ H[pi(.|x_k)] ]

    junction   unsure now, sure after the reveal            -> g > 0, waiting pays
    corridor   sure now (the plan is visible), lost later   -> g < 0, waiting costs

The naive version of this signal, "would the policy act differently after k steps", is wrong: in a
corridor the policy also acts differently, because it forgot the plan rather than because anything
was learned. Uncertainty change separates the two, which is exactly the separation the four-forces
account asks for: branching (shortens k) against information loss (lengthens k).

Using it: a short commitment is admissible only where information actually arrives.

    kappa(x, c) = argmax over { k : g(x, k) > tau } union {H}   of  Q(x, c, k)

so the value still chooses, but it may only choose to react where reacting can buy something.

Pre-registered (fixed before running):
  V1 The uncertainty gap has the predicted SIGN by state type: positive at a branch entry, negative
     inside a plan segment. Rejected if the signs are absent or reversed.
  V2 On a critic trained only on leaky demonstrations, gating restores junction reaction (rate near
     1) where the value-only selector fails, without shortening corridor commitments.
  V3 On a critic trained on self-rollouts (already unbiased), gating changes little: it is a repair,
     not a source of gain. Rejected if gating helps there just as much, which would mean it is doing
     something other than what is claimed.

    python slurm/probes/toy_voi.py --seeds 6 --out /scratch/jellyho/acrft/probes/toy_voi
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "slurm/probes")

import numpy as np
import torch
import torch.nn as nn
import toy_cfac_nn as T  # noqa: N812


class HeteroPolicy(nn.Module):
    """A chunk policy that also reports how unsure it is, per state.

    The shared log-std of the original policy makes entropy a constant and the signal we need
    invisible. Here the spread is predicted from the observation and fitted by Gaussian likelihood,
    so it reflects how varied the demonstrations were at that state, which is what "the policy does
    not know yet" means in this setting.
    """

    def __init__(self, hid=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(T.OBS_DIM, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.mu = nn.Linear(hid, T.H * T.ADIM)
        self.ls = nn.Linear(hid, T.H * T.ADIM)

    def forward(self, o):
        return self.mu(self.body(o))

    def dist(self, o):
        h = self.body(o)
        return self.mu(h), self.ls(h).clamp(-4.0, 1.0)

    def sample(self, o, rng_t=None):
        mu, ls = self.dist(o)
        return mu + torch.randn_like(mu) * ls.exp()

    def uncertainty(self, o):
        """How unsure the policy is about the NEXT action.

        Averaging over the whole chunk would fold in the spread of actions several steps out, which
        moves for reasons that have nothing to do with the decision at hand (a window overlapping
        the next segment picks up its variety). The commitment question is about what happens now.
        """
        _mu, ls = self.dist(o)
        return ls[..., : T.ADIM].mean(-1)


def train_bc_hetero(data, steps, seed):
    torch.manual_seed(seed)
    pi = HeteroPolicy()
    opt = torch.optim.Adam(pi.parameters(), 1e-3)
    n = len(data["o"])
    for _ in range(steps):
        i = torch.randint(0, n, (256,))
        mu, ls = pi.dist(data["o"][i])
        # Gaussian negative log likelihood: the spread has to explain the demonstrations' variety
        loss = (0.5 * ((data["ch"][i] - mu) / ls.exp()) ** 2 + ls).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi


@torch.no_grad()
def uncertainty_table(data, pi):
    """Average uncertainty, and its k-step-ahead value, keyed by the decision point.

    Everything comes from the dataset: for a transition at time t the k-step successor is the
    recorded observation at t+k, so no privileged access to the environment is used.
    """
    n_ep = len(data["o"]) // T.T
    obs = data["o"].view(n_ep, T.T, -1)
    u = pi.uncertainty(data["o"]).view(n_ep, T.T)
    now = np.zeros(T.T)
    ahead = np.zeros((T.T, T.H + 1))
    for t in range(T.T):
        now[t] = float(u[:, t].mean())
        for k in range(1, T.H + 1):
            tt = min(t + k, T.T - 1)
            ahead[t, k] = float(u[:, tt].mean())
    return {"now": now, "ahead": ahead, "obs_shape": tuple(obs.shape)}


def decision_index(o):
    """Recover the timestep from the observation's segment and step one-hots."""
    n_seg = len(T.SEGS)
    seg = int(np.argmax(o[:n_seg]))
    step = int(np.argmax(o[n_seg : n_seg + 6]))
    # segment boundaries are cumulative lengths; the suite exposes them through the env, but the
    # plain toy has equal-length segments, so recompute conservatively
    return seg, step


@torch.no_grad()
def rollout_gated(seed, pi, q, table, *, tau=0.0, gate=True, n_ep=300):
    """Deployment with the admissible-set gate: react only where waiting reduces uncertainty."""
    rng = np.random.default_rng(seed)
    env = T.PlanReach(rng)
    rets, kc, kj, react, seen = [], [], [], [], []
    for _ in range(n_ep):
        o = env.reset()
        hist, disc = [], 0.0
        while True:
            seg, step = env._seg_step()
            if step == 0:
                hist = []
            t_now = env.t
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            ht = torch.as_tensor(T.hist_feat(hist), dtype=torch.float32).unsqueeze(0)
            c = pi.sample(ot)
            qv = q(ot, ht, c)[0].numpy()
            allowed = []
            for k in range(1, T.H + 1):
                if not gate or k == T.H:
                    allowed.append(k)
                    continue
                g = table["now"][t_now] - table["ahead"][t_now, k]
                if g > tau:  # uncertainty drops if we wait: reacting can buy something
                    allowed.append(k)
            best = max(qv[k - 1] for k in allowed)
            k = max(k for k in allowed if qv[k - 1] >= best - T.TIE_EPS)
            if step == 0:
                (kj if T.SEGS[seg] == "J" else kc).append(k)
            if T.SEGS[seg] == "J" and step >= 1:
                react.append(1.0)
            acts = c.view(T.H, T.ADIM).numpy()
            done = False
            for j in range(min(k, T.T - env.t)):
                a = np.clip(acts[j], -1, 1)
                o2, r, done = env.step(acts[j])
                disc += T.GAMMA ** (env.t - 1) * r
                hist = [] if env.t % T.H == 0 else [*hist, a]
                o = o2
                if done:
                    break
            if done:
                break
        rets.append(disc)
        seen.append(1.0 if react else 0.0)
        react.clear()
    return {
        "ret": float(np.mean(rets)),
        "k_corridor": float(np.mean(kc)) if kc else float("nan"),
        "k_junction": float(np.mean(kj)) if kj else float("nan"),
        "react_rate": float(np.mean(seen)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--rollouts", type=int, default=800)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eval-eps", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/probes/toy_voi"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    rows, gaps = [], []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        demos = T.gen_demos(rng, a.episodes)
        pi = train_bc_hetero(demos, a.steps, seed)
        table = uncertainty_table(demos, pi)
        gaps.append({"now": table["now"].tolist(), "ahead": table["ahead"].tolist()})

        rolls = T.gen_self_rollouts(20_000 + seed, pi, a.rollouts)
        q_demo = T.train_cfac_critic(demos, pi, a.steps, seed, use_hist=True, interventional=False)
        q_roll = T.train_cfac_critic(rolls, pi, a.steps, seed, use_hist=True, interventional=False)

        res = {}
        for tag, q in (("demo", q_demo), ("roll", q_roll)):
            for gate in (False, True):
                name = f"{tag}_{'gated' if gate else 'plain'}"
                res[name] = rollout_gated(9000 + seed, pi, q, table, tau=a.tau, gate=gate, n_ep=a.eval_eps)
        res["oracle"] = T.rollout(9000 + seed, pi, oracle=True, n_ep=a.eval_eps)
        rows.append(res)
        print(
            f"seed {seed}: demo plain {res['demo_plain']['ret']:.2f}/react {res['demo_plain']['react_rate']:.2f}"
            f" -> gated {res['demo_gated']['ret']:.2f}/react {res['demo_gated']['react_rate']:.2f}"
            f" | roll plain {res['roll_plain']['ret']:.2f} -> gated {res['roll_gated']['ret']:.2f}",
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

    def paired(x, y):
        d = [r[x]["ret"] - r[y]["ret"] for r in rows]
        return [float(np.mean(d)), float(np.std(d)), int(sum(v > 0 for v in d)), len(d)]

    summary["_paired"] = {
        "demo_gated-demo_plain": paired("demo_gated", "demo_plain"),
        "roll_gated-roll_plain": paired("roll_gated", "roll_plain"),
        "demo_gated-roll_plain": paired("demo_gated", "roll_plain"),
    }
    # the sign check: uncertainty gap at a branch entry versus inside a plan segment
    now = np.mean([g["now"] for g in gaps], 0)
    ahead = np.mean([g["ahead"] for g in gaps], 0)
    sign = {}
    for t in range(T.T):
        seg = min(t // T.H, len(T.SEGS) - 1)
        step = t - seg * T.H
        key = f"{'branch' if T.SEGS[seg] == 'J' else 'plan'}_step{step}"
        sign[key] = float(now[t] - ahead[t, 1])
    summary["_uncertainty_gap_1step"] = sign
    (a.out / "results.json").write_text(json.dumps({"summary": summary, "per_seed": rows}, indent=1))
    print(json.dumps({"paired": summary["_paired"], "gap_sign": sign}, indent=1))


if __name__ == "__main__":
    main()
