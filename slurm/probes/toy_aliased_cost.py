"""Aliased-cost corridor: when is HISTORY conditioning actually necessary?

The tunnel-fork toy (toy_tunnel_fork.py) showed the honest bootstrap, not history, is the operative
correction -- because every decision there sat at an observed point. This probe isolates when history
IS necessary, and the answer sharpens the framework:

  In a selection-only setting, committing short is always success-safe (you merely re-query more), so
  TASK SUCCESS alone can never make history necessary -- a Markov critic can always play short and not
  fail. History becomes necessary only under a RE-PLAN COST, where committing long is beneficial and the
  critic must tell, from history, how long it is safe to commit through an observation-aliased region.
  This is exactly the event-triggered-control tradeoff (compute vs performance).

Environment. A start signal reveals the episode type t in {safe, forky}; afterwards every cell shows
the same 'plain' observation (aliased across t). The correct action is 0 everywhere except, in a forky
episode, a single FORK cell that reveals a branch b only when standing on it (correct = b there).
  * safe : commit the whole corridor from cell 1 -> 1 re-query, no risk.
  * forky: commit up to the fork, re-query there to react to b, commit the rest.
Objective = reached-goal  -  lambda * (number of re-queries).

A Markov critic at cell 1 sees 'plain' for BOTH types -> one commitment length. Committing long fails
the forky branch (SR drop); committing short is SR-safe but pays the re-query cost on the safe type. It
cannot be optimal for both. A history critic knows t and commits long-when-safe, short-to-the-fork-when-
forky. Pre-registered: history ties the oracle objective; Markov cannot.
"""

# ruff: noqa: N802, N803, N806  (Q is the value-function symbol)
import argparse
import json
import pathlib

import numpy as np

L = 8  # corridor length (cells 1..L are 'plain'); goal at L+1
FORK_AT = 5  # in a forky episode, cell FORK_AT is the fork
GAMMA = 0.99
DEMO_ERR = 0.02
H = L  # can commit the whole corridor
GOAL = L + 1


def draw(rng):
    t = "forky" if rng.random() < 0.5 else "safe"
    b = int(rng.integers(2))
    return {"t": t, "b": b}


def correct_action(idx, lat):
    if idx == 0:
        return 0  # start signal cell
    if lat["t"] == "forky" and idx == FORK_AT:
        return lat["b"]
    return 0


def observation(idx, lat):
    if idx == 0:
        return ("start", lat["t"])  # type visible ONLY here
    if lat["t"] == "forky" and idx == FORK_AT:
        return ("fork", lat["b"])  # branch visible only while standing on the fork
    return ("plain",)  # everything else aliased across type and position


def latent_key(idx, lat):
    if idx == 0:
        return ("start", lat["t"])
    if lat["t"] == "forky" and idx == FORK_AT:
        return ("fork", lat["b"])
    return ("plain", idx, lat["t"])  # history recovers position and type


def propose(idx, lat):
    """Proposer sees the observation. Proposes 0 everywhere except it plays b when standing on the fork."""
    chunk = []
    for d in range(H):
        j = idx + d
        if j == FORK_AT and lat["t"] == "forky" and j == idx:
            chunk.append(lat["b"])  # knows b only when standing on the fork
        else:
            chunk.append(0)
    return chunk


def step_commit(idx, lat, chunk, k):
    cur = idx
    for j in range(k):
        if cur >= GOAL:
            break
        if chunk[j] == correct_action(cur, lat):
            cur += 1
            if cur == GOAL:
                return GOAL, True
        else:
            return None, False
    return cur, True


def rollout(selector, lat, rng, lam, start=0):
    """Returns (reached_goal, n_requery). Objective = goal - lam * n_requery."""
    idx, guard, nq = start, 0, 0
    while idx < GOAL and guard < 4 * GOAL:
        guard += 1
        nq += 1
        chunk = propose(idx, lat)
        k = selector(idx, lat)
        noisy = [a if rng.random() > DEMO_ERR else 1 - a for a in chunk]
        nxt, alive = step_commit(idx, lat, noisy, k)
        if not alive:
            return 0, nq
        idx = nxt
        if idx == GOAL:
            return 1, nq
    return (1 if idx == GOAL else 0), nq


def oracle(idx, lat):
    if idx == 0:
        return 1
    if lat["t"] == "safe":
        return H  # commit the whole corridor
    # forky: commit up to (not into) the fork, else react on the fork, else run to goal
    if idx < FORK_AT:
        return FORK_AT - idx
    if idx == FORK_AT:
        return 1
    return H


def gen_demo(rng, n):
    data = []
    for _ in range(n):
        lat = draw(rng)
        idx, dead, traj = 0, False, []
        while idx < GOAL:
            a = correct_action(idx, lat)
            if rng.random() < DEMO_ERR:
                a = 1 - a
            traj.append(idx)
            if a == correct_action(idx, lat):
                idx += 1
            else:
                dead = True
                break
        for t, cell in enumerate(traj):
            ret = GAMMA ** (len(traj) - t) if not dead else 0.0
            data.append((cell, lat, ret))
    return data


def build_Q(rng, demo, n_mc, honest, hist, lam):
    def inp(idx, lat):
        return latent_key(idx, lat) if hist else observation(idx, lat)

    by = {}
    vdat = {}
    for cell, lat, ret in demo:
        x = inp(cell, lat)
        by.setdefault(x, []).append((cell, lat))
        vdat.setdefault(x, []).append(ret)
    vdat = {x: float(np.mean(r)) for x, r in vdat.items()}

    vpol = {}
    if honest:
        for x, mem in by.items():
            idxs = rng.integers(0, len(mem), size=min(n_mc, 4 * len(mem)))
            vals = []
            for i in idxs:
                cell, lat = mem[i]
                g, nq = rollout(oracle, lat, rng, lam, start=cell)
                vals.append(g - lam * nq)  # objective-value of the deployment process
            vpol[x] = float(np.mean(vals))
    vboot = vpol if honest else vdat

    Q = {}
    for x, mem in by.items():
        idxs = rng.integers(0, len(mem), size=min(n_mc, 4 * len(mem)))
        samp = [mem[i] for i in idxs]
        Q[x] = {}
        for k in range(1, H + 1):
            acc = []
            for cell, lat in samp:
                chunk = (
                    propose(cell, lat) if honest else [correct_action(min(cell + d, GOAL - 1), lat) for d in range(H)]
                )
                nxt, alive = step_commit(cell, lat, chunk, k)
                imm = -lam  # this commitment cost one re-query
                if not alive:
                    acc.append(imm)
                elif nxt == GOAL:
                    acc.append(imm + GAMMA**k * 1.0)
                else:
                    acc.append(imm + GAMMA**k * vboot.get(inp(nxt, lat), 0.0))
            Q[x][k] = float(np.mean(acc))
    return Q


def arm_selector(Q, hist, eps=1e-4):
    def sel(idx, lat):
        x = latent_key(idx, lat) if hist else observation(idx, lat)
        if x not in Q:
            return 1
        qk = Q[x]
        best = max(qk.values())
        return max(k for k, v in qk.items() if v >= best - eps)

    return sel


def run_seed(seed, n_demo, n_mc, n_eval, lam):
    rng = np.random.default_rng(seed)
    demo = gen_demo(rng, n_demo)
    arms = {"A0_obs_naive": (False, False), "A2_obs_honest": (True, False), "A3_RCV_hist": (True, True)}
    out = {}

    def evaluate(sel):
        ev = np.random.default_rng(seed * 100 + 3)
        gs, nqs = [], []
        for _ in range(n_eval):
            g, nq = rollout(sel, draw(ev), ev, lam)
            gs.append(g)
            nqs.append(nq)
        gs, nqs = np.array(gs), np.array(nqs)
        return float(gs.mean()), float(nqs.mean()), float((gs - lam * nqs).mean())

    for name, (honest, hist) in arms.items():
        Q = build_Q(rng, demo, n_mc, honest, hist, lam)
        sr, nq, obj = evaluate(arm_selector(Q, hist))
        # commit length chosen at cell 1 for each type (aliased for the Markov arm)
        k_safe = arm_selector(Q, hist)(1, {"t": "safe", "b": 0})
        k_forky = arm_selector(Q, hist)(1, {"t": "forky", "b": 0})
        out[name] = {"sr": sr, "nq": nq, "obj": obj, "k1_safe": k_safe, "k1_forky": k_forky}
    sr, nq, obj = evaluate(oracle)
    out["oracle"] = {"sr": sr, "nq": nq, "obj": obj, "k1_safe": H, "k1_forky": FORK_AT - 1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-demo", type=int, default=4000)
    ap.add_argument("--n-mc", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=0.05, help="re-plan cost per re-query")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/toy_aliased_cost.json"))
    a = ap.parse_args()
    ps = [run_seed(s, a.n_demo, a.n_mc, a.n_eval, a.lam) for s in range(a.seeds)]

    def agg(key, f):
        v = np.array([p[key][f] for p in ps])
        return v.mean(), v.std()

    print(f"aliased-cost toy | {a.seeds} seeds | L={L} fork@{FORK_AT} lambda={a.lam}")
    print(f"objective = SR - {a.lam}*re-queries ; oracle k1: safe={H} forky={FORK_AT - 1}\n")
    print(f"{'arm':16s} {'SR':>13s} {'re-queries':>12s} {'objective':>13s} {'k1@safe':>8s} {'k1@forky':>9s}")
    for name in ["A0_obs_naive", "A2_obs_honest", "A3_RCV_hist", "oracle"]:
        sr_m, sr_s = agg(name, "sr")
        nq_m, nq_s = agg(name, "nq")
        ob_m, ob_s = agg(name, "obj")
        ks, _ = agg(name, "k1_safe")
        kf, _ = agg(name, "k1_forky")
        print(f"{name:16s} {sr_m:.3f}±{sr_s:.3f} {nq_m:7.2f}±{nq_s:.2f} {ob_m:+.3f}±{ob_s:.3f} {ks:8.1f} {kf:9.1f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"config": vars(a) | {"out": str(a.out)}, "per_seed": ps}, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
