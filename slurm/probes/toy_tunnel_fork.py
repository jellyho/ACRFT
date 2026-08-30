"""Tunnel-Fork corridor: does honest value evaluation stop TD from preferring short horizons?

A pre-registered tabular test of Reactive-Commitment Value (docs/reactive_commitment_value.md). One
corridor mixes the two regimes so the optimal commitment kappa*(s) is genuinely state-dependent and
graded, not "always 1" or "always max":

  idx:  0    1     2    3    4    5      6    7     8    9      10   11
  type: S  TentA Tin  Tin  Tin  TexitA  S  TentB Tin  TexitB  F    G
        |<------- tunnel A (len 5) ------>|    |<- tun B (3) ->|

  * Tunnel: a key z in {0,1} is drawn and OBSERVED only at the entrance; the interior observation is
    'dark' (position and z hidden). The correct action at each cell is pattern(z)[depth] -- a multi-step
    plan that must be carried blind. Re-querying inside the dark proposes a coin flip. So the whole
    tunnel must be committed FROM the entrance. Oracle: commit each tunnel to its exit (5, then 3).
  * Fork: a branch b in {0,1} is drawn and OBSERVED only on arrival; correct action = b. Committing
    THROUGH the fork guesses b (wrong w.p. 1/2); re-querying at the fork sees b. Oracle: commit 1.
  * Straight: observed, deterministic, correct = 0.
  A wrong action ends the episode at 0; reaching the goal pays 1. H = 8, gamma = 0.97, demo error 3%.

Four critic arms, a 2x2 factorial (matches the framework's corrections):
  axis HIST  : critic input = observation (Markov)   vs  latent (history-conditioned)  -> H
  axis HONEST: bootstrap = demo-return V + demo-chunk-conditioned reward (naive)       vs
               deployment-policy value + model-marginal committed reward (honest)      -> S + P
  A0 = (obs, naive)   A1 = (latent, naive)   A2 = (obs, honest)   A3 = RCV = (latent, honest)
plus fixed-k (k=1..H) and the oracle selector.

Everything is Monte-Carlo over latents sampled from the demo visitation given the critic's input, so no
information leaks to a Markov arm. 8 seeds; the table is recomputed from the returned records.
"""
# ruff: noqa: N803, N806, PERF401, RET504  (probe script: Q is the value-fn symbol)

import argparse
import json
import pathlib

import numpy as np

# ---- corridor layout -------------------------------------------------------------------------------
# each cell: (type, tunnel_id or fork flag). Tunnels share a per-tunnel key z; forks a per-fork b.
LAYOUT = [
    ("S", None),  # 0
    ("Tent", "A"),  # 1  tunnel A entrance (len 5: idx 1..5)
    ("Tin", "A"),  # 2
    ("Tin", "A"),  # 3
    ("Tin", "A"),  # 4
    ("Texit", "A"),  # 5
    ("S", None),  # 6
    ("Tent", "B"),  # 7  tunnel B entrance (len 3: idx 7..9)
    ("Tin", "B"),  # 8
    ("Texit", "B"),  # 9
    ("F", None),  # 10 fork
    ("G", None),  # 11 goal
]
GOAL = len(LAYOUT) - 1
H = 8
GAMMA = 0.97
DEMO_ERR = 0.03
TUN = {"A": list(range(1, 6)), "B": list(range(7, 10))}  # cell indices per tunnel
TENT = {1: "A", 7: "B"}  # entrance idx -> tunnel
FORK = 10
# tunnel plan: z-keyed alternating pattern, long enough for either tunnel
PATTERN = {0: [0, 1, 0, 1, 0], 1: [1, 0, 1, 0, 1]}


def tunnel_depth(idx, tid):
    return TUN[tid].index(idx)


def correct_action(idx, latent):
    """latent = dict(zA, zB, b). The right action at idx given the drawn latents."""
    typ, tid = LAYOUT[idx]
    if typ in ("Tent", "Tin", "Texit"):
        z = latent["z" + tid]
        return PATTERN[z][tunnel_depth(idx, tid)]
    if typ == "F":
        return latent["b"]
    return 0  # straight


def observation(idx, latent):
    """What a Markov critic / the deploy proposer sees. Occlusion loses position and key inside tunnels."""
    typ, tid = LAYOUT[idx]
    if typ == "Tent":
        return ("Tent", idx, latent["z" + tid])  # key visible at entrance
    if typ in ("Tin", "Texit"):
        return ("dark",)  # position AND key hidden
    if typ == "F":
        return ("F", idx, latent["b"])  # branch visible on arrival
    return (typ, idx)


def latent_key(idx, latent):
    """History-conditioned critic input = the sufficient statistic (never occluded)."""
    typ, tid = LAYOUT[idx]
    if typ in ("Tent", "Tin", "Texit"):
        return ("T", idx, latent["z" + tid])
    if typ == "F":
        return ("F", idx, latent["b"])
    return (typ, idx)


def draw_latent(rng):
    return {"zA": int(rng.integers(2)), "zB": int(rng.integers(2)), "b": int(rng.integers(2))}


# ---- the deployment proposer: what chunk the frozen policy proposes at a re-query --------------------
def propose(idx, latent, seen_dark):
    """H actions proposed from the observation at idx. Knows structure + locally-revealed latent; guesses
    future keys/branches. At 'dark' it knows neither position nor key, so it coin-flips the tunnel."""
    obs = observation(idx, latent)
    chunk = []
    if obs[0] == "dark":
        # lost: does not know where it is or the key. Best it can do is a fixed guess.
        return [0] * H
    for d in range(H):
        j = idx + d
        if j >= GOAL:
            chunk.append(0)
            continue
        typ, tid = LAYOUT[j]
        if typ in ("Tent", "Tin", "Texit"):
            # knows this tunnel's key only if the proposal starts at (or the key is carried from) its
            # entrance -- i.e. idx is the entrance of this tunnel. Otherwise it is guessing.
            if idx in TENT and TENT[idx] == tid:
                chunk.append(PATTERN[latent["z" + tid]][tunnel_depth(j, tid)])
            else:
                chunk.append(0)  # future tunnel: key unknown
        elif typ == "F":
            chunk.append(latent["b"] if j == idx else 0)  # knows b only if standing on the fork
        else:
            chunk.append(0)
    return chunk


# ---- rollout of a committed chunk in the true model -------------------------------------------------
def step_commit(idx, latent, chunk, k):
    """Execute the first k actions of chunk from idx. Returns (discounted committed reward, end idx or
    None if dead, survived)."""
    r = 0.0
    cur = idx
    for j in range(k):
        if cur >= GOAL:
            break
        a = chunk[j]
        if a == correct_action(cur, latent):
            cur += 1
            if cur == GOAL:
                r += GAMMA**j * 1.0
                return r, GOAL, True
        else:
            return r, None, False  # dead
    return r, cur, True


def rollout(selector, latent, rng, start=0):
    """Deploy: propose, select k, commit, re-query. selector(idx, latent) -> k. Returns 1 if goal."""
    idx = start
    guard = 0
    while idx < GOAL and guard < 3 * len(LAYOUT):
        guard += 1
        chunk = propose(idx, latent, seen_dark=False)
        # demo-style execution noise on the committed actions
        k = selector(idx, latent)
        noisy = [a if rng.random() > DEMO_ERR else 1 - a for a in chunk]
        _, nxt, alive = step_commit(idx, latent, noisy, k)
        if not alive:
            return 0
        idx = nxt
        if idx == GOAL:
            return 1
    return 1 if idx == GOAL else 0


# ---- demo data + per-arm value tables ---------------------------------------------------------------
def gen_demo(rng, n):
    """Demo = oracle-given-latent with 3% flips. Records (input_key, obs_key, mc_return) per visited idx,
    and per (input, k) the demo committed return -- for the naive/confounded estimates."""
    data = []
    for _ in range(n):
        latent = draw_latent(rng)
        traj = []
        idx, dead = 0, False
        while idx < GOAL:
            a = correct_action(idx, latent)
            if rng.random() < DEMO_ERR:
                a = 1 - a
            traj.append(idx)
            if a == correct_action(idx, latent):
                idx += 1
            else:
                dead = True
                break
        ret_from = {}
        if not dead:  # reached goal
            for t, cell in enumerate(traj):
                steps_to_goal = len(traj) - t
                ret_from[cell] = GAMMA**steps_to_goal
        else:
            for cell in traj:
                ret_from[cell] = 0.0
        for cell in traj:
            data.append((cell, latent, ret_from[cell]))
    return data


def build_tables(rng, demo, n_mc, honest, hist):
    """Return Q[input_key][k] for k=1..H under the arm (honest bool, hist bool)."""

    def inp(idx, latent):
        return latent_key(idx, latent) if hist else observation(idx, latent)

    # demo visitation grouped by the arm's input, with the latents seen there (for MC given input)
    by_input = {}
    v_data = {}
    for cell, latent, ret in demo:
        x = inp(cell, latent)
        by_input.setdefault(x, []).append((cell, latent))
        v_data.setdefault(x, []).append(ret)
    v_data = {x: float(np.mean(r)) for x, r in v_data.items()}

    # V_polexp[x]: value of the DEPLOY process from states with input x (MC, honest bootstrap)
    v_polexp = {}
    if honest:
        for x, members in by_input.items():
            samp = [members[i] for i in rng.integers(0, len(members), size=min(n_mc, 4 * len(members)))]
            vals = []
            for cell, latent in samp:
                vals.append(rollout(oracle_selector, latent, rng, start=cell))  # deploy from THIS cell
            v_polexp[x] = float(np.mean(vals)) if vals else 0.0

    vboot = v_polexp if honest else v_data

    Q = {}
    for x, members in by_input.items():
        idxs = rng.integers(0, len(members), size=min(n_mc, 4 * len(members)))
        samp = [members[i] for i in idxs]
        Q[x] = {}
        for k in range(1, H + 1):
            acc = []
            for cell, latent in samp:
                if honest:  # synthetic: model-marginal reward of the DEPLOY proposer's chunk
                    chunk = propose(cell, latent, seen_dark=False)
                else:  # confounded: reward assuming the DEMO (correct) chunk -- optimistic
                    chunk = [correct_action(min(cell + d, GOAL - 1), latent) for d in range(H)]
                r, nxt, alive = step_commit(cell, latent, chunk, k)
                if not alive or nxt is None:
                    acc.append(r)  # died in the window; no bootstrap
                elif nxt == GOAL:
                    acc.append(r)
                else:
                    xb = inp(nxt, latent)
                    acc.append(r + GAMMA**k * vboot.get(xb, 0.0))
            Q[x][k] = float(np.mean(acc))
    return Q, vboot


# ---- selectors --------------------------------------------------------------------------------------
def oracle_selector(idx, latent):
    typ, tid = LAYOUT[idx]
    if typ == "Tent":
        return len(TUN[tid])  # commit the whole tunnel
    if typ == "F":
        return 1  # react to the branch
    if typ in ("Tin", "Texit"):
        return 1
    # straight: commit up to (not into) the next fork/tunnel-interior decision
    k = 1
    while idx + k < GOAL and LAYOUT[idx + k][0] in ("S",):
        k += 1
    return max(1, k)


def make_arm_selector(Q, hist, eps=1e-3):
    def sel(idx, latent):
        x = latent_key(idx, latent) if hist else observation(idx, latent)
        if x not in Q:
            return 1
        qk = Q[x]
        best = max(qk.values())
        longest = max(k for k, v in qk.items() if v >= best - eps)  # lexicographic longest within eps
        return longest

    return sel


def make_fixed_selector(k):
    return lambda idx, latent: k


# ---- experiment -------------------------------------------------------------------------------------
def run_seed(seed, n_demo, n_mc, n_eval):
    rng = np.random.default_rng(seed)
    demo = gen_demo(rng, n_demo)
    arms = {
        "A0_obs_naive": (False, False),
        "A1_lat_naive": (False, True),
        "A2_obs_honest": (True, False),
        "A3_RCV": (True, True),
    }
    out = {}
    # per-arm learned selectors + the tunnel-entrance advantage sign (P1/P5)
    for name, (honest, hist) in arms.items():
        Q, _ = build_tables(rng, demo, n_mc, honest, hist)
        sel = make_arm_selector(Q, hist)
        ev = np.random.default_rng(seed * 100 + 7)
        sr = np.mean([rollout(sel, draw_latent(ev), ev) for _ in range(n_eval)])
        # commit length chosen at each tunnel entrance (avg over the two key latents), and at the fork
        kA = np.mean([sel(1, {"zA": z, "zB": 0, "b": 0}) for z in (0, 1)])
        kB = np.mean([sel(7, {"zA": 0, "zB": z, "b": 0}) for z in (0, 1)])
        kPre = np.mean([sel(9, {"zA": 0, "zB": zb, "b": b}) for zb in (0, 1) for b in (0, 1)])  # pre-fork stop
        # advantage at tunnel-A entrance: Q_5 - Q_1 (sign = does the arm prefer the long commit?)
        adv = []
        for z in (0, 1):
            lat = {"zA": z, "zB": 0, "b": 0}
            x = latent_key(1, lat) if hist else observation(1, lat)
            if x in Q:
                adv.append(Q[x][5] - Q[x][1])
        out[name] = {
            "sr": float(sr),
            "kA": float(kA),
            "kB": float(kB),
            "kPre": float(kPre),
            "advA_5m1": float(np.mean(adv)) if adv else float("nan"),
        }
    # fixed-k sweep + oracle
    for k in range(1, H + 1):
        ev = np.random.default_rng(seed * 100 + 7)
        out[f"fixed_k{k}"] = {
            "sr": float(np.mean([rollout(make_fixed_selector(k), draw_latent(ev), ev) for _ in range(n_eval)]))
        }
    ev = np.random.default_rng(seed * 100 + 7)
    out["oracle"] = {"sr": float(np.mean([rollout(oracle_selector, draw_latent(ev), ev) for _ in range(n_eval)]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-demo", type=int, default=4000)
    ap.add_argument("--n-mc", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/toy_tunnel_fork.json"))
    a = ap.parse_args()

    per_seed = [run_seed(s, a.n_demo, a.n_mc, a.n_eval) for s in range(a.seeds)]

    def agg(key, field="sr"):
        v = np.array([ps[key][field] for ps in per_seed if field in ps[key]])
        return float(v.mean()), float(v.std())

    print(f"tunnel-fork toy | {a.seeds} seeds | H={H} gamma={GAMMA} | oracle kappa*: A=5 B=3 fork=1\n")
    print(f"{'arm':16s} {'SR':>14s} {'k@TentA(5)':>11s} {'k@TentB(3)':>11s} {'k@pre-fork(1)':>13s} {'Q5-Q1@A':>10s}")
    for name in ["A0_obs_naive", "A1_lat_naive", "A2_obs_honest", "A3_RCV"]:
        sr_m, sr_s = agg(name)
        kA_m, _ = agg(name, "kA")
        kB_m, _ = agg(name, "kB")
        kPre_m, _ = agg(name, "kPre")
        adv_m, _ = agg(name, "advA_5m1")
        print(f"{name:16s} {sr_m:.3f} ± {sr_s:.3f} {kA_m:11.2f} {kB_m:11.2f} {kPre_m:10.2f} {adv_m:+10.3f}")
    o_m, o_s = agg("oracle")
    print(f"{'oracle':16s} {o_m:.3f} ± {o_s:.3f}")
    print("\nfixed-k sweep (SR):")
    for k in range(1, H + 1):
        m, s = agg(f"fixed_k{k}")
        print(f"  k={k}: {m:.3f} ± {s:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"config": vars(a) | {"out": str(a.out)}, "per_seed": per_seed}, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
