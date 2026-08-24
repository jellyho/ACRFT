"""CFAC toy: can a critic price non-Markovian commitment and reactiveness fairly?

Environment ("plan maze", segments [corridor, junction, corridor], 4 steps each, H=4 chunks):
  - corridor: latent plan z ~ Bern(.5) is OBSERVABLE ONLY AT SEGMENT ENTRY (then hidden).
    Correct action = z every step; a wrong action fails the episode. A demonstrator remembers z
    (non-Markov w.r.t. observations); a Markov BC policy queried mid-corridor is a 50/50 mixture.
    => commitment carries private information (the plan); reactivity destroys it.
  - junction: event b ~ Bern(.5) is revealed AFTER the first step. Steps 1-3 must match b.
    => committing across the reveal guesses b (50%); reacting after the reveal wins.
  - reward 1 on reaching the end, gamma = 0.95. Demonstrator has per-step error EPS_DEMO.

Critic arms (all per-prefix, SMDP bookkeeping, lexicographic longest-within-eps selector):
  A0 naive : obs-conditioned chunk-outcome regression + dataset-V bootstrap
             (the standard chunked critic trained on closed-loop demos)
  A1 +hist : A0 with history conditioning (obs + actions since segment entry)
  A2 +pol  : A1 with policy-expectation bootstrap at the requery leaf (FQE of the deployed system)
  A3 CFAC  : A2 with the within-chunk value COMPOSED from one-step backups in history space
             (unconfounds future reveals) -- the full proposal.
Baselines: fixed k in {1,2,3,4}; oracle kappa* (commit corridors, react at junctions).

Pre-registered predictions (hub entry cfac-proposal):
  T1 A0 believes ~everything succeeds (hindsight leak) and over-commits junctions; its
     belief-minus-realized gap is the largest of all arms.
  T2 A1/A2 fix the corridor but still over-commit the junction (entry-time conditioning on the
     chunk selects b => the leak survives history at states where the latent is still future).
  T3 Only A3 both commits corridors and reacts at junctions; SR(A3) ~ SR(oracle) > all fixed k.
  T4 Fixed-k sweep is non-monotone with interior loser/winner pattern per segment mix.
Rejected if: A3 fails to separate from A0-A2 (then the composition/bootstrap story is wrong).

Run: python slurm/probes/toy_cfac.py --seeds 8 --episodes 1000 --out /scratch/jellyho/acrft/probes/toy_cfac
"""

import argparse
import collections
import json
import pathlib

import numpy as np

GAMMA = 0.95
H = 4  # chunk length = segment length
SEGS = ("C", "J", "C")  # corridor, junction, corridor
T = len(SEGS) * H
EPS_DEMO = 0.04
TIE_EPS = 0.02


# ---------------------------------------------------------------- environment
def obs_of(seg, step, z, b):
    """Observation: (segment index, type, step, cue). Cue: corridor z only at step 0;
    junction b only from step 1 on; hidden (-1) otherwise."""
    typ = SEGS[seg]
    cue = (z if step == 0 else -1) if typ == "C" else b if step >= 1 else -1
    return (seg, typ, step, cue)


class Env:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.lat = [int(self.rng.random() < 0.5) for _ in SEGS]  # z or b per segment
        self.t = 0
        self.failed = False
        return self._obs()

    def _obs(self):
        seg, step = divmod(self.t, H)
        return obs_of(seg, step, self.lat[seg], self.lat[seg])

    def correct(self):
        seg, step = divmod(self.t, H)
        if SEGS[seg] == "C":
            return self.lat[seg]
        return None if step == 0 else self.lat[seg]  # junction step 0: anything goes

    def step(self, a):
        c = self.correct()
        if c is not None and a != c:
            self.failed = True
        self.t += 1
        done = self.failed or self.t >= T
        r = 1.0 if (done and not self.failed and self.t >= T) else 0.0
        return (None if done else self._obs()), r, done


# ---------------------------------------------------------------- demonstrator + dataset
def gen_demos(rng, n):
    episodes = []
    env = Env(rng)
    for _ in range(n):
        o = env.reset()
        traj = []  # (obs, hist_key, action, reward)
        hist = ()  # actions since segment entry
        while True:
            seg, step = divmod(env.t, H)
            if step == 0:
                hist = ()
            c = env.correct()
            a = int(rng.random() < 0.5) if c is None else (c if rng.random() > EPS_DEMO else 1 - c)
            h = (o, hist)
            o2, r, done = env.step(a)
            traj.append((o, h, a, r))
            hist = (*hist, a)
            o = o2
            if done:
                break
        episodes.append(traj)
    return episodes


def build_tables(episodes):
    """Everything the critics need, from data only."""
    cont = collections.defaultdict(list)  # obs -> next-H action arrays (BC chunk policy)
    cont_h = collections.defaultdict(list)  # hist-key -> same (history-conditioned chunk policy)
    ret_o = collections.defaultdict(list)  # obs -> discounted return-to-go   (dataset V, Markov)
    ret_h = collections.defaultdict(list)  # hist -> discounted return-to-go  (dataset V, history)
    model = collections.defaultdict(list)  # (hist,a) -> (next hist key or None, reward)
    for traj in episodes:
        rews = [x[3] for x in traj]
        for i, (o, h, a, _r) in enumerate(traj):
            acts = [traj[j][2] for j in range(i, min(i + H, len(traj)))]
            acts = acts + [0] * (H - len(acts))
            cont[o].append(tuple(acts))
            cont_h[h].append(tuple(acts))
            g = sum(GAMMA**j * rews[i + j] for j in range(len(traj) - i))
            ret_o[o].append(g)
            ret_h[h].append(g)
            nxt = traj[i + 1][1] if i + 1 < len(traj) else None
            model[(h, a)].append((nxt, rews[i]))
    return {"cont": cont, "cont_h": cont_h, "ret_o": ret_o, "ret_h": ret_h, "model": model}


# ---------------------------------------------------------------- critics
def chunk_regression(episodes, tables, key_mode):
    """A0/A1: Q(key, c_{1:k}) = mean over data of [within-chunk discounted reward
    + gamma^k * dataset-V(landing key)]. key_mode: 'obs' or 'hist'."""
    ret = tables["ret_o"] if key_mode == "obs" else tables["ret_h"]
    acc = collections.defaultdict(list)
    for traj in episodes:
        rews = [x[3] for x in traj]
        for i, (o, h, _a, _r) in enumerate(traj):
            key = o if key_mode == "obs" else h
            for k in range(1, H + 1):
                if i + k > len(traj):
                    # episode ended inside the prefix: realized rewards only
                    pref = tuple(x[2] for x in traj[i:])
                    if len(pref) == 0:
                        continue
                    y = sum(GAMMA**j * rews[i + j] for j in range(len(traj) - i))
                    acc[(key, pref, k)].append(y)
                    continue
                pref = tuple(traj[i + j][2] for j in range(k))
                y = sum(GAMMA**j * rews[i + j] for j in range(k))
                if i + k < len(traj):
                    land = traj[i + k][0] if key_mode == "obs" else traj[i + k][1]
                    y += GAMMA**k * float(np.mean(ret[land]))
                acc[(key, pref, k)].append(y)
    return {kk: float(np.mean(v)) for kk, v in acc.items()}


def select_k(qfn, key, c, tie_eps=TIE_EPS):
    """Lexicographic: longest k within tie_eps of the max. Unseen prefix values -> None."""
    vals = [qfn(key, tuple(c[:k]), k) for k in range(1, H + 1)]
    seen = [(k, v) for k, v in zip(range(1, H + 1), vals, strict=True) if v is not None]
    if not seen:
        return 1, None
    vmax = max(v for _, v in seen)
    best = max(k for k, v in seen if v >= vmax - tie_eps)
    return best, vmax


def fqe_policy_bootstrap(episodes, tables, composed, iters=40):
    """A2 (composed=False) / A3 (composed=True): history-conditioned per-prefix values with a
    policy-expectation bootstrap at the requery leaf. A2 keeps the chunk-conditioned within-chunk
    estimate (leaked at future-latent states); A3 composes one-step empirical backups in history
    space, so future reveals enter with their marginal distribution."""
    model = tables["model"]
    cont = tables["cont"]
    # states where the deployed system can requery = history keys seen in data
    hkeys = {h for (h, _a) in model}

    # empirical one-step model p(next_h, r | h, a)
    step_model = {}
    for (h, a), lst in model.items():
        cnt = collections.Counter(lst)
        tot = sum(cnt.values())
        step_model[(h, a)] = [(nx, r, c / tot) for (nx, r), c in cnt.items()]

    V = collections.defaultdict(float)  # V_pi_hat(h): value of requerying at h

    def q_composed(h, c, k):
        """Exact composition through the empirical h-space model, bootstrap at depth k."""
        if k == 0:
            return V[h] if h in hkeys else 0.0
        if (h, c[0]) not in step_model:
            return None
        out = 0.0
        for nx, r, p in step_model[(h, c[0])]:
            val = r
            if nx is not None:
                sub = q_composed(nx, c[1:], k - 1)
                if sub is None:
                    sub = 0.0  # unseen branch: pessimistic
                val += GAMMA * sub
            out += p * val
        return out

    # within-chunk-only table (no bootstrap) for A2; the policy bootstrap is resolved
    # per-iteration against the current V.
    if not composed:
        acc = collections.defaultdict(list)
        for traj in episodes:
            rews = [x[3] for x in traj]
            for i, (_o, h, _a, _r) in enumerate(traj):
                for k in range(1, H + 1):
                    kk = min(k, len(traj) - i)
                    pref = tuple(traj[i + j][2] for j in range(kk))
                    if len(pref) < k and i + kk >= len(traj):
                        y = sum(GAMMA**j * rews[i + j] for j in range(kk))
                        acc[(h, pref, k)].append((y, None))
                        continue
                    y = sum(GAMMA**j * rews[i + j] for j in range(k))
                    land = traj[i + k][1] if i + k < len(traj) else None
                    acc[(h, pref, k)].append((y, land))
        # keep lists: the bootstrap depends on V, resolved per-iteration
        within_only = dict(acc)

    def q_fn(h, c, k):
        if composed:
            return q_composed(h, tuple(c), k)
        lst = within_only.get((h, tuple(c[:k]), k))
        if lst is None:
            return None
        vals = []
        for y, land in lst:
            v = y
            if land is not None:
                v += GAMMA**k * V[land]
            vals.append(v)
        return float(np.mean(vals))

    # iterate: V(h) = E_{c ~ BC(obs(h))}[ max-k Q(h, c, k) ]  (deployed policy requeries with
    # the MARKOV chunk policy -- that is what deployment does)
    for _ in range(iters):
        newV = {}
        for h in hkeys:
            o = h[0]
            chunks = cont.get(o, [])
            if not chunks:
                newV[h] = 0.0
                continue
            vs = []
            for c in set(chunks):
                w = chunks.count(c) / len(chunks)
                _k, vmax = select_k(q_fn, h, c)
                vs.append(w * (vmax if vmax is not None else 0.0))
            newV[h] = float(sum(vs))
        V.update(newV)
    return q_fn


# ---------------------------------------------------------------- deployment
def deploy(env_rng, tables, qfn, key_mode, n_eps, *, kappa_fixed=None, oracle=False):
    env = Env(env_rng)
    cont = tables["cont"]
    succ = 0
    k_at = {"C_entry": [], "J_entry": []}
    beliefs, realized = [], []
    for _ in range(n_eps):
        o = env.reset()
        hist = ()
        disc_ret, first_belief = 0.0, None
        while True:
            seg, step = divmod(env.t, H)
            if step == 0:
                hist = ()
            chunks = cont.get(o)
            c = list(chunks[env_rng.integers(len(chunks))]) if chunks else [0] * H
            if oracle:
                # commit exactly the remaining segment; at a junction entry, react (k=1)
                k = (H - step) if not (SEGS[seg] == "J" and step == 0) else 1
            elif kappa_fixed is not None:
                k = kappa_fixed
            else:
                key = o if key_mode == "obs" else (o, hist)
                k, vmax = select_k(qfn, key, c)
                if first_belief is None and vmax is not None:
                    first_belief = vmax
            if SEGS[seg] == "C" and step == 0:
                k_at["C_entry"].append(k)
            if SEGS[seg] == "J" and step == 0:
                k_at["J_entry"].append(k)
            done = False
            for j in range(min(k, T - env.t)):
                o2, r, done = env.step(c[j])
                disc_ret += GAMMA ** (env.t - 1) * r
                # history = actions since the CURRENT segment's entry (reset on crossing)
                hist = () if env.t % H == 0 else (*hist, c[j])
                o = o2
                if done:
                    break
            if done:
                break
        succ += int(not env.failed)
        realized.append(disc_ret)
        beliefs.append(first_belief if first_belief is not None else 0.0)
    return {
        "sr": succ / n_eps,
        "mean_k_C": float(np.mean(k_at["C_entry"])) if k_at["C_entry"] else None,
        "mean_k_J": float(np.mean(k_at["J_entry"])) if k_at["J_entry"] else None,
        "belief_gap": float(np.mean(beliefs) - np.mean(realized)),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--eval-eps", type=int, default=2000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/probes/toy_cfac"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    arms = ["A0_naive", "A1_hist", "A2_polboot", "A3_cfac", "k1", "k2", "k3", "k4", "oracle"]
    results = {arm: [] for arm in arms}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        episodes = gen_demos(rng, a.episodes)
        tables = build_tables(episodes)

        q_a0 = chunk_regression(episodes, tables, "obs")
        q_a1 = chunk_regression(episodes, tables, "hist")

        def fn_a0(key, pref, k, _t=q_a0):
            return _t.get((key, pref, k))

        def fn_a1(key, pref, k, _t=q_a1):
            return _t.get((key, pref, k))

        fn_a2 = fqe_policy_bootstrap(episodes, tables, composed=False)
        fn_a3 = fqe_policy_bootstrap(episodes, tables, composed=True)

        ev = np.random.default_rng(10_000 + seed)
        results["A0_naive"].append(deploy(ev, tables, fn_a0, "obs", a.eval_eps))
        results["A1_hist"].append(deploy(ev, tables, fn_a1, "hist", a.eval_eps))
        results["A2_polboot"].append(deploy(ev, tables, fn_a2, "hist", a.eval_eps))
        results["A3_cfac"].append(deploy(ev, tables, fn_a3, "hist", a.eval_eps))
        for k in range(1, H + 1):
            results[f"k{k}"].append(deploy(ev, tables, None, "obs", a.eval_eps, kappa_fixed=k))
        results["oracle"].append(deploy(ev, tables, None, "obs", a.eval_eps, oracle=True))
        print(f"seed {seed} done", flush=True)

    summary = {}
    for arm in arms:
        rs = results[arm]
        summary[arm] = {
            m: (
                float(np.mean([r[m] for r in rs if r[m] is not None])),
                float(np.std([r[m] for r in rs if r[m] is not None])),
            )
            for m in ("sr", "mean_k_C", "mean_k_J", "belief_gap")
            if any(r[m] is not None for r in rs)
        }
    (a.out / "results.json").write_text(json.dumps({"summary": summary, "per_seed": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
