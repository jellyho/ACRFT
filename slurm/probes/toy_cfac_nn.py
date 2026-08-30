"""CFAC under function approximation: does the actual algorithm work, and does it improve the policy?

The tabular toy (toy_cfac.py) showed the MECHANISM with exact enumeration and an empirical model.
This one runs the ALGORITHM we would ship: neural per-prefix critic, composed per-step TD (no model),
policy-expectation bootstrap, AWR chunk-policy improvement against that critic, lexicographic selector.

Environment "PlanReach" (continuous, so tabular enumeration is impossible):
  3 segments x H=4 steps. Action a in R^2. Target direction g ~ Uniform(unit circle) per segment.
    - corridor  (segments 0, 2): g is in the observation ONLY at the segment's first step, then hidden.
                                 Every step is scored against g.  => a PAST latent: commitment carries
                                 the plan, a Markov requery mid-corridor cannot recover it.
    - junction  (segment 1):     g is revealed only AFTER the first step (steps 1..3 scored, step 0 free).
                                 => a FUTURE latent: committing at entry must guess g, reacting wins.
  reward r = exp(-2 ||a - g||^2) on scored steps, 0 on the free step; gamma = 0.95.
  Demonstrator plays g + N(0, sigma^2) on scored steps (it remembers, so the data is non-Markovian),
  and a random direction on the free step.

Arms (all offline, same demo dataset, same BC chunk policy as the starting policy):
  bc_k1/2/4        fixed execution length with the BC chunk policy
  bc_oracle        BC policy + hand-crafted kappa* (commit corridors, react after the junction reveal)
  naive_sel        obs-keyed chunk-outcome regression + dataset-V bootstrap, selector only (policy frozen)
  cfac_sel         history-keyed critic, composed per-step TD, policy-expectation bootstrap, selector only
  cfac_joint       cfac critic + AWR improvement of the FULL chunk (Lemma B), selector, iterated

Pre-registered predictions (fixed before running; hub entry cfac-nn):
  V1 cfac_sel > naive_sel in deployed return, and the junction commitment drops for cfac only.
  V2 cfac_joint > cfac_sel: selection alone cannot absorb policy error (Lemma B / absorption).
  V3 curriculum: across improvement rounds the mean corridor commitment increases, while the junction
     commitment stays near 1. (This is prediction P3 of the four-forces synthesis, never yet measured.)
  V4 naive_sel over-commits at the junction (mean k >= 3) because chunk-outcome regression is confounded.
Rejected if: cfac_sel does not separate from naive_sel (V1), or cfac_joint does not exceed cfac_sel (V2),
or the corridor commitment fails to grow while returns improve (V3).

Run: python slurm/probes/toy_cfac_nn.py --seeds 6 --out /scratch/jellyho/acrft/probes/toy_cfac_nn
"""

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as torch_F  # noqa: N812

F = torch_F

H = 4
SEGS = ("C", "J", "C")
T = len(SEGS) * H
GAMMA = 0.95
DEMO_SIGMA = 0.25
MID_CUE_SIGMA = 0.0  # >0: the corridor keeps a NOISY cue mid-segment, so requerying is not fatal
TIE_EPS = 0.02
ADIM = 2
OBS_DIM = len(SEGS) + H + 2  # segment one-hot, step one-hot, cue (zeros when hidden)


# ------------------------------------------------------------------ environment
class PlanReach:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        ang = self.rng.uniform(0, 2 * np.pi, size=len(SEGS))
        self.g = np.stack([np.cos(ang), np.sin(ang)], 1)  # (3, 2)
        self.t = 0
        return self.obs()

    def _seg_step(self):
        return divmod(self.t, H)

    def scored(self, seg=None, step=None):
        seg, step = self._seg_step() if seg is None else (seg, step)
        return SEGS[seg] == "C" or step >= 1

    def obs(self):
        seg, step = self._seg_step()
        o = np.zeros(OBS_DIM, np.float32)
        o[seg] = 1.0
        o[len(SEGS) + step] = 1.0
        visible = (SEGS[seg] == "C" and step == 0) or (SEGS[seg] == "J" and step >= 1)
        if visible:
            o[-2:] = self.g[seg]
        elif MID_CUE_SIGMA > 0 and SEGS[seg] == "C":
            # a degraded cue: a Markov requery can partially recover the plan, so the corridor
            # trade-off becomes "keep a possibly-bad committed chunk" vs "resample policy noise"
            o[-2:] = self.g[seg] + self.rng.normal(0, MID_CUE_SIGMA, 2)
        return o

    def step(self, a):
        seg, step = self._seg_step()
        a = np.clip(a, -1, 1)
        r = float(np.exp(-2.0 * np.sum((a - self.g[seg]) ** 2))) if self.scored(seg, step) else 0.0
        self.t += 1
        done = self.t >= T
        return (None if done else self.obs()), r, done


def hist_feat(hist):
    """Actions taken since the current segment's entry, fixed-width with a validity mask."""
    f = np.zeros((H, ADIM + 1), np.float32)
    for i, a in enumerate(hist[-H:]):
        f[i, :ADIM] = a
        f[i, ADIM] = 1.0
    return f.reshape(-1)


HIST_DIM = H * (ADIM + 1)


# ------------------------------------------------------------------ data
def gen_demos(rng, n_ep):
    """Non-Markovian demonstrations: the demonstrator remembers g inside a corridor."""
    env = PlanReach(rng)
    obs_l, hi_l, act_l, rew_l, obs2_l, h2_l, done_l, ch_l = [], [], [], [], [], [], [], []
    for _ in range(n_ep):
        o = env.reset()
        hist, ep = [], []
        while True:
            seg, step = env._seg_step()
            if step == 0:
                hist = []
            g = env.g[seg]
            a = (g + rng.normal(0, DEMO_SIGMA, ADIM)) if env.scored() else rng.normal(0, 0.7, ADIM)
            a = np.clip(a, -1, 1).astype(np.float32)
            hf = hist_feat(hist)
            o2, r, done = env.step(a)
            hist = [] if env.t % H == 0 else [*hist, a]
            ep.append((o, hf, a, r, o2, hist_feat(hist), done))
            o = o2
            if done:
                break
        acts = [e[2] for e in ep]
        for i, (o_, hf, a, r, o2, hf2, d) in enumerate(ep):
            chunk = acts[i : i + H]
            chunk = chunk + [np.zeros(ADIM, np.float32)] * (H - len(chunk))
            obs_l.append(o_)
            hi_l.append(hf)
            act_l.append(a)
            rew_l.append(r)
            obs2_l.append(o2 if o2 is not None else np.zeros(OBS_DIM, np.float32))
            h2_l.append(hf2)
            done_l.append(float(d))
            ch_l.append(np.concatenate(chunk))
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)  # noqa: E731
    return {
        "o": t(obs_l),
        "h": t(hi_l),
        "a": t(act_l),
        "r": t(rew_l),
        "o2": t(obs2_l),
        "h2": t(h2_l),
        "d": t(done_l),
        "ch": t(ch_l),
    }


@torch.no_grad()
def gen_self_rollouts(rng_seed, pi, n_ep, exec_k=H):
    """Roll the current policy out and record what it did, in the same schema as gen_demos.

    The point is provenance, not performance: these actions were chosen without seeing the event
    that arrives inside the window, so a chunk-conditioned value fitted on them is not confounded.
    Human demonstrations do not have that property, which is what the interventional pairing was
    working around.
    """
    rng = np.random.default_rng(rng_seed)
    env = PlanReach(rng)
    obs_l, hi_l, act_l, rew_l, obs2_l, h2_l, done_l, ch_l = [], [], [], [], [], [], [], []
    for _ in range(n_ep):
        o = env.reset()
        hist, ep = [], []
        while True:
            seg, step = env._seg_step()
            if step == 0:
                hist = []
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            chunk = pi.sample(ot).view(H, ADIM).numpy()
            for j in range(min(exec_k, T - env.t)):
                a = np.clip(chunk[j], -1, 1).astype(np.float32)
                hf = hist_feat(hist)
                o2, r, done = env.step(a)
                hist = [] if env.t % H == 0 else [*hist, a]
                ep.append((o, hf, a, r, o2, hist_feat(hist), done))
                o = o2
                if done:
                    break
            if len(ep) and ep[-1][6]:
                break
        acts = [e[2] for e in ep]
        for i, (o_, hf, a, r, o2, hf2, d) in enumerate(ep):
            chunk_i = acts[i : i + H]
            chunk_i = chunk_i + [np.zeros(ADIM, np.float32)] * (H - len(chunk_i))
            obs_l.append(o_)
            hi_l.append(hf)
            act_l.append(a)
            rew_l.append(r)
            obs2_l.append(o2 if o2 is not None else np.zeros(OBS_DIM, np.float32))
            h2_l.append(hf2)
            done_l.append(float(d))
            ch_l.append(np.concatenate(chunk_i))
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)  # noqa: E731
    return {
        "o": t(obs_l),
        "h": t(hi_l),
        "a": t(act_l),
        "r": t(rew_l),
        "o2": t(obs2_l),
        "h2": t(h2_l),
        "d": t(done_l),
        "ch": t(ch_l),
    }


# ------------------------------------------------------------------ networks
class ChunkPolicy(nn.Module):
    """Markov chunk policy (the VLA analogue): obs -> H actions."""

    def __init__(self, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, H * ADIM)
        )
        self.log_std = nn.Parameter(torch.full((H * ADIM,), -1.5))

    def forward(self, o):
        return self.net(o)

    def sample(self, o, rng_t=None):
        mu = self.net(o)
        return mu + torch.randn_like(mu) * self.log_std.exp()


class PrefixCritic(nn.Module):
    """Per-prefix critic: key (obs [+ history]) x chunk -> one value per prefix length.

    Causal by construction: prefix k's head sees a running encoding of only the first k actions.
    """

    def __init__(self, use_hist, hid=128):
        super().__init__()
        self.use_hist = use_hist
        kdim = OBS_DIM + (HIST_DIM if use_hist else 0)
        self.key = nn.Sequential(nn.Linear(kdim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.cell = nn.GRUCell(ADIM, hid)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, o, h, chunk):
        k = self.key(torch.cat([o, h], -1) if self.use_hist else o)
        acts = chunk.view(-1, H, ADIM)
        out, s = [], k
        for j in range(H):
            s = self.cell(acts[:, j], s)
            out.append(self.head(s))
        return torch.cat(out, -1)  # (B, H): value of committing 1..H actions


def select_k(q, tie_eps=TIE_EPS):
    """Lexicographic: the longest prefix within tie_eps of the best."""
    best = q.max(-1, keepdim=True).values
    ok = (q >= best - tie_eps).float()
    idx = torch.arange(1, H + 1, dtype=torch.float32, device=q.device)
    return (ok * idx).max(-1).values.long()


# ------------------------------------------------------------------ training
def train_bc(data, steps, seed):
    torch.manual_seed(seed)
    pi = ChunkPolicy()
    opt = torch.optim.Adam(pi.parameters(), 1e-3)
    n = len(data["o"])
    for _ in range(steps):
        i = torch.randint(0, n, (256,))
        loss = F.mse_loss(pi(data["o"][i]), data["ch"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi


def dataset_v(data, steps, seed):
    """Markov regression of the discounted return-to-go (the naive arm's bootstrap)."""
    torch.manual_seed(seed + 1)
    v = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(v.parameters(), 1e-3)
    # discounted return-to-go within the episode (episodes are contiguous blocks of T)
    r = data["r"].view(-1, T)
    g = torch.zeros_like(r)
    acc = torch.zeros(r.shape[0])
    for j in reversed(range(T)):
        acc = r[:, j] + GAMMA * acc
        g[:, j] = acc
    g = g.reshape(-1, 1)
    n = len(data["o"])
    for _ in range(steps):
        i = torch.randint(0, n, (256,))
        loss = F.mse_loss(v(data["o"][i]), g[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return v


def train_naive_critic(data, vnet, steps, seed):
    """Chunk-outcome regression: y_k = sum_{j<k} gamma^j r_{t+j} + gamma^k V_data(s_{t+k}).

    Confounded by construction: the target conditions on the executed chunk, which the closed-loop
    demonstrator chose using information the observation does not carry.
    """
    torch.manual_seed(seed + 2)
    q = PrefixCritic(use_hist=False)
    opt = torch.optim.Adam(q.parameters(), 1e-3)
    r = data["r"].view(-1, T)
    n_ep = r.shape[0]
    with torch.no_grad():
        vall = vnet(data["o"]).view(-1, T)
    # y[ep, t, k]
    y = torch.zeros(n_ep, T, H)
    for t in range(T):
        acc = torch.zeros(n_ep)
        for k in range(1, H + 1):
            if t + k - 1 < T:
                acc = acc + GAMMA ** (k - 1) * r[:, t + k - 1]
            boot = GAMMA**k * vall[:, t + k] if t + k < T else torch.zeros(n_ep)
            y[:, t, k - 1] = acc + boot
    y = y.view(-1, H)
    n = len(data["o"])
    for _ in range(steps):
        i = torch.randint(0, n, (256,))
        pred = q(data["o"][i], data["h"][i], data["ch"][i])
        loss = F.mse_loss(pred, y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return q


def train_cfac_critic(data, pi, steps, seed, *, use_hist=True, interventional=True, q_init=None):
    """Composed per-step TD with an INTERVENTIONAL pairing and a policy-expectation bootstrap.

      Q_k(h_t, do(c_{1:k})) <- r_t + gamma * E_{s' ~ p(.|decision point)} [ Q_{k-1}(s', h_{t+1}, c_{2:k}) ]
      Q_1(h_t, do(c_1))     <- r_t + gamma * E_{s'} E_{c ~ pi(.|s')} [ Q_{kappa}(s', h_{t+1}, c) ]

    The pairing is the causal ingredient. Bootstrapping from the demonstration's OWN successor keeps
    the correlation between the executed tail and the event revealed inside the window (the
    demonstrator chose that tail knowing it), so the chunk value stays confounded even though the
    backup is composed. Resampling the successor among transitions at the same decision point,
    while holding the candidate tail fixed, integrates the revelation with its marginal: it is the
    do-operator, implemented without a model. `interventional=False` is the ablation that keeps the
    demonstration's own successor.
    """
    torch.manual_seed(seed + 3)
    q = PrefixCritic(use_hist=use_hist)
    if q_init is not None:
        q.load_state_dict(q_init.state_dict())
    tgt = PrefixCritic(use_hist=use_hist)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), 1e-3)
    n_ep = len(data["o"]) // T
    idx_all = torch.arange(len(data["o"])).view(n_ep, T)
    n_alt = 4  # successor resamples per update
    for it in range(steps):
        ep = torch.randint(0, n_ep, (128,))
        t = torch.randint(0, T - 1, (128,))
        i = idx_all[ep, t]
        with torch.no_grad():
            ys = []
            for _ in range(n_alt if interventional else 1):
                if interventional:
                    ep2 = torch.randint(0, n_ep, (128,))  # another episode at the SAME decision point
                    i_alt = idx_all[ep2, t]
                else:
                    i_alt = i  # the demonstration's own successor (ablation)
                o_next = data["o2"][i_alt]  # exogenous revelation, resampled
                h_next = data["h2"][i]  # our own executed history (the tail's context)
                c_pi = pi.sample(o_next)
                qn = tgt(o_next, h_next, c_pi)
                kk = select_k(qn)
                v_next = qn.gather(-1, kk.unsqueeze(-1) - 1).squeeze(-1)
                tail = torch.roll(data["ch"][i].view(-1, H, ADIM), -1, dims=1).reshape(-1, H * ADIM)
                q_tail = tgt(o_next, h_next, tail)  # Q_j of the SAME candidate tail at a resampled successor
                y = torch.zeros(len(i), H)
                y[:, 0] = data["r"][i] + GAMMA * (1 - data["d"][i]) * v_next
                for k in range(2, H + 1):
                    y[:, k - 1] = data["r"][i] + GAMMA * (1 - data["d"][i]) * q_tail[:, k - 2]
                ys.append(y)
            y = torch.stack(ys).mean(0)
        pred = q(data["o"][i], data["h"][i], data["ch"][i])
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            tgt.load_state_dict(q.state_dict())
    return q


def improve_actor(data, pi, q, steps, seed, beta=1.0):
    """AWR on the FULL chunk (Lemma B: improve what the deployed lower bound measures).

    Weights come from the same critic that picks the commitment, so both output dimensions are
    optimized against one value. No sampling-and-ranking happens at deployment.
    """
    torch.manual_seed(seed + 4)
    pi2 = ChunkPolicy()
    pi2.load_state_dict(pi.state_dict())
    opt = torch.optim.Adam(pi2.parameters(), 3e-4)
    n = len(data["o"])
    with torch.no_grad():
        adv = torch.zeros(n)
        for s in range(0, n, 4096):
            sl = slice(s, min(s + 4096, n))
            qa = q(data["o"][sl], data["h"][sl], data["ch"][sl])[:, -1]  # full-chunk value of the data chunk
            qb = torch.stack(  # V_pi(s): average over policy chunks, not a single sample
                [q(data["o"][sl], data["h"][sl], pi.sample(data["o"][sl]))[:, -1] for _ in range(4)]
            ).mean(0)
            adv[sl] = qa - qb
        w = torch.exp(torch.clamp(adv / beta, -5, 5))
        w = w / w.mean()
    for _ in range(steps):
        i = torch.randint(0, n, (256,))
        loss = (w[i].unsqueeze(-1) * (pi2(data["o"][i]) - data["ch"][i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi2


# ------------------------------------------------------------------ deployment
@torch.no_grad()
def rollout(seed, pi, q=None, *, fixed_k=None, oracle=False, n_ep=300):
    rng = np.random.default_rng(seed)
    env = PlanReach(rng)
    rets, kc, kj, react, seen_reveal = [], [], [], [], []
    for _ in range(n_ep):
        o = env.reset()
        hist, disc = [], 0.0
        while True:
            seg, step = env._seg_step()
            if step == 0:
                hist = []
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            ht = torch.as_tensor(hist_feat(hist), dtype=torch.float32).unsqueeze(0)
            c = pi.sample(ot)
            if oracle:
                k = 1 if (SEGS[seg] == "J" and step == 0) else H - step
            elif fixed_k is not None:
                k = fixed_k
            else:
                k = int(select_k(q(ot, ht, c))[0].item())
            if step == 0:
                (kj if SEGS[seg] == "J" else kc).append(k)
            if SEGS[seg] == "J" and step >= 1:
                react.append(1.0)  # a query happened after the reveal: g was visible when choosing
            acts = c.view(H, ADIM).numpy()
            done = False
            for j in range(min(k, T - env.t)):
                o2, r, done = env.step(acts[j])
                disc += GAMMA ** (env.t - 1) * r
                hist = [] if env.t % H == 0 else [*hist, np.clip(acts[j], -1, 1)]
                o = o2
                if done:
                    break
            if done:
                break
        rets.append(disc)
        seen_reveal.append(1.0 if react else 0.0)
        react.clear()
    return dict(  # noqa: C408
        ret=float(np.mean(rets)),
        k_corridor=float(np.mean(kc)) if kc else float("nan"),
        k_junction=float(np.mean(kj)) if kj else float("nan"),
        react_rate=float(np.mean(seen_reveal)),  # fraction of episodes that re-queried after the junction reveal
    )


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--eval-eps", type=int, default=300)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/probes/toy_cfac_nn"))
    ap.add_argument("--mid-cue-sigma", type=float, default=0.0, help="curriculum variant: noisy mid-corridor cue")
    ap.add_argument("--demo-sigma", type=float, default=DEMO_SIGMA, help="demonstrator action noise")
    a = ap.parse_args()
    globals()["MID_CUE_SIGMA"] = a.mid_cue_sigma
    globals()["DEMO_SIGMA"] = a.demo_sigma
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    per_seed = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        data = gen_demos(rng, a.episodes)
        pi_bc = train_bc(data, a.steps, seed)
        vnet = dataset_v(data, a.steps, seed)
        q_naive = train_naive_critic(data, vnet, a.steps, seed)
        # 2x2: history conditioning x interventional composition (plus the naive outcome-regression arm)
        cells = {
            "cfac": {"use_hist": True, "interventional": True},
            "cfac_nointerv": {"use_hist": True, "interventional": False},
            "cfac_nohist": {"use_hist": False, "interventional": True},
            "cfac_neither": {"use_hist": False, "interventional": False},
        }
        critics = {n: train_cfac_critic(data, pi_bc, a.steps, seed, **kw) for n, kw in cells.items()}
        q_cfac = critics["cfac"]

        res = {}
        for k in (1, 2, 4):
            res[f"bc_k{k}"] = rollout(9000 + seed, pi_bc, fixed_k=k, n_ep=a.eval_eps)
        res["bc_oracle"] = rollout(9000 + seed, pi_bc, oracle=True, n_ep=a.eval_eps)
        res["naive_sel"] = rollout(9000 + seed, pi_bc, q=q_naive, n_ep=a.eval_eps)
        for n, qq in critics.items():
            res[f"{n}_sel"] = rollout(9000 + seed, pi_bc, q=qq, n_ep=a.eval_eps)

        # joint: alternate policy improvement and critic re-evaluation
        pi_j, q_j, curriculum = pi_bc, q_cfac, []
        for rd in range(a.rounds):
            pi_j = improve_actor(data, pi_j, q_j, a.steps, seed + 10 * rd)
            q_j = train_cfac_critic(data, pi_j, a.steps, seed + 10 * rd, q_init=q_j)
            r = rollout(9000 + seed, pi_j, q=q_j, n_ep=a.eval_eps)
            curriculum.append(r)
            res[f"cfac_joint_r{rd + 1}"] = r
        res["cfac_joint"] = curriculum[-1]
        res["_curriculum"] = curriculum
        per_seed.append(res)
        print(
            f"seed {seed}: naive {res['naive_sel']['ret']:.2f}/react {res['naive_sel']['react_rate']:.2f} | "
            f"cfac {res['cfac_sel']['ret']:.2f}/react {res['cfac_sel']['react_rate']:.2f}/kC {res['cfac_sel']['k_corridor']:.2f} | "
            f"no-interv {res['cfac_nointerv_sel']['ret']:.2f}/react {res['cfac_nointerv_sel']['react_rate']:.2f} | "
            f"no-hist {res['cfac_nohist_sel']['ret']:.2f}/kC {res['cfac_nohist_sel']['k_corridor']:.2f} | "
            f"joint {res['cfac_joint']['ret']:.2f}/kC {res['cfac_joint']['k_corridor']:.2f}",
            flush=True,
        )

    arms = [k for k in per_seed[0] if not k.startswith("_")]
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

    # paired deltas (the verdicts)
    def paired(x, y):
        d = [s[x]["ret"] - s[y]["ret"] for s in per_seed]
        return [float(np.mean(d)), float(np.std(d)), int(sum(v > 0 for v in d)), len(d)]

    summary["_paired"] = {
        "cfac_sel-naive_sel": paired("cfac_sel", "naive_sel"),
        "cfac_sel-cfac_nointerv_sel": paired("cfac_sel", "cfac_nointerv_sel"),
        "cfac_sel-cfac_nohist_sel": paired("cfac_sel", "cfac_nohist_sel"),
        "cfac_joint-cfac_sel": paired("cfac_joint", "cfac_sel"),
        "cfac_joint-bc_oracle": paired("cfac_joint", "bc_oracle"),
    }
    (a.out / "results.json").write_text(json.dumps({"summary": summary, "per_seed": per_seed}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
