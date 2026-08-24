"""Train the baseline ladder on a MimicGen task.

Every demonstration in these datasets succeeds, so a success filter does nothing. What still varies
is how efficiently the task was done, and a discounted return exposes that without extra labels, so
the ladder is rebuilt on that axis:

    bc        imitate everything
    fast      imitate the quickest third of the episodes (the filter analogue)
    awr       weight each recorded chunk by its advantage under the critic
    cfac      the same weighting, plus a commitment chosen by the same critic at deployment

The critic is trained on the SMDP backup with the requery leaf priced by the current policy, which
is the part that makes prefix values comparable across lengths.

    uv run python slurm/probes/mimicgen_train.py --task three_piece_assembly_d0 --arm awr
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "slurm/probes")

from mimicgen_models import ChunkPolicy
from mimicgen_models import PrefixCritic
from mimicgen_models import select_k
import numpy as np
import torch


def load(task, root, horizon, hist_len, device):
    z = np.load(root / f"{task}.npz")
    meta = json.loads((root / f"{task}_meta.json").read_text())
    obs, act = z["obs"], z["action"]
    o_mu, o_sd = obs.mean(0), obs.std(0) + 1e-6
    obs = (obs - o_mu) / o_sd
    ep = z["episode"]
    starts = np.flatnonzero(np.diff(ep, prepend=-1))
    ends = np.append(starts[1:], len(ep))

    # index of every decision point that has a full chunk inside its own episode
    idx, chunks, hists, nxt = [], [], [], []
    for s, e in zip(starts, ends, strict=True):
        for t in range(s, e - horizon):
            idx.append(t)
            chunks.append(act[t : t + horizon])
            h = np.zeros((hist_len, act.shape[1] + 1), np.float32)
            for j in range(min(hist_len, t - s)):
                h[j, :-1] = act[t - 1 - j]
                h[j, -1] = 1.0
            hists.append(h.reshape(-1))
            nxt.append(t + 1)
    t_ = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    data = {
        "obs": t_(obs[idx]),
        "chunk": t_(chunks),
        "hist": t_(hists),
        "reward": t_(z["reward"][idx]),
        "rtg": t_(z["rtg"][idx]),
        "obs_next": t_(obs[nxt]),
        "ep": torch.as_tensor(ep[idx], device=device),
        "ep_len": torch.as_tensor(z["ep_lengths"], device=device),
    }
    # the successor's history is this one shifted by the action just executed
    hn = np.asarray(hists).copy()
    hn = np.roll(hn.reshape(len(hn), hist_len, -1), 1, axis=1)
    hn[:, 0, :-1] = act[idx]
    hn[:, 0, -1] = 1.0
    data["hist_next"] = t_(hn.reshape(len(hn), -1))
    data["norm"] = {"o_mu": o_mu.tolist(), "o_sd": o_sd.tolist()}
    data["meta"] = meta
    return data


def train_critic(data, pi, steps, horizon, gamma, device, *, use_hist=True, lr=3e-4, batch=256, log=None):
    obs_dim = data["obs"].shape[1]
    act_dim = data["chunk"].shape[2]
    hist_len = data["hist"].shape[1] // (act_dim + 1)
    q = PrefixCritic(obs_dim, act_dim, horizon, use_hist=use_hist, hist_len=hist_len).to(device)
    tgt = PrefixCritic(obs_dim, act_dim, horizon, use_hist=use_hist, hist_len=hist_len).to(device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr)
    n = len(data["obs"])
    for it in range(steps):
        i = torch.randint(0, n, (batch,), device=device)
        with torch.no_grad():
            o2, h2 = data["obs_next"][i], data["hist_next"][i]
            c2 = pi.sample(o2, steps=4)  # what the policy would do if queried again
            qn = tgt(o2, h2, c2)
            v_next = qn.gather(-1, (select_k(qn) - 1).unsqueeze(-1)).squeeze(-1)
            # the same chunk shifted by one: continuing the plan rather than requerying
            shifted = torch.roll(data["chunk"][i], -1, dims=1)
            q_tail = tgt(o2, h2, shifted)
            y = torch.zeros(batch, horizon, device=device)
            y[:, 0] = data["reward"][i] + gamma * v_next
            for k in range(2, horizon + 1):
                y[:, k - 1] = data["reward"][i] + gamma * q_tail[:, k - 2]
        loss = ((q(data["obs"][i], data["hist"][i], data["chunk"][i]) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0:
            tgt.load_state_dict(q.state_dict())
            if log is not None:
                log.append({"step": it, "critic_loss": float(loss)})
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="three_piece_assembly_d0")
    ap.add_argument("--arm", default="bc", choices=["bc", "fast", "awr", "cfac"])
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/mimicgen/lowdim"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/mimicgen/runs"))
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--hist-len", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--critic-steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    data = load(a.task, a.root, a.horizon, a.hist_len, device)
    obs_dim, act_dim = data["obs"].shape[1], data["chunk"].shape[2]
    print(f"{a.task}: {len(data['obs'])} decision points, obs {obs_dim}, act {act_dim}, device {device}", flush=True)

    pi = ChunkPolicy(obs_dim, act_dim, a.horizon).to(device)
    opt = torch.optim.Adam(pi.parameters(), 3e-4)
    n = len(data["obs"])

    # weights per decision point, one per arm
    weights = torch.ones(n, device=device)
    critic = None
    if a.arm == "fast":
        # the filter analogue: keep the quickest third of the episodes
        lens = data["ep_len"].float()
        cut = torch.quantile(lens, 1 / 3)
        keep = (lens[data["ep"]] <= cut).float()
        weights = keep * (n / keep.sum().clamp(min=1))
        print(f"fast filter keeps {float(keep.mean()):.2f} of the data", flush=True)
    elif a.arm in ("awr", "cfac"):
        # a short behaviour-cloning warm start gives the critic a policy to price the requery with
        for _ in range(a.steps // 4):
            i = torch.randint(0, n, (a.batch,), device=device)
            loss = pi.loss(data["obs"][i], data["chunk"][i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        critic = train_critic(data, pi, a.critic_steps, a.horizon, a.gamma, device)
        with torch.no_grad():
            adv = torch.zeros(n, device=device)
            for s in range(0, n, 4096):
                sl = slice(s, min(s + 4096, n))
                qa = critic(data["obs"][sl], data["hist"][sl], data["chunk"][sl])[:, -1]
                qb = torch.stack(
                    [
                        critic(data["obs"][sl], data["hist"][sl], pi.sample(data["obs"][sl], steps=4))[:, -1]
                        for _ in range(2)
                    ]
                ).mean(0)
                adv[sl] = qa - qb
            weights = torch.exp((adv / (adv.std() + 1e-6) / a.beta).clamp(-5, 5))
            weights = weights / weights.mean()
        print(f"advantage weights: mean {float(weights.mean()):.2f} max {float(weights.max()):.2f}", flush=True)

    hist_log = []
    for it in range(a.steps):
        i = torch.randint(0, n, (a.batch,), device=device)
        loss = pi.loss(data["obs"][i], data["chunk"][i], weights=weights[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 1000 == 0:
            hist_log.append({"step": it, "loss": float(loss)})
            print(f"  step {it} loss {float(loss):.4f}", flush=True)

    out = a.out / f"{a.task}_{a.arm}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": pi.state_dict(),
            "critic": critic.state_dict() if critic is not None else None,
            "cfg": {
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "horizon": a.horizon,
                "hist_len": a.hist_len,
                "arm": a.arm,
                "task": a.task,
                "gamma": a.gamma,
                "env_name": data["meta"]["env_name"],
            },
            "norm": data["norm"],
        },
        out / "ckpt.pt",
    )
    (out / "train_log.json").write_text(json.dumps({"loss": hist_log}, indent=1))
    print("saved", out / "ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
