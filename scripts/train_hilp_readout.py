"""HILP as a READOUT on a frozen representation: train only phi, never reshape z.

Answers: does the frozen RLT token already CONTAIN the reachability metric (just not
surfaced), or does the space itself need shaping? Only a 2-layer phi is trained with the
same expectile-TD loss used in train_cheap_z; the base z is untouched.
"""
import argparse, json, pathlib
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_cache_PrepareCoffee"))
ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/rlt_hilp_readout"))
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--gamma", type=float, default=0.98)
ap.add_argument("--tau", type=float, default=0.9)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--heldout-frac", type=float, default=0.0,
                help="fraction of EPISODES excluded from TD training (generalization check)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
meta = json.loads((a.cache/"meta.json").read_text()); n, D = meta["num_frames"], meta["dim"]
z = np.load(a.cache/"features.npy").astype(np.float32)
z = (z - z.mean(0)) / (z.std(0) + 1e-6)
ep = np.load(a.cache/"episode_index.npy")
z_t = torch.from_numpy(z).to(dev)

eps_u, starts = np.unique(ep, return_index=True)
ends = np.r_[starts[1:], n]
ep_end = np.zeros(n, dtype=np.int64)
for e,s,t in zip(eps_u, starts, ends): ep_end[s:t] = t-1
held = np.array([], dtype=eps_u.dtype)
train_mask = np.ones(n, dtype=bool)
if a.heldout_frac > 0:
    perm = rng.permutation(eps_u)
    held = perm[: max(1, int(len(eps_u) * a.heldout_frac))]
    train_mask = ~np.isin(ep, held)
    print(f"holding out {len(held)} episodes from TD training", flush=True)
train_rows = np.flatnonzero(train_mask)

phi = nn.Sequential(nn.Linear(D,256), nn.GELU(), nn.Linear(256,a.dim)).to(dev)
tgt = nn.Sequential(nn.Linear(D,256), nn.GELU(), nn.Linear(256,a.dim)).to(dev)
tgt.load_state_dict(phi.state_dict())
for q in tgt.parameters(): q.requires_grad_(False)
opt = torch.optim.AdamW(phi.parameters(), lr=3e-4, weight_decay=1e-5)

for step in range(a.steps):
    B = 512
    s = rng.choice(train_rows, B); s = np.where(s+1 <= ep_end[s], s, s-1)
    nx = s+1
    u = rng.random(B)
    fut = np.minimum(s + rng.geometric(1-a.gamma, B), ep_end[s])
    g = np.where(u<0.2, nx, np.where(u<0.7, fut, rng.choice(train_rows, B)))
    si,ni,gi = (torch.from_numpy(x).to(dev) for x in (s,nx,g))
    v  = -torch.norm(phi(z_t[si]) - phi(z_t[gi]), dim=-1)
    with torch.no_grad():
        vn = -torch.norm(tgt(z_t[ni]) - tgt(z_t[gi]), dim=-1)
    ng = (si!=gi).float()
    td = (-ng) + a.gamma*vn*ng - v
    w = torch.abs(a.tau - (td<0).float())
    loss = (w*td**2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 5 == 0:
        with torch.no_grad():
            for q,p_ in zip(tgt.parameters(), phi.parameters()): q.mul_(0.995).add_(p_, alpha=0.005)
    if step % 5000 == 0: print(f"step {step}: td {loss.item():.4f}", flush=True)

a.out.mkdir(parents=True, exist_ok=True)
torch.save(phi.state_dict(), a.out/"phi.pt")
with torch.no_grad():
    ph = np.concatenate([phi(z_t[i:i+8192]).cpu().numpy() for i in range(0,n,8192)])
np.save(a.out/"z.npy", ph)  # probe scripts read z.npy — phi IS the space here
np.save(a.out/"held_episodes.npy", held)
print("saved phi-space as z.npy for probing")
