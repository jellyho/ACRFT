"""Small chunk policy and per-prefix critic for MimicGen low-dim tasks.

Sized for this cluster: a 3B VLA needed four GPUs and still ran out of memory, while these tasks are
51-dimensional. A small flow-matching transformer over the action chunk plays the same structural
role (one query returns H actions) at a scale that trains in minutes, so the questions we care
about, which are about commitment and value rather than perception, can be iterated on quickly.

  ChunkPolicy   obs -> H actions, trained by rectified-flow matching, sampled by Euler steps
  PrefixCritic  (obs, executed history, chunk) -> one value per prefix length, causal over the chunk
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], -1)


class Block(nn.Module):
    """Pre-norm transformer block with optional causal masking."""

    def __init__(self, dim, heads=4, *, causal=False):
        super().__init__()
        self.causal = causal
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.n1(x)
        mask = None
        if self.causal:
            n = x.shape[1]
            mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), 1)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ChunkPolicy(nn.Module):
    """Flow-matching policy over an action chunk.

    Rectified flow rather than a diffusion ladder: the straight-line target makes few-step and
    one-step sampling behave, which is the property the larger stack is also pursuing.
    """

    def __init__(self, obs_dim, act_dim, horizon, dim=256, depth=4, heads=4):
        super().__init__()
        self.h, self.a = horizon, act_dim
        self.obs = nn.Sequential(nn.Linear(obs_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.inp = nn.Linear(act_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)
        self.tim = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, act_dim))
        self.dim = dim

    def velocity(self, obs, x, t):
        c = self.obs(obs)[:, None] + self.tim(timestep_embedding(t, self.dim))[:, None]
        z = self.inp(x) + self.pos + c
        for b in self.blocks:
            z = b(z)
        return self.out(z)

    def loss(self, obs, chunk, weights=None):
        """Rectified flow: regress the straight-line velocity from noise to data."""
        b = chunk.shape[0]
        t = torch.rand(b, device=chunk.device)
        x0 = torch.randn_like(chunk)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * chunk
        v = self.velocity(obs, xt, t)
        per = ((v - (chunk - x0)) ** 2).mean((1, 2))
        return (per * weights).mean() if weights is not None else per.mean()

    @torch.no_grad()
    def sample(self, obs, steps=8):
        b = obs.shape[0]
        x = torch.randn(b, self.h, self.a, device=obs.device)
        for i in range(steps):
            t = torch.full((b,), i / steps, device=obs.device)
            x = x + self.velocity(obs, x, t) / steps
        return x


class PrefixCritic(nn.Module):
    """One pass over the chunk, one value per prefix length.

    The key carries the observation and, optionally, the actions already executed since the last
    query; the chunk is read causally so prefix k depends on the first k actions only.
    """

    def __init__(self, obs_dim, act_dim, horizon, *, use_hist=True, hist_len=8, dim=256, depth=3, heads=4):
        super().__init__()
        self.use_hist, self.hist_len, self.h = use_hist, hist_len, horizon
        kdim = obs_dim + (hist_len * (act_dim + 1) if use_hist else 0)
        self.key = nn.Sequential(nn.Linear(kdim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.inp = nn.Linear(act_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, causal=True) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, obs, hist, chunk):
        k = self.key(torch.cat([obs, hist], -1) if self.use_hist else obs)
        z = self.inp(chunk) + self.pos + k[:, None]
        for b in self.blocks:
            z = b(z)
        return self.out(z).squeeze(-1)  # (B, H)


def select_k(q, tie_eps=0.02, menu=None):
    """Longest admissible prefix within tie_eps of the best."""
    if menu is not None:
        mask = torch.full_like(q, float("-inf"))
        for k in menu:
            mask[:, k - 1] = q[:, k - 1]
        q = mask
    best = q.max(-1, keepdim=True).values
    ok = (q >= best - tie_eps) & torch.isfinite(q)
    idx = torch.arange(1, q.shape[1] + 1, device=q.device, dtype=torch.float32)
    return (ok.float() * idx).max(-1).values.long()
