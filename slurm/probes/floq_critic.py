# ruff: noqa
"""Faithful floq (arXiv:2509.06863) on OUR ARQ critic architecture — only the bootstrap+head swap.

Keeps the ARQCritic trunk (causal transformer over [obs, action-macro] tokens). The head becomes a
velocity field v(t, z | s, a) (flow_head=True, with the paper's collapse-prevention: categorical z +
Fourier t). Compared against the SAME ARQCritic with the standard scalar head (num_atoms=1, expectile
TD), so the only difference is the value head + bootstrap.

floq loss (Eq 4.2):  E_{z~U[l,u], t~U(0,1)} || v(t, z(t)|s,a) - (y - z) ||²,  z(t)=(1-t)z + t·y
  y = TD target = r_to_h + γ^h · V_next(1-ended) with mc_floor; V_next read by INTEGRATING the
  TARGET velocity field over the next-state candidates (min-over-ensemble not used here; single trunk).
Q read: integrate v from z0~U[l,u] over K steps to t=1.

Single-commit (full chunk) TD for a clean floq-vs-scalar isolation — the ARQ prefix machinery is
orthogonal to floq. Metrics on held-out (by episode): Spearman(Q,mc), action-sensitivity, demo_winrate.

Usage: floq_critic.py [steps=30000]
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from openpi.rlt_critic.critic import ARQCritic
from openpi.rlt_critic.critic import HLGauss

C = os.environ["CACHE_DIR"]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
D = f"{C}/annot/mixed"
meta = json.load(open(f"{D}/meta.json"))
T, Dt = meta["num_frames"], meta["token_dim"]
H, A, NS = meta["horizon"], meta["action_dim"], meta["num_samples"]
GAMMA = float(os.environ.get("GAMMA_OVR", "0.997"))  # override dataset 0.99 for a longer effective horizon
GH = GAMMA**H  # full-chunk commit
LO, HI = -0.5, 1.0  # floq design-choice-1 interval around the mc range [0,1] (u=Q_max=1)
KI = 10  # integration steps

tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
cand = np.memmap(f"{D}/base_action.dat", dtype=np.float32, mode="r", shape=(T, NS, H, A))
mc = np.asarray(np.memmap(f"{D}/mc_return.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
rew = np.asarray(np.memmap(f"{D}/reward.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
done = np.zeros(T, np.float32)
ep = np.asarray(np.memmap(f"{D}/episode_index.dat", dtype=np.int32, mode="r", shape=(T,)))

rng = np.random.default_rng(0)
hold = set(rng.choice(np.unique(ep), max(1, len(np.unique(ep)) // 5), replace=False).tolist())
ishold = np.array([e in hold for e in ep])
tr_idx = np.flatnonzero(~ishold)
ho_idx = np.flatnonzero(ishold)

POOL = min(200000, len(tr_idx))
tr = np.sort(rng.choice(tr_idx, POOL, replace=False))
tmu, tsd = np.asarray(tok[tr]).mean(0), np.asarray(tok[tr]).std(0) + 1e-6
asub = np.asarray(act[tr[:40000]], np.float32).reshape(-1, A)
amu, asd = asub.mean(0), asub.std(0) + 1e-6


def zof(gidx):
    return ((np.asarray(tok[gidx], np.float32) - tmu) / tsd).astype(np.float32)


def aof(chunk):
    return ((chunk.reshape(-1, A) - amu) / asd).reshape(chunk.shape).astype(np.float32)


# next-state index for the full-chunk landing (clamp within episode)
nxt = np.minimum(np.arange(T) + H, T - 1)
same_ep_next = ep[nxt] == ep
ended = (~same_ep_next).astype(np.float32)  # left the episode within the chunk

MG = 2
NA = 51  # HL-Gauss atoms — matched to floq flow_bins for equal head capacity
scalar_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=1)
hlg_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=NA)
hlg = HLGauss(v_min=0.0, v_max=1.0, num_atoms=NA)  # returns live in [0,1] (sparse reward + mc_floor)
flow_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, flow_head=True, flow_lo=LO, flow_hi=HI)
LAST = H // MG - 1  # index of the full-chunk prefix (last macro)


def integrate(apply, params, z_obs, a_norm, key, n=KI):
    """Read Q by ODE integration of the flow head from z0~U[l,u] to t=1 (full-chunk prefix)."""
    B = z_obs.shape[0]
    q = jax.random.uniform(key, (B,), minval=LO, maxval=HI)
    dt = 1.0 / n
    for i in range(n):
        t = jnp.full((B,), i * dt)
        # broadcast z,t across prefixes; take the LAST (full-chunk) prefix value
        v = apply(
            params,
            z_obs,
            a_norm,
            jnp.broadcast_to(q[:, None], (B, H // MG)),
            jnp.broadcast_to(t[:, None], (B, H // MG)),
        )
        q = q + dt * v[:, LAST]
    return q


def main():
    key = jax.random.key(0)
    Ztr, Atr = jnp.asarray(zof(tr)), jnp.asarray(aof(np.asarray(act[tr], np.float32)))
    mc_tr = jnp.asarray(mc[tr])
    r_tr = jnp.asarray(rew[tr])
    ended_tr = jnp.asarray(ended[tr])
    # next-state obs + executed next chunk (for V_next bootstrap, single action = the demo's next)
    Znx = jnp.asarray(zof(nxt[tr]))
    Anx = jnp.asarray(aof(np.asarray(act[nxt[tr]], np.float32)))

    # ---------- scalar ARQ critic (expectile TD baseline) ----------
    def train_scalar():
        p = scalar_net.init(key, jnp.zeros((1, Dt)), jnp.zeros((1, H, A)))
        tgt = p
        tx = optax.adam(3e-4)
        opt = tx.init(p)

        @jax.jit
        def step(p, tgt, opt, k):
            b = jax.random.randint(k, (256,), 0, Ztr.shape[0])
            vnext = scalar_net.apply(tgt, Znx[b], Anx[b])[:, LAST]
            y = r_tr[b] + GH * (1 - ended_tr[b]) * vnext
            y = jnp.maximum(y, mc_tr[b])

            def loss(pp):
                q = scalar_net.apply(pp, Ztr[b], Atr[b])[:, LAST]
                u = y - q
                w = jnp.abs(0.9 - (u < 0).astype(jnp.float32))  # expectile 0.9
                return jnp.mean(w * u**2)

            v, g = jax.value_and_grad(loss)(p)
            up, opt2 = tx.update(g, opt)
            p2 = optax.apply_updates(p, up)
            tgt2 = jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2)
            return p2, tgt2, opt2, v

        k = key
        for s in range(STEPS):
            k, kk = jax.random.split(k)
            p, tgt, opt, _ = step(p, tgt, opt, kk)
        return lambda zo, an: scalar_net.apply(p, zo, an)[:, LAST]

    # ---------- HL-Gauss ARQ critic (categorical classification head, same trunk) ----------
    def train_hlgauss():
        p = hlg_net.init(key, jnp.zeros((1, Dt)), jnp.zeros((1, H, A)))
        tgt = p
        tx = optax.adam(3e-4)
        opt = tx.init(p)

        @jax.jit
        def step(p, tgt, opt, k):
            b = jax.random.randint(k, (256,), 0, Ztr.shape[0])
            vnext = hlg.from_logits(hlg_net.apply(tgt, Znx[b], Anx[b])[:, LAST])  # read scalar V_next
            y = r_tr[b] + GH * (1 - ended_tr[b]) * vnext
            y = jnp.maximum(y, mc_tr[b])
            probs = hlg.to_probs(y)  # Gaussian-smoothed categorical target [256, NA]

            def loss(pp):
                logits = hlg_net.apply(pp, Ztr[b], Atr[b])[:, LAST]  # [256, NA]
                return -jnp.mean(jnp.sum(probs * jax.nn.log_softmax(logits, -1), -1))

            v, g = jax.value_and_grad(loss)(p)
            up, opt2 = tx.update(g, opt)
            p2 = optax.apply_updates(p, up)
            tgt2 = jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2)
            return p2, tgt2, opt2, v

        k = key
        for s in range(STEPS):
            k, kk = jax.random.split(k)
            p, tgt, opt, _ = step(p, tgt, opt, kk)
        return lambda zo, an: hlg.from_logits(hlg_net.apply(p, zo, an)[:, LAST])

    # ---------- floq ARQ critic ----------
    def train_floq():
        p = flow_net.init(
            key, jnp.zeros((1, Dt)), jnp.zeros((1, H, A)), jnp.zeros((1, H // MG)), jnp.zeros((1, H // MG))
        )
        tgt = p
        tx = optax.adam(3e-4)
        opt = tx.init(p)

        @jax.jit
        def step(p, tgt, opt, k):
            k1, k2, k3, k4 = jax.random.split(k, 4)
            b = jax.random.randint(k1, (256,), 0, Ztr.shape[0])
            vnext = integrate(flow_net.apply, tgt, Znx[b], Anx[b], k2)  # bootstrap via TARGET flow integration
            y = r_tr[b] + GH * (1 - ended_tr[b]) * vnext
            y = jnp.maximum(y, mc_tr[b])
            z0 = jax.random.uniform(k3, (256,), minval=LO, maxval=HI)
            t = jax.random.uniform(k4, (256,))
            zt = (1 - t) * z0 + t * y  # interpolant
            target_v = y - z0  # linear flow-matching velocity

            def loss(pp):
                v = flow_net.apply(
                    pp,
                    Ztr[b],
                    Atr[b],
                    jnp.broadcast_to(zt[:, None], (256, H // MG)),
                    jnp.broadcast_to(t[:, None], (256, H // MG)),
                )[:, LAST]
                return jnp.mean((v - target_v) ** 2)

            val, g = jax.value_and_grad(loss)(p)
            up, opt2 = tx.update(g, opt)
            p2 = optax.apply_updates(p, up)
            tgt2 = jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2)
            return p2, tgt2, opt2, val

        k = key
        for s in range(STEPS):
            k, kk = jax.random.split(k)
            p, tgt, opt, _ = step(p, tgt, opt, kk)
        rk = [jax.random.key(7)]

        def read(zo, an):
            rk[0], kk = jax.random.split(rk[0])
            return integrate(flow_net.apply, p, zo, an, kk)

        return read

    def evaluate(qfn, name):
        ho = np.sort(rng.choice(ho_idx, min(5000, len(ho_idx)), replace=False))
        qe = np.asarray(qfn(jnp.asarray(zof(ho)), jnp.asarray(aof(np.asarray(act[ho], np.float32)))))
        sp = float(spearmanr(qe, mc[ho]).statistic)
        sub = ho[:1500]
        cf = np.asarray(cand[sub], np.float32)
        qc = np.stack([np.asarray(qfn(jnp.asarray(zof(sub)), jnp.asarray(aof(cf[:, j])))) for j in range(NS)], 1)
        qd = qe[np.searchsorted(ho, sub)]
        out = {
            "spearman_Q_mc": round(sp, 4),
            "action_sensitivity": round(float(np.mean(np.std(qc, 1))), 5),
            "demo_winrate": round(float(np.mean(qd[:, None] > qc)), 4),
        }
        print("FLOQC", name, json.dumps(out), flush=True)
        return out

    res = {
        "gamma": GAMMA,
        "interval": [LO, HI],
        "K": KI,
        "hlg_atoms": NA,
        "scalar_arq": evaluate(train_scalar(), "scalar_arq"),
        "hlgauss_arq": evaluate(train_hlgauss(), "hlgauss_arq"),
        "floq_arq": evaluate(train_floq(), "floq_arq"),
    }
    json.dump(res, open(f"{C}/probes/floq_critic.json", "w"), indent=2)
    print("FLOQC_DONE")


if __name__ == "__main__":
    main()
