# ruff: noqa
"""DEAS-style value learning (arXiv:2510.07730) on OUR critic + BoN — does it beat the VLA?

The td-max probe (eval_bon_pfx.py) FAILED because V_next = max_j Q(s', cand_j) maximizes the very
overestimation DEAS warns against ("directly adopting such sequences ... introduces excessive value
overestimation, which we address through detached value learning that steers value estimates toward
in-distribution actions that achieve high return"). DEAS replaces the max bootstrap with an
IQL-expectile state value V and bootstraps Q FROM V — never maxing over OOD candidates.

Faithful to gr00t/model/action_head/deas_critic.py:
  V loss (expectile + HL-Gauss):  g = where(q_demo >= V, tau, 1-tau);  L_V = mean( g * CE(V_logits, HLGauss(q_demo)) )
      q_demo = min(Q1_target, Q2_target)(s, a_demo)      # in-distribution demo action, detached
  Q loss (bootstrap from V):       target = sum_i g1^i r_i + g2^H (1-done) V(s');  L_Q = (CE(Q1,target)+CE(Q2,target))/2
  double critic (min), EMA targets, HL-Gauss on BOTH V and Q.

Three Q-head variants share this DEAS value learning (V is HL-Gauss+expectile throughout):
  scalar  : Q scalar, L_Q = MSE to target;  V scalar, expectile MSE.
  hlgauss : Q HL-Gauss CE to target;        V HL-Gauss expectile-CE.   (DEAS canonical)
  floq    : Q flow-matching to target;      V HL-Gauss expectile-CE.
Deploy = BoN: score = min(Q1,Q2)(z, cand) over N VLA candidates, arg-max (full chunk, as DEAS).
Controls: vla (candidate 0), rand. Scene-paired, Wilson + McNemar. Single-commit full chunk.

Usage: eval_deas.py [head=hlgauss] [trials=25] [N=8] [seed=8000] [steps=20000]
"""

import dataclasses
import json
import math
import os
import pathlib
import sys

import flax.linen as fnn
import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "examples/robocasa")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "third_party/robocasa")
from openpi.rlt_critic.critic import ARQCritic, HLGauss

C = os.environ["CACHE_DIR"]
HEAD = sys.argv[1] if len(sys.argv) > 1 else "hlgauss"
NTRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
NCAND = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
GAMMA = float(os.environ.get("GAMMA_OVR", "0.997"))
CKPT = "/scratch/jellyho/acrft/checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_noprop/70000"
CONFIG = "pi05_robocasa_PrepareCoffee_rlt"
SAVE = pathlib.Path(f"{C}/critic_runs/deas")
SAVE.mkdir(parents=True, exist_ok=True)

D = f"{C}/annot/mixed"
meta = json.load(open(f"{D}/meta.json"))
T, Dt = meta["num_frames"], meta["token_dim"]
H, A, NS = meta["horizon"], meta["action_dim"], meta["num_samples"]
MG, NA, KI, TAU = 2, 51, 8, 0.9  # macro group, atoms, floq integration steps, expectile
LO, HI = -0.5, 1.0
VMIN, VMAX = 0.0, 1.0
GH = GAMMA**H
LAST = H // MG - 1
hlg = HLGauss(v_min=VMIN, v_max=VMAX, num_atoms=NA)

ATOMS = {"scalar": 1, "hlgauss": NA, "floq": 1}[HEAD]
FLOW = HEAD == "floq"
q_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=ATOMS, flow_head=FLOW, flow_lo=LO, flow_hi=HI)


class ValueMLP(fnn.Module):
    atoms: int = NA

    @fnn.compact
    def __call__(self, z):
        x = fnn.LayerNorm()(z)
        for h in (512, 512):
            x = fnn.gelu(fnn.Dense(h)(x))
        out = fnn.Dense(self.atoms)(x)
        return out if self.atoms > 1 else jnp.squeeze(out, -1)


v_net = ValueMLP(atoms=(1 if HEAD == "scalar" else NA))


def q_integrate(params, z, a, key, n=KI):
    lead = z.shape[:-1]
    q = jax.random.uniform(key, (*lead, H // MG), minval=LO, maxval=HI)
    dt = 1.0 / n
    for i in range(n):
        t = jnp.full((*lead, H // MG), i * dt)
        q = q + dt * q_net.apply(params, z, a, q, t)
    return q[..., LAST]


def q_read(params, z, a, key=None):
    if FLOW:
        return q_integrate(params, z, a, key)
    out = q_net.apply(params, z, a)  # [...,P] or [...,P,NA]
    return (hlg.from_logits(out) if ATOMS > 1 else out)[..., LAST]


def v_read(vp, z):
    out = v_net.apply(vp, z)
    return hlg.from_logits(out) if HEAD != "scalar" else out


def build_stats():
    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(T, min(200000, T), replace=False))
    tmu, tsd = np.asarray(tok[s]).mean(0), np.asarray(tok[s]).std(0) + 1e-6
    asub = np.asarray(act[s[:40000]], np.float32).reshape(-1, A)
    amu, asd = asub.mean(0), asub.std(0) + 1e-6
    return tmu.astype(np.float32), tsd.astype(np.float32), amu.astype(np.float32), asd.astype(np.float32)


def train():
    tmu, tsd, amu, asd = build_stats()
    np.savez(SAVE / "stats.npz", tmu=tmu, tsd=tsd, amu=amu, asd=asd)
    qpath, vpath = SAVE / f"{HEAD}_q.msgpack", SAVE / f"{HEAD}_v.msgpack"
    if qpath.exists() and vpath.exists():
        print("LOAD", HEAD, flush=True)
        return fser.msgpack_restore(qpath.read_bytes()), fser.msgpack_restore(vpath.read_bytes())

    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    rew = np.asarray(np.memmap(f"{D}/reward.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
    ep = np.asarray(np.memmap(f"{D}/episode_index.dat", dtype=np.int32, mode="r", shape=(T,)))
    done = np.asarray(np.memmap(f"{D}/done.dat", dtype=np.uint8, mode="r", shape=(T,)), np.float32)
    zof = lambda g: ((np.asarray(tok[g], np.float32) - tmu) / tsd).astype(np.float32)
    aof = lambda ch: ((ch.reshape(-1, A) - amu) / asd).reshape(ch.shape).astype(np.float32)

    rng = np.random.default_rng(1)
    pool = np.sort(rng.choice(T, min(200000, T), replace=False))
    nxt = np.minimum(pool + H, T - 1)
    win = np.clip(pool[:, None] + np.arange(H)[None, :], 0, T - 1)
    in_ep = (ep[win] == ep[pool][:, None]).astype(np.float32)
    disc1 = (GAMMA ** np.arange(H)).astype(np.float32)
    scaled_r = np.sum(rew[win] * in_ep * disc1[None, :], axis=1).astype(np.float32)  # inner-MDP reward
    ended = (ep[nxt] != ep[pool]).astype(np.float32)
    done_land = done[nxt].astype(np.float32)
    B_all = len(pool)
    Z = jnp.asarray(zof(pool))
    Adem = jnp.asarray(aof(np.asarray(act[pool], np.float32)))
    Znx = jnp.asarray(zof(nxt))
    SR = jnp.asarray(scaled_r)
    EN = jnp.asarray(ended)
    DL = jnp.asarray(done_land)

    key0 = jax.random.key(0)
    qp1 = q_net.init(
        key0,
        jnp.zeros((1, Dt)),
        jnp.zeros((1, H, A)),
        *((jnp.zeros((1, H // MG)), jnp.zeros((1, H // MG))) if FLOW else ()),
    )
    qp2 = q_net.init(
        jax.random.key(2),
        jnp.zeros((1, Dt)),
        jnp.zeros((1, H, A)),
        *((jnp.zeros((1, H // MG)), jnp.zeros((1, H // MG))) if FLOW else ()),
    )
    vp = v_net.init(key0, jnp.zeros((1, Dt)))
    tq1, tq2 = qp1, qp2
    txq, txv = optax.adam(3e-4), optax.adam(3e-4)
    oq1, oq2, ov = txq.init(qp1), txq.init(qp2), txv.init(vp)

    def hlg_ce(logits, target):
        return -jnp.mean(jnp.sum(hlg.to_probs(jnp.clip(target, VMIN, VMAX)) * jax.nn.log_softmax(logits, -1), -1))

    @jax.jit
    def step(qp1, qp2, vp, tq1, tq2, oq1, oq2, ov, k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        b = jax.random.randint(k1, (256,), 0, B_all)
        z, a, znx = Z[b], Adem[b], Znx[b]

        # ----- V loss: expectile toward min-double target-Q(z, a_demo) -----
        q1d = q_read(tq1, z, a, k2)
        q2d = q_read(tq2, z, a, k3)
        q_demo = jnp.minimum(q1d, q2d)  # in-distribution, detached

        def v_loss_fn(vp):
            vs = v_read(vp, z)
            g = jnp.where(q_demo >= vs, TAU, 1 - TAU)
            if HEAD == "scalar":
                return jnp.mean(g * (jax.lax.stop_gradient(q_demo) - vs) ** 2)
            logits = v_net.apply(vp, z)  # [B,NA]
            ce = -jnp.sum(hlg.to_probs(jnp.clip(q_demo, VMIN, VMAX)) * jax.nn.log_softmax(logits, -1), -1)
            return jnp.mean(g * ce)

        vloss, gv = jax.value_and_grad(v_loss_fn)(vp)
        upv, ov = txv.update(gv, ov)
        vp = optax.apply_updates(vp, upv)

        # ----- Q loss: bootstrap from V(z') (NOT max over candidates) -----
        vnext = v_read(vp, znx)
        target = jnp.clip(SR[b] + GH * (1 - EN[b]) * (1 - DL[b]) * vnext, VMIN, VMAX)
        target = jax.lax.stop_gradient(target)

        def q_loss_fn(qp, k):
            if FLOW:
                z0 = jax.random.uniform(k, target.shape, minval=LO, maxval=HI)
                t = jax.random.uniform(jax.random.fold_in(k, 1), target.shape)
                zt = (1 - t) * z0 + t * target
                zb = jnp.broadcast_to(zt[:, None], (target.shape[0], H // MG))
                tb = jnp.broadcast_to(t[:, None], (target.shape[0], H // MG))
                v = q_net.apply(qp, z, a, zb, tb)[:, LAST]
                return jnp.mean((v - (target - z0)) ** 2)
            out = q_net.apply(qp, z, a)
            if ATOMS > 1:
                return hlg_ce(out[:, LAST], target)
            return jnp.mean((out[:, LAST] - target) ** 2)

        q1loss, g1 = jax.value_and_grad(q_loss_fn)(qp1, k4)
        q2loss, g2 = jax.value_and_grad(q_loss_fn)(qp2, jax.random.fold_in(k4, 7))
        up1, oq1 = txq.update(g1, oq1)
        qp1 = optax.apply_updates(qp1, up1)
        up2, oq2 = txq.update(g2, oq2)
        qp2 = optax.apply_updates(qp2, up2)
        tq1 = jax.tree.map(lambda a, b: 0.995 * a + 0.005 * b, tq1, qp1)
        tq2 = jax.tree.map(lambda a, b: 0.995 * a + 0.005 * b, tq2, qp2)
        return qp1, qp2, vp, tq1, tq2, oq1, oq2, ov, vloss, q1loss

    k = key0
    for s in range(STEPS):
        k, kk = jax.random.split(k)
        qp1, qp2, vp, tq1, tq2, oq1, oq2, ov, vl, ql = step(qp1, qp2, vp, tq1, tq2, oq1, oq2, ov, kk)
    print(f"trained {HEAD}  v_loss={float(vl):.4f} q_loss={float(ql):.4f}", flush=True)
    qpath.write_bytes(fser.msgpack_serialize({"q1": qp1, "q2": qp2}))
    vpath.write_bytes(fser.msgpack_serialize(vp))
    return {"q1": qp1, "q2": qp2}, vp


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    qparams, vp = train()
    if os.environ.get("SMOKE"):
        print(f"SMOKE {HEAD} ok")
        print("SMOKE_DONE")
        return
    st = np.load(SAVE / "stats.npz")
    tmu, tsd, amu, asd = st["tmu"], st["tsd"], st["amu"], st["asd"]
    rk = [jax.random.key(0)]

    def score(zc, ac):
        if FLOW:
            rk[0], k1 = jax.random.split(rk[0])
            rk[0], k2 = jax.random.split(rk[0])
            return np.minimum(
                np.asarray(q_integrate(qparams["q1"], zc, ac, k1)), np.asarray(q_integrate(qparams["q2"], zc, ac, k2))
            )
        q1 = np.asarray(q_read(qparams["q1"], zc, ac))
        q2 = np.asarray(q_read(qparams["q2"], zc, ac))
        return np.minimum(q1, q2)

    import eval_critic as EC
    import rollout as RO

    _base = EC._config.get_config(CONFIG)
    if hasattr(_base.data, "include_progress"):
        _cfg = dataclasses.replace(_base, data=dataclasses.replace(_base.data, include_progress=False))
        _orig = EC._config.get_config
        EC._config.get_config = lambda name: _cfg if name == CONFIG else _orig(name)
    vla = EC.VLA(
        CONFIG,
        pathlib.Path(CKPT),
        num_samples=NCAND,
        flow_steps=10,
        seed=SEED,
        model_overrides={"rlt_decoder_mode": "parallel", "rlt_include_proprio": False},
    )
    assert vla.raw_dim == A
    rng = np.random.default_rng(SEED)

    def make_policy(mode):
        def fn(element):
            z, cand = vla.token_and_candidates(element)
            N = len(cand)
            if mode == "vla":
                return cand[0], vla.H, None
            if mode == "rand":
                return cand[int(rng.integers(N))], vla.H, None
            zc = jnp.asarray(np.broadcast_to(((np.asarray(z, np.float32)[0] - tmu) / tsd)[None], (N, tmu.shape[0])))
            ac = jnp.asarray(((np.asarray(cand, np.float32) - amu) / asd))
            q = score(zc, ac)  # [N]
            return cand[int(np.argmax(q))], vla.H, None

        return fn

    env = RO.make_env("PrepareCoffee", seed=SEED)
    modes = ["vla", "rand", HEAD]
    results = {}
    for mode in modes:
        vla.reset_rng(SEED)
        res = RO.run_trials(
            env, make_policy(mode), task="PrepareCoffee", num_trials=NTRIALS, seed=SEED, replan_steps=vla.H
        )
        succ = np.array([t["success"] for t in res["trials"]], bool)
        lo, hi = wilson(int(succ.sum()), len(succ))
        results[mode] = {
            "success_rate": res["success_rate"],
            "successes": int(succ.sum()),
            "num_trials": len(succ),
            "wilson95": [round(lo, 3), round(hi, 3)],
            "per_trial": succ.astype(int).tolist(),
        }
        print(
            f"DEAS[{HEAD}] mode={mode:8s} succ={succ.sum():2d}/{len(succ)} = {res['success_rate']:.3f} CI[{lo:.2f},{hi:.2f}]",
            flush=True,
        )
    base = np.array(results["vla"]["per_trial"], bool)
    for mode in modes[1:]:
        m = np.array(results[mode]["per_trial"], bool)
        results[mode]["vs_vla_mcnemar"] = {
            "gain": int((~base & m).sum()),
            "loss": int((base & ~m).sum()),
            "delta_rate": round(float(m.mean() - base.mean()), 3),
        }
        print(
            f"  vs_vla {mode:8s} gain={int((~base & m).sum())} loss={int((base & ~m).sum())} dRate={m.mean()-base.mean():+.3f}",
            flush=True,
        )
    results["meta"] = {
        "head": HEAD,
        "gamma": GAMMA,
        "N": NCAND,
        "seed": SEED,
        "steps": STEPS,
        "expectile": TAU,
        "objective": "DEAS (expectile-V bootstrap, double-min, HL-Gauss)",
        "task": "PrepareCoffee",
    }
    out = f"{C}/gr1_eval/deas_{HEAD}.json"
    json.dump(results, open(out, "w"), indent=2)
    print("DEAS_DONE", out)


if __name__ == "__main__":
    main()
