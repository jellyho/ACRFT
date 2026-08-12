# ruff: noqa
"""Closed-loop Best-of-N eval: does a critic actually improve the VLA on PrepareCoffee?

Trains three critics on the SAME AQC trunk (only head+bootstrap differ) — scalar (expectile-TD
regression), HL-Gauss (categorical classification), floq (flow-matching) — then plugs each into a
Best-of-N rollout: at every replan the VLA samples N action-chunk candidates, the critic scores them,
and the argmax candidate is executed. Compared against the VLA baseline (execute candidate 0) and a
`rand` null (execute a uniformly-drawn candidate) — so a BoN mode beating `vla` but not `rand` would
only prove that resampling helps, not that the critic ranks.

Scenes are pinned by (seed, trial) inside run_trials, so every mode faces the IDENTICAL set of
initial scenes (paired at the scene level). Reports success_rate + Wilson 95% CI per mode and the
paired McNemar table vs vla. Critics are saved to {C}/critic_runs/floq3/ and re-loaded if present, so
the eval can be re-run without retraining.

Usage: eval_bon.py [num_trials=25] [N=8] [seed=8000] [steps=30000]
"""

import dataclasses
import json
import math
import os
import pathlib
import sys

import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "examples/robocasa")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "third_party/robocasa")
from openpi.rlt_critic.critic import ARQCritic
from openpi.rlt_critic.critic import HLGauss

C = os.environ["CACHE_DIR"]
NTRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
NCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
GAMMA = float(os.environ.get("GAMMA_OVR", "0.997"))
CKPT = "/scratch/jellyho/acrft/checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_noprop/70000"
CONFIG = "pi05_robocasa_PrepareCoffee_rlt"
SAVE = pathlib.Path(f"{C}/critic_runs/floq3")
SAVE.mkdir(parents=True, exist_ok=True)

D = f"{C}/annot/mixed"
meta = json.load(open(f"{D}/meta.json"))
T, Dt = meta["num_frames"], meta["token_dim"]
H, A = meta["horizon"], meta["action_dim"]
GH = GAMMA**H
LO, HI, MG, NA, KI = -0.5, 1.0, 2, 51, 10
LAST = H // MG - 1

scalar_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=1)
hlg_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=NA)
hlg = HLGauss(v_min=0.0, v_max=1.0, num_atoms=NA)
flow_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, flow_head=True, flow_lo=LO, flow_hi=HI)


def integrate(apply, params, z_obs, a_norm, key, n=KI):
    B = z_obs.shape[0]
    q = jax.random.uniform(key, (B,), minval=LO, maxval=HI)
    dt = 1.0 / n
    for i in range(n):
        t = jnp.full((B,), i * dt)
        v = apply(
            params,
            z_obs,
            a_norm,
            jnp.broadcast_to(q[:, None], (B, H // MG)),
            jnp.broadcast_to(t[:, None], (B, H // MG)),
        )
        q = q + dt * v[:, LAST]
    return q


def build_stats():
    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(T, min(200000, T), replace=False))
    tmu, tsd = np.asarray(tok[tr]).mean(0), np.asarray(tok[tr]).std(0) + 1e-6
    asub = np.asarray(act[tr[:40000]], np.float32).reshape(-1, A)
    amu, asd = asub.mean(0), asub.std(0) + 1e-6
    return tr, tmu.astype(np.float32), tsd.astype(np.float32), amu.astype(np.float32), asd.astype(np.float32)


def train_all():
    """Train (or load) the three critics; return dict name -> params. Saves to SAVE/."""
    tr, tmu, tsd, amu, asd = build_stats()
    np.savez(SAVE / "stats.npz", tmu=tmu, tsd=tsd, amu=amu, asd=asd, gamma=GAMMA, H=H, A=A, LO=LO, HI=HI, NA=NA)
    paths = {n: SAVE / f"{n}.msgpack" for n in ("scalar", "hlgauss", "floq")}
    if all(p.exists() for p in paths.values()):
        print("LOAD critics from disk", flush=True)
        return {n: fser.msgpack_restore(p.read_bytes()) for n, p in paths.items()}

    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    mc = np.asarray(np.memmap(f"{D}/mc_return.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
    rew = np.asarray(np.memmap(f"{D}/reward.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
    ep = np.asarray(np.memmap(f"{D}/episode_index.dat", dtype=np.int32, mode="r", shape=(T,)))
    nxt = np.minimum(np.arange(T) + H, T - 1)
    ended = (ep[nxt] != ep).astype(np.float32)
    zof = lambda g: ((np.asarray(tok[g], np.float32) - tmu) / tsd).astype(np.float32)
    aof = lambda ch: ((ch.reshape(-1, A) - amu) / asd).reshape(ch.shape).astype(np.float32)

    key = jax.random.key(0)
    Ztr, Atr = jnp.asarray(zof(tr)), jnp.asarray(aof(np.asarray(act[tr], np.float32)))
    Znx, Anx = jnp.asarray(zof(nxt[tr])), jnp.asarray(aof(np.asarray(act[nxt[tr]], np.float32)))
    mc_tr, r_tr, en_tr = jnp.asarray(mc[tr]), jnp.asarray(rew[tr]), jnp.asarray(ended[tr])

    def run(init, stepfn):
        p = init
        tgt = p
        opt = optax.adam(3e-4).init(p)
        k = key
        for _ in range(STEPS):
            k, kk = jax.random.split(k)
            p, tgt, opt = stepfn(p, tgt, opt, kk)
        return p

    tx = optax.adam(3e-4)

    @jax.jit
    def step_scalar(p, tgt, opt, k):
        b = jax.random.randint(k, (256,), 0, Ztr.shape[0])
        y = jnp.maximum(r_tr[b] + GH * (1 - en_tr[b]) * scalar_net.apply(tgt, Znx[b], Anx[b])[:, LAST], mc_tr[b])

        def loss(pp):
            u = y - scalar_net.apply(pp, Ztr[b], Atr[b])[:, LAST]
            return jnp.mean(jnp.abs(0.9 - (u < 0).astype(jnp.float32)) * u**2)

        g = jax.grad(loss)(p)
        up, opt2 = tx.update(g, opt)
        p2 = optax.apply_updates(p, up)
        return p2, jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2), opt2

    @jax.jit
    def step_hlg(p, tgt, opt, k):
        b = jax.random.randint(k, (256,), 0, Ztr.shape[0])
        vnext = hlg.from_logits(hlg_net.apply(tgt, Znx[b], Anx[b])[:, LAST])
        y = jnp.maximum(r_tr[b] + GH * (1 - en_tr[b]) * vnext, mc_tr[b])
        probs = hlg.to_probs(y)

        def loss(pp):
            logits = hlg_net.apply(pp, Ztr[b], Atr[b])[:, LAST]
            return -jnp.mean(jnp.sum(probs * jax.nn.log_softmax(logits, -1), -1))

        g = jax.grad(loss)(p)
        up, opt2 = tx.update(g, opt)
        p2 = optax.apply_updates(p, up)
        return p2, jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2), opt2

    @jax.jit
    def step_floq(p, tgt, opt, k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        b = jax.random.randint(k1, (256,), 0, Ztr.shape[0])
        vnext = integrate(flow_net.apply, tgt, Znx[b], Anx[b], k2)  # single-sample bootstrap
        y = jnp.maximum(r_tr[b] + GH * (1 - en_tr[b]) * vnext, mc_tr[b])
        z0 = jax.random.uniform(k3, (256,), minval=LO, maxval=HI)
        t = jax.random.uniform(k4, (256,))
        zt = (1 - t) * z0 + t * y

        def loss(pp):
            v = flow_net.apply(
                pp,
                Ztr[b],
                Atr[b],
                jnp.broadcast_to(zt[:, None], (256, H // MG)),
                jnp.broadcast_to(t[:, None], (256, H // MG)),
            )[:, LAST]
            return jnp.mean((v - (y - z0)) ** 2)

        g = jax.grad(loss)(p)
        up, opt2 = tx.update(g, opt)
        p2 = optax.apply_updates(p, up)
        return p2, jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2), opt2

    out = {}
    out["scalar"] = run(scalar_net.init(key, jnp.zeros((1, Dt)), jnp.zeros((1, H, A))), step_scalar)
    print("trained scalar", flush=True)
    out["hlgauss"] = run(hlg_net.init(key, jnp.zeros((1, Dt)), jnp.zeros((1, H, A))), step_hlg)
    print("trained hlgauss", flush=True)
    out["floq"] = run(
        flow_net.init(key, jnp.zeros((1, Dt)), jnp.zeros((1, H, A)), jnp.zeros((1, H // MG)), jnp.zeros((1, H // MG))),
        step_floq,
    )
    print("trained floq", flush=True)
    for n, p in out.items():
        (SAVE / f"{n}.msgpack").write_bytes(fser.msgpack_serialize(p))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    params = train_all()
    st = np.load(SAVE / "stats.npz")
    tmu, tsd, amu, asd = st["tmu"], st["tsd"], st["amu"], st["asd"]

    # score functions: raw live token/candidates -> per-candidate full-chunk value [N]
    def zc_of(z_raw, n):
        zc = (np.asarray(z_raw, np.float32)[0] - tmu) / tsd
        return jnp.broadcast_to(jnp.asarray(zc)[None], (n, zc.shape[0]))

    def ac_of(cand):
        return jnp.asarray((np.asarray(cand, np.float32) - amu) / asd)

    rk = [jax.random.key(0)]

    @jax.jit
    def q_scalar(zc, ac):
        return scalar_net.apply(params["scalar"], zc, ac)[:, LAST]

    @jax.jit
    def q_hlg(zc, ac):
        return hlg.from_logits(hlg_net.apply(params["hlgauss"], zc, ac)[:, LAST])

    def q_floq(zc, ac):
        rk[0], kk = jax.random.split(rk[0])
        return integrate(flow_net.apply, params["floq"], zc, ac, kk)

    scorers = {"scalar": q_scalar, "hlgauss": q_hlg, "floq": q_floq}

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
    assert vla.raw_dim == A, f"action dim mismatch: vla {vla.raw_dim} vs critic {A}"
    rng = np.random.default_rng(SEED)

    def make_policy(mode):
        def fn(element):
            z, cand = vla.token_and_candidates(element)  # z [1,D], cand [N,H,A]
            if mode == "vla":
                return cand[0], vla.H, None
            if mode == "rand":
                return cand[int(rng.integers(len(cand)))], vla.H, None
            zc, ac = zc_of(z, len(cand)), ac_of(cand)
            q = np.asarray(scorers[mode](zc, ac))  # [N]
            return cand[int(np.argmax(q))], vla.H, None

        return fn

    env = RO.make_env("PrepareCoffee", seed=SEED)
    modes = ["vla", "rand", "scalar", "hlgauss", "floq"]
    results = {}
    for mode in modes:
        vla.reset_rng(SEED)  # identical first-step candidate draws per scene across modes
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
            f"BON mode={mode:8s} succ={succ.sum():2d}/{len(succ)} = {res['success_rate']:.3f}  CI[{lo:.2f},{hi:.2f}]",
            flush=True,
        )

    # paired McNemar vs vla (scene-level)
    base = np.array(results["vla"]["per_trial"], bool)
    for mode in ("rand", "scalar", "hlgauss", "floq"):
        m = np.array(results[mode]["per_trial"], bool)
        b01 = int((~base & m).sum())  # vla fail, mode success
        b10 = int((base & ~m).sum())  # vla success, mode fail
        results[mode]["vs_vla_mcnemar"] = {"gain": b01, "loss": b10, "delta_rate": round((m.mean() - base.mean()), 3)}
        print(f"  vs_vla {mode:8s} gain={b01} loss={b10} dRate={m.mean() - base.mean():+.3f}", flush=True)

    results["meta"] = {"gamma": GAMMA, "N": NCAND, "seed": SEED, "steps": STEPS, "task": "PrepareCoffee"}
    out = f"{C}/gr1_eval/bon_critic_compare.json"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("BON_DONE", out)


if __name__ == "__main__":
    main()
