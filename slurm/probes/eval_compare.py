# ruff: noqa
"""Head-to-head, ONE paired run: our td-max bootstrap vs DEAS expectile-V bootstrap, vs VLA/rand.

Everything is held fixed — HL-Gauss head, our pi05 backbone, DEAS setup (negative reward, dual
discount γ1=0.9/γ2=0.99, support [-1/(1-γ2),0], double-min, EMA) — and ONLY the bootstrap operator
differs, so any gap is attributable to the operator alone:

  td-max (ours):  V_next = max_j min(Q1,Q2)_target(s', cand_j)   over the stored VLA candidates
  DEAS:           Q bootstraps from an expectile state value V(s')  (loaded from eval_deas run)

Both deploy as BoN (score = min(Q1,Q2)(z, cand), arg-max over N candidates). Controls: vla (candidate
0), rand (random candidate). Scenes pinned by (seed, trial) => all four arms face the identical scenes
(scene-paired). Wilson 95% + paired McNemar vs vla AND td-max-vs-deas. Cross-run VLA drift (0.56-0.80
at n=25) is why this must be one run. The DEAS critic is loaded from critic_runs/deas/hlgauss_q.msgpack
(trained by eval_deas.py); only the td-max critic is trained here.

Usage: eval_compare.py [trials=25] [N=10] [seed=8000] [steps=20000]
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
from openpi.rlt_critic.critic import ARQCritic, HLGauss

C = os.environ["CACHE_DIR"]
NTRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
NCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
CKPT = "/scratch/jellyho/acrft/checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_noprop/70000"
CONFIG = "pi05_robocasa_PrepareCoffee_rlt"
DEAS_DIR = pathlib.Path(f"{C}/critic_runs/deas")  # DEAS hlgauss critic (from eval_deas.py)
SAVE = pathlib.Path(f"{C}/critic_runs/compare")
SAVE.mkdir(parents=True, exist_ok=True)

D = f"{C}/annot/mixed"
meta = json.load(open(f"{D}/meta.json"))
T, Dt = meta["num_frames"], meta["token_dim"]
H, A, NS = meta["horizon"], meta["action_dim"], meta["num_samples"]
MG, NA = 2, 51
G1, G2, TAU = 0.9, 0.99, 0.7
NEG = True
VMIN, VMAX = (-1.0 / (1.0 - G2), 0.0)
GH = G2**H
LAST = H // MG - 1
NB, POOL, BATCH = 8, 60000, 128
hlg = HLGauss(v_min=VMIN, v_max=VMAX, num_atoms=NA, sigma_frac=0.1)
q_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=NA)


def q_read(params, z, a):
    return hlg.from_logits(q_net.apply(params, z, a))[..., LAST]


def hlg_ce(logits, target):
    return -jnp.mean(jnp.sum(hlg.to_probs(jnp.clip(target, VMIN, VMAX)) * jax.nn.log_softmax(logits, -1), -1))


def build_stats():
    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(T, min(200000, T), replace=False))
    tmu, tsd = np.asarray(tok[s]).mean(0), np.asarray(tok[s]).std(0) + 1e-6
    asub = np.asarray(act[s[:40000]], np.float32).reshape(-1, A)
    amu, asd = asub.mean(0), asub.std(0) + 1e-6
    return tmu.astype(np.float32), tsd.astype(np.float32), amu.astype(np.float32), asd.astype(np.float32)


def train_tdmax(tmu, tsd, amu, asd):
    """Double-Q HL-Gauss with candidate-max bootstrap (our operator), DEAS setup otherwise."""
    p = SAVE / "tdmax_q.msgpack"
    if p.exists():
        print("LOAD tdmax", flush=True)
        return fser.msgpack_restore(p.read_bytes())
    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    cand = np.memmap(f"{D}/base_action.dat", dtype=np.float32, mode="r", shape=(T, NS, H, A))
    rew = np.asarray(np.memmap(f"{D}/reward.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
    ep = np.asarray(np.memmap(f"{D}/episode_index.dat", dtype=np.int32, mode="r", shape=(T,)))
    done = np.asarray(np.memmap(f"{D}/done.dat", dtype=np.uint8, mode="r", shape=(T,)), np.float32)
    zof = lambda g: ((np.asarray(tok[g], np.float32) - tmu) / tsd).astype(np.float32)
    aof = (
        lambda ch: ((np.asarray(ch, np.float32).reshape(-1, A) - amu) / asd)
        .reshape(np.asarray(ch).shape)
        .astype(np.float32)
    )

    rng = np.random.default_rng(1)
    pool = np.sort(rng.choice(T, POOL, replace=False))
    nxt = np.minimum(pool + H, T - 1)
    win = np.clip(pool[:, None] + np.arange(H)[None, :], 0, T - 1)
    in_ep = (ep[win] == ep[pool][:, None]).astype(np.float32)
    disc1 = (G1 ** np.arange(H)).astype(np.float32)
    rstep = (rew[win] - 1.0) if NEG else rew[win]
    scaled_r = np.sum(rstep * in_ep * disc1[None, :], axis=1).astype(np.float32)
    ended = (ep[nxt] != ep[pool]).astype(np.float32)
    done_land = done[nxt].astype(np.float32)

    key0 = jax.random.key(0)
    qp1 = q_net.init(key0, jnp.zeros((1, Dt)), jnp.zeros((1, H, A)))
    qp2 = q_net.init(jax.random.key(2), jnp.zeros((1, Dt)), jnp.zeros((1, H, A)))
    tq1, tq2 = qp1, qp2
    tx = optax.adam(3e-4)
    o1, o2 = tx.init(qp1), tx.init(qp2)

    def gather(b):
        pb = pool[b]
        Zb = jnp.asarray(zof(pb))
        Ab = jnp.asarray(aof(act[pb]))
        nb = nxt[b]
        Zl = jnp.asarray(zof(nb))  # [B,Dt]
        Cl = jnp.asarray(aof(np.asarray(cand[nb], np.float32)[:, :NB]))  # [B,NB,H,A]
        return Zb, Ab, Zl, Cl, jnp.asarray(scaled_r[b]), jnp.asarray(ended[b]), jnp.asarray(done_land[b])

    @jax.jit
    def step(qp1, qp2, tq1, tq2, o1, o2, batch):
        Zb, Ab, Zl, Cl, sr, en, dl = batch
        zrep = jnp.broadcast_to(Zl[:, None, :], (Zl.shape[0], NB, Dt))
        q1n = q_read(tq1, zrep, Cl)
        q2n = q_read(tq2, zrep, Cl)  # [B,NB]
        v_next = jnp.max(jnp.minimum(q1n, q2n), axis=1)  # candidate-max of min-double
        target = jnp.clip(sr + GH * (1 - en) * (1 - dl) * v_next, VMIN, VMAX)
        target = jax.lax.stop_gradient(target)
        l1, g1 = jax.value_and_grad(lambda pp: hlg_ce(q_net.apply(pp, Zb, Ab)[:, LAST], target))(qp1)
        l2, g2 = jax.value_and_grad(lambda pp: hlg_ce(q_net.apply(pp, Zb, Ab)[:, LAST], target))(qp2)
        u1, o1 = tx.update(g1, o1)
        qp1 = optax.apply_updates(qp1, u1)
        u2, o2 = tx.update(g2, o2)
        qp2 = optax.apply_updates(qp2, u2)
        tq1 = jax.tree.map(lambda a, b: 0.995 * a + 0.005 * b, tq1, qp1)
        tq2 = jax.tree.map(lambda a, b: 0.995 * a + 0.005 * b, tq2, qp2)
        return qp1, qp2, tq1, tq2, o1, o2, l1

    gr = np.random.default_rng(7)
    for s in range(STEPS):
        qp1, qp2, tq1, tq2, o1, o2, l = step(qp1, qp2, tq1, tq2, o1, o2, gather(gr.integers(0, POOL, BATCH)))
    print(f"trained tdmax q_loss={float(l):.4f}", flush=True)
    out = {"q1": qp1, "q2": qp2}
    p.write_bytes(fser.msgpack_serialize(out))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    pp = k / n
    d = 1 + z * z / n
    c = pp + z * z / (2 * n)
    h = z * math.sqrt(pp * (1 - pp) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    st = np.load(DEAS_DIR / "stats.npz")
    tmu, tsd, amu, asd = st["tmu"], st["tsd"], st["amu"], st["asd"]
    deas = fser.msgpack_restore((DEAS_DIR / "hlgauss_q.msgpack").read_bytes())  # {q1,q2}
    tdmax = train_tdmax(tmu, tsd, amu, asd)
    if os.environ.get("SMOKE"):
        print("SMOKE tdmax+deas loaded ok")
        print("SMOKE_DONE")
        return

    @jax.jit
    def score_pair(q1p, q2p, zc, ac):
        return jnp.minimum(q_read(q1p, zc, ac), q_read(q2p, zc, ac))

    def score(params, zc, ac):
        return np.asarray(score_pair(params["q1"], params["q2"], zc, ac))

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
    critics = {"tdmax": tdmax, "deas": deas}

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
            q = score(critics[mode], zc, ac)
            return cand[int(np.argmax(q))], vla.H, None

        return fn

    seeds = [int(s) for s in os.environ.get("SEEDS", str(SEED)).split(",")]
    env = RO.make_env("PrepareCoffee", seed=seeds[0])
    modes = ["vla", "rand", "tdmax", "deas"]
    per_seed = {m: [] for m in modes}  # per-seed success RATE (run-level unit)
    per_trial = {m: [] for m in modes}  # pooled per-trial for McNemar
    for sd in seeds:
        for mode in modes:
            vla.reset_rng(sd)
            res = RO.run_trials(
                env, make_policy(mode), task="PrepareCoffee", num_trials=NTRIALS, seed=sd, replan_steps=vla.H
            )
            succ = np.array([t["success"] for t in res["trials"]], bool)
            per_seed[mode].append(float(succ.mean()))
            per_trial[mode].extend(succ.astype(int).tolist())
            print(f"CMP seed={sd} mode={mode:6s} {succ.sum():2d}/{len(succ)} = {succ.mean():.3f}", flush=True)

    def tci(x):  # run-level mean +/- 95% t-CI over seeds
        x = np.array(x)
        n = len(x)
        m = float(x.mean())
        if n < 2:
            return m, 0.0
        from scipy.stats import t as _t

        se = float(x.std(ddof=1) / math.sqrt(n))
        return m, float(_t.ppf(0.975, n - 1) * se)

    results = {
        "meta": {
            "g1": G1,
            "g2": G2,
            "expectile": TAU,
            "neg_reward": NEG,
            "support": [VMIN, VMAX],
            "N": NCAND,
            "seeds": seeds,
            "trials_per_seed": NTRIALS,
            "steps": STEPS,
            "head": "hlgauss",
            "isolation": "only bootstrap operator differs (td-max vs DEAS expectile-V)",
            "task": "PrepareCoffee",
        }
    }
    for mode in modes:
        m, ci = tci(per_seed[mode])
        results[mode] = {
            "run_level_mean": round(m, 3),
            "run_level_t95": round(ci, 3),
            "per_seed": [round(v, 3) for v in per_seed[mode]],
            "per_trial": per_trial[mode],
            "pooled_rate": round(float(np.mean(per_trial[mode])), 3),
            "n_total": len(per_trial[mode]),
        }
        print(f"RUNLEVEL {mode:6s} mean={m:.3f} ±{ci:.3f} (per-seed {results[mode]['per_seed']})", flush=True)
    # paired run-level deltas over seeds
    ps = {m: np.array(per_seed[m]) for m in modes}
    for a, b in [("tdmax", "vla"), ("deas", "vla"), ("deas", "tdmax")]:
        d = ps[a] - ps[b]
        md, cid = tci(list(d))
        print(f"PAIRED {a}-{b}: Δ̄={md:+.3f} ±{cid:.3f} (per-seed {[round(v,3) for v in d]})", flush=True)
        results.setdefault("paired", {})[f"{a}_vs_{b}"] = {"mean_delta": round(md, 3), "t95": round(cid, 3)}
    out = f"{C}/gr1_eval/compare_tdmax_deas.json"
    json.dump(results, open(out, "w"), indent=2)
    print("CMP_DONE", out)


if __name__ == "__main__":
    main()
