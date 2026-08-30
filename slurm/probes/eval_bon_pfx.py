# ruff: noqa
"""Per-prefix, TD-max critic comparison (scalar / HL-Gauss / floq) + joint (candidate x prefix) BoN.

Fixes the flaw of eval_bon.py: that probe bootstrapped V_next from the DEMO's next action (SARSA-style),
learning the demo-policy value, not the optimal value. Here the bootstrap is the production trainer's
**TD-max over the stored VLA candidates** — the offline stand-in for max_a Q the user prescribed ("if TD,
sample dataset actions and take the max"). Faithful to openpi.rlt_critic.training.targets:

  prefixes p in {MG, 2MG, ..., H};  gamma_h = gamma^p
  y_p = sum_{i<p} gamma^i r_{t+i}  +  gamma^p * (1 - ended_p) * V_next(landing_p)          [per prefix]
  V_next(s') = max over {NB candidates x P' successor-prefixes} of Q_target(s', cand, p')   [td-max]
  y_p = max(y_p, mc_return[t]);  clamp to the value support                                 [mc floor]

Three heads on the SAME AQC trunk, all trained against this identical per-prefix TD-max target:
  scalar (L2), HL-Gauss (cross-entropy), floq (flow-matching; V read by ODE integration).

Deployment: at each replan the VLA samples N candidates, the critic scores [N, P], and a JOINT arg-max
over (candidate, prefix) picks the candidate AND the commit length n_exec=(p+1)*MG. Controls: vla, rand,
and randh (random candidate AND random prefix — the honest null for a joint arg-max). Scene-paired.

Memory: candidate tensors are gathered per-batch from the memmaps on the host (the full landing-candidate
tensor is TB-scale); NB<=NS candidates are subsampled for the bootstrap. Single critic (no ensemble) — a
documented simplification. Usage: eval_bon_pfx.py [trials=25] [N=8] [seed=8000] [steps=15000]
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
NCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 15000
GAMMA = float(os.environ.get("GAMMA_OVR", "0.997"))
CKPT = "/scratch/jellyho/acrft/checkpoints/pi05_robocasa_PrepareCoffee_rlt/PrepareCoffee_rlt5_pardec_noprop/70000"
CONFIG = "pi05_robocasa_PrepareCoffee_rlt"
SAVE = pathlib.Path(f"{C}/critic_runs/pfx3")
SAVE.mkdir(parents=True, exist_ok=True)

D = f"{C}/annot/mixed"
meta = json.load(open(f"{D}/meta.json"))
T, Dt = meta["num_frames"], meta["token_dim"]
H, A, NS = meta["horizon"], meta["action_dim"], meta["num_samples"]
MG, NA, KI = 2, 51, 8
P = H // MG
NB = 8  # bootstrap candidate subsample (<= NS) — keeps the per-step gather affordable
POOL = 60000
BATCH = 128
LO, HI = -0.5, 1.0
VMIN, VMAX = 0.0, 1.0
prefixes = np.arange(MG, H + 1, MG)
gamma_h = (GAMMA**prefixes).astype(np.float32)
step_discount = (GAMMA ** np.arange(H)).astype(np.float32)

scalar_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=1)
hlg_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, num_atoms=NA)
hlg = HLGauss(v_min=VMIN, v_max=VMAX, num_atoms=NA)
flow_net = ARQCritic(action_dim=A, horizon=H, macro_group_size=MG, flow_head=True, flow_lo=LO, flow_hi=HI)


def integrate(params, z_obs, a_norm, key, n=KI):
    """floq Q read: integrate velocity from noise to t=1, per prefix. Returns Q [..., P]."""
    lead = z_obs.shape[:-1]
    q = jax.random.uniform(key, (*lead, P), minval=LO, maxval=HI)
    dt = 1.0 / n
    for i in range(n):
        t = jnp.full((*lead, P), i * dt)
        q = q + dt * flow_net.apply(params, z_obs, a_norm, q, t)
    return q


def q_read(name, params, z_obs, a_norm, key=None):
    if name == "scalar":
        return scalar_net.apply(params, z_obs, a_norm)
    if name == "hlgauss":
        return hlg.from_logits(hlg_net.apply(params, z_obs, a_norm))
    return integrate(params, z_obs, a_norm, key)


def build_stats():
    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(T, min(200000, T), replace=False))
    tmu, tsd = np.asarray(tok[s]).mean(0), np.asarray(tok[s]).std(0) + 1e-6
    asub = np.asarray(act[s[:40000]], np.float32).reshape(-1, A)
    amu, asd = asub.mean(0), asub.std(0) + 1e-6
    return tmu.astype(np.float32), tsd.astype(np.float32), amu.astype(np.float32), asd.astype(np.float32)


def train_all():
    tmu, tsd, amu, asd = build_stats()
    np.savez(
        SAVE / "stats.npz", tmu=tmu, tsd=tsd, amu=amu, asd=asd, gamma=GAMMA, H=H, A=A, MG=MG, LO=LO, HI=HI, NA=NA, P=P
    )
    paths = {n: SAVE / f"{n}.msgpack" for n in ("scalar", "hlgauss", "floq")}
    if all(p.exists() for p in paths.values()):
        print("LOAD critics from disk", flush=True)
        return {n: fser.msgpack_restore(p.read_bytes()) for n, p in paths.items()}

    tok = np.memmap(f"{D}/rl_token.dat", dtype=np.float32, mode="r", shape=(T, Dt))
    act = np.memmap(f"{D}/action_chunk.dat", dtype=np.float32, mode="r", shape=(T, H, A))
    cand = np.memmap(f"{D}/base_action.dat", dtype=np.float32, mode="r", shape=(T, NS, H, A))
    mc = np.asarray(np.memmap(f"{D}/mc_return.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
    rew = np.asarray(np.memmap(f"{D}/reward.dat", dtype=np.float32, mode="r", shape=(T,)), np.float32)
    ep = np.asarray(np.memmap(f"{D}/episode_index.dat", dtype=np.int32, mode="r", shape=(T,)))
    done = np.asarray(np.memmap(f"{D}/done.dat", dtype=np.uint8, mode="r", shape=(T,)), np.float32)  # 1-byte flags
    znorm = lambda g: ((np.asarray(tok[g], np.float32) - tmu) / tsd).astype(np.float32)
    anorm = (
        lambda ch: ((np.asarray(ch, np.float32).reshape(-1, A) - amu) / asd)
        .reshape(np.asarray(ch).shape)
        .astype(np.float32)
    )

    rng = np.random.default_rng(1)
    pool = np.sort(rng.choice(T, POOL, replace=False))
    # small per-frame host arrays (cheap)
    win = np.clip(pool[:, None] + np.arange(H)[None, :], 0, T - 1)
    in_ep = (ep[win] == ep[pool][:, None]).astype(np.float32)
    r2h_h = np.cumsum(rew[win] * in_ep * step_discount[None, :], axis=1)[:, prefixes - 1].astype(np.float32)  # [POOL,P]
    landing_h = np.clip(pool[:, None] + prefixes[None, :], 0, T - 1).astype(np.int64)  # [POOL,P]
    ended_h = (ep[landing_h] != ep[pool][:, None]).astype(np.float32)  # [POOL,P]
    mc_h = mc[pool].astype(np.float32)
    done_land_h = done[landing_h].astype(np.float32)
    rew_land_h = rew[landing_h].astype(np.float32)
    gh = jnp.asarray(gamma_h)

    def gather(b):
        """Host gather of one minibatch -> device arrays."""
        pb = pool[b]
        Zb = jnp.asarray(znorm(pb))  # [B,Dt]
        Ab = jnp.asarray(anorm(act[pb]))  # [B,H,A]
        lb = landing_h[b]  # [B,P]
        Zl = jnp.asarray(znorm(lb.reshape(-1)).reshape(len(b), P, Dt))  # [B,P,Dt]
        cr = np.asarray(cand[lb.reshape(-1)], np.float32)[:, :NB]  # [B*P, NB, H, A]
        Cl = jnp.asarray(anorm(cr).reshape(len(b), P, NB, H, A))  # [B,P,NB,H,A]
        return (
            Zb,
            Ab,
            Zl,
            Cl,
            jnp.asarray(r2h_h[b]),
            jnp.asarray(ended_h[b]),
            jnp.asarray(mc_h[b]),
            jnp.asarray(done_land_h[b]),
            jnp.asarray(rew_land_h[b]),
        )

    def make_step(name, loss_fn):
        @jax.jit
        def step(params, tgt, opt, batch, key):
            Zb, Ab, Zl, Cl, r2h_b, ended_b, mc_b, dl_b, rl_b = batch
            zrep = jnp.broadcast_to(Zl[:, :, None, :], (Zl.shape[0], P, NB, Dt))
            qn = q_read(name, tgt, zrep, Cl, key)  # [B,P,NB,P']
            v_next = jnp.clip(jnp.max(qn.reshape(qn.shape[0], P, NB * P), axis=-1), VMIN, VMAX)  # [B,P]
            v_next = jnp.where(dl_b > 0, rl_b, v_next)
            y = r2h_b + gh[None, :] * (1 - ended_b) * v_next
            y = jnp.clip(jnp.maximum(y, mc_b[:, None]), VMIN, VMAX)
            y = jax.lax.stop_gradient(y)
            v, g = jax.value_and_grad(lambda pp: loss_fn(pp, Zb, Ab, y, key))(params)
            up, opt2 = tx.update(g, opt)
            p2 = optax.apply_updates(params, up)
            return p2, jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, tgt, p2), opt2, v

        return step

    def loss_scalar(pp, Zb, Ab, y, key):
        return jnp.mean((scalar_net.apply(pp, Zb, Ab) - y) ** 2)

    def loss_hlg(pp, Zb, Ab, y, key):
        logits = hlg_net.apply(pp, Zb, Ab)
        return -jnp.mean(jnp.sum(hlg.to_probs(jnp.clip(y, VMIN, VMAX)) * jax.nn.log_softmax(logits, -1), -1))

    def loss_floq(pp, Zb, Ab, y, key):
        k1, k2 = jax.random.split(key)
        z0 = jax.random.uniform(k1, y.shape, minval=LO, maxval=HI)
        t = jax.random.uniform(k2, y.shape)
        zt = (1 - t) * z0 + t * y
        return jnp.mean((flow_net.apply(pp, Zb, Ab, zt, t) - (y - z0)) ** 2)

    tx = optax.adam(3e-4)
    key0 = jax.random.key(0)

    def run(name, init, loss_fn):
        p = init
        tgt = p
        opt = tx.init(p)
        k = key0
        step = make_step(name, loss_fn)
        gr = np.random.default_rng(hash(name) % (2**31))
        for s in range(STEPS):
            k, kk = jax.random.split(k)
            b = gr.integers(0, POOL, BATCH)
            p, tgt, opt, _ = step(p, tgt, opt, gather(b), kk)
        return p

    out = {}
    out["scalar"] = run("scalar", scalar_net.init(key0, jnp.zeros((1, Dt)), jnp.zeros((1, H, A))), loss_scalar)
    print("trained scalar", flush=True)
    out["hlgauss"] = run("hlgauss", hlg_net.init(key0, jnp.zeros((1, Dt)), jnp.zeros((1, H, A))), loss_hlg)
    print("trained hlgauss", flush=True)
    out["floq"] = run(
        "floq",
        flow_net.init(key0, jnp.zeros((1, Dt)), jnp.zeros((1, H, A)), jnp.zeros((1, P)), jnp.zeros((1, P))),
        loss_floq,
    )
    print("trained floq", flush=True)
    for n, pp in out.items():
        (SAVE / f"{n}.msgpack").write_bytes(fser.msgpack_serialize(pp))
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
    if os.environ.get("SMOKE"):
        for n, p in params.items():
            print("SMOKE trained", n, "ok", flush=True)
        print("SMOKE_DONE")
        return
    st = np.load(SAVE / "stats.npz")
    tmu, tsd, amu, asd = st["tmu"], st["tsd"], st["amu"], st["asd"]
    rk = [jax.random.key(0)]

    @jax.jit
    def score_scalar(zc, ac):
        return scalar_net.apply(params["scalar"], zc, ac)

    @jax.jit
    def score_hlg(zc, ac):
        return hlg.from_logits(hlg_net.apply(params["hlgauss"], zc, ac))

    def score_floq(zc, ac):
        rk[0], kk = jax.random.split(rk[0])
        return integrate(params["floq"], zc, ac, kk)

    scorers = {"scalar": score_scalar, "hlgauss": score_hlg, "floq": score_floq}

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
            if mode == "randh":
                p = int(rng.integers(P))
                return cand[int(rng.integers(N))], (p + 1) * MG, None
            zc = jnp.asarray(np.broadcast_to(((np.asarray(z, np.float32)[0] - tmu) / tsd)[None], (N, tmu.shape[0])))
            ac = jnp.asarray(((np.asarray(cand, np.float32) - amu) / asd))
            q = np.asarray(scorers[mode](zc, ac))  # [N,P]
            i, p = np.unravel_index(int(np.argmax(q)), q.shape)  # joint (candidate, prefix)
            return cand[int(i)], (int(p) + 1) * MG, None

        return fn

    env = RO.make_env("PrepareCoffee", seed=SEED)
    modes = ["vla", "rand", "randh", "scalar", "hlgauss", "floq"]
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
            f"PFX mode={mode:8s} succ={succ.sum():2d}/{len(succ)} = {res['success_rate']:.3f}  CI[{lo:.2f},{hi:.2f}]",
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
            f"  vs_vla {mode:8s} gain={int((~base & m).sum())} loss={int((base & ~m).sum())} dRate={m.mean() - base.mean():+.3f}",
            flush=True,
        )

    results["meta"] = {
        "gamma": GAMMA,
        "N": NCAND,
        "seed": SEED,
        "steps": STEPS,
        "P": P,
        "MG": MG,
        "NB_boot": NB,
        "objective": "td-max-over-candidates",
        "select": "joint(candidate,prefix) argmax",
        "task": "PrepareCoffee",
    }
    out = f"{C}/gr1_eval/bon_pfx_compare.json"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("PFX_DONE", out)


if __name__ == "__main__":
    main()
