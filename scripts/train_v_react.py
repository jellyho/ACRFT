"""Train the honest reactive value V_react(o) by pure 1-step observation TD -- no chunk, no Q.

The framework's V_react must be the observation-belief value (Theorem 4 in docs/reactive_commitment_value.md):
    V(o_t) <- r_t + gamma * V(o_{t+1})
bootstrapping V from V on the cache's real transitions, with the cost_to_goal reward analytic. The point
is what it does NOT do: it never conditions on the chunk and never bootstraps a chunk-conditioned Q, so
by Lemma 1 it cannot inherit the belief-shift leak. The current PatchV does the opposite -- it is fit by
expectile to the chunk-conditioned Q -- which is why V(s0) splits success/failure by 986 at a frame
where the observation is identical. This trainer's checkpoint should shrink that gap; that shrinkage is
the on-data test of Theorem 4.

Reuses the cache/preproc/homing plumbing of train_patch_critic_cached.py; distributional HL-Gauss V,
EMA target. Saves a V-only checkpoint (v_params.msgpack + config.json with input_spec) that
measure_reactive_map.py --v-react can use as the honest boundary.

    uv run python scripts/train_v_react.py --cache /data1/jellyho/pc_cache/yam_s347 \
        --outcomes .scratch/yam_outcomes_347.jsonl --homing-onsets .scratch/yam_homing_onsets.json \
        --input-mode pi05 --norm-stats <...>/norm_stats.json --proprio-dims pos --steps 20000 --out <dir>
"""

import argparse
import json
import pathlib
import time

import numpy as np

import openpi.training.outcomes as _outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument(
        "--outcomes",
        default=None,
        help="legacy outcomes.jsonl (deprecated: the verdict is read from the dataset's next.success / next.done)",
    )
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=None)
    ap.add_argument("--truncate-homing", choices=["all", "failure", "none"], default="all")
    ap.add_argument("--num-atoms", type=int, default=101)
    ap.add_argument("--discount", type=float, default=0.99964)
    ap.add_argument("--h-goal", type=int, default=30)
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--v-max", type=float, default=0.0)
    ap.add_argument("--failure-reward", type=float, default=None)
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--input-mode", choices=["raw", "pi05"], default="pi05")
    ap.add_argument("--norm-stats", type=pathlib.Path, default=None)
    ap.add_argument("--proprio-dims", default="pos")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="v-react")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/v_react"))
    a = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.patch_critic import preproc as critic_preproc
    from openpi.patch_critic import spec as critic_spec
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchV

    v_min = a.v_min if a.v_min is not None else -1.0 / (1.0 - a.discount)
    failure_reward = a.failure_reward if a.failure_reward is not None else v_min

    meta = json.loads((a.cache / "meta.json").read_text())
    N, npatch, emb, sd_raw, _ad = meta["N"], meta["npatch"], meta["emb"], meta["sd"], meta["ad"]
    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, npatch, emb))
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, sd_raw))

    pre = None
    if a.input_mode == "pi05":
        if a.norm_stats is None:
            raise SystemExit("--input-mode pi05 needs --norm-stats")
        from openpi.policies import yam_policy

        pre = critic_preproc.Pi05Preproc.build(a.norm_stats, yam_policy.joint_delta_reference())
    pidx = critic_preproc.PROPRIO_SETS[a.proprio_dims]
    pidx = None if pidx is None else np.asarray(pidx)
    sd = len(pidx) if pidx is not None else sd_raw

    outc = _outcomes.cache_outcomes(meta, legacy_jsonl=a.outcomes)
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets else None

    # flat table of valid transitions (cur, nxt) within an episode's task span, with analytic reward/done
    cur_g, nxt_g, rew, done = [], [], [], []
    for e_str, info in meta["episodes"].items():
        e = int(e_str)
        if e not in outc:
            continue
        full, off = info["full_len"], info["offset"]
        is_succ = outc[e] == "success"
        eff = full
        if (
            homing is not None
            and str(e) in homing
            and (a.truncate_homing == "all" or (a.truncate_homing == "failure" and not is_succ))
        ):
            eff = int(homing[str(e)]["homing_onset"])
        goal_start = eff - a.h_goal
        for pos in range(eff):
            g0 = off + pos
            r = 0.0 if (is_succ and pos >= goal_start) else -1.0
            if is_succ and pos >= goal_start:  # goal region -> absorbing 0
                cur_g.append(g0)
                nxt_g.append(g0)
                rew.append(0.0)
                done.append(1.0)
            elif not is_succ and pos == eff - 1:  # failure terminal
                cur_g.append(g0)
                nxt_g.append(g0)
                rew.append(failure_reward)
                done.append(1.0)
            else:
                cur_g.append(g0)
                nxt_g.append(min(g0 + 1, off + eff - 1))
                rew.append(r)
                done.append(0.0)
    cur_g = np.asarray(cur_g, np.int64)
    nxt_g = np.asarray(nxt_g, np.int64)
    rew = np.asarray(rew, np.float32)
    done = np.asarray(done, np.float32)
    M = len(cur_g)
    print(
        f"V-react: {M} transitions ({int((done > 0).sum())} terminal) v=[{v_min:.0f},{a.v_max}] h_goal={a.h_goal}",
        flush=True,
    )

    def prep_state(gidx):
        st = np.asarray(states[gidx])
        if pre is not None:
            st = pre.state(st)
        if pidx is not None:
            st = st[..., pidx]
        return st

    hl = HLGauss(v_min, a.v_max, a.num_atoms)
    centers = jnp.asarray(hl.centers)
    net = PatchV(num_atoms=a.num_atoms)
    rng = jax.random.key(0)
    v_params = net.init(rng, jnp.zeros((2, npatch, emb)), jnp.zeros((2, sd)))
    tgt = v_params
    tx = optax.adam(a.lr)
    opt = tx.init(v_params)

    def from_logits(lg):
        return jnp.sum(jax.nn.softmax(lg, -1) * centers, -1)

    @jax.jit
    def step(v_params, tgt, opt, pc, sc, pn, sn, r, d):
        vlog_n = net.apply(tgt, pn.astype(jnp.float32), sn)
        vnext = from_logits(vlog_n)
        y = jnp.clip(r + a.discount * (1.0 - d) * vnext, v_min, a.v_max)
        ytgt = jax.lax.stop_gradient(hl.to_probs(y))

        def loss_fn(vp):
            vlog = net.apply(vp, pc.astype(jnp.float32), sc)
            ce = -jnp.sum(ytgt * jax.nn.log_softmax(vlog, -1), -1)
            return jnp.mean(ce), from_logits(vlog).mean()

        (loss, vmean), grad = jax.value_and_grad(loss_fn, has_aux=True)(v_params)
        up, opt2 = tx.update(grad, opt, v_params)
        v_params2 = optax.apply_updates(v_params, up)
        tgt2 = optax.incremental_update(v_params2, tgt, a.target_tau)
        return v_params2, tgt2, opt2, loss, vmean

    wb = None
    if a.wandb:
        import wandb

        wb = wandb.init(project="acrft-critic", group=a.wandb_group, name=a.wandb_name, config=vars(a))

    rng_np = np.random.default_rng(0)
    t0 = time.time()
    for s in range(a.steps):
        idx = rng_np.integers(0, M, size=a.batch)
        gc, gn = cur_g[idx], nxt_g[idx]
        v_params, tgt, opt, loss, vmean = step(
            v_params,
            tgt,
            opt,
            jnp.asarray(np.asarray(feats[gc])),
            jnp.asarray(prep_state(gc)),
            jnp.asarray(np.asarray(feats[gn])),
            jnp.asarray(prep_state(gn)),
            jnp.asarray(rew[idx]),
            jnp.asarray(done[idx]),
        )
        if s % 500 == 0 or s == a.steps - 1:
            rate = (s + 1) / (time.time() - t0)
            print(f"step {s:6d}  v_loss {float(loss):.4f}  v_mean {float(vmean):.2f}  ({rate:.2f} it/s)", flush=True)
            if wb is not None:
                wb.log({"v_loss": float(loss), "v_mean": float(vmean)}, step=s)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "v_params.msgpack").write_bytes(flax.serialization.msgpack_serialize(jax.device_get(v_params)))
    spec = critic_spec.input_spec(meta, horizon=30)
    if pre is not None:
        spec.update(pre.spec(a.norm_stats))
        spec["norm_stats_file"] = "pi05_norm_stats.json"
        (a.out / "pi05_norm_stats.json").write_text(json.dumps(pre.embedded()))
    spec["proprio_dims"] = a.proprio_dims
    spec["proprio_indices"] = None if pidx is None else pidx.tolist()
    spec["proprio_dim"] = sd
    spec["truncate_homing"] = a.truncate_homing
    cfg = {
        "kind": "v_react_obs_td",
        "num_atoms": a.num_atoms,
        "discount": a.discount,
        "h_goal": a.h_goal,
        "v_min": v_min,
        "v_max": a.v_max,
        "steps": a.steps,
        "input_spec": spec,
    }
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved -> {a.out}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
