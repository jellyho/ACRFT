"""Train the patch-critic from PRECOMPUTED DINOv2 features (scripts/cache_patch_features.py).

Same model, same analytic cost_to_goal targets, same losses as train_patch_critic_clip.py -- only the
data source changes: instead of decoding video + running frozen DINOv2 every step (the ~0.68 it/s wall),
this reads pooled patch features from the cache memmap and runs only the small critic transformer.
~20-40x faster; scientifically identical (the backbone output is byte-for-byte what the clip trainer
would have computed). Homing truncation is applied HERE (eff_len), so the full-episode cache is reused.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from openpi.patch_critic import preproc as critic_preproc
from openpi.patch_critic import spec as critic_spec

# reuse the validated target math + checkpoint writer from the clip trainer
from scripts.train_patch_critic_clip import _save
from scripts.train_patch_critic_clip import analytic_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, required=True, help="dir from cache_patch_features.py")
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=None)
    ap.add_argument(
        "--truncate-homing",
        choices=["all", "failure", "none"],
        default="all",
        help="drop the trailing return-to-home motion. all (default): from every episode -- a success's "
        "homing frames are not task progress, and with h_goal=30 they would otherwise make up most of "
        "the goal region, teaching the critic that 'arms back home' IS the goal. failure: the old "
        "behaviour, which left success homing in.",
    )
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--num-atoms", type=int, default=101)
    ap.add_argument("--macro-group-size", type=int, default=5)
    ap.add_argument("--num-critics", type=int, default=2)
    ap.add_argument("--reward-scheme", choices=["cost_to_goal"], default="cost_to_goal")
    ap.add_argument("--h-goal", type=int, default=3)
    ap.add_argument("--discount", type=float, default=0.99964)
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--v-max", type=float, default=0.0)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--mc-floor", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256, help="transitions per step")
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--failure-reward", type=float, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="acrft-critic")
    ap.add_argument("--wandb-group", default="patch-critic-cached")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/patch_critic_cached"))
    ap.add_argument("--init-params", type=pathlib.Path, default=None)
    ap.add_argument(
        "--input-mode",
        choices=critic_preproc.MODES,
        default="pi05",
        help="pi05: state/actions go through the base VLA's own preprocessing (joint delta + "
        "quantile norm), so the sampler's output IS the critic's input. raw: legacy dataset units.",
    )
    ap.add_argument("--norm-stats", type=pathlib.Path, default=None, help="norm_stats.json (required for pi05)")
    ap.add_argument(
        "--proprio-dims",
        choices=sorted(critic_preproc.PROPRIO_SETS),
        default="pos",
        help="which proprio channels the critic sees. pos (default): joint positions + grippers only, "
        "matching what ALOHA/Libero/DROID feed. all: every channel including velocity and effort -- "
        "extra sensors the baselines do not get, and effort in particular leaks grasp success.",
    )
    ap.add_argument(
        "--preload",
        action="store_true",
        help="materialize the feature/state/action memmaps into RAM (needs ~feature-cache GB of --mem; "
        "turns per-step NFS gathers into RAM-speed reads -- the throughput win)",
    )
    a = ap.parse_args()

    v_min = a.v_min if a.v_min is not None else -1.0 / (1.0 - a.discount)
    failure_reward = a.failure_reward if a.failure_reward is not None else v_min
    H = a.horizon

    meta = json.loads((a.cache / "meta.json").read_text())
    N, npatch, emb, sd_raw, ad = meta["N"], meta["npatch"], meta["emb"], meta["sd"], meta["ad"]
    sd = sd_raw  # network proprio width; --proprio-dims may narrow it below
    # fields _save() expects (it was written for the clip trainer)
    a.backbone = meta["backbone"]
    a.clip_len = 0  # not applicable to the cached path
    a.repo_id = f"cache:{a.cache.name}"
    a.loader = "critic_cached"  # this run read the feature cache, not the clip loader
    # the input contract + the reference distribution, saved with every checkpoint (see critic_spec)
    spec = critic_spec.input_spec(meta, horizon=H)
    spec["cache"] = str(a.cache)
    spec["n_episodes"] = len(meta["episodes"])
    pidx = critic_preproc.PROPRIO_SETS[a.proprio_dims]
    if pidx is not None:
        pidx = np.asarray(pidx)
        if int(pidx.max()) >= sd_raw:
            raise SystemExit(f"--proprio-dims {a.proprio_dims} needs state dim > {int(pidx.max())}, cache has {sd_raw}")
        sd = len(pidx)  # what the network actually sees
    spec["proprio_dims"] = a.proprio_dims
    spec["proprio_indices"] = None if pidx is None else pidx.tolist()
    spec["proprio_dim"] = sd
    spec["truncate_homing"] = a.truncate_homing
    stats = critic_spec.norm_stats(a.cache, meta)
    pre = None
    embedded = None
    if a.input_mode == "pi05":
        if a.norm_stats is None:
            raise SystemExit("--input-mode pi05 needs --norm-stats (the base checkpoint's norm_stats.json)")
        from openpi.policies import yam_policy

        pre = critic_preproc.Pi05Preproc.build(a.norm_stats, yam_policy.joint_delta_reference())
        spec.update(pre.spec(a.norm_stats))
        embedded = pre.embedded()
        print(f"input mode: pi05 preprocessing (joint delta + quantile norm) from {a.norm_stats}", flush=True)
    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(N, npatch, emb))
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, sd_raw))
    actions = np.memmap(a.cache / "action.dat", np.float32, "r", shape=(N, ad))
    if a.preload:
        import time as _t

        _t0 = _t.time()
        feats = np.ascontiguousarray(feats)  # -> RAM (float16); ~N*npatch*emb*2 bytes
        states = np.ascontiguousarray(states)
        actions = np.ascontiguousarray(actions)
        print(f"preloaded cache into RAM ({feats.nbytes / 1e9:.0f}GB features) in {_t.time() - _t0:.0f}s", flush=True)

    outc = {}
    for line in pathlib.Path(a.outcomes).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outc[int(r["episode"])] = r["outcome"]
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets is not None else None

    # Build the flat table of valid CURRENT frames (homing tail dropped for failures at train time).
    cur_g0, cur_pos, cur_eff, cur_full, cur_succ = [], [], [], [], []
    for e_str, info in meta["episodes"].items():
        e = int(e_str)
        if e not in outc:
            continue
        full = info["full_len"]
        succ = outc[e] == "success"
        eff = full
        if (
            homing is not None
            and str(e) in homing
            and (a.truncate_homing == "all" or (a.truncate_homing == "failure" and not succ))
        ):
            eff = int(homing[str(e)]["homing_onset"])
        off = info["offset"]
        cur_g0.append(np.full(eff, off))
        cur_pos.append(np.arange(eff))
        cur_eff.append(np.full(eff, eff))
        cur_full.append(np.full(eff, full))
        cur_succ.append(np.full(eff, succ))
    cur_g0 = np.concatenate(cur_g0).astype(np.int64)
    cur_pos = np.concatenate(cur_pos).astype(np.int64)
    cur_eff = np.concatenate(cur_eff).astype(np.int64)
    cur_full = np.concatenate(cur_full).astype(np.int64)
    cur_succ = np.concatenate(cur_succ).astype(bool)
    M = len(cur_pos)
    print(
        f"cache N={N} frames, {M} valid current frames "
        f"({int(cur_succ.sum())} success / {M - int(cur_succ.sum())} fail) "
        f"discount={a.discount:.5f} v=[{v_min:.1f},{a.v_max:.1f}] failure_reward={failure_reward:.1f}",
        flush=True,
    )

    import jax
    import jax.numpy as jnp
    import optax

    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble
    from openpi.patch_critic.critic import PatchV

    net = PatchCriticEnsemble(
        action_dim=ad, horizon=H, num_critics=a.num_critics, macro_group_size=a.macro_group_size, num_atoms=a.num_atoms
    )
    v_net = PatchV(num_atoms=a.num_atoms)
    hl = HLGauss(v_min, a.v_max, a.num_atoms)
    centers = jnp.asarray(hl.centers)
    prefixes = list(range(a.macro_group_size, H + 1, a.macro_group_size))
    P_ = len(prefixes)

    rng = jax.random.key(0)
    p2 = jnp.zeros((2, npatch, emb), jnp.float32)
    params = net.init(rng, p2, jnp.zeros((2, H, ad)), jnp.zeros((2, sd)))
    v_params = v_net.init(rng, p2, jnp.zeros((2, sd)))
    if a.init_params is not None:
        import flax.serialization

        params = flax.serialization.msgpack_restore((a.init_params / "params.msgpack").read_bytes())
        vpf = a.init_params / "v_params.msgpack"
        if vpf.exists():
            v_params = flax.serialization.msgpack_restore(vpf.read_bytes())
            print(f"warm-start: loaded Q + V from {a.init_params}", flush=True)
    tgt = params
    tx = optax.adam(a.lr)
    tx_v = optax.adam(a.lr)
    opt = tx.init(params)
    v_opt = tx_v.init(v_params)

    def from_logits(x):
        return jnp.sum(jax.nn.softmax(x, -1) * centers, -1)

    def loss_fn(params, v_params, tgt_p, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc):
        vlog = v_net.apply(
            jax.lax.stop_gradient(v_params), pnxt.reshape(-1, npatch, emb).astype(jnp.float32), snxt.reshape(-1, sd)
        ).reshape(-1, P_, a.num_atoms)
        vprob = jax.nn.softmax(vlog, -1)
        gam = a.discount ** jnp.asarray(prefixes, jnp.float32)
        z = cum[..., None] + gam[None, :, None] * centers[None, None, :]
        phi = hl.to_probs(jnp.clip(z, v_min, a.v_max))
        tgt = jnp.einsum("bpj,bpja->bpa", vprob, phi)
        tgt = jnp.where((done_nxt > 0)[..., None], hl.to_probs(reward_nxt), tgt)
        if a.mc_floor:
            tmean = jnp.sum(tgt * centers, -1)
            floor = mc[:, None] > tmean
            tgt = jnp.where(floor[..., None], hl.to_probs(jnp.broadcast_to(mc[:, None], tmean.shape)), tgt)
        tgt = jax.lax.stop_gradient(tgt)
        pred = net.apply(params, pcur.astype(jnp.float32), chunk, scur)
        per = -jnp.sum(tgt[None] * jax.nn.log_softmax(pred, -1), -1)
        q_loss = jnp.sum(per * valid[None]) / (jnp.sum(valid) * pred.shape[0] + 1e-8)
        qd_log = net.apply(jax.lax.stop_gradient(tgt_p), pcur.astype(jnp.float32), chunk, scur)[:, :, -1, :]
        qd_probs = jnp.mean(jax.nn.softmax(qd_log, -1), 0)
        qbar = jnp.min(from_logits(qd_log), 0)
        vlog_c = v_net.apply(v_params, pcur.astype(jnp.float32), scur)
        vbar = from_logits(vlog_c)
        u = jax.lax.stop_gradient(qbar) - vbar
        wexp = jnp.abs(a.expectile - (u < 0).astype(jnp.float32))
        v_ce = -jnp.sum(jax.lax.stop_gradient(qd_probs) * jax.nn.log_softmax(vlog_c, -1), -1)
        v_loss = jnp.sum(wexp * v_ce) / u.shape[0]
        return q_loss + v_loss, {
            "q_loss": q_loss,
            "v_loss": v_loss,
            "q_mean": jnp.mean(from_logits(pred)),
            "v_mean": jnp.mean(vbar),
        }

    @jax.jit
    def step(carry, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc):
        params, tgt, opt, v_params, v_opt = carry
        (_, info), (gp, gv) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            params, v_params, tgt, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc
        )
        up, opt = tx.update(gp, opt, params)
        params = optax.apply_updates(params, up)
        uv, v_opt = tx_v.update(gv, v_opt, v_params)
        v_params = optax.apply_updates(v_params, uv)
        tgt = optax.incremental_update(params, tgt, a.target_tau)
        return (params, tgt, opt, v_params, v_opt), info

    wb = None
    if a.wandb:
        import wandb

        wb = wandb.init(project=a.wandb_project, group=a.wandb_group, name=a.wandb_name, config=vars(a))

    a.out.mkdir(parents=True, exist_ok=True)
    carry = (params, tgt, opt, v_params, v_opt)
    rng_np = np.random.default_rng(0)
    ar_h = np.arange(H)
    pref = np.asarray(prefixes)
    t0 = time.time()
    for s in range(a.steps):
        idx = rng_np.integers(0, M, size=a.batch)
        g0 = cur_g0[idx]
        pos = cur_pos[idx]
        eff = cur_eff[idx]
        full = cur_full[idx]
        succ = cur_succ[idx]
        CL = int(pos.max()) + int(pref[-1]) + 2
        pad_row = np.zeros((a.batch, CL), bool)
        cum, reward_nxt, done_nxt, valid, mc, jnxt = analytic_targets(
            pos, eff, succ, pos, pad_row, list(prefixes), a.discount, a.h_goal, v_min, a.reward_scheme, failure_reward
        )
        gcur = g0 + pos
        nxt_pos = np.clip(pos[:, None] + pref[None], 0, full[:, None] - 1)
        gnxt = g0[:, None] + nxt_pos  # [T, P]
        hpos = pos[:, None] + ar_h[None]  # [T, H]
        # Clamp to the TRUNCATED end, not the raw one: past eff lie the homing frames, and reading them
        # would put the return-to-home motion back into the chunk we just removed from the frame pool.
        gch = g0[:, None] + np.clip(hpos, 0, eff[:, None] - 1)
        chunk = np.asarray(actions[gch.reshape(-1)]).reshape(a.batch, H, ad)
        s_cur_raw = np.asarray(states[gcur])
        s_nxt_raw = np.asarray(states[gnxt.reshape(-1)]).reshape(a.batch, P_, sd_raw)
        if pre is not None:
            # delta is taken against the chunk's BASE frame, exactly as the base VLA does
            chunk = pre.actions(chunk, s_cur_raw)  # delta needs the FULL state (ref hits idx 21..27)
            s_cur_raw, s_nxt_raw = pre.state(s_cur_raw), pre.state(s_nxt_raw)
        if pidx is not None:
            s_cur_raw, s_nxt_raw = s_cur_raw[..., pidx], s_nxt_raw[..., pidx]
        # No zero-fill. The clamp above already HOLDS the last valid action, which is exactly what
        # LeRobot's delta_timestamps does (`max(ep_start, min(ep_end - 1, idx + delta))`) and therefore
        # what pi05 itself trains on. Writing 0.0 was wrong in the normalized space: a true "no motion"
        # action is not the zero vector there -- the gripper is absolute, so holding it normalizes to
        # -1.0 -- which made the pad a constant, recognisable pattern on exactly the frames carrying
        # the failure v_min anchor.
        # transfer features as float16 (half the PCIe traffic); the model upcasts to f32 on-device.
        pcur = jnp.asarray(np.asarray(feats[gcur]))
        pnxt = jnp.asarray(np.asarray(feats[gnxt.reshape(-1)]).reshape(a.batch, P_, npatch, emb))
        scur = jnp.asarray(s_cur_raw)
        snxt = jnp.asarray(s_nxt_raw)
        carry, info = step(
            carry,
            pcur,
            pnxt,
            jnp.asarray(chunk),
            scur,
            snxt,
            jnp.asarray(cum),
            jnp.asarray(reward_nxt),
            jnp.asarray(done_nxt),
            jnp.asarray(valid),
            jnp.asarray(mc),
        )
        if wb is not None and s % 100 == 0:
            wb.log({k: float(v) for k, v in info.items()}, step=s)
        if s == 20:
            t_warm, s_warm = time.time(), s  # reset clock after JIT compile for a clean steady rate
        if s % 200 == 0 or s == a.steps - 1:
            i = jax.tree.map(lambda x: float(x), info)
            rate = (s + 1) / (time.time() - t0)
            if s > 20:
                rate = (s - s_warm) / (time.time() - t_warm)
            print(
                f"step {s:6d}  q_loss {i['q_loss']:.4f}  v_loss {i['v_loss']:.4f}  q_mean {i['q_mean']:.2f}  "
                f"v_mean {i['v_mean']:.2f}  ({rate:.2f} it/s)",
                flush=True,
            )
        if a.save_every and (s + 1) % a.save_every == 0:
            a._step = s + 1
            _save(a, carry[0], carry[3], npatch, v_min, prefixes, ad, spec=spec, stats=stats, embedded=embedded)
    a._step = a.steps
    _save(a, carry[0], carry[3], npatch, v_min, prefixes, ad, spec=spec, stats=stats, embedded=embedded)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
