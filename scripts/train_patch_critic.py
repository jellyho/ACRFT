"""Train the standalone patch-critic (DINOv2 + ARQ distributional value) with IQL on pi05 rollouts.

Mirrors the validated ``scripts/train_rlt_critic.py`` IQL recipe, with ONE change: the observation is
a FROZEN DINOv2 patch grid (computed once up front and cached) instead of the RLT VLA token.

Data: the per-step transition dirs written by ``examples/robocasa/collect_rollouts_patchcritic.py``
(images[N,3,224,224,3] state[N,16] action[N,12] reward[N] done[N] episode_index[N]). The action
CHUNK for frame t is action[t:t+H] within the same episode (zero-padded past the episode end); the
per-prefix / MC targets come from the dense sparse reward.

    uv run python scripts/train_patch_critic.py \
        --data /data5/jellyho/pc_rollouts/OpenDrawer /data5/jellyho/pc_rollouts/CoffeeSetupMug \
        --horizon 16 --steps 40000 --out .scratch/patch_critic
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np


@dataclasses.dataclass
class Cfg:
    horizon: int = 16
    macro_group_size: int = 2
    num_atoms: int = 51
    num_critics: int = 2
    discount: float = 0.99
    expectile: float = 0.7
    target_tau: float = 0.005
    lr: float = 3e-4
    batch: int = 128
    steps: int = 40000
    v_min: float = 0.0
    v_max: float = 1.0
    backbone: str = "small"


def load_dirs(dirs, horizon, discount, frames_cap=0):
    """Concatenate per-step rollout dirs -> images, state, chunk[t]=action[t:t+H], reward, done, mc, ep."""
    IM, ST, AC, RW, DN, EP = [], [], [], [], [], []
    ep_off = 0
    for d0 in dirs:
        d = pathlib.Path(d0)
        m = json.loads((d / "meta.json").read_text())
        n = m["num_steps"]
        IM.append(np.asarray(np.memmap(d / "images.dat", np.uint8, "r", shape=(n, 3, m["img_size"], m["img_size"], 3))))
        ST.append(np.asarray(np.memmap(d / "state.dat", np.float32, "r", shape=(n, m["state_dim"]))))
        AC.append(np.asarray(np.memmap(d / "action.dat", np.float32, "r", shape=(n, m["action_dim"]))))
        RW.append(np.asarray(np.memmap(d / "reward.dat", np.float32, "r", shape=(n,))))
        DN.append(np.asarray(np.memmap(d / "done.dat", np.int8, "r", shape=(n,))))
        EP.append(np.asarray(np.memmap(d / "episode_index.dat", np.int32, "r", shape=(n,))) + ep_off)
        ep_off = int(EP[-1].max()) + 1
    images = np.concatenate(IM)
    state = np.concatenate(ST)
    action = np.concatenate(AC)
    reward = np.concatenate(RW)
    done = np.concatenate(DN).astype(np.int32)
    ep = np.concatenate(EP)
    # optional EPISODE-level subsample (keep whole episodes so within-episode timesteps stay
    # contiguous): all success episodes first, then random failures until frames_cap is reached.
    if frames_cap and len(action) > frames_cap:
        rng = np.random.default_rng(0)
        eps = np.unique(ep)
        succ = np.array([reward[ep == e].max() > 0.5 for e in eps])
        order = np.concatenate([eps[succ], rng.permutation(eps[~succ])])
        length = {int(e): int((ep == e).sum()) for e in eps}
        keep, tot = [], 0
        for e in order:
            keep.append(int(e))
            tot += length[int(e)]
            if tot >= frames_cap:
                break
        sel = np.sort(np.flatnonzero(np.isin(ep, keep)))  # preserve original contiguous order
        images, state, action, reward, done = (x[sel] for x in (images, state, action, reward, done))
        _, ep = np.unique(ep[sel], return_inverse=True)
        ep = ep.astype(np.int32)
        print(f"  subsampled -> {len(action)} frames from {len(keep)} episodes ({int(succ.sum())} success eps kept)")
    n, A = action.shape
    H = horizon
    # chunk[t] = action[t:t+H], zeroed where it would cross into another episode
    chunk = np.zeros((n, H, A), np.float32)
    for i in range(H):
        j = np.minimum(np.arange(n) + i, n - 1)
        same = ep[j] == ep
        chunk[:, i] = np.where(same[:, None], action[j], 0.0)
    # discounted MC return-to-go within episode (sparse reward)
    mc = np.zeros(n, np.float32)
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        g = 0.0
        for t in idx[::-1]:
            g = reward[t] + discount * g
            mc[t] = g
    return {
        "images": images,
        "state": state,
        "chunk": chunk,
        "reward": reward,
        "done": done,
        "mc": mc,
        "ep": ep,
        "n": n,
        "A": A,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--backbone", default="small")
    ap.add_argument("--max-frames", type=int, default=0, help="cap total frames (smoke test)")
    ap.add_argument("--num-workers", type=int, default=6, help="prefetch worker threads (host gather + H2D)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/patch_critic"))
    a = ap.parse_args()
    cfg = Cfg(
        horizon=a.horizon, steps=a.steps, batch=a.batch, discount=a.discount, expectile=a.expectile, backbone=a.backbone
    )

    import jax
    import jax.numpy as jnp
    import optax

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.patch_critic.backbone import to_nchw
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble
    from openpi.patch_critic.critic import PatchV

    D = load_dirs(a.data, cfg.horizon, cfg.discount, a.max_frames)  # max_frames caps by whole episodes
    n = D["n"]
    print(
        f"loaded {n} transitions from {len(a.data)} task(s); {int((D['reward'] > 0).sum())} success steps", flush=True
    )

    # --- precompute frozen DINOv2 patches once (cached on device as f16; critic casts up) ---
    # 256 patches/cam (16x16) is O(P^2)-expensive in the critic's attention; 2x2 avg-pool -> 64/cam
    # (192 total), which keeps spatial layout but cuts attention memory ~16x.
    bb = DinoV2Backbone(cfg.backbone)
    grid = int(bb.num_patches(224) ** 0.5)  # 16
    pooled = grid // 2  # 8
    npatch = 3 * pooled * pooled  # 192

    def pool(p):  # [b, 3cam*256, D] -> [b, 3*64, D]
        b, _, d = p.shape
        p = p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d)
        return p.mean((3, 5)).reshape(b, npatch, d)

    # Disk cache: DINOv2 is frozen+deterministic, so the pooled patches for a fixed (data, backbone,
    # pool, frame-count) never change -> compute once, reuse across every re-run / hyperparam sweep.
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / f"patches_{cfg.backbone}_p{npatch}_n{n}.npy"
    if cache.exists():
        print(f"loading cached patches <- {cache}", flush=True)
        patches = np.load(cache, mmap_mode="r")
    else:
        patches = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16, shape=(n, npatch, bb.embed_dim))
        bs = 64
        for s in range(0, n, bs):
            imgs = jnp.asarray(to_nchw(D["images"][s : s + bs].transpose(0, 1, 3, 4, 2).reshape(-1, 224, 224, 3)))
            imgs = imgs.reshape(-1, 3, 3, 224, 224)  # [b,3cam,3,H,W]
            patches[s : s + bs] = np.asarray(pool(bb(imgs)), np.float16)
            if s % (bs * 50) == 0:
                print(f"  dino patches {s}/{n}", flush=True)
        patches.flush()
        print(f"cached patches -> {cache}", flush=True)
    # patches live on the GPU as f16 (pooled 192 tokens -> 22 GB, well inside the reserved budget);
    # gathering a batch's patch sets is then an on-device op, so the GPU never stalls on a host->device
    # copy. This is ~10x faster than host-gather (which starves the GPU: it sat at 0% util, 2.6 it/s).
    patches_dev = jnp.asarray(np.asarray(patches), jnp.float16)  # [n, P, D] on device
    state = jnp.asarray(D["state"])
    chunk = jnp.asarray(D["chunk"])
    reward = jnp.asarray(D["reward"])
    done = jnp.asarray(D["done"])
    mc = jnp.asarray(D["mc"])
    ep = jnp.asarray(D["ep"])
    P = patches_dev.shape[1]
    D["images"] = None  # free 63 GB of raw images now that patches are cached
    print(f"patches on device (f16): {patches_dev.shape} ({patches_dev.nbytes / 1e9:.1f} GB)", flush=True)

    net = PatchCriticEnsemble(
        action_dim=D["A"],
        horizon=cfg.horizon,
        num_critics=cfg.num_critics,
        macro_group_size=cfg.macro_group_size,
        num_atoms=cfg.num_atoms,
    )
    v_net = PatchV(num_atoms=cfg.num_atoms)  # distributional V (HL-Gauss), like Q
    hl = HLGauss(cfg.v_min, cfg.v_max, cfg.num_atoms)
    centers = jnp.asarray(hl.centers)
    g = cfg.macro_group_size
    prefixes = list(range(g, cfg.horizon + 1, g))
    prefixes_a = jnp.asarray(prefixes)
    P_ = len(prefixes)

    def pf(pd, i):  # gather + cast f16->f32. pd is PASSED IN (not a closure constant), so XLA never
        return pd[i].astype(jnp.float32)  # embeds the multi-GB patch array into the graph (>2GB proto -> segfault)

    rng = jax.random.key(0)
    params = net.init(rng, pf(patches_dev, jnp.arange(2)), chunk[:2], state[:2])
    v_params = v_net.init(rng, pf(patches_dev, jnp.arange(2)), state[:2])
    tgt = params
    tx = optax.adam(cfg.lr)
    tx_v = optax.adam(cfg.lr)
    opt = tx.init(params)
    v_opt = tx_v.init(v_params)

    def from_logits(x):
        return jnp.sum(jax.nn.softmax(x, -1) * centers, -1)

    def targets(idx, tgt_p, v_p, pd):
        # per-prefix cumulative discounted reward (same-episode masked)
        cum = jnp.zeros((idx.shape[0], P_))
        for pi, h in enumerate(prefixes):
            disc = cfg.discount ** jnp.arange(h)
            js = jnp.minimum(idx[:, None] + jnp.arange(h)[None], n - 1)
            same = ep[js] == ep[idx][:, None]
            cum = cum.at[:, pi].set(jnp.sum(jnp.where(same, reward[js] * disc[None], 0.0), -1))
        nxt = jnp.minimum(idx[:, None] + prefixes_a[None], n - 1)  # [B, P_]
        valid = ep[nxt] == ep[idx][:, None]
        gam = cfg.discount ** prefixes_a.astype(jnp.float32)  # [P_]
        # DISTRIBUTIONAL Bellman: bootstrap V(s')'s whole distribution. Transform its support by
        # z = cum + gamma^h * center, then project back onto the fixed atoms (weighted by V's probs).
        vlog = v_net.apply(v_p, pf(pd, nxt.reshape(-1)), state[nxt.reshape(-1)]).reshape(
            idx.shape[0], P_, cfg.num_atoms
        )
        vprob = jax.nn.softmax(vlog, -1)  # [B, P_, atoms]
        z = cum[..., None] + gam[None, :, None] * centers[None, None, :]  # [B, P_, atoms]
        phi = hl.to_probs(jnp.clip(z, cfg.v_min, cfg.v_max))  # [B, P_, atoms(support j), atoms(a)]
        tgt = jnp.einsum("bpj,bpja->bpa", vprob, phi)  # [B, P_, atoms]
        # terminal successor -> deterministic mass at its own reward
        term = (done[nxt] > 0)[..., None]  # [B, P_, 1]
        tgt = jnp.where(term, hl.to_probs(reward[nxt]), tgt)
        # Cal-QL floor at the distribution level: where the behaviour's MC return exceeds the bootstrap
        # mean, replace the target with a HL-Gauss spike at that return (success-only data needs this).
        tmean = jnp.sum(tgt * centers, -1)  # [B, P_]
        floor = mc[idx][:, None] > tmean
        tgt = jnp.where(floor[..., None], hl.to_probs(jnp.broadcast_to(mc[idx][:, None], tmean.shape)), tgt)
        return tgt, valid.astype(jnp.float32)  # [B, P_, atoms], [B, P_]

    def loss_fn(params, v_params, tgt_p, idx, rng, pd):
        tgt_probs, w = targets(idx, jax.lax.stop_gradient(tgt_p), jax.lax.stop_gradient(v_params), pd)
        tgt_probs = jax.lax.stop_gradient(tgt_probs)
        pb = pf(pd, idx)
        pred = net.apply(params, pb, chunk[idx], state[idx])  # [K,B,P_,atoms]
        per = -jnp.sum(tgt_probs[None] * jax.nn.log_softmax(pred, -1), -1)  # [K,B,P_]
        q_loss = jnp.sum(per * w[None]) / (jnp.sum(w) * pred.shape[0] + 1e-8)
        # distributional V: expectile-weighted CE toward Q(s, demonstrated chunk)'s full-prefix dist.
        qd_log = net.apply(jax.lax.stop_gradient(tgt_p), pb, chunk[idx], state[idx])[:, :, -1, :]  # [K,B,atoms]
        qd_probs = jnp.mean(jax.nn.softmax(qd_log, -1), 0)  # [B,atoms] ensemble-mean dist
        qbar = jnp.min(from_logits(qd_log), 0)  # [B] ensemble-min scalar (deployed value read)
        vlog = v_net.apply(v_params, pb, state[idx])  # [B,atoms]
        vbar = from_logits(vlog)  # [B]
        u = jax.lax.stop_gradient(qbar) - vbar
        wexp = jnp.abs(cfg.expectile - (u < 0).astype(jnp.float32))  # [B]
        v_ce = -jnp.sum(jax.lax.stop_gradient(qd_probs) * jax.nn.log_softmax(vlog, -1), -1)  # [B]
        v_loss = jnp.sum(wexp * v_ce) / u.shape[0]
        return q_loss + v_loss, {
            "q_loss": q_loss,
            "v_loss": v_loss,
            "q_mean": jnp.mean(from_logits(pred)),
            "v_mean": jnp.mean(vbar),
        }

    @jax.jit
    def step(carry, rng, pd):
        params, tgt, opt, v_params, v_opt = carry
        idx = jax.random.randint(rng, (cfg.batch,), 0, n)
        (_, info), (gp, gv) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            params, v_params, tgt, idx, rng, pd
        )
        up, opt = tx.update(gp, opt, params)
        params = optax.apply_updates(params, up)
        uv, v_opt = tx_v.update(gv, v_opt, v_params)
        v_params = optax.apply_updates(v_params, uv)
        tgt = optax.incremental_update(params, tgt, cfg.target_tau)
        return (params, tgt, opt, v_params, v_opt), info

    carry = (params, tgt, opt, v_params, v_opt)
    import time as _time

    t0 = _time.perf_counter()
    for s in range(cfg.steps):
        rng, k = jax.random.split(rng)
        carry, info = step(carry, k, patches_dev)
        if s % 1000 == 0 or s == cfg.steps - 1:
            i = jax.tree.map(lambda x: float(x), info)
            rate = (s + 1) / (_time.perf_counter() - t0)
            print(
                f"step {s:6d}  q_loss {i['q_loss']:.4f}  v_loss {i['v_loss']:.4f}  "
                f"q_mean {i['q_mean']:.3f}  v_mean {i['v_mean']:.3f}  ({rate:.1f} it/s)",
                flush=True,
            )

    params = carry[0]
    a.out.mkdir(parents=True, exist_ok=True)
    import flax.serialization

    (a.out / "params.msgpack").write_bytes(flax.serialization.msgpack_serialize(jax.device_get(params)))
    (a.out / "config.json").write_text(json.dumps({**dataclasses.asdict(cfg), "num_patches": int(P)}, indent=2))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
