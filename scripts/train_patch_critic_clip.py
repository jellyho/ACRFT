"""Patch-critic trainer on openpi's LeRobot stack via CriticClipDataset (clip -> transitions).

Reuses openpi's verified decode path (LeRobotDataset + pyav + delta_timestamps, wrapped in a plain
torch DataLoader with num_workers) through openpi.training.critic_data_loader.CriticClipDataset. Each
loaded clip is `clip_len` contiguous frames (decoded once); the trainer encodes all clip frames with
frozen DINOv2 ONCE on the GPU, then explodes the clip into transitions -- current frame + its H-step
action chunk + the bootstrap futures at prefixes {g,2g,...,H}, all indexed WITHIN the clip so no frame
is re-decoded. The distributional IQL target (HL-Gauss, expectile V, MC floor) is identical to
train_patch_critic.py; the cost_to_goal reward is computed analytically from each frame's episode
position + outcome (no next-state relabel buffer needed).

    LEROBOT_VIDEO_BACKEND=pyav uv run python scripts/train_patch_critic_clip.py \
      --repo-id jellyho/yam_lego_taxi --root /data5/jellyho/yam_v2/lerobot \
      --outcomes <outcomes.jsonl> --reward-scheme cost_to_goal --discount 0.99964 --num-atoms 101 \
      --horizon 30 --clip-len 60 --clip-batch 8 --trans-per-clip 32 --num-workers 8 \
      --steps 120000 --out .scratch/patch_critic_yam_clip
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

CAMS = ["observation.images.agentview", "observation.images.wrist_left", "observation.images.wrist_right"]


def analytic_targets(pos, ep_len, succ, jloc, img_pad_row, prefixes, discount, h_goal, v_min, scheme, failure_reward):
    """Per-transition cost_to_goal target pieces, computed from episode geometry (no next-state decode).

    pos: [T] episode-frame index of each transition's current frame. ep_len,succ: [T]. jloc: [T] the
    clip-local index of the current frame. img_pad_row: [T, clip_len] bool pad mask of the clip.
    Returns cum[T,P], reward_nxt[T,P], done_nxt[T,P], valid[T,P], mc[T]."""
    T = pos.shape[0]
    P = len(prefixes)
    cl = img_pad_row.shape[1]

    def rew(q, elen, sc):  # reward at absolute episode-frame q (vectorized over T)
        # cost_to_goal: -1 per step, 0 in the last h_goal frames of a SUCCESS episode
        goal = sc & (q >= (elen - h_goal))
        return np.where(goal, 0.0, -1.0).astype(np.float32)

    cum = np.zeros((T, P), np.float32)
    # cumulative discounted reward over the h steps leading to each prefix
    maxh = prefixes[-1]
    disc_pow = discount ** np.arange(maxh)
    within = pos[:, None] + np.arange(maxh)[None] < ep_len[:, None]  # [T,maxh] still inside episode
    rr = rew(np.clip(pos[:, None] + np.arange(maxh)[None], 0, ep_len[:, None] - 1), ep_len[:, None], succ[:, None])
    rr = np.where(within, rr * disc_pow[None], 0.0)
    csum = np.cumsum(rr, axis=1)  # [T,maxh]
    for pi, h in enumerate(prefixes):
        cum[:, pi] = csum[:, h - 1]
    nxt = pos[:, None] + np.asarray(prefixes)[None]  # [T,P] episode-frame of the bootstrap successor
    jnxt = jloc[:, None] + np.asarray(prefixes)[None]  # [T,P] clip-local index of it
    in_ep = nxt < ep_len[:, None]
    in_clip = jnxt < cl
    not_pad = np.take_along_axis(img_pad_row, np.clip(jnxt, 0, cl - 1), axis=1) == False  # noqa: E712
    valid = (in_ep & in_clip & not_pad).astype(np.float32)
    reward_nxt = rew(np.clip(nxt, 0, ep_len[:, None] - 1), ep_len[:, None], succ[:, None])
    done_nxt = (succ[:, None] & (nxt >= (ep_len[:, None] - h_goal)) & in_ep).astype(np.float32)
    # FAILURE terminal: the last frame of a failure is an ABSORBING terminal with a large-negative
    # failure_reward (done=1). Keeps the -1/step cost, but anchors the failure deep so a short failure
    # cannot bootstrap to a shallow value and get preferred over a long success. Valid even when the
    # successor lands past the (truncated) episode end -- it is the absorbing terminal, not a real frame.
    fail_term = (~succ[:, None]) & (nxt >= ep_len[:, None] - 1) & in_clip & not_pad
    reward_nxt = np.where(fail_term, failure_reward, reward_nxt).astype(np.float32)
    done_nxt = np.where(fail_term, 1.0, done_nxt).astype(np.float32)
    valid = np.where(fail_term, 1.0, valid).astype(np.float32)
    # MC return-to-go from the current frame (exact for cost_to_goal deterministic outcomes)
    if scheme == "cost_to_goal":
        goal_start = ep_len - h_goal
        steps_to_goal = np.maximum(0, goal_start - pos)  # -1 per step until goal region, then 0
        mc = -(1 - discount**steps_to_goal) / (1 - discount)
        mc = np.where(succ, mc, v_min).astype(np.float32)  # failure: pin to floor (not a valid bound)
    else:
        mc = np.zeros(T, np.float32)
    return cum, reward_nxt, done_nxt, valid, mc, jnxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jellyho/yam_lego_taxi")
    ap.add_argument("--root", default="/data5/jellyho/yam_v2/lerobot")
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--backbone", default="small")
    ap.add_argument(
        "--dino-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="frozen DINOv2 compute dtype (bf16 ~2x on L40S)",
    )
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--num-atoms", type=int, default=101)
    ap.add_argument("--macro-group-size", type=int, default=2)
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
    ap.add_argument("--clip-len", type=int, default=60)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--clip-batch", type=int, default=8, help="clips per step")
    ap.add_argument("--trans-per-clip", type=int, default=32, help="transitions sampled per clip")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="acrft-critic")
    ap.add_argument("--wandb-group", default="patch-critic-clip")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/patch_critic_clip"))
    ap.add_argument(
        "--init-params",
        type=pathlib.Path,
        default=None,
        help="warm-start from a checkpoint dir (loads Q, and V if v_params.msgpack present)",
    )
    ap.add_argument(
        "--homing-onsets",
        type=pathlib.Path,
        default=None,
        help="json from compute_homing_onsets.py; truncates the homing tail of FAILURE episodes",
    )
    ap.add_argument(
        "--failure-reward",
        type=float,
        default=None,
        help="large-negative terminal reward at a FAILURE's last frame (done=1); default v_min (the floor)",
    )
    a = ap.parse_args()
    os.environ.setdefault("LEROBOT_VIDEO_BACKEND", "pyav")

    v_min = a.v_min if a.v_min is not None else -1.0 / (1.0 - a.discount)
    failure_reward = a.failure_reward if a.failure_reward is not None else v_min  # floor by default
    H = a.horizon
    print(
        f"discount={a.discount:.5f} v=[{v_min:.1f},{a.v_max:.1f}] mc_floor={a.mc_floor} "
        f"failure_reward={failure_reward:.1f} clip_len={a.clip_len}",
        flush=True,
    )

    import jax
    import jax.numpy as jnp
    import optax
    import torch

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble
    from openpi.patch_critic.critic import PatchV
    from openpi.training.critic_data_loader import CriticClipDataset

    outc = {}
    for line in pathlib.Path(a.outcomes).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outc[int(r["episode"])] = r["outcome"]

    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets is not None else None
    ds = CriticClipDataset(
        a.repo_id,
        root=a.root,
        episodes=list(outc),
        image_keys=CAMS,
        horizon=H,
        clip_len=a.clip_len,
        stride=a.stride,
        outcomes=outc,
        img_size=a.img_size,
        homing_onsets=homing,
    )
    print(f"{len(ds)} clips from {len(outc)} episodes{' (failure homing truncated)' if homing else ''}", flush=True)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=a.clip_batch,
        num_workers=a.num_workers,
        shuffle=True,
        collate_fn=CriticClipDataset.collate,
        drop_last=True,
        persistent_workers=a.num_workers > 0,
        prefetch_factor=2 if a.num_workers > 0 else None,
    )

    bb = DinoV2Backbone(a.backbone, dtype=getattr(jnp, a.dino_dtype))
    grid = int(bb.num_patches(224) ** 0.5)
    pooled = grid // 2
    npatch = 3 * pooled * pooled
    emb = bb.embed_dim
    ncam = len(CAMS)

    def pool(p):
        b, _, d = p.shape
        p = p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d)
        return p.mean((3, 5)).reshape(b, npatch, d)

    # DINOv2 wants [M, ncam, 3, S, S] in [0,1]; build it inside jit from the uint8 HWC clip frames.
    @jax.jit
    def encode(imgs_u8):  # [M, ncam, S, S, 3] uint8 -> [M, npatch, emb] f16
        x = imgs_u8.astype(jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 1, 4, 2, 3))  # -> [M, ncam, 3, S, S]
        return pool(bb(x)).astype(jnp.float16)

    ad = int(np.asarray(ds[0]["action"]).shape[1])
    net = PatchCriticEnsemble(
        action_dim=ad,
        horizon=H,
        num_critics=a.num_critics,
        macro_group_size=a.macro_group_size,
        num_atoms=a.num_atoms,
    )
    v_net = PatchV(num_atoms=a.num_atoms)
    hl = HLGauss(v_min, a.v_max, a.num_atoms)
    centers = jnp.asarray(hl.centers)
    prefixes = list(range(a.macro_group_size, H + 1, a.macro_group_size))
    P_ = len(prefixes)

    rng = jax.random.key(0)
    sd = np.asarray(ds[0]["state"]).shape[1]
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
        else:
            print(f"warm-start: loaded Q from {a.init_params} (V re-init: earlier ckpt saved no v_params)", flush=True)
    tgt = params
    tx = optax.adam(a.lr)
    tx_v = optax.adam(a.lr)
    opt = tx.init(params)
    v_opt = tx_v.init(v_params)

    def from_logits(x):
        return jnp.sum(jax.nn.softmax(x, -1) * centers, -1)

    def loss_fn(params, v_params, tgt_p, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc):
        # bootstrap V(s') distribution, shift by cumulative reward, project (distributional Bellman)
        vlog = v_net.apply(
            jax.lax.stop_gradient(v_params), pnxt.reshape(-1, npatch, emb).astype(jnp.float32), snxt.reshape(-1, sd)
        ).reshape(-1, P_, a.num_atoms)
        vprob = jax.nn.softmax(vlog, -1)
        gam = a.discount ** jnp.asarray(prefixes, jnp.float32)
        z = cum[..., None] + gam[None, :, None] * centers[None, None, :]
        phi = hl.to_probs(jnp.clip(z, v_min, a.v_max))
        tgt = jnp.einsum("bpj,bpja->bpa", vprob, phi)
        # terminal successor -> deterministic mass at its own reward (success goal = 0; failure end =
        # a large negative failure_reward, both set in analytic_targets).
        tgt = jnp.where((done_nxt > 0)[..., None], hl.to_probs(reward_nxt), tgt)
        if a.mc_floor:
            tmean = jnp.sum(tgt * centers, -1)
            floor = mc[:, None] > tmean
            tgt = jnp.where(floor[..., None], hl.to_probs(jnp.broadcast_to(mc[:, None], tmean.shape)), tgt)
        tgt = jax.lax.stop_gradient(tgt)
        pred = net.apply(params, pcur.astype(jnp.float32), chunk, scur)  # [K,B,P_,atoms]
        per = -jnp.sum(tgt[None] * jax.nn.log_softmax(pred, -1), -1)  # [K,B,P_]
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
    t0 = time.time()
    s = 0
    while s < a.steps:
        for clip in dl:
            if s >= a.steps:
                break
            imgs = clip["images"].numpy()  # [B, cl, ncam, S, S, 3] u8
            B, cl = imgs.shape[0], imgs.shape[1]
            patches = np.asarray(encode(jnp.asarray(imgs.reshape(B * cl, ncam, a.img_size, a.img_size, 3))))
            patches = patches.reshape(B, cl, npatch, emb)  # [B, cl, npatch, emb] f16
            state = clip["state"].numpy()  # [B, cl, sd]
            action = clip["action"].numpy()  # [B, cl+H, ad]
            img_pad = clip["img_pad"].numpy()  # [B, cl] bool
            pos0 = clip["clip_pos0"].numpy()  # [B]
            ep_len = clip["ep_len"].numpy()  # [B]
            succ = clip["is_success"].numpy()  # [B]
            # sample transitions: valid current positions (not pad) per clip
            bi, ji, pos, epl, sc = [], [], [], [], []
            for b in range(B):
                # current frames: real (not pad) AND within the episode's EFFECTIVE length (for FAILURE
                # clips this drops the homing tail; ep_len already carries the truncation from the loader).
                valid_j = np.flatnonzero((~img_pad[b]) & (pos0[b] + np.arange(cl) < ep_len[b]))
                if len(valid_j) == 0:
                    continue
                pick = rng_np.integers(0, len(valid_j), size=a.trans_per_clip)
                jj = valid_j[pick]
                bi.append(np.full(a.trans_per_clip, b))
                ji.append(jj)
                pos.append(pos0[b] + jj)
                epl.append(np.full(a.trans_per_clip, ep_len[b]))
                sc.append(np.full(a.trans_per_clip, succ[b]))
            if not bi:
                continue
            bi = np.concatenate(bi)
            ji = np.concatenate(ji)
            pos = np.concatenate(pos).astype(np.int64)
            epl = np.concatenate(epl).astype(np.int64)
            sc = np.concatenate(sc).astype(bool)
            cum, reward_nxt, done_nxt, valid, mc, jnxt = analytic_targets(
                pos, epl, sc, ji, img_pad[bi], prefixes, a.discount, a.h_goal, v_min, a.reward_scheme, failure_reward
            )
            jnxt = np.clip(jnxt, 0, cl - 1)
            # gather patches / state (device)
            pcur = jnp.asarray(patches[bi, ji])  # [T, npatch, emb]
            pnxt = jnp.asarray(patches[bi[:, None], jnxt])  # [T, P_, npatch, emb]
            scur = jnp.asarray(state[bi, ji])  # [T, sd]
            snxt = jnp.asarray(state[bi[:, None], jnxt])  # [T, P_, sd]
            # chunk = action[b, j:j+H]
            chunk = np.stack([action[bi[k], ji[k] : ji[k] + H] for k in range(len(bi))]).astype(np.float32)
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
            if s % 500 == 0 or s == a.steps - 1:
                i = jax.tree.map(lambda x: float(x), info)
                rate = (s + 1) / (time.time() - t0)
                print(
                    f"step {s:6d}  q_loss {i['q_loss']:.4f}  v_loss {i['v_loss']:.4f}  q_mean {i['q_mean']:.2f}  "
                    f"v_mean {i['v_mean']:.2f}  ({rate:.2f} it/s)",
                    flush=True,
                )
            if a.save_every and (s + 1) % a.save_every == 0:
                _save(a, carry[0], carry[3], npatch, v_min, prefixes, ad)
            s += 1
    _save(a, carry[0], carry[3], npatch, v_min, prefixes, ad)
    if wb is not None:
        wb.finish()


def _git_stamp():
    """branch@hash(+dirty) so a checkpoint can be traced back to the code that produced it."""
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parents[1])

    def _run(*cmd):
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        stamp = f"{_run('git', 'rev-parse', '--abbrev-ref', 'HEAD')}@{_run('git', 'rev-parse', '--short', 'HEAD')}"
        return stamp + ("+dirty" if _run("git", "status", "--porcelain") else "")
    except Exception:  # a checkpoint is still worth saving if git is unavailable
        return "unknown"


def _save(a, params, v_params, npatch, v_min, prefixes, ad, *, spec=None, stats=None, embedded=None):
    import flax.serialization
    import jax

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "params.msgpack").write_bytes(flax.serialization.msgpack_serialize(jax.device_get(params)))
    # also persist the value net so --init-params can warm-start BOTH Q and V on a later continuation.
    (a.out / "v_params.msgpack").write_bytes(flax.serialization.msgpack_serialize(jax.device_get(v_params)))
    cfg = {
        "horizon": a.horizon,
        "macro_group_size": a.macro_group_size,
        "num_atoms": a.num_atoms,
        "num_critics": a.num_critics,
        "discount": a.discount,
        "expectile": a.expectile,
        "target_tau": a.target_tau,
        "lr": a.lr,
        "v_min": v_min,
        "v_max": a.v_max,
        "backbone": a.backbone,
        "reward_scheme": a.reward_scheme,
        "h_goal": a.h_goal,
        "mc_floor": a.mc_floor,
        "action_dim": ad,
        "num_patches": int(npatch),
        "clip_len": a.clip_len,
        "source": a.repo_id,
        "loader": "critic_clip",
        "git": _git_stamp(),
    }
    # The INPUT CONTRACT. Without this the only record of "what scale does this critic eat" lives in two
    # unrelated files (the dataset loader and the serving wrapper), which agree only by convention -- a
    # critic trained on normalized inputs would be served raw and fail SILENTLY. Serving validates it.
    if spec is not None:
        cfg["input_spec"] = spec
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    if stats is not None:
        (a.out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    if embedded is not None:
        # the base VLA's stats, copied IN so the checkpoint is portable (a path is not)
        (a.out / "pi05_norm_stats.json").write_text(json.dumps(embedded))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
