"""Programmatic success-vs-failure discrimination for a patch-critic (no eyeballing single episodes).

Scores every LABELLED episode in a pc_rollouts dir: at each (strided) frame reads the critic's
ensemble-min full-horizon value Q(s, executed chunk), aggregates per episode (mean and max = closest
approach to goal), then reports the success-vs-fail separation: ROC-AUC (P[a random success episode
scores above a random fail]), group means, and a house-style figure.

    uv run python scripts/score_critic_auc.py --critic <ckpt> --data <pc_rollouts dir> --out <dir>
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", type=pathlib.Path, required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/critic_auc"))
    a = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp

    from openpi.patch_critic.backbone import DinoV2Backbone
    from openpi.patch_critic.backbone import to_nchw
    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    cc = json.loads((a.critic / "config.json").read_text())
    H, gsz, atoms, nc = cc["horizon"], cc["macro_group_size"], cc["num_atoms"], cc["num_critics"]
    m = json.loads((a.data / "meta.json").read_text())
    adim, sdim, n = m["action_dim"], m["state_dim"], m["num_steps"]
    net = PatchCriticEnsemble(action_dim=adim, horizon=H, num_critics=nc, macro_group_size=gsz, num_atoms=atoms)
    params = flax.serialization.msgpack_restore((a.critic / "params.msgpack").read_bytes())
    hl = HLGauss(cc["v_min"], cc["v_max"], atoms)
    centers = np.asarray(hl.centers)
    bb = DinoV2Backbone(cc["backbone"])
    grid = int(bb.num_patches(224) ** 0.5)
    pooled = grid // 2
    npatch = 3 * pooled * pooled

    def pool(p):
        b, _, d = p.shape
        return p.reshape(b, 3, grid, grid, d).reshape(b, 3, pooled, 2, pooled, 2, d).mean((3, 5)).reshape(b, npatch, d)

    @jax.jit
    def value_of(imgs_nchw, chunk, state):  # ensemble-min full-horizon mean value
        patches = pool(bb(imgs_nchw))
        out = net.apply(params, patches, chunk, state)  # [K,B,mh,atoms]
        sm = jax.nn.softmax(out, -1)
        return jnp.min(jnp.sum(sm[:, :, -1, :] * jnp.asarray(centers), -1), 0)  # [B]

    images = np.memmap(a.data / "images.dat", np.uint8, "r", shape=(n, 3, 224, 224, 3))
    state = np.asarray(np.memmap(a.data / "state.dat", np.float32, "r", shape=(n, sdim)))
    action = np.asarray(np.memmap(a.data / "action.dat", np.float32, "r", shape=(n, adim)))
    reward = np.asarray(np.memmap(a.data / "reward.dat", np.float32, "r", shape=(n,)))
    ep = np.asarray(np.memmap(a.data / "episode_index.dat", np.int32, "r", shape=(n,)))

    ep_mean, ep_max, ep_succ = [], [], []
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        succ = reward[idx].max() > 0.5
        sel = idx[:: a.stride]
        vals = []
        for s in range(0, len(sel), 64):
            b = sel[s : s + 64]
            chunks = np.zeros((len(b), H, adim), np.float32)
            for i, t in enumerate(b):
                loc = t - idx[0]
                k = min(H, len(idx) - loc)
                chunks[i, :k] = action[idx[0] + loc : idx[0] + loc + k]
            imgs = jnp.asarray(to_nchw(images[b].transpose(0, 1, 3, 4, 2).reshape(-1, 224, 224, 3))).reshape(
                -1, 3, 3, 224, 224
            )
            vals.append(np.asarray(value_of(imgs, jnp.asarray(chunks), jnp.asarray(state[b]))))
        vals = np.concatenate(vals)
        ep_mean.append(float(vals.mean()))
        ep_max.append(float(vals.max()))
        ep_succ.append(bool(succ))
        print(f"  ep{int(e):3d} {'S' if succ else 'F'}  mean {vals.mean():8.1f}  max {vals.max():8.1f}", flush=True)

    ep_mean, ep_max, ep_succ = np.array(ep_mean), np.array(ep_max), np.array(ep_succ)

    def auc(scores, labels):  # P[success scored above fail]; 0.5 = chance
        s, f = scores[labels], scores[~labels]
        if len(s) == 0 or len(f) == 0:
            return float("nan")
        return float((s[:, None] > f[None, :]).mean() + 0.5 * (s[:, None] == f[None, :]).mean())

    res = {
        "n_success": int(ep_succ.sum()),
        "n_fail": int((~ep_succ).sum()),
        "auc_mean_value": auc(ep_mean, ep_succ),
        "auc_max_value": auc(ep_max, ep_succ),
        "succ_mean_of_mean": float(ep_mean[ep_succ].mean()),
        "fail_mean_of_mean": float(ep_mean[~ep_succ].mean()),
        "succ_mean_of_max": float(ep_max[ep_succ].mean()),
        "fail_mean_of_max": float(ep_max[~ep_succ].mean()),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "auc.json").write_text(json.dumps(res, indent=2))
    print("\n=== DISCRIMINATION ===")
    print(json.dumps(res, indent=2), flush=True)

    # house-style figure: per-episode value by outcome (mean & max), AUC in title
    try:
        import matplotlib.pyplot as plt
        import report_style as rs

        rs.use()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, vals, tag, au in [
            (axes[0], ep_mean, "episode-mean value", res["auc_mean_value"]),
            (axes[1], ep_max, "episode-max value (closest approach)", res["auc_max_value"]),
        ]:
            rng = np.random.default_rng(0)
            for lab, col, name in [(True, rs.GREEN, "success"), (False, rs.RED, "fail")]:
                v = vals[ep_succ == lab]
                ax.scatter(
                    rng.uniform(-0.08, 0.08, len(v)) + (0 if lab else 1), v, s=18, color=col, alpha=0.7, label=name
                )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["success", "fail"])
            ax.set_ylabel("critic value")
            ax.set_title(f"{tag}\nAUC = {au:.3f}", fontsize=10)
        fig.suptitle(f"347-critic discrimination ({res['n_success']}S / {res['n_fail']}F episodes)", fontsize=11)
        rs.save(fig, a.out / "discrimination.png")
        print(f"figure -> {a.out / 'discrimination.png'}", flush=True)
    except Exception as ex:
        print(f"(figure skipped: {type(ex).__name__}: {ex})", flush=True)


if __name__ == "__main__":
    main()
