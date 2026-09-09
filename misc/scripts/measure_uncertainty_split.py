"""P2 measurement: the aleatoric/epistemic split across training (uncertainty-split entry, prediction 2).

For each frame of the chosen episodes and each checkpoint, decompose the ensemble's return
distributions by the law of total variance:

    u_alea(s,k) = mean_m Var[Z_m(s, a_1:k)]      (within-member: survives training)
    u_epis(s,k) = Var_m  E[Z_m(s, a_1:k)]        (across-member: training should shrink it)

Prediction 2 says: from the 20k checkpoint to the 120k continuation, u_epis drops while u_alea
stays. Values are computed from the yam_s347 feature cache (no DINOv2 recompute); the deployed
ensemble is K=2, so u_epis here is a 2-member disagreement -- a weak but honest estimate (the
head_ensemble route to K>=8 is noted in the entry).

    uv run python scripts/measure_uncertainty_split.py \
      --critic-a /data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_g5_pi05 --label-a 20k \
      --critic-b .scratch/patch_critic_yam_s347_g5_pi05_cont --label-b 120k \
      --episodes 320 79 23 214 5 141 --out .scratch/p2_uncertainty
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic-a", type=pathlib.Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--critic-b", type=pathlib.Path, default=None)
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/p2_uncertainty"))
    a = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp

    from openpi.patch_critic.critic import HLGauss
    from openpi.patch_critic.critic import PatchCriticEnsemble

    m = json.loads((a.cache / "meta.json").read_text())
    npatch, emb, sd, ad, n_total = m["npatch"], m["emb"], m["sd"], m["ad"], m["N"]
    eps = {int(k): v for k, v in m["episodes"].items()}
    feats = np.memmap(a.cache / "features.dat", np.float16, "r", shape=(n_total, npatch, emb))
    state = np.asarray(np.memmap(a.cache / "state.dat", np.float32, "r", shape=(n_total, sd)))
    action = np.asarray(np.memmap(a.cache / "action.dat", np.float32, "r", shape=(n_total, ad)))

    def load(cdir):
        cc = json.loads((cdir / "config.json").read_text())
        net = PatchCriticEnsemble(
            action_dim=ad,
            horizon=cc["horizon"],
            num_critics=cc["num_critics"],
            macro_group_size=cc["macro_group_size"],
            num_atoms=cc["num_atoms"],
        )
        params = flax.serialization.msgpack_restore((cdir / "params.msgpack").read_bytes())
        hl = HLGauss(cc["v_min"], cc["v_max"], cc["num_atoms"])
        return net, params, np.asarray(hl.centers), cc

    def split_fn(net, params, centers):
        c = jnp.asarray(centers)

        @jax.jit
        def f(patches, chunk, st):
            logits = net.apply(params, patches, chunk, st)  # [K,B,P,atoms]
            probs = jax.nn.softmax(logits, -1)
            mean_m = jnp.sum(probs * c, -1)  # [K,B,P] per-member mean
            var_m = jnp.sum(probs * (c - mean_m[..., None]) ** 2, -1)  # [K,B,P] per-member variance
            u_alea = jnp.mean(var_m, 0)  # [B,P]
            u_epis = jnp.var(mean_m, 0)  # [B,P]  (K=2 disagreement)
            return u_alea, u_epis, jnp.mean(mean_m, 0)

        return f

    critics = [(a.label_a, a.critic_a)] + ([(a.label_b, a.critic_b)] if a.critic_b else [])
    out = {}
    for label, cdir in critics:
        net, params, centers, cc = load(cdir)
        H = cc["horizon"]
        fn = split_fn(net, params, centers)
        per_ep = {}
        for e in a.episodes:
            off, n = eps[e]["offset"], eps[e]["full_len"]
            sel = np.arange(0, n, a.stride)
            chunks = np.zeros((len(sel), H, ad), np.float32)
            for i, t in enumerate(sel):
                k = min(H, n - t)
                chunks[i, :k] = action[off + t : off + t + k]
            ua, ue, qm = [], [], []
            B = 256
            for s0 in range(0, len(sel), B):
                fr = off + sel[s0 : s0 + B]
                r = fn(
                    jnp.asarray(np.asarray(feats[fr]), jnp.float32),
                    jnp.asarray(chunks[s0 : s0 + B]),
                    jnp.asarray(state[fr]),
                )
                ua.append(np.asarray(r[0]))
                ue.append(np.asarray(r[1]))
                qm.append(np.asarray(r[2]))
            per_ep[e] = {
                "success": bool(eps[e]["success"]),
                "frames": sel.tolist(),
                "u_alea": np.concatenate(ua).tolist(),
                "u_epis": np.concatenate(ue).tolist(),
                "q_mean": np.concatenate(qm).tolist(),
            }
            fu = np.concatenate(ua)[:, -1]
            fe = np.concatenate(ue)[:, -1]
            print(
                f"[{label}] ep{e} ({'succ' if eps[e]['success'] else 'fail'}): "
                f"u_alea(full-prefix) median {np.median(fu):.0f}  u_epis median {np.median(fe):.0f}",
                flush=True,
            )
        out[label] = per_ep
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "p2_split.json").write_text(json.dumps(out))
    print("wrote", a.out / "p2_split.json", flush=True)


if __name__ == "__main__":
    main()
