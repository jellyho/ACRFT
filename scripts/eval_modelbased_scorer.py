"""Does routing the value through dynamics restore action sensitivity?

The offline critic battery found BOTH critics (VLA-z and cheap-z, same IQL recipe) ignore the
action: with all-success demos every demonstrated chunk leads to success, so Q(z, a) degenerates
to a state-value function and best-of-N has nothing to select. The model-based hypothesis is that

    score_mb(z, a) = V(d(z, a))          (dynamics ensemble mean, V = the critic's state-value)

recovers action sensitivity structurally: different chunks predict different futures, and V of a
different future differs even when Q was flat. This runs the SAME discrimination battery on the
model-based score so the two scoring rules are directly comparable:

  * within-state spread of score over the 16+8 candidate chunks (was ~0.0013 for Q)
  * demonstrated-vs-shuffled discrimination: score(z, demo chunk) vs score(z, random other
    frame's chunk) — AUC and mean margin (Q FAILED this)
  * per-prefix score curves for the commit decision (does score vary with prefix?)
  * disagreement-weighted variant: score - beta * ensemble disagreement (the conservative form)

Cheap: everything runs on cached arrays. One GPU, minutes.
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_v(critic_dir: pathlib.Path, zdim: int):
    """Rebuild the critic's V(z) head from its saved params (msgpack flax tree -> torch-free eval).

    We only need forward evaluation, so the flax params are applied manually with numpy/torch.
    The V net is an MLP saved in vparams.msgpack; config.json records the layer sizes.
    """
    import flax.serialization as fser

    cfg = json.loads((critic_dir / "config.json").read_text())
    raw = (critic_dir / "vparams.msgpack").read_bytes()
    tree = fser.msgpack_restore(raw)
    return tree, cfg


def mlp_apply(tree, x: torch.Tensor) -> torch.Tensor:
    """Mirror critic.py's V exactly: LayerNorm_0 on the INPUT, then Dense_i+gelu, last Dense linear.

    (Verified against the saved tree: LayerNorm_0 has the input width, and _mlp in critic.py is
    ``x = gelu(Dense(d)(x))`` per hidden layer with a final linear Dense.)
    """
    g = torch.from_numpy(np.asarray(tree["LayerNorm_0"]["scale"])).to(x.device, x.dtype)
    be = torch.from_numpy(np.asarray(tree["LayerNorm_0"]["bias"])).to(x.device, x.dtype)
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(var + 1e-6) * g + be
    dense = sorted((k for k in tree if k.startswith("Dense")), key=lambda s: int(s.split("_")[1]))
    for i, k in enumerate(dense):
        w = torch.from_numpy(np.asarray(tree[k]["kernel"])).to(x.device, x.dtype)
        b = torch.from_numpy(np.asarray(tree[k]["bias"])).to(x.device, x.dtype)
        h = h @ w + b
        if i < len(dense) - 1:
            h = torch.nn.functional.gelu(h)
    return h.squeeze(-1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-dir", type=pathlib.Path, default=pathlib.Path(".scratch/cheap_z_v4b"))
    ap.add_argument("--annot", type=pathlib.Path, default=pathlib.Path(".scratch/annot_noprop"))
    ap.add_argument("--dyn", type=pathlib.Path, default=pathlib.Path(".scratch/cheapz_dyn_v0"))
    ap.add_argument("--critic", type=pathlib.Path, default=pathlib.Path(".scratch/critic_cheapz_v4b"),
                    help="critic dir whose V head scores the predicted futures (must match --z-dir's z!)")
    ap.add_argument("--num-states", type=int, default=2048)
    ap.add_argument("--beta", type=float, default=1.0, help="disagreement penalty for the conservative score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/mb_scorer_eval.json"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.annot / "meta.json").read_text())
    n, H, A, N = meta["num_frames"], meta["horizon"], meta["action_dim"], meta["num_samples"]
    z = np.load(args.z_dir / "z.npy")
    zdim = z.shape[1]
    chunk = np.memmap(args.annot / "action_chunk.dat", dtype=np.float32, mode="r", shape=(n, H, A))
    cand = np.memmap(args.annot / "base_action.dat", dtype=np.float32, mode="r", shape=(n, N, H, A))
    ep = np.array(np.memmap(args.annot / "episode_index.dat", dtype=np.int32, mode="r", shape=(n,)))
    pro = np.memmap(args.annot / "proprio.dat", dtype=np.float32, mode="r", shape=(n, meta["proprio_dim"]))

    # dynamics ensemble
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_cheapz_dynamics import Dyn

    state = torch.load(args.dyn / "ensemble.pt", map_location=dev)
    models = []
    for k in sorted(state.keys()):
        m = Dyn(zdim, H, A).to(dev)
        m.load_state_dict(state[k])
        m.eval()
        models.append(m)

    vtree, vcfg = load_v(args.critic, zdim)
    logger.info(f"critic V config keys: {list(vcfg)[:6]}...  vtree top: {list(vtree)[:4]}")

    # The critic's V consumes [z ; z-scored proprio]. Reproduce that normalization.
    mu, sd = pro.mean(0), pro.std(0)
    pro_n = np.where(sd > 1e-6, (pro - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0).astype(np.float32)

    def V(z_batch, pro_batch):
        x = torch.cat([z_batch, pro_batch], dim=-1)
        # vtree may nest under a top-level module name
        tree = vtree[next(iter(vtree))] if len(vtree) == 1 else vtree
        return mlp_apply(tree, x)

    # sample states away from episode ends so t+H exists
    ep_end = np.zeros(int(ep.max()) + 1, dtype=np.int64)
    for e in np.unique(ep):
        ep_end[e] = np.flatnonzero(ep == e).max()
    valid = np.flatnonzero((np.arange(n) + H) <= ep_end[ep])
    rows = rng.choice(valid, size=min(args.num_states, len(valid)), replace=False)

    z_t = torch.from_numpy(z).to(dev)
    zt = z_t[torch.from_numpy(rows).to(dev)]
    pr = torch.from_numpy(pro_n[rows]).to(dev)

    def mb_score(chunks):  # chunks: [B, K, H, A] -> mean score, disagreement [B, K]
        B, K = chunks.shape[:2]
        c = torch.from_numpy(np.ascontiguousarray(chunks)).to(dev).view(B * K, H, A)
        zz = zt[:, None].expand(B, K, zdim).reshape(B * K, zdim)
        pp = pr[:, None].expand(B, K, pr.shape[-1]).reshape(B * K, -1)
        pfx = torch.full((B * K,), 1.0, device=dev)
        with torch.no_grad():
            preds = torch.stack([m(zz, c, pfx) for m in models])  # [M, B*K, z]
            vs = torch.stack([V(preds[i], pp) for i in range(len(models))])  # [M, B*K]
        disag = torch.norm(preds - preds.mean(0, keepdim=True), dim=-1).mean(0)
        return vs.mean(0).view(B, K), disag.view(B, K)

    # 1) within-state spread over policy candidates
    s_cand, d_cand = mb_score(np.asarray(cand[rows]))  # [B, N]
    spread = s_cand.std(dim=1).mean().item()

    # 2) demonstrated vs shuffled chunk discrimination
    demo = np.asarray(chunk[rows])[:, None]  # [B,1,H,A]
    shuf = np.asarray(chunk[rng.permutation(rows)])[:, None]
    s_demo, d_demo = mb_score(demo)
    s_shuf, d_shuf = mb_score(shuf)
    margin = (s_demo - s_shuf).mean().item()
    auc = float((s_demo.squeeze(1)[:, None] > s_shuf.squeeze(1)[None, :]).float().mean().item())
    cons_margin = ((s_demo - args.beta * d_demo) - (s_shuf - args.beta * d_shuf)).mean().item()

    # Disagreement ALONE as the discriminator (score = -disagreement): the v2c run showed the value
    # path is flat but the uncertainty path separates matched from mismatched chunks — quantify it.
    dd, ds = d_demo.squeeze(1), d_shuf.squeeze(1)
    auc_disag = float(((-dd)[:, None] > (-ds)[None, :]).float().mean().item())
    spread_disag = d_cand.std(dim=1).mean().item()

    report = {
        "within_state_spread_mb": spread,
        "within_state_spread_disagreement": spread_disag,
        "demo_vs_shuffled_margin": margin,
        "demo_vs_shuffled_margin_conservative": cons_margin,
        "demo_vs_shuffled_auc": auc,
        "demo_vs_shuffled_auc_disagreement_only": auc_disag,
        "mean_disagreement_demo": d_demo.mean().item(),
        "mean_disagreement_shuffled": d_shuf.mean().item(),
    }
    args.out.write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(f"{k:42s} {v:+.4f}")
    print("(reference: model-free Q within-state spread was ~0.0013 and demo-vs-shuffled FAILED)")


if __name__ == "__main__":
    main()
