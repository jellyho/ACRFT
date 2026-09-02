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
    ap.add_argument(
        "--critic-arch",
        choices=["independent", "shared"],
        default="independent",
        help="independent: K full PatchARQCritics (the deployed architecture). shared: ONE trunk over "
        "the action-independent patch/proprio tokens plus K action heads. The trunk split is an exact "
        "refactor -- patch tokens already cannot attend to the action token -- and it is what makes a "
        "large K affordable (K=10 independent OOMs an L40S with a single 38.6 GB allocation).",
    )
    ap.add_argument("--trunk-layers", type=int, default=3, help="shared arch only")
    ap.add_argument(
        "--edac-batch", type=int, default=64, help="rows carrying the EDAC penalty (it costs a second backward)"
    )
    ap.add_argument("--head-layers", type=int, default=2, help="shared arch only")
    ap.add_argument(
        "--edac-weight",
        type=float,
        default=0.0,
        help="EDAC ensemble-diversity penalty on grad_a Q (An et al., NeurIPS 2021; their `eta`). "
        "Penalises the pairwise cosine similarity of the members' action-gradients, which is the one "
        "direction our critics agree along when they should not: worker B measures Q inflating +32.8 "
        "along grad_a Q against -0.13 along random directions, in all 9 critics, and our own pairwise "
        "probe puts the cosine at 0.33 for members that share nothing but the recipe (chance 0.049).",
    )
    ap.add_argument("--reward-scheme", choices=["cost_to_goal"], default="cost_to_goal")
    ap.add_argument("--h-goal", type=int, default=3)
    ap.add_argument("--discount", type=float, default=0.99964)
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--v-max", type=float, default=0.0)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument(
        "--q-reduction",
        choices=["min", "mean"],
        default="min",
        help="how the ensemble is reduced for the expectile comparison. min is the usual pessimism, but "
        "it pulls against the tau>0.5 optimism that is supposed to stop a failure's value propagating "
        "back through states it shares with successes; mean removes that tug-of-war.",
    )
    ap.add_argument(
        "--feat-dropout",
        type=float,
        default=0.0,
        help="probability of zeroing a whole patch TOKEN (occlusion in feature space). The backbone is "
        "frozen and its features are cached, so image-space augmentation is impossible; this is the "
        "stand-in that forces the head to generalise across visually similar frames instead of "
        "memorising which episode a frame came from.",
    )
    ap.add_argument(
        "--feat-noise", type=float, default=0.0, help="gaussian noise on patch features, as a fraction of their std"
    )
    ap.add_argument(
        "--alpha-cql",
        type=float,
        default=0.0,
        help="weight of the CQL conservative term, DIMENSIONLESS: the raw value-unit gap is divided "
        "by the value span (v_max - v_min) before scaling, because our TD loss is a cross-entropy in "
        "nats and official CQL's is an MSE in value units squared -- see the comment at the term. "
        "0 = off = plain IQL, which NEVER queries an action outside the dataset and therefore leaves "
        "the Q of every sampled chunk unconstrained at serving time. This is the term that pushes those "
        "down. Provenance: aviralkumar2907/CQL rlkit/torch/sac/cql.py -- "
        "`min_qf1_loss = logsumexp(cat_q1/temp).mean()*min_q_weight*temp - q1_pred.mean()*min_q_weight`.",
    )
    ap.add_argument(
        "--cql-negatives",
        choices=["shuffle", "uniform", "both", "bank"],
        default="shuffle",
        help="where the OOD action candidates come from. shuffle: in-batch permutation, i.e. a REAL "
        "chunk paired with the WRONG state -- free, and on-manifold so the critic cannot reject it on "
        "action statistics alone. uniform: U(-1,1) in normalized action space, official CQL's "
        "`random_actions_tensor ... .uniform_(-1,1)` with num_random=10. bank: chunks actually drawn "
        "from the frozen BC policy (scripts/sample_policy_chunks.py) -- the distribution our arms "
        "really score, and therefore the correctly targeted negative. both: shuffle + uniform.",
    )
    ap.add_argument("--cql-n", type=int, default=8, help="negatives per state (official CQL: num_random=10)")
    ap.add_argument("--cql-temp", type=float, default=1.0, help="official CQL `temp` (default 1.0)")
    ap.add_argument(
        "--cql-batch",
        type=int,
        default=64,
        help="how many rows of the batch carry the conservative term. The term costs one extra critic "
        "forward per negative, so this caps its price at cql_batch*cql_n/batch extra forwards; the batch "
        "is already a uniform draw, so the leading rows are an unbiased subsample.",
    )
    ap.add_argument(
        "--calql",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Cal-QL calibration: lower-bound the OOD Q-values by the data trajectory's MC return before "
        "the logsumexp, so conservatism never pushes a value below what the behaviour policy demonstrably "
        "achieves. nakamotoo/Cal-QL JaxCQL/conservative_sac.py applies `jnp.maximum(., lower_bounds)` to "
        "the POLICY-sampled Q's (current and next actions) and NOT to the uniform-random ones; we follow "
        "that split. `lower_bounds` is the same mc return array --mc-floor already uses.",
    )
    ap.add_argument("--cql-bank", type=pathlib.Path, default=None, help="dir from scripts/sample_policy_chunks.py")
    ap.add_argument(
        "--backup",
        choices=["scalar", "distributional"],
        default="scalar",
        help="scalar (DEFAULT, and what csmile-1006/DEAS-FQL does in agents/deas.py critic_loss): "
        "scalarize V(s'), form a SCALAR TD target, apply the HL-Gauss kernel ONCE, cross-entropy. "
        "distributional: the old path -- shift every atom by r + gamma^k, re-project with the same "
        "HL-Gauss kernel, mix by the next-state probabilities, i.e. a C51 operator with an HL-Gauss "
        "kernel. Kept only to reproduce the critics trained before this was fixed. It DIFFUSES: "
        "HL-Gauss's sigma is 0.75 x bin width by design, meant to be applied once to a scalar, so "
        "using it as a projection kernel convolves a Gaussian into the target at every backup. "
        "Measured on our support ([-2778, 0], 101 atoms) the target std grows 22.1 -> 31.3 -> 73.3 "
        "-> 222.1 after 0/1/10/100 backups, exactly 22.1*sqrt(n). Away from the boundary the mean "
        "survives; near the goal the clip at v_max=0 truncates the spreading mass and drags it down "
        "(from Q=-80 to -264 after 200 backups) -- a bias on the near-goal states selection cares "
        "about, with the same sign as worker B's 11-37% under-estimate of remaining time.",
    )
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
    spec["feat_dropout"] = a.feat_dropout
    spec["feat_noise"] = a.feat_noise
    spec["q_reduction"] = a.q_reduction
    spec["backup"] = a.backup
    spec["critic_arch"] = a.critic_arch
    spec["edac_weight"] = a.edac_weight
    if a.critic_arch == "shared":
        spec["trunk_layers"] = a.trunk_layers
        spec["head_layers"] = a.head_layers
    spec["alpha_cql"] = a.alpha_cql
    spec["cql_negatives"] = a.cql_negatives if a.alpha_cql > 0 else None
    spec["calql"] = bool(a.calql) if a.alpha_cql > 0 else None
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
    from openpi.patch_critic.critic import SharedTrunkCriticEnsemble

    _common = {
        "action_dim": ad,
        "horizon": H,
        "num_critics": a.num_critics,
        "macro_group_size": a.macro_group_size,
        "num_atoms": a.num_atoms,
    }
    if a.critic_arch == "shared":
        _common |= {"trunk_layers": a.trunk_layers, "head_layers": a.head_layers}
        net = SharedTrunkCriticEnsemble(**_common)
        # Same parameter tree, but each member differentiates its own copy of the action -- the JAX
        # form of EDAC's `actions_tile ... .requires_grad_(True)` (snu-mllab/EDAC sac.py).
        net_pm = SharedTrunkCriticEnsemble(**_common, per_member_actions=True)
    else:
        net = PatchCriticEnsemble(**_common)
        net_pm = PatchCriticEnsemble(**_common, per_member_actions=True)
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

    def _negatives(key, chunk, bank_negs):
        """OOD action candidates for the conservative term, [Bc, cql_n, H, ad].

        `shuffle` pairs a real chunk with the wrong state: it is on the action manifold, so the
        critic cannot reject it by action statistics alone, only by whether it fits THIS state --
        which is the discrimination we actually need at serving. (A row can draw its own chunk with
        probability 1/batch; at batch 256 that is a 0.4% self-pairing, well inside the noise, and
        official CQL's candidate set includes the data action outright in min_q_version < 3.)
        """
        Bc = min(a.cql_batch, chunk.shape[0])
        if a.cql_negatives == "bank":
            return bank_negs
        out = []
        if a.cql_negatives in ("shuffle", "both"):
            out.extend(chunk[jax.random.permutation(k, chunk.shape[0])][:Bc] for k in jax.random.split(key, a.cql_n))
        if a.cql_negatives in ("uniform", "both"):
            key = jax.random.fold_in(key, 1)
            out.extend(
                jax.random.uniform(k, (Bc, H, ad), minval=-1.0, maxval=1.0) for k in jax.random.split(key, a.cql_n)
            )
        return jnp.stack(out, 1)

    def loss_fn(params, v_params, tgt_p, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc, negs):
        vlog = v_net.apply(
            jax.lax.stop_gradient(v_params), pnxt.reshape(-1, npatch, emb).astype(jnp.float32), snxt.reshape(-1, sd)
        ).reshape(-1, P_, a.num_atoms)
        gam = a.discount ** jnp.asarray(prefixes, jnp.float32)
        if a.backup == "scalar":
            # DEAS-FQL agents/deas.py::critic_loss -- Farebrother et al.'s HL-Gauss used as intended,
            # a CLASSIFICATION loss on a scalar target with the kernel applied exactly once:
            #     next_v   = self.transform_from_probs(next_v_probs)
            #     target_v = rewards + discount**(nstep*action_sequence) * masks * next_v
            #     critic_loss = cross_entropy_loss_on_scalar(q_dists, target_v, transform_to_probs)
            # In scalar space the terminal case and the MC floor stop needing distribution surgery
            # and become a select and a maximum.
            next_v = hl.from_logits(vlog)  # [B, P_]
            y = cum + gam[None, :] * next_v
            y = jnp.where(done_nxt > 0, reward_nxt, y)
            if a.mc_floor:
                y = jnp.maximum(y, mc[:, None])
            tgt = jax.lax.stop_gradient(hl.to_probs(jnp.clip(y, v_min, a.v_max)))
        else:
            vprob = jax.nn.softmax(vlog, -1)
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
        # ---- conservative (CQL / Cal-QL) term -------------------------------------------------
        # IQL's expectile above only ever evaluates the DEMONSTRATED chunk, so Q is unconstrained
        # off-support. This pushes down the log-sum-exp of Q over OOD candidates while pulling up
        # the data action: official CQL computes
        #   logsumexp(cat_q/temp).mean()*temp - q_pred.mean(), scaled by min_q_weight,
        # with cat_q built from random + policy actions and NOT containing q_pred (min_q_version 3).
        # We omit CQL's importance-sampling density correction (`random_density = log(0.5**d)`):
        # it is calibrated for a per-STEP action space, and at chunk level d = H*ad = 420 makes it a
        # ~291-nat constant that would swamp every Q in the logsumexp. Dropping it is CQL's
        # min_q_version<3 form, and matches how the chunk-level critics in the VLA literature write
        # the term (Q-VGM arXiv 2606.08015v1 App. A: "log sum_A exp Q_m(s,A)", no density term).
        cql_loss = jnp.asarray(0.0)
        cql_gap = jnp.asarray(0.0)
        if a.alpha_cql > 0.0:
            Bc = negs.shape[0]
            nneg = negs.shape[1]
            qn = from_logits(
                net.apply(
                    params,
                    jnp.repeat(pcur[:Bc].astype(jnp.float32), nneg, 0),
                    negs.reshape(Bc * nneg, H, ad),
                    jnp.repeat(scur[:Bc], nneg, 0),
                )[:, :, -1, :]
            ).reshape(-1, Bc, nneg)  # [K, Bc, n]
            if a.calql and a.cql_negatives != "uniform":
                # Cal-QL Eq. 6: never penalise below the behaviour policy's own achieved return.
                qn = jnp.maximum(qn, jax.lax.stop_gradient(mc[:Bc])[None, :, None])
            q_data = from_logits(pred[:, :Bc, -1, :])  # [K, Bc]
            ood = a.cql_temp * jax.scipy.special.logsumexp(qn / a.cql_temp, axis=-1)
            cql_gap = jnp.mean(ood - q_data)  # raw VALUE units, so it stays interpretable in the log
            # Scale conversion, required and not present in official CQL. There the TD loss is
            # MSE on Q, so a value-unit conservative term is naturally commensurate. Ours is a
            # cross-entropy over 101 HL-Gauss atoms (~5 nats), while the gap is in value units
            # (~100 on this reward scale): alpha_cql=1.0 unconverted would make the conservative
            # term ~20x the TD loss and simply collapse the critic. Dividing by the value span
            # makes alpha_cql dimensionless and portable across reward scales -- alpha_cql=1 then
            # means "one span-fraction of conservatism per nat of TD loss".
            cql_loss = a.alpha_cql * cql_gap / (a.v_max - v_min)

        # ---- EDAC ensemble-diversity penalty on grad_a Q ---------------------------------------
        # An et al., "Uncertainty-Based Offline RL with Diversified Q-Ensemble" (NeurIPS 2021).
        # Transcribed from the official snu-mllab/EDAC sac.py: tile the action so every member
        # differentiates its OWN copy, L2-normalise each member's gradient, take all pairwise dot
        # products, zero the diagonal, sum, average over the batch, divide by (num_qs - 1):
        #     qs_pred_grads = qs_pred_grads / (torch.norm(..., p=2, dim=2).unsqueeze(-1) + 1e-10)
        #     qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        #     qs_pred_grads = (1 - masks) * qs_pred_grads
        #     grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1,2))) / (self.num_qs - 1)
        # Our gradient is taken on the FULL-CHUNK prefix, which is the value steering differentiates.
        edac_loss = jnp.asarray(0.0)
        edac_cos = jnp.asarray(0.0)
        if a.edac_weight > 0.0 and a.num_critics > 1:
            K = a.num_critics
            Be = min(a.edac_batch, pcur.shape[0])
            pc_e, sc_e, ch_e = pcur[:Be].astype(jnp.float32), scur[:Be], chunk[:Be]

            def _q_per_member(ck):
                return from_logits(net_pm.apply(params, pc_e, ck, sc_e)[:, :, -1, :]).sum()

            g = jax.grad(_q_per_member)(jnp.broadcast_to(ch_e, (K, *ch_e.shape)))  # [K, Be, H, ad]
            gf = g.reshape(K, Be, -1)
            gf = gf / (jnp.linalg.norm(gf, axis=-1, keepdims=True) + 1e-10)
            sim = jnp.einsum("bik,bjk->bij", jnp.transpose(gf, (1, 0, 2)), jnp.transpose(gf, (1, 0, 2)))
            sim = sim * (1.0 - jnp.eye(K))
            edac_cos = jnp.sum(sim) / (Be * K * (K - 1))  # mean off-diagonal cosine, for the log
            edac_loss = a.edac_weight * jnp.mean(jnp.sum(sim, axis=(1, 2))) / (K - 1)

        qd_log = net.apply(jax.lax.stop_gradient(tgt_p), pcur.astype(jnp.float32), chunk, scur)[:, :, -1, :]
        # V's target distribution has to come from the SAME member the expectile comparison used.
        # DEAS-FQL value_loss picks the arg-min member's whole distribution when q_agg == "min":
        #     min_q_idx = jnp.argmin(qs, axis=0); q_prob = q_probs[min_q_idx, batch_indices]
        # We were taking the ensemble MEAN distribution while comparing against the ensemble MIN
        # scalar, so the weight said "be pessimistic" about a target that was not.
        qs_all = from_logits(qd_log)  # [K, B]
        if a.q_reduction == "min":
            _i = jnp.argmin(qs_all, axis=0)
            _b = jnp.arange(qs_all.shape[1])
            qbar = qs_all[_i, _b]
            qd_probs = jax.nn.softmax(qd_log, -1)[_i, _b]
        else:
            qbar = jnp.mean(qs_all, 0)
            qd_probs = jnp.mean(jax.nn.softmax(qd_log, -1), 0)
        vlog_c = v_net.apply(v_params, pcur.astype(jnp.float32), scur)
        vbar = from_logits(vlog_c)
        u = jax.lax.stop_gradient(qbar) - vbar
        wexp = jnp.abs(a.expectile - (u < 0).astype(jnp.float32))
        v_ce = -jnp.sum(jax.lax.stop_gradient(qd_probs) * jax.nn.log_softmax(vlog_c, -1), -1)
        v_loss = jnp.sum(wexp * v_ce) / u.shape[0]
        return q_loss + v_loss + cql_loss + edac_loss, {
            "q_loss": q_loss,
            "v_loss": v_loss,
            "cql_gap": cql_gap,
            "edac_cos": edac_cos,
            "q_mean": jnp.mean(from_logits(pred)),
            "v_mean": jnp.mean(vbar),
        }

    def _augment(key, x):
        """Occlusion + noise on frozen patch tokens. Scale-free: noise is relative to the batch std."""
        if a.feat_dropout <= 0.0 and a.feat_noise <= 0.0:
            return x
        x = x.astype(jnp.float32)
        k1, k2 = jax.random.split(key)
        if a.feat_dropout > 0.0:
            keep = jax.random.uniform(k1, (*x.shape[:-1], 1)) >= a.feat_dropout
            x = x * keep  # occlusion, not dropout: no 1/(1-p) rescale
        if a.feat_noise > 0.0:
            x = x + a.feat_noise * jnp.std(x) * jax.random.normal(k2, x.shape, x.dtype)
        return x

    @jax.jit
    def step(carry, key, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc, bank_negs):
        kc, kn, kq = jax.random.split(key, 3)
        pcur, pnxt = _augment(kc, pcur), _augment(kn, pnxt)
        negs = _negatives(kq, chunk, bank_negs) if a.alpha_cql > 0.0 else bank_negs
        params, tgt, opt, v_params, v_opt = carry
        (_, info), (gp, gv) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            params, v_params, tgt, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc, negs
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
    aug_key = jax.random.key(1)
    ar_h = np.arange(H)
    pref = np.asarray(prefixes)
    # ---- CQL negative bank (frozen-BC policy samples), if requested ---------------------------
    if a.alpha_cql > 0.0 and a.cql_negatives == "bank":
        if a.cql_bank is None:
            raise SystemExit("--cql-negatives bank needs --cql-bank <dir from sample_policy_chunks.py>")
        bank_chunks = np.load(a.cql_bank / "chunks.npy", mmap_mode="r")  # [Nb, K, H, ad]
        bank_rows = np.load(a.cql_bank / "idx.npy")
        # The bank is strided over frames, so a training row's own frame is usually not in it; map
        # each row to the NEAREST sampled frame. The drift is bounded by the bank stride (reported
        # below); at 30 Hz a stride of s frames is s/30 s of staleness in the conditioning image,
        # which is well inside the horizon the chunk itself spans.
        nearest = np.searchsorted(bank_rows, np.arange(N)).clip(0, len(bank_rows) - 1)
        left = (nearest - 1).clip(0)
        pick = np.where(
            np.abs(bank_rows[nearest] - np.arange(N)) <= np.abs(bank_rows[left] - np.arange(N)), nearest, left
        )
        drift = np.abs(bank_rows[pick] - np.arange(N))
        print(
            f"cql bank: {len(bank_rows)} sampled frames, k={bank_chunks.shape[1]}, "
            f"row->bank drift mean {drift.mean():.1f} max {drift.max()} frames",
            flush=True,
        )
        n_use = min(a.cql_n, bank_chunks.shape[1])

        def bank_negs_of(gcur):
            return jnp.asarray(np.asarray(bank_chunks[pick[gcur[: a.cql_batch]]][:, :n_use], np.float32))
    else:
        _dummy = jnp.zeros((min(a.cql_batch, a.batch), 1, H, ad), jnp.float32)

        def bank_negs_of(gcur):
            return _dummy

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
        aug_key, k_step = jax.random.split(aug_key)
        carry, info = step(
            carry,
            k_step,
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
            bank_negs_of(gcur),
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
            cqls = f"  cql_gap {i['cql_gap']:.2f}" if a.alpha_cql > 0.0 else ""
            cqls += f"  edac_cos {i['edac_cos']:.3f}" if a.edac_weight > 0.0 else ""
            print(
                f"step {s:6d}  q_loss {i['q_loss']:.4f}  v_loss {i['v_loss']:.4f}  q_mean {i['q_mean']:.2f}  "
                f"v_mean {i['v_mean']:.2f}{cqls}  ({rate:.2f} it/s)",
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
