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
import openpi.training.outcomes as _outcomes

# reuse the validated target math + checkpoint writer from the clip trainer
from scripts.train_patch_critic_clip import _save
from scripts.train_patch_critic_clip import analytic_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, required=True, help="dir from cache_patch_features.py")
    ap.add_argument(
        "--outcomes",
        default=None,
        help="legacy outcomes.jsonl. DEPRECATED: the verdict now lives in the dataset schema as the "
        "per-frame next.success / next.done features, aggregated per episode in meta/episodes, and "
        "openpi.training.outcomes reads it from there. Kept only for pre-migration copies.",
    )
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
        "MEASURED RANGE: 10 and 30 DESTROY the value function -- see below -- and the usable range is "
        "well under 1. "
        "0 = off = plain IQL, which NEVER queries an action outside the dataset and therefore leaves "
        "the Q of every sampled chunk unconstrained at serving time. This is the term that pushes those "
        "down. Provenance: aviralkumar2907/CQL rlkit/torch/sac/cql.py -- "
        "`min_qf1_loss = logsumexp(cat_q1/temp).mean()*min_q_weight*temp - q1_pred.mean()*min_q_weight`. "
        "WHY THE RANGE MATTERS. At alpha 10 and 30 the term wins outright: the critic separates the "
        "executed chunk from wrong-state chunks almost perfectly (ranking accuracy 0.549 -> 0.994) "
        "and stops being a value function while doing it. Spearman(Q, -time_to_goal) on demonstrated "
        "chunks falls from +0.991 to +0.136 (alpha 10) and +0.082 (alpha 30), and Q's spread across "
        "states collapses from 381 to 79. It is not an over-training effect: at alpha 10 the "
        "correlation is already +0.287 by step 20k and never recovers, so there is no early stop that "
        "rescues it. What those critics learned is 'is this the action that was demonstrated here', a "
        "discriminator, not a value. Note also that measuring them with the selection-bias probe alone "
        "would have shown a spectacular win, because that probe scores the very objective the term "
        "optimises -- the value-correlation check is what catches it.",
    )
    ap.add_argument(
        "--cql-negatives",
        choices=["shuffle", "within", "uniform", "both", "bank"],
        default="shuffle",
        help="where the OOD action candidates come from. shuffle: in-batch permutation, i.e. a REAL "
        "chunk paired with the WRONG state -- free, and on-manifold so the critic cannot reject it on "
        "action statistics alone. WITH ONE SHORTCUT: a batch drawn uniformly from 938k frames almost "
        "always pairs across EPISODES, and a linear probe recovers episode identity from these frozen "
        "DINOv2 features at accuracy 1.000 against a chance of 0.0019, with 70% of a frame's nearest "
        "neighbours from its own episode (.scratch/probe_cheap_z.json). So a cross-episode negative can "
        "be pushed down by noticing the episode rather than by understanding the state -- a cue that does "
        "not exist at serving time, where the policy's own sample carries no episode. That is one "
        "mechanism behind what alpha 10/30 did above: a discriminator with a collapsed value function. "
        "within: the same construction drawn from the SAME EPISODE at a different time, which removes "
        "that cue and leaves only the discrimination we actually need. uniform: U(-1,1) in normalized "
        "action space, official CQL's "
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
        "--inv-weight",
        type=float,
        default=0.0,
        help="AC-State multi-step inverse auxiliary on the SHARED TRUNK (Lamb et al., arXiv 2207.08229; "
        "official alexmlamb/ControllableLatentState). 0 = off. Predict the action taken at t from the "
        "pair (z_t, z_{t+k}) -- repnet.py: `a_hat = self.multi_step_inv_model(z0, z1)` against the "
        "action AT t, with invnet.py concatenating `context = torch.cat((z0, z1), -1)` and NOT taking k "
        "as an input. WHY HERE, with the sourcing corrected: the episode_acc 1.000 / knn_purity 0.698 "
        "figures are RoboCasa PrepareCoffee, not YAM (see --cql-negatives). On YAM's own pooled DINOv2 "
        "features knn_purity is 0.061 with 94% of neighbours cross-episode, so 'the representation splits "
        "the task by episode' is NOT established here and this arm's motivation is weaker than first "
        "written. What does hold on YAM: cost_to_goal is a frame-index target whose residual at matched "
        "task state is CV 0.188 across episodes (p10-p90 1.51x), which only episode-specific pace "
        "explains. An inverse objective keeps only "
        "what is needed to infer the ACTION, which is invariant to background, lighting and -- unlike our "
        "cost_to_goal target -- to how long this particular episode happened to take.",
    )
    ap.add_argument(
        "--inv-max-k",
        type=int,
        default=90,
        help="largest gap, in frames, between the two observations. DEVIATION FROM THE OFFICIAL CODE, "
        "stated because it is a real one: buffer.py sets `randk = maxk` -- always the LARGEST valid gap, "
        "clamped by `steps_to_goal`. In their gridworld that is a handful of steps. Here it would be the "
        "whole episode: at 30 Hz with ~2900-frame episodes it would ask which 14-dim action was taken at t "
        "given a frame 90 SECONDS later, which carries no information about it. We sample k log-uniformly "
        "in [1, inv_max_k] instead, so the pair spans easy and hard gaps. 90 frames = 3 s = 3x the chunk.",
    )
    ap.add_argument(
        "--inv-bottleneck",
        type=int,
        default=64,
        help="width of the projection the inverse head reads. The official information bottleneck is a "
        "VECTOR QUANTIZER applied to z0/z1 (repnet.py: `z0, zq_loss0, ind = self.vq_layer(z0)`), and its "
        "lossiness is what forces the encoder to DISCARD; a linear projection in the auxiliary branch only "
        "adds pressure to RETAIN control-relevant information, it does not force anything out of the trunk. "
        "That distinction is the main reason this arm might fail, and it is pre-registered as such: expect "
        "act_cos to move before episode_acc does.",
    )
    ap.add_argument(
        "--inv-zero-proprio",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="zero the proprio token in the AUXILIARY branch only. Our arms move smoothly, so the action at "
        "t is largely predictable from proprio by interpolation -- the inverse task would then be solved "
        "without the visual encoder learning anything, the same shape of shortcut as the episode cue in "
        "--cql-negatives shuffle. Zeroing forces the pixels to carry it.",
    )
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
    ap.add_argument(
        "--discount2",
        type=float,
        default=None,
        help="DEAS's INTER-OPTION discount gamma2, used for the bootstrap exponent, while --discount "
        "becomes the INTRA-OPTION gamma1 that sums rewards inside one chunk and sets the MC floor. "
        "DEAS carries the two separately end to end (utils/smdp.py:12-13, datasets.py:240-241, "
        "agents/deas.py:159) and at scale they differ: scripts/large/deas.sh:11-12 sets "
        "DISCOUNT1=0.9, DISCOUNT2=0.999. None = gamma2 is gamma1, i.e. today's single-discount "
        "behaviour, so every existing checkpoint stays reproducible.",
    )
    ap.add_argument(
        "--hlg-sigma-frac",
        type=float,
        default=0.75,
        help="HL-Gauss kernel width in bin widths. Ours realises N(y, sigma^2) with sigma = frac*bin "
        "(rlt_critic/critic.py:281-284). DEAS divides by sqrt(2)*sigma before the STANDARD-NORMAL cdf "
        "(utils/hlg.py:90-93), where the canonical form divides before erf, so its realised kernel is "
        "sqrt(2) WIDER. Pass 1.06066 (= 0.75*sqrt(2)) to match DEAS exactly.",
    )
    ap.add_argument(
        "--terminal",
        choices=["replace", "deas"],
        default="replace",
        help="replace (today's): on a terminal the target BECOMES reward_nxt, discarding the reward "
        "accumulated over the chunk. deas: only the BOOTSTRAP is masked, the accumulated reward is "
        "kept -- agents/deas.py:157-162 `rewards + gamma^(nstep*L) * masks * next_v`.",
    )
    ap.add_argument(
        "--support",
        choices=["fixed", "smdp"],
        default="fixed",
        help="fixed (today's): v_min = -1/(1-discount). smdp: DEAS's default `universal` support, "
        "derived from BOTH discounts and the option length (main.py:150-154, utils/smdp.py:18-34). "
        "DEAS hardcodes H=1000 for OGBench; ours must be given via --smdp-h.",
    )
    ap.add_argument("--smdp-h", type=int, default=5941, help="episode length for the smdp support; our max task length")
    ap.add_argument(
        "--zero-init-head",
        action="store_true",
        help="zero the critic's output kernel at init, as DEAS does via use_zero_output=True "
        "(agents/deas.py:380-387, utils/networks.py:50,55-59). Only affects the first few thousand steps.",
    )
    ap.add_argument(
        "--value-head",
        choices=["categorical", "floq"],
        default="categorical",
        help="categorical: HL-Gauss logits, read once. floq: the Q-value is the ENDPOINT of a flow "
        "over a scalar, integrated K Euler steps from uniform noise (arXiv 2509.06863, official "
        "CMU-AIRe/floq). floq REPLACES the categorical read-out -- HL-Gauss survives only as the "
        "INPUT encoding of the interpolant, which is what floq itself does.",
    )
    ap.add_argument("--floq-steps", type=int, default=8, help="K, floq/agents/floq.py:477; tuned over {4,8,16}")
    ap.add_argument("--floq-noise-samples", type=int, default=8, help="m, floq/agents/floq.py:475")
    ap.add_argument(
        "--floq-kappa",
        type=float,
        default=0.1,
        help="noise_coverage: the interval is [kappa*Q_min, kappa*Q_max] (floq.py:438-439), which "
        "makes kappa identically the paper's (u-l)/(Q_max-Q_min). Default 0.1 (floq.py:476), swept "
        "over {0.1, 0.25} by their README. On our support that is l = -277.78, u = 0. Do not jump to "
        "1.0 for target coverage: floq ablated kappa over {0.01..1.0} and found an interior optimum, "
        "the stated reason being flow CURVATURE rather than overlap.",
    )
    ap.add_argument("--floq-bins", type=int, default=51, help="floq/agents/floq.py:484; 51 EDGES -> a 50-wide encoding")
    ap.add_argument("--floq-embed-sigma", type=float, default=16.0, help="in BIN WIDTHS, floq/agents/floq.py:485")
    ap.add_argument(
        "--floq-reward-offset",
        type=float,
        default=0.01,
        help="pads the z-encoding support only (floq.py:369-370,486), NOT the noise interval",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds parameter init, batch sampling and augmentation together. Seed replicates must "
        "vary ONLY this -- if the recipe moves too, the seed term absorbs the recipe term and the "
        "run-level CI stops meaning what it says.",
    )
    ap.add_argument(
        "--heldout-frac",
        type=float,
        default=0.0,
        help="fraction of EPISODES held out of training and scored periodically. 0 = off, which is what "
        "every critic in .scratch was trained with -- there is no validation split anywhere in this "
        "repo's critic trainers, and at 200k x 256 = 51.2M draws over 937,993 frames that is 55 visits "
        "per frame against ~10 parameters per training frame. EPISODES, not frames: cost_to_goal is "
        "deterministic in (episode, frame), so a frame-level split leaks the answer from the neighbours "
        "on either side (diag_action_identifiability.py:30-33 documents the same point). The split is "
        "stratified by outcome so both sides carry failures, and the held-out episode list is written "
        "into config.json so the diagnostics and scorers can consume the same one.",
    )
    ap.add_argument("--heldout-seed", type=int, default=12345, help="fixed so a re-run holds out the same episodes")
    ap.add_argument(
        "--eval-every",
        type=int,
        default=5000,
        help="steps between held-out evaluations. Needs --heldout-frac > 0. What is scored is the thing "
        "the serving path consumes and the thing the audit found does not transfer: Spearman(Q(demo "
        "chunk), -time_to_goal) globally AND within-episode (the within-episode number is the honest "
        "one -- across episodes the target itself disagrees at CV 0.188), plus success-vs-failure AUC "
        "on early frames, where in-sample was 0.87-0.89 and episode-grouped CV on the same features "
        "gave 0.490.",
    )
    ap.add_argument(
        "--eval-frames", type=int, default=4096, help="fixed anchor set drawn once from the held-out episodes"
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
    ap.add_argument(
        "--deas-faithful",
        action="store_true",
        help="preset: everything the DEAS audit found we differ on, at once. Sets --backup scalar, "
        "--terminal deas, --support smdp, --hlg-sigma-frac 1.06066 (their sqrt(2)-wider kernel), "
        "--zero-init-head, --no-mc-floor, --alpha-cql 0, --edac-weight 0, --feat-dropout 0, "
        "--feat-noise 0. It does NOT set the discounts, expectile or batch: those are values DEAS "
        "tunes per benchmark, not structural choices, so they stay explicit.",
    )
    a = ap.parse_args()
    if a.deas_faithful:
        a.backup, a.terminal, a.support = "scalar", "deas", "smdp"
        a.hlg_sigma_frac, a.zero_init_head, a.mc_floor = 1.0606601717798212, True, False
        a.alpha_cql, a.edac_weight, a.feat_dropout, a.feat_noise = 0.0, 0.0, 0.0, 0.0
        print(
            "--deas-faithful: scalar backup, deas terminal, smdp support, sqrt(2) kernel, "
            "zero-init head; mc-floor/CQL/EDAC/augmentation OFF",
            flush=True,
        )

    g1 = a.discount
    g2 = a.discount2 if a.discount2 is not None else a.discount

    def smdp_return_range(r_min, r_max, ln, horizon, gamma1, gamma2):
        """Transcribed from DEAS-FQL utils/smdp.py:18-34 -- the reachable return range of an SMDP
        whose options last `ln` primitive steps, discounting inside an option by gamma1 and between
        options by gamma2^ln."""
        m, r = horizon // ln, horizon % ln
        s_l = (1 - gamma1**ln) / (1 - gamma1)
        s_r = (1 - gamma1**r) / (1 - gamma1) if r > 0 else 0.0
        weight = (1 - gamma2 ** (m * ln)) / (1 - gamma2**ln)
        total = s_l * weight + ((gamma2 ** (m * ln)) * s_r if r > 0 else 0.0)
        return r_min * total, r_max * total

    if a.support == "smdp":
        v_min, _v_max_smdp = smdp_return_range(-1.0, 0.0, a.macro_group_size, a.smdp_h, g1, g2)
        print(f"smdp support: v_min={v_min:.2f} (L={a.macro_group_size}, H={a.smdp_h}, g1={g1}, g2={g2})", flush=True)
    else:
        v_min = a.v_min if a.v_min is not None else -1.0 / (1.0 - g1)
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
    spec["discount2"] = g2
    spec["hlg_sigma_frac"] = a.hlg_sigma_frac
    spec["terminal"] = a.terminal
    spec["support"] = a.support
    spec["zero_init_head"] = a.zero_init_head
    spec["seed"] = a.seed
    spec["value_head"] = a.value_head
    if a.value_head == "floq":
        spec["floq"] = {
            "steps": a.floq_steps,
            "noise_samples": a.floq_noise_samples,
            "kappa": a.floq_kappa,
            "bins": a.floq_bins,
            "embed_sigma": a.floq_embed_sigma,
        }
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

    # Verdicts come from the dataset schema (next.success / next.done in meta/episodes), with the
    # legacy sidecar accepted only for pre-migration copies. The reader keeps "unknown" distinct from
    # "fail" -- an episode kept without being judged must not be counted as a failure.
    outc = _outcomes.cache_outcomes(meta, legacy_jsonl=a.outcomes)
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets is not None else None

    # ---- episode-level held-out split ---------------------------------------------------------
    _eps_all = sorted(int(k) for k in meta["episodes"] if int(k) in outc)
    heldout: set[int] = set()
    if a.heldout_frac > 0:
        _r = np.random.default_rng(a.heldout_seed)
        for want in (True, False):  # stratify, so the held-out side carries failures too
            grp = np.array([e for e in _eps_all if (outc[e] == "success") == want])
            k = round(len(grp) * a.heldout_frac)
            if k:
                heldout.update(int(x) for x in _r.choice(grp, k, replace=False))
        print(
            f"held out {len(heldout)}/{len(_eps_all)} episodes "
            f"({sum(outc[e] == 'success' for e in heldout)} success / "
            f"{sum(outc[e] != 'success' for e in heldout)} fail), seed {a.heldout_seed}",
            flush=True,
        )

    # Written into the checkpoint so the scorers and diagnostics can consume the SAME split rather
    # than re-deriving one and quietly evaluating on frames the critic trained on.
    spec["heldout_episodes"] = sorted(heldout) if heldout else None
    spec["heldout_seed"] = a.heldout_seed if heldout else None

    # Build the flat table of valid CURRENT frames (homing tail dropped for failures at train time).
    cur_g0, cur_pos, cur_eff, cur_full, cur_succ = [], [], [], [], []
    ho_g0, ho_pos, ho_eff, ho_succ = [], [], [], []
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
        if e in heldout:
            ho_g0.append(np.full(eff, off))
            ho_pos.append(np.arange(eff))
            ho_eff.append(np.full(eff, eff))
            ho_succ.append(np.full(eff, succ))
            continue
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
        "zero_init_head": a.zero_init_head,
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

    # ---- AC-State multi-step inverse auxiliary -------------------------------------------------
    inv_mod = inv_trunk = None
    if a.inv_weight > 0.0:
        if a.critic_arch != "shared":
            raise SystemExit(
                "--inv-weight needs --critic-arch shared: it trains the shared TRUNK, and "
                "PatchARQCritic has no action-independent half to attach to"
            )
        import flax.linen as _nn

        from openpi.patch_critic.critic import PatchTrunk as _PatchTrunk

        # The same module class the ensemble builds internally, applied to the ensemble's own trunk
        # subtree. Reusing the weights rather than the module object keeps critic.py untouched and
        # keeps the parameter tree byte-identical to a run without this flag.
        inv_trunk = _PatchTrunk(num_layers=a.trunk_layers, num_heads=8, head_dim=48, mlp_dim=1024)

        class _InvHead(_nn.Module):
            """(z_t, z_{t+k}) -> the action taken AT t.

            invnet.py concatenates the two latents and runs an MLP to action logits; `k` is not an
            input, so the head cannot condition on how far apart the pair is and must read the
            displacement out of the representation itself. Ours regresses a continuous 14-dim joint
            delta instead of classifying one of n_actions, which is the only change the action space
            forces.
            """

            bottleneck: int
            out_dim: int

            @_nn.compact
            def __call__(self, z0, z1):
                proj = _nn.Dense(self.bottleneck)
                h = jnp.concatenate([proj(z0), proj(z1)], axis=-1)  # shared projection: the bottleneck
                h = _nn.gelu(_nn.Dense(256)(_nn.LayerNorm()(h)))
                h = _nn.gelu(_nn.Dense(256)(h))
                return _nn.Dense(self.out_dim)(h)

        inv_mod = _InvHead(bottleneck=a.inv_bottleneck, out_dim=ad)

    # ---- floq: a trunk + a velocity field replace the categorical read-out ----------------------
    floq_net = floq_trunk = None
    if a.value_head == "floq":
        from openpi.patch_critic.critic import PatchFloqVelocity
        from openpi.patch_critic.critic import PatchTrunk

        # z-encoding support is PADDED past the value range on purpose (floq/agents/floq.py:369-370):
        #   q_min = (r_min - offset)/(1-g),  q_max = (r_max + offset)/(1-g)
        _fq_min = (-1.0 - a.floq_reward_offset) / (1.0 - g1)
        _fq_max = (0.0 + a.floq_reward_offset) / (1.0 - g1)
        # the NOISE interval is a different range, scaled by kappa at both ends (floq.py:438-439)
        noise_lo, noise_hi = a.floq_kappa * (-1.0 / (1.0 - g1)), a.floq_kappa * 0.0
        print(
            f"floq: K={a.floq_steps} m={a.floq_noise_samples} kappa={a.floq_kappa} "
            f"noise=[{noise_lo:.2f},{noise_hi:.2f}] zbins={a.floq_bins} zsupport=[{_fq_min:.1f},{_fq_max:.1f}]",
            flush=True,
        )
        floq_trunk = PatchTrunk(num_layers=a.trunk_layers)
        floq_net = PatchFloqVelocity(
            action_dim=ad,
            horizon=H,
            macro_group_size=a.macro_group_size,
            num_layers=a.head_layers,
            num_bins=a.floq_bins,
            sigma=a.floq_embed_sigma,
            q_min=_fq_min,
            q_max=_fq_max,
        )
    hl = HLGauss(v_min, a.v_max, a.num_atoms, sigma_frac=a.hlg_sigma_frac)
    centers = jnp.asarray(hl.centers)
    prefixes = list(range(a.macro_group_size, H + 1, a.macro_group_size))
    P_ = len(prefixes)

    rng = jax.random.key(a.seed)
    p2 = jnp.zeros((2, npatch, emb), jnp.float32)
    params = net.init(rng, p2, jnp.zeros((2, H, ad)), jnp.zeros((2, sd)))
    if inv_mod is not None:
        # Carried inside `params` under a key the ensemble never looks up: flax.apply resolves only
        # the paths it needs, so an extra subtree is inert for `net.apply` while `tx.init(params)`
        # picks it up and the existing optimizer trains it. That avoids threading a third parameter
        # tree and optimizer through the carry, the jit signature and both loss paths.
        _z = inv_trunk.apply({"params": params["params"]["PatchTrunk_0"]}, p2, jnp.zeros((2, sd)))
        _zp = _z.mean(axis=-2)
        params["params"]["inv_head"] = inv_mod.init(rng, _zp, _zp)["params"]
        print(
            f"AC-State inverse aux: lambda={a.inv_weight} max_k={a.inv_max_k} "
            f"bottleneck={a.inv_bottleneck} zero_proprio={a.inv_zero_proprio}",
            flush=True,
        )
    if a.value_head == "floq":
        _mh = H // a.macro_group_size
        _tp = floq_trunk.init(rng, p2, jnp.zeros((2, sd)))
        _zs = floq_trunk.apply(_tp, p2, jnp.zeros((2, sd)))
        _vp = floq_net.init(rng, _zs, jnp.zeros((2, H, ad)), jnp.zeros((2, _mh)), jnp.zeros(()))
        params = {"trunk": _tp, "vel": _vp}

        def flow_q(prm, feats, chunk, prop, *, steps):
            """Q = psi(1): K Euler steps from uniform noise. floq/agents/floq.py:244-249.

            The trunk is hoisted out of the loop. That is exact, not an approximation: the trunk sees
            neither the interpolant nor the flow time, so K velocity evaluations share one encoding.
            """
            zs = floq_trunk.apply(prm["trunk"], feats, prop)
            zf = jnp.broadcast_to(jnp.asarray(0.5 * (noise_lo + noise_hi)), (feats.shape[0], _mh))
            for i in range(steps):
                zf = zf + floq_net.apply(prm["vel"], zs, chunk, zf, jnp.asarray(i / steps, jnp.float32)) / steps
            return zf

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
        if a.cql_negatives in ("bank", "within"):
            return bank_negs  # built host-side by bank_negs_of
        out = []
        if a.cql_negatives in ("shuffle", "both"):
            out.extend(chunk[jax.random.permutation(k, chunk.shape[0])][:Bc] for k in jax.random.split(key, a.cql_n))
        if a.cql_negatives in ("uniform", "both"):
            key = jax.random.fold_in(key, 1)
            out.extend(
                jax.random.uniform(k, (Bc, H, ad), minval=-1.0, maxval=1.0) for k in jax.random.split(key, a.cql_n)
            )
        return jnp.stack(out, 1)

    def loss_fn(
        params, v_params, tgt_p, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc, negs, pinv, sinv
    ):
        vlog = v_net.apply(
            jax.lax.stop_gradient(v_params), pnxt.reshape(-1, npatch, emb).astype(jnp.float32), snxt.reshape(-1, sd)
        ).reshape(-1, P_, a.num_atoms)
        gam = g2 ** jnp.asarray(prefixes, jnp.float32)  # gamma2: the inter-option bootstrap exponent
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
            if a.terminal == "deas":
                # agents/deas.py:157-162 masks only the BOOTSTRAP; the reward accumulated over the
                # chunk is kept, and the terminal reward is added at the option boundary.
                y = (
                    cum
                    + jnp.where(done_nxt > 0, gam[None, :] * reward_nxt, 0.0)
                    + gam[None, :] * (1.0 - done_nxt) * next_v
                )
            else:
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
        cql_clamped = jnp.asarray(0.0)
        cql_exceeds = jnp.asarray(0.0)
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
            # How much of the term is LIVE, reported every step because a silently dead
            # conservative term looks exactly like a conservative term that did not help.
            # `clamped` is the fraction of negatives Cal-QL floors at the MC return: those
            # contribute a stop_gradient'd constant, so at clamped -> 1 this whole loss has no
            # gradient and the run is a no-op with a plausible-looking loss curve. `exceeds` is the
            # fraction scoring above the DEMONSTRATED chunk -- what there is to push down at all.
            cql_clamped = jnp.mean(qn < jax.lax.stop_gradient(mc[:Bc])[None, :, None])
            if a.calql and a.cql_negatives != "uniform":
                # Cal-QL Eq. 6: never penalise below the behaviour policy's own achieved return.
                qn = jnp.maximum(qn, jax.lax.stop_gradient(mc[:Bc])[None, :, None])
            else:
                cql_clamped = jnp.asarray(0.0)
            q_data = from_logits(pred[:, :Bc, -1, :])  # [K, Bc]
            cql_exceeds = jnp.mean(qn > q_data[..., None])
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

        # ---- AC-State multi-step inverse auxiliary ---------------------------------------------
        inv_loss = jnp.asarray(0.0)
        inv_r2 = jnp.asarray(0.0)
        if a.inv_weight > 0.0:
            tp = {"params": params["params"]["PatchTrunk_0"]}
            sc = jnp.zeros_like(scur) if a.inv_zero_proprio else scur
            si = jnp.zeros_like(sinv) if a.inv_zero_proprio else sinv
            z0 = inv_trunk.apply(tp, pcur.astype(jnp.float32), sc).mean(axis=-2)  # [B, d]
            z1 = inv_trunk.apply(tp, pinv.astype(jnp.float32), si).mean(axis=-2)
            a_hat = inv_mod.apply({"params": params["params"]["inv_head"]}, z0, z1)
            # the action taken AT t, which for a chunk policy is the chunk's first step
            tgt_a = chunk[:, 0, :]
            inv_loss = jnp.mean(jnp.square(a_hat - tgt_a))
            # MSE alone cannot tell learning from regression-to-the-mean, and the mean action here is
            # a strong predictor (the arms move smoothly). R2 against the batch's own variance is the
            # readable form: <=0 means the head is not beating the constant predictor, and if it never
            # leaves 0 the auxiliary is shaping nothing no matter how good the loss curve looks.
            inv_r2 = 1.0 - inv_loss / (jnp.mean(jnp.var(tgt_a, axis=0)) + 1e-8)

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
        return q_loss + v_loss + cql_loss + edac_loss + a.inv_weight * inv_loss, {
            "q_loss": q_loss,
            "inv_loss": inv_loss,
            "inv_r2": inv_r2,
            "v_loss": v_loss,
            "cql_gap": cql_gap,
            "cql_clamped": cql_clamped,
            "cql_exceeds": cql_exceeds,
            "edac_cos": edac_cos,
            "q_mean": jnp.mean(from_logits(pred)),
            "v_mean": jnp.mean(vbar),
        }

    def floq_loss_fn(params, v_params, tgt_p, key, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc):
        """floq's flow-matching critic loss, transcribed from floq/agents/floq.py:56-97.

        The categorical read-out is GONE: there are no logits and no cross-entropy. HL-Gauss survives
        only inside PatchFloqVelocity, as the encoding of the interpolant.

        V stays a separate network and supplies the bootstrap, which is the IQL substitution for
        floq's `a' ~ pi(s')`; the arithmetic of the target is otherwise floq's
        (floq.py:41-52) -- a SCALAR y, used directly as the Dirac endpoint x_1, never pushed through
        a histogram. floq clips only x_0 to [l,u] (floq.py:66) and never clips x_1, so neither do we.
        """
        m = a.floq_noise_samples
        B = pcur.shape[0]
        # ---- scalar target, exactly the DEAS-shaped backup the categorical mode builds ----------
        vlog = v_net.apply(
            jax.lax.stop_gradient(v_params), pnxt.reshape(-1, npatch, emb).astype(jnp.float32), snxt.reshape(-1, sd)
        ).reshape(-1, P_, a.num_atoms)
        gam = g2 ** jnp.asarray(prefixes, jnp.float32)
        next_v = hl.from_logits(vlog)
        y = cum + gam[None, :] * next_v
        if a.terminal == "deas":
            y = cum + jnp.where(done_nxt > 0, gam[None, :] * reward_nxt, 0.0) + gam[None, :] * (1.0 - done_nxt) * next_v
        else:
            y = jnp.where(done_nxt > 0, reward_nxt, y)
        if a.mc_floor:
            y = jnp.maximum(y, mc[:, None])
        y = jax.lax.stop_gradient(y)  # [B, P_]

        # ---- flow matching over m independent noise draws (floq.py:58-87) -----------------------
        k1, k2 = jax.random.split(key)
        ratios = jax.random.uniform(k1, (B, m, P_))
        x0 = (1.0 - ratios) * noise_lo + ratios * noise_hi  # floq.py:66
        x1 = jnp.broadcast_to(y[:, None, :], (B, m, P_))  # the same scalar for every draw = Dirac
        tt = jax.random.uniform(k2, (B, m))  # floq.py:74-77
        xt = (1.0 - tt[..., None]) * x0 + tt[..., None] * x1  # floq.py:79
        vel = x1 - x0  # floq.py:81
        zs = floq_trunk.apply(params["trunk"], pcur.astype(jnp.float32), scur)
        # fold the m draws into the batch; the trunk encoding is shared across them
        zs_m = jnp.repeat(zs, m, axis=0)
        ch_m = jnp.repeat(chunk, m, axis=0)
        pred = floq_net.apply(params["vel"], zs_m, ch_m, xt.reshape(B * m, P_), tt.reshape(B * m))
        per = jnp.sum((pred.reshape(B, m, P_) - vel) ** 2, axis=1)  # floq.py:87 SUMS over draws
        q_loss = jnp.sum(per * valid) / (jnp.sum(valid) + 1e-8)

        # ---- V: the IQL expectile against the TARGET flow's Q, on scalars -----------------------
        qbar = jax.lax.stop_gradient(flow_q(tgt_p, pcur.astype(jnp.float32), chunk, scur, steps=a.floq_steps))[:, -1]
        vbar = hl.from_logits(v_net.apply(v_params, pcur.astype(jnp.float32), scur))
        u = qbar - vbar
        wexp = jnp.abs(a.expectile - (u < 0).astype(jnp.float32))
        v_loss = jnp.mean(wexp * u**2)  # standard IQL squared expectile; Q is a scalar now
        return q_loss + v_loss, {
            "q_loss": q_loss,
            "v_loss": v_loss,
            "cql_gap": jnp.asarray(0.0),
            "edac_cos": jnp.asarray(0.0),
            "q_mean": jnp.mean(qbar),
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
    def step(carry, key, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc, bank_negs, pinv, sinv):
        kc, kn, kq = jax.random.split(key, 3)
        pcur, pnxt = _augment(kc, pcur), _augment(kn, pnxt)
        negs = _negatives(kq, chunk, bank_negs) if a.alpha_cql > 0.0 else bank_negs
        params, tgt, opt, v_params, v_opt = carry
        if a.value_head == "floq":
            (_, info), (gp, gv) = jax.value_and_grad(floq_loss_fn, argnums=(0, 1), has_aux=True)(
                params, v_params, tgt, kq, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc
            )
            up, opt = tx.update(gp, opt, params)
            params = optax.apply_updates(params, up)
            uv, v_opt = tx_v.update(gv, v_opt, v_params)
            v_params = optax.apply_updates(v_params, uv)
            tgt = optax.incremental_update(params, tgt, a.target_tau)
            return (params, tgt, opt, v_params, v_opt), info
        (_, info), (gp, gv) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            params, v_params, tgt, pcur, pnxt, chunk, scur, snxt, cum, reward_nxt, done_nxt, valid, mc, negs, pinv, sinv
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
    rng_np = np.random.default_rng(a.seed)
    aug_key = jax.random.key(a.seed + 1)
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

        def bank_negs_of(gcur, g0=None, eff=None, s_raw=None):
            return jnp.asarray(np.asarray(bank_chunks[pick[gcur[: a.cql_batch]]][:, :n_use], np.float32))
    elif a.alpha_cql > 0.0 and a.cql_negatives == "within":
        Bc_ = min(a.cql_batch, a.batch)

        def bank_negs_of(gcur, g0=None, eff=None, s_raw=None):
            """`cql_n` chunks executed elsewhere in the SAME episode, presented as candidates HERE.

            Normalized against the CURRENT state, not their own base frame: a candidate at this state
            is a set of absolute joint targets expressed as a delta from where the arms are now, which
            is the space the policy's own proposal lives in. Normalizing against the negative's own
            origin would hand the critic a chunk whose delta is measured from somewhere else --
            rejectable as a coordinate artifact rather than as a bad action.
            """
            e = np.maximum(eff[:Bc_], 1)
            pos2 = rng_np.integers(0, e[:, None], size=(Bc_, a.cql_n))
            g = g0[:Bc_, None, None] + np.clip(pos2[..., None] + ar_h[None, None, :], 0, e[:, None, None] - 1)
            raw = np.asarray(actions[g.reshape(-1)]).reshape(Bc_ * a.cql_n, H, ad)
            if pre is not None:
                raw = pre.actions(raw, np.repeat(s_raw[:Bc_], a.cql_n, 0))
            return jnp.asarray(raw.reshape(Bc_, a.cql_n, H, ad), jnp.float32)
    else:
        _dummy = jnp.zeros((min(a.cql_batch, a.batch), 1, H, ad), jnp.float32)

        def bank_negs_of(gcur, g0=None, eff=None, s_raw=None):
            return _dummy

    # Inert placeholders when the auxiliary is off: shapes must still be static for the jit, and
    # `a.inv_weight > 0.0` is a closure constant so the branch that reads them is traced out.
    _inv_dummy_p = jnp.zeros((a.batch, npatch, emb), jnp.float32)
    _inv_dummy_s = jnp.zeros((a.batch, sd), jnp.float32)

    # ---- held-out evaluation ------------------------------------------------------------------
    # A fixed anchor set drawn ONCE, so the curve across steps is a curve about the critic and not
    # about which frames were sampled. Anchors come only from held-out episodes.
    evalset = None
    if heldout and a.eval_every:
        ho_g0_a = np.concatenate(ho_g0).astype(np.int64)
        ho_pos_a = np.concatenate(ho_pos).astype(np.int64)
        ho_eff_a = np.concatenate(ho_eff).astype(np.int64)
        ho_succ_a = np.concatenate(ho_succ).astype(bool)
        _r = np.random.default_rng(a.heldout_seed + 1)
        sel = _r.choice(len(ho_pos_a), min(a.eval_frames, len(ho_pos_a)), replace=False)
        e_g0, e_pos, e_eff, e_succ = ho_g0_a[sel], ho_pos_a[sel], ho_eff_a[sel], ho_succ_a[sel]
        e_gcur = e_g0 + e_pos
        e_gch = e_g0[:, None] + np.clip(e_pos[:, None] + ar_h[None], 0, e_eff[:, None] - 1)
        e_chunk = np.asarray(actions[e_gch.reshape(-1)]).reshape(len(sel), H, ad)
        e_state = np.asarray(states[e_gcur])
        if pre is not None:
            e_chunk = pre.actions(e_chunk, e_state)
            e_state = pre.state(e_state)
        if pidx is not None:
            e_state = e_state[..., pidx]
        e_feat = np.asarray(feats[e_gcur])
        # -time_to_goal in the TRUNCATED frame: the quantity cost_to_goal is a monotone function of.
        e_ttg = (e_eff - e_pos).astype(np.float64)
        e_epi = e_g0  # episode identity, for the within-episode correlation
        evalset = (e_feat, e_chunk, e_state, e_ttg, e_succ, e_epi, e_pos)
        print(f"held-out anchors: {len(sel)} frames from {len(np.unique(e_g0))} episodes", flush=True)

    @jax.jit
    def _eval_q(params, pcur, chunk, scur):
        qd = net.apply(params, pcur.astype(jnp.float32), chunk, scur)[:, :, -1, :]
        return from_logits(qd).min(0) if a.q_reduction == "min" else from_logits(qd).mean(0)

    def _spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and rx.std() and ry.std() else float("nan")

    def _auc(score, label):
        if label.all() or not label.any():
            return float("nan")
        r = np.argsort(np.argsort(score)).astype(np.float64) + 1
        n1 = int(label.sum())
        return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1)))

    def heldout_metrics(params):
        f_, c_, s_, ttg, sc, epi, pos = evalset
        q = np.asarray(_eval_q(params, jnp.asarray(f_), jnp.asarray(c_), jnp.asarray(s_)), np.float64)
        # within-episode is the honest correlation: across episodes the TARGET itself disagrees
        # (CV 0.188 in remaining time at matched task state), so a global number mixes the critic's
        # skill with the pace of whichever episode a frame came from.
        wi = [_spearman(q[epi == e], -ttg[epi == e]) for e in np.unique(epi) if (epi == e).sum() >= 8]
        early = pos < 60
        return {
            "ho_spearman": _spearman(q, -ttg),
            "ho_spearman_within_ep": float(np.nanmean(wi)) if wi else float("nan"),
            "ho_auc_early": _auc(q[early], sc[early]),
            "ho_q_mean": float(q.mean()),
        }

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
            pos,
            eff,
            succ,
            pos,
            pad_row,
            list(prefixes),
            g2,
            a.h_goal,
            v_min,
            a.reward_scheme,
            failure_reward,
            discount1=g1,
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
        # Built BEFORE pre.state() rewrites s_cur_raw below: `within` needs the raw state to express
        # another timestep's absolute joint targets as a delta from here.
        host_negs = bank_negs_of(gcur, g0, eff, s_cur_raw)
        if a.inv_weight > 0.0:
            # k log-uniform in [1, inv_max_k], clamped to what is left of the TRUNCATED episode --
            # past eff lie the homing frames, and a pair straddling them would be asking which action
            # was taken at t given a frame from the return-to-home motion.
            room = np.maximum(eff - pos - 1, 1)
            kk = np.minimum(np.exp(rng_np.uniform(0.0, np.log(a.inv_max_k + 1.0), size=a.batch)).astype(np.int64), room)
            ginv = gcur + np.maximum(kk, 1)
            pinv = jnp.asarray(np.asarray(feats[ginv]))
            s_inv_raw = np.asarray(states[ginv])
            if pre is not None:
                s_inv_raw = pre.state(s_inv_raw)
            if pidx is not None:
                s_inv_raw = s_inv_raw[..., pidx]
            sinv = jnp.asarray(s_inv_raw)
        else:
            pinv, sinv = _inv_dummy_p, _inv_dummy_s
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
            host_negs,
            pinv,
            sinv,
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
            invs = f"  inv {i['inv_loss']:.4f} r2 {i['inv_r2']:+.3f}" if a.inv_weight > 0.0 else ""
            cqls = (
                f"  cql_gap {i['cql_gap']:.2f} clamped {i['cql_clamped']:.2f} exceeds {i['cql_exceeds']:.2f}"
                if a.alpha_cql > 0.0
                else ""
            )
            cqls += f"  edac_cos {i['edac_cos']:.3f}" if a.edac_weight > 0.0 else ""
            print(
                f"step {s:6d}  q_loss {i['q_loss']:.4f}  v_loss {i['v_loss']:.4f}  q_mean {i['q_mean']:.2f}  "
                f"v_mean {i['v_mean']:.2f}{invs}{cqls}  ({rate:.2f} it/s)",
                flush=True,
            )
        if evalset is not None and (s % a.eval_every == 0 or s == a.steps - 1):
            hm = heldout_metrics(carry[0])
            print(
                f"  [held-out @ {s}] spearman {hm['ho_spearman']:+.3f} "
                f"(within-ep {hm['ho_spearman_within_ep']:+.3f})  early-AUC {hm['ho_auc_early']:.3f}  "
                f"q_mean {hm['ho_q_mean']:.1f}",
                flush=True,
            )
            if wb is not None:
                wb.log(hm, step=s)
        if a.save_every and (s + 1) % a.save_every == 0:
            a._step = s + 1
            _save(a, carry[0], carry[3], npatch, v_min, prefixes, ad, spec=spec, stats=stats, embedded=embedded)
    a._step = a.steps
    _save(a, carry[0], carry[3], npatch, v_min, prefixes, ad, spec=spec, stats=stats, embedded=embedded)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
