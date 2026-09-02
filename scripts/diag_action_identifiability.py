"""Does knowing the ACTION predict the return beyond knowing the STATE? Model-free, episode-held-out.

Why this and not a neighbour analysis. The natural way to ask "is state->action one-to-one" is to
find repeated states across episodes and look at the spread of actions there. We tried that and the
premise did not survive verification: the cross-episode pairs that pass a tight visual+proprio gate
are ~89% episode-BOUNDARY frames -- the robot parked at its home pose before or after the task --
against a 3% base rate, and removing them collapses the strict tier from 783 pairs to 6. Whatever
"same state" means at that gate, it is mostly "not doing the task".

This measurement avoids the premise entirely. It never asks whether two frames are the same state.
It asks a regression question:

    R2_state         predict time-to-goal from the state representation
    R2_state_action  predict it from the state AND the action chunk

If the action adds nothing on held-out EPISODES, no critic can rank actions on this data, because
there is no action-conditioned return information to learn. If it adds something, that increment is
a model-free LOWER BOUND on what a Q(s,a) could extract beyond V(s) -- the probe here is a ridge, so
a flexible critic could only do better, never worse. (An earlier draft of this file called it an
upper bound. That was wrong: a linear probe bounds the achievable signal from below.)

TWO CONTROLS, because the obvious objection is that the action leaks the demonstrator's PACE rather
than the quality of the action. A slow demonstrator has both a longer time-to-goal and a recognisable
action style, so a model could score the gap without knowing anything about action quality.
  pace-corrected target   time-to-goal rescaled by median_eff / eff of its own episode, which
                          removes the "this episode is simply slow" component of the target.
  style control           the episode's MEAN action chunk is added to the STATE side, so the
                          increment being tested is the action's deviation from its own episode's
                          style rather than the style itself.

Held out by EPISODE, never by frame. Frames within an episode are ~30 Hz samples of a continuous
trajectory, so a frame-level split lets the model memorise a trajectory it has already seen and
report a gap that is pure leakage. Both splits are reported so the size of that leakage is visible.

Homing frames are excluded throughout: the return-to-home motion after the task is where the
boundary artifact above lives, and it is not task progress.
"""

# ruff: noqa: PLC0415

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k",
        help="only for its pi05 action normalisation, so the action space matches the critic's",
    )
    ap.add_argument("--homing-onsets", type=pathlib.Path, default=R / ".scratch/yam_homing_onsets.json")
    ap.add_argument("--frames", type=int, default=60000, help="frames sampled across all episodes")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/diag_action_identifiability.json")
    a = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from openpi.extraction import critic_q as cq

    critic = cq.load(a.critic)
    meta = json.loads((a.cache / "meta.json").read_text())
    N, H, AD, SD = meta["N"], a.horizon, meta["ad"], meta["sd"]
    feats = np.load(a.cache / "features_pooled_f32.npy", mmap_mode="r")
    states = np.memmap(a.cache / "state.dat", np.float32, "r", shape=(N, SD))
    actions = np.memmap(a.cache / "action.dat", np.float32, "r", shape=(N, AD))
    homing = json.loads(a.homing_onsets.read_text()) if a.homing_onsets.exists() else {}

    # ---- eligible frames: successful episodes, task portion only, room for a full chunk -----------
    rows, ttg, epi = [], [], []
    for k, e in meta["episodes"].items():
        if not e["success"]:
            continue  # time-to-goal is undefined without a goal
        off, ln = e["offset"], e["full_len"]
        _h = homing.get(str(k)) or {}
        eff = int(_h.get("homing_onset", ln))  # task portion ends at the homing onset
        eff = min(eff, ln)
        if eff <= H + 2:
            continue
        t = np.arange(0, eff - H - 1)
        rows.append(off + t)
        ttg.append((eff - 1 - t).astype(np.float32))  # steps remaining to the goal
        epi.append(np.full(len(t), int(k)))
    rows, ttg, epi = np.concatenate(rows), np.concatenate(ttg), np.concatenate(epi)
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(rows), size=min(a.frames, len(rows)), replace=False)
    rows, ttg, epi = rows[sel], ttg[sel], epi[sel]
    order = np.argsort(rows)
    rows, ttg, epi = rows[order], ttg[order], epi[order]
    print(f"{len(rows)} frames from {len(np.unique(epi))} successful episodes (homing excluded)", flush=True)

    # ---- features ---------------------------------------------------------------------------------
    ar = np.arange(H)
    raw_state = np.asarray(states[rows])
    ends = {}
    for k, e in meta["episodes"].items():
        ends[int(k)] = e["offset"] + e["full_len"] - 1
    end_of = np.array([ends[c] for c in epi])
    gch = np.clip(rows[:, None] + ar[None], 0, end_of[:, None])
    chunk = np.asarray(actions[gch.reshape(-1)]).reshape(len(rows), H, AD)
    A = np.asarray(critic.pre.actions(chunk, raw_state))[..., :AD].reshape(len(rows), -1)  # the critic's space
    Sdino = np.asarray(feats[rows], np.float32)
    Sprop = np.asarray(critic_proprio(critic, raw_state))
    print(f"state: dino {Sdino.shape[1]} + proprio {Sprop.shape[1]}  | action {A.shape[1]}", flush=True)

    def r2(feat, y, groups, *, by_episode):
        """Cross-validated R2. by_episode=True holds out whole episodes; False holds out frames."""
        if by_episode:
            uniq = np.unique(groups)
            rng2 = np.random.default_rng(a.seed)
            rng2.shuffle(uniq)
            fold_of = {e: i % a.folds for i, e in enumerate(uniq)}
            f = np.array([fold_of[g] for g in groups])
        else:
            f = np.random.default_rng(a.seed).integers(0, a.folds, len(y))
        preds = np.empty_like(y)
        for i in range(a.folds):
            tr, te = f != i, f == i
            sc = StandardScaler().fit(feat[tr])
            m = Ridge(alpha=10.0).fit(sc.transform(feat[tr]), y[tr])
            preds[te] = m.predict(sc.transform(feat[te]))
        ss_res = float(((y - preds) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    S = np.concatenate([Sdino, Sprop], 1)
    SA = np.concatenate([S, A], 1)
    # per-episode mean action = that demonstrator's style on this episode
    style = np.zeros_like(A)
    for c in np.unique(epi):
        m = epi == c
        style[m] = A[m].mean(0)
    S_style = np.concatenate([S, style], 1)
    eff_of = {}
    for k, e in meta["episodes"].items():
        _h2 = homing.get(str(k)) or {}
        eff_of[int(k)] = min(int(_h2.get("homing_onset", e["full_len"])), e["full_len"])
    med_eff = float(np.median([eff_of[c] for c in np.unique(epi)]))
    ttg_pace = ttg * np.array([med_eff / eff_of[c] for c in epi], np.float32)
    res = {"n_frames": len(rows), "n_episodes": len(np.unique(epi)), "horizon": H}
    for split, by_ep in (("episode_heldout", True), ("frame_heldout", False)):
        r_s = r2(S, ttg, epi, by_episode=by_ep)
        r_sa = r2(SA, ttg, epi, by_episode=by_ep)
        r_a = r2(A, ttg, epi, by_episode=by_ep)
        res[split] = {"state": r_s, "state_action": r_sa, "action_only": r_a, "gap": r_sa - r_s}
        print(
            f"\n[{split}]  R2(state) {r_s:+.4f}   R2(state+action) {r_sa:+.4f}   GAP {r_sa - r_s:+.4f}   R2(action only) {r_a:+.4f}"
        )

    # ---- THE NULL: is the gap capacity rather than signal? ---------------------------------------
    # Adding 420 action columns to a ridge at a FIXED alpha is not a free comparison -- the larger
    # design spreads the same regularisation over more directions, so part of a gap can be capacity.
    # Cross-validation bounds that but does not zero it. Permuting the action block across frames
    # destroys the state-action correspondence while preserving the block's marginal distribution and
    # its collinearity structure exactly, so whatever the gap survives at is capacity alone.
    # Two versions; the within-episode one is stricter because it also preserves that demonstrator's
    # pace and style. Control contributed by the ACRFT-WS session, which ran it first.
    rngp = np.random.default_rng(a.seed + 1)
    A_glob = A[rngp.permutation(len(A))]
    A_wep = A.copy()
    for c in np.unique(epi):
        m = np.flatnonzero(epi == c)
        A_wep[m] = A[m[rngp.permutation(len(m))]]
    null = {}
    for lab, Ap in (("global", A_glob), ("within_episode", A_wep)):
        r_n = r2(np.concatenate([S, Ap], 1), ttg, epi, by_episode=True)
        null[lab] = {"state_action_permuted": r_n, "gap": r_n - r2(S, ttg, epi, by_episode=True)}
        print(f"  null ({lab:14s}) : R2 {r_n:+.4f}  GAP {null[lab]['gap']:+.4f}")
    res["permutation_null"] = null

    # ---- controls: is the gap just the demonstrator's pace or style leaking through? -------------
    print("\n--- permutation null and controls, episode-held-out ---")
    ctl = {}
    r_sp, r_sap = r2(S, ttg_pace, epi, by_episode=True), r2(SA, ttg_pace, epi, by_episode=True)
    ctl["pace_corrected"] = {"state": r_sp, "state_action": r_sap, "gap": r_sap - r_sp}
    print(f"  pace-corrected target : R2(state) {r_sp:+.4f}  R2(state+action) {r_sap:+.4f}  GAP {r_sap - r_sp:+.4f}")
    SsA = np.concatenate([S_style, A], 1)
    r_ss, r_ssa = r2(S_style, ttg, epi, by_episode=True), r2(SsA, ttg, epi, by_episode=True)
    ctl["style_controlled"] = {"state_plus_style": r_ss, "plus_action": r_ssa, "gap": r_ssa - r_ss}
    print(f"  episode-style control : R2(state+style) {r_ss:+.4f}  R2(+action) {r_ssa:+.4f}  GAP {r_ssa - r_ss:+.4f}")
    r_bp, r_bap = r2(S_style, ttg_pace, epi, by_episode=True), r2(SsA, ttg_pace, epi, by_episode=True)
    ctl["both"] = {"state_plus_style": r_bp, "plus_action": r_bap, "gap": r_bap - r_bp}
    print(f"  both controls         : R2(state+style) {r_bp:+.4f}  R2(+action) {r_bap:+.4f}  GAP {r_bap - r_bp:+.4f}")
    res["controls"] = ctl

    # ---- is the action a deterministic function of the state? -------------------------------------
    ridge_a = []
    uniq = np.unique(epi)
    rng2 = np.random.default_rng(a.seed)
    rng2.shuffle(uniq)
    fold_of = {e: i % a.folds for i, e in enumerate(uniq)}
    f = np.array([fold_of[g] for g in epi])
    pred_A = np.empty_like(A)
    for i in range(a.folds):
        tr, te = f != i, f == i
        sc = StandardScaler().fit(S[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(S[tr]), A[tr])
        pred_A[te] = m.predict(sc.transform(S[te]))
    ss_res = ((A - pred_A) ** 2).sum(0)
    ss_tot = ((A - A.mean(0)) ** 2).sum(0)
    ridge_a = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    res["action_from_state_r2"] = {"mean": float(ridge_a.mean()), "median": float(np.median(ridge_a))}
    print(f"\nR2 of predicting the ACTION CHUNK from the state (episode-held-out): mean {ridge_a.mean():.4f}")
    print("  1.0 would mean the action is fully determined by the state -- nothing left to attribute value to.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")


def critic_proprio(critic, raw_state):
    from openpi.patch_critic import preproc as critic_preproc

    return critic_preproc.critic_proprio(critic.pre, critic.proprio_idx, raw_state)


if __name__ == "__main__":
    main()
