"""Independent replication of the action-identifiability measurement, plus the null it needs.

The question ACRFT-D asked: on YAM demonstrations, does knowing the action chunk predict
time-to-goal any better than knowing the state alone? If it does not, no critic can ever pick an
action here, and the size of the improvement is the ceiling on what any critic could extract.
Their answer was a ridge R2 gap of +0.034 (episode-held-out), measured with pooled DINOv2 features
as the state.

This script exists for two reasons.

1. INDEPENDENCE. It shares no code with theirs. It reads the LeRobot parquet directly -- no patch
   cache, no critic checkpoint, no DINO backbone, hence no GPU -- and re-derives every quantity.
   The state here is proprioception only (observation.state), so the absolute R2 is not comparable
   to theirs; what is comparable is whether a gap of this size survives the control below.

2. THE NULL THEY DID NOT RUN, which is the part that actually decides the question. Adding 420
   action columns to a ridge with a FIXED alpha is not a free comparison: the larger design gets
   the same regularisation spread over more directions, so some of the apparent gain can be
   capacity rather than signal. Cross-validation bounds that but does not zero it. The test is to
   re-run with the action block PERMUTED across frames, which destroys the state-action
   correspondence while preserving the block's marginal distribution and its collinearity
   structure exactly. Two permutations:

     global    a chunk from a random frame of a random episode
     within    a chunk from a random frame of the SAME episode, which additionally preserves
               whatever is constant within an episode (that demonstrator's pace and style)

   A real signal survives both. A capacity artefact shows the same gap under permutation.

Reported per split: R2(state), R2(state+action), the gap, and the gap under each null.
"""

import argparse
import glob
import json
import pathlib

import numpy as np


def _episode_table(ds_dir: pathlib.Path) -> dict[int, str]:
    """Episode verdicts from the dataset's own schema (next.success / next.done)."""
    import openpi.training.outcomes as _outcomes

    outcomes = _outcomes.episode_outcomes(ds_dir)
    if outcomes is None:
        raise SystemExit(
            f"{ds_dir} carries no next.success / next.done features. Migrate it with the recorder's "
            "`workstation/yam-data migrate-outcomes <dir>` (i2rt_rllab)."
        )
    return outcomes


def load(ds_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(state, action, episode_index, frame_index) for the whole dataset, in global frame order."""
    import pandas as pd

    files = sorted(glob.glob(str(ds_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {ds_dir}/data")
    cols = ["observation.state", "action", "episode_index", "frame_index"]
    parts = [pd.read_parquet(f, columns=cols) for f in files]
    frames = pd.concat(parts, ignore_index=True)
    frames = frames.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    state = np.stack(frames["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(frames["action"].to_numpy()).astype(np.float32)
    return state, action, frames["episode_index"].to_numpy(), frames["frame_index"].to_numpy()


def r2_cv(feat: np.ndarray, y: np.ndarray, groups: np.ndarray, *, by_episode: bool, folds: int, seed: int) -> float:
    """Cross-validated R2. ``by_episode`` holds out whole episodes, else individual frames."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if by_episode:
        uniq = np.unique(groups)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        fold_of = {e: i % folds for i, e in enumerate(uniq)}
        f = np.array([fold_of[g] for g in groups])
    else:
        f = np.random.default_rng(seed).integers(0, folds, len(y))
    pred = np.empty_like(y)
    for i in range(folds):
        tr, te = f != i, f == i
        sc = StandardScaler().fit(feat[tr])
        model = Ridge(alpha=10.0).fit(sc.transform(feat[tr]), y[tr])
        pred[te] = model.predict(sc.transform(feat[te]))
    return 1.0 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def permute(blocks: np.ndarray, epi: np.ndarray, *, within: bool, seed: int) -> np.ndarray:
    """The action block, reassigned to the wrong frames. ``within`` keeps it inside its episode."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(blocks))
    if not within:
        rng.shuffle(idx)
        return blocks[idx]
    out = idx.copy()
    for e in np.unique(epi):
        m = np.flatnonzero(epi == e)
        out[m] = rng.permutation(m)
    return blocks[out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("~/lerobot_data/yam_lego_taxi").expanduser())
    ap.add_argument("--frames", type=int, default=60_000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail", type=int, default=0, help="frames to drop at each episode end (a homing stand-in)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    ds_dir = a.dataset.expanduser()
    verdicts = _episode_table(ds_dir)
    success = {e for e, v in verdicts.items() if v == "success"}
    state, action, epi_all, _ = load(ds_dir)
    lengths = {int(e): int((epi_all == e).sum()) for e in np.unique(epi_all)}
    print(f"{ds_dir}: {len(lengths)} episodes, {len(state)} frames; {len(success)} success", flush=True)

    # ---- eligible frames: successful episodes, room for a full chunk, optional tail dropped ------
    H = a.horizon
    rows, ttg, epi = [], [], []
    base = 0
    for e in sorted(lengths):
        n = lengths[e]
        if e in success:
            eff = n - a.tail
            if eff > H + 2:
                t = np.arange(0, eff - H - 1)
                rows.append(base + t)
                ttg.append((eff - 1 - t).astype(np.float32))
                epi.append(np.full(len(t), e))
        base += n
    rows, ttg, epi = np.concatenate(rows), np.concatenate(ttg), np.concatenate(epi)
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(len(rows), size=min(a.frames, len(rows)), replace=False))
    rows, ttg, epi = rows[sel], ttg[sel], epi[sel]
    print(f"{len(rows)} frames from {len(np.unique(epi))} successful episodes (tail={a.tail})", flush=True)

    S = state[rows]
    A = action[(rows[:, None] + np.arange(H)[None]).reshape(-1)].reshape(len(rows), -1)
    print(f"state {S.shape[1]} dims (proprio only)   action {A.shape[1]} dims ({H} x {action.shape[1]})", flush=True)

    res: dict = {
        "dataset": str(ds_dir),
        "n_frames": int(len(rows)),
        "n_episodes": int(len(np.unique(epi))),
        "horizon": H,
        "tail_dropped": a.tail,
        "state_dims": int(S.shape[1]),
        "action_dims": int(A.shape[1]),
    }
    kw = {"folds": a.folds, "seed": a.seed}
    for split, by_ep in (("episode_heldout", True), ("frame_heldout", False)):
        r_s = r2_cv(S, ttg, epi, by_episode=by_ep, **kw)
        r_sa = r2_cv(np.concatenate([S, A], 1), ttg, epi, by_episode=by_ep, **kw)
        res[split] = {"state": r_s, "state_action": r_sa, "gap": r_sa - r_s}
        print(f"\n[{split}]  R2(state) {r_s:+.4f}   R2(state+action) {r_sa:+.4f}   GAP {r_sa - r_s:+.4f}", flush=True)

    # ---- the null: the same 420 columns, attached to the wrong frames ----------------------------
    print("\n--- nulls, episode-held-out (the gap a capacity artefact would still show) ---", flush=True)
    r_s = res["episode_heldout"]["state"]
    nulls = {}
    for name, within in (("global_permutation", False), ("within_episode_permutation", True)):
        Ap = permute(A, epi, within=within, seed=a.seed + 1)
        r_p = r2_cv(np.concatenate([S, Ap], 1), ttg, epi, by_episode=True, **kw)
        nulls[name] = {"state_action": r_p, "gap": r_p - r_s}
        print(f"  {name:28s} R2 {r_p:+.4f}   GAP {r_p - r_s:+.4f}", flush=True)
    res["nulls"] = nulls
    real = res["episode_heldout"]["gap"]
    worst = max(v["gap"] for v in nulls.values())
    res["gap_above_null"] = real - worst
    print(f"\ngap above the strongest null: {real:+.4f} - {worst:+.4f} = {real - worst:+.4f}", flush=True)
    print("  > 0 by a clear margin means the action carries real information about time-to-goal.", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
