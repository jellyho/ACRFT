"""Has the BC policy collapsed its per-state action distribution?

The question this answers. Best-of-N picks among chunks the policy samples at ONE state, so its
whole premise is that those chunks differ. If flow-matching BC has learned a near-deterministic map
-- every noise draw landing on the same chunk -- then N=8 is eight copies of one action and no
critic, however good, can select. The failure would be on the POLICY side, not the critic's.

The comparison that makes the number mean something. At a task state, how much did the HUMANS vary?
That is measurable without any model: take cross-episode frames matched on arm configuration
(the critic's own proprio slice) and remaining time -- neither of which involves the action, so
finding action differences among them is not circular -- and measure the spread of what was actually
demonstrated. Both spreads are reported in the same per-dimension units, from one script, so the
ratio is not assembled from two papers.

Run it on more than one checkpoint step and the over-training hypothesis becomes a curve: if the
policy's spread shrinks from 100k to 200k while the demonstrators' spread is fixed, the collapse is
progressive and the deployed step is a choice, not a given.
"""

# ruff: noqa: PLC0415

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
BC = pathlib.Path("/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps",
        nargs="+",
        default=["100000", "200000"],
        help="BC checkpoint steps, EARLIEST FIRST. 200000 is the step every robot evaluation ran; 100000 is "
        "the earlier point that turns the single number into an over-training curve.",
    )
    ap.add_argument("--ckpt-root", type=pathlib.Path, default=BC)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--critic", type=pathlib.Path, default=R / ".scratch/patch_critic_yam_s347_fixed_tau9_min_200k")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--samples", type=int, default=16, help="chunks drawn per state, from ONE prefix pass")
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--proprio-thr", type=float, default=0.10, help="per-dim distance for 'same arm configuration'")
    ap.add_argument("--ttg-frac", type=float, default=0.10, help="remaining-time tolerance for 'same task phase'")
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/policy_spread.json")
    a = ap.parse_args()

    import jax

    from openpi.extraction import critic_q as cq

    critic = cq.load(a.critic)
    view = cq.CacheView(a.cache)
    meta = view.meta
    n_all, hor, ad = meta["N"], critic.config["horizon"], critic.config["action_dim"]
    items = list(meta["episodes"].items())
    epid = np.empty(n_all, np.int32)
    ep_end = np.empty(n_all, np.int64)
    eff_end = np.empty(n_all, np.int64)
    succ = np.zeros(len(items), bool)
    hom = json.loads((R / ".scratch/yam_homing_onsets.json").read_text())
    for i, (k, e) in enumerate(items):
        lo, hi = e["offset"], e["offset"] + e["full_len"]
        epid[lo:hi], ep_end[lo:hi], succ[i] = i, hi - 1, bool(e["success"])
        h = hom.get(k, hom.get(str(k)))
        eff_end[lo:hi] = lo + (int(h) if isinstance(h, int | float) else e["full_len"]) - 1

    pool = np.arange(0, n_all, 12)
    pool = pool[succ[epid[pool]]]
    pool = pool[pool < eff_end[pool] - hor]
    _, raws, props = view.rows(pool, critic)
    ttg = (eff_end[pool] - pool).astype(float)
    gch = np.clip(pool[:, None] + np.arange(hor)[None], 0, ep_end[pool][:, None])
    raw_chunks = np.asarray(view.actions[gch.reshape(-1)]).reshape(len(pool), hor, ad)

    # ---- the demonstrators' conditional spread, model-free -------------------------------------
    rng = np.random.default_rng(0)
    anc = rng.choice(len(pool), min(400, len(pool)), replace=False)
    dpro = np.linalg.norm(props[anc][:, None, :] - props[None, :, :], axis=2) / np.sqrt(props.shape[1])
    match = (
        (epid[pool][None, :] != epid[pool][anc][:, None])
        & (np.abs(ttg[None, :] - ttg[anc][:, None]) <= a.ttg_frac * ttg[anc][:, None])
        & (dpro < a.proprio_thr)
    )
    human = []
    for r, i in enumerate(anc):
        j = np.flatnonzero(match[r])
        if len(j) < 2:
            continue
        # every partner's motion expressed as a delta from THIS state, which is the space a policy
        # proposal lives in -- otherwise the spread would include where the arms happened to be
        ch = critic.pre.actions(raw_chunks[j], np.repeat(raws[i][None], len(j), 0))[..., :ad]
        human.append(float(np.sqrt(np.mean(ch.var(0)))))
    human = np.array(human)

    res = {
        "human_conditional_spread": {
            "per_dim_std_median": float(np.median(human)),
            "p10": float(np.percentile(human, 10)),
            "p90": float(np.percentile(human, 90)),
            "n_states": len(human),
            "proprio_thr": a.proprio_thr,
            "ttg_frac": a.ttg_frac,
        },
        "policy": {},
    }
    print(
        f"demonstrators at a matched task state: per-dim std "
        f"{res['human_conditional_spread']['per_dim_std_median']:.4f} "
        f"(n={len(human)} states, proprio<{a.proprio_thr}, ttg within {a.ttg_frac:.0%})",
        flush=True,
    )

    # ---- the policy's conditional spread, per checkpoint ----------------------------------------
    import flax.nnx as nnx
    import torch

    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    sel = np.sort(rng.choice(len(pool), a.states, replace=False))
    rows = pool[sel]
    dataset, cfg = exdata.make_bc_dataset(str(a.ckpt_root / a.steps[0] / "assets"))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, rows.tolist()), batch_size=4, shuffle=False, num_workers=4
    )

    def _np(v):
        if isinstance(v, dict):
            return {k: _np(x) for k, x in v.items()}
        arr = np.asarray(v)
        return arr.astype(np.float32) if arr.dtype == np.float64 else arr

    for step in a.steps:
        model = cfg.model.create(jax.random.key(0))
        gd, st = nnx.split(model)
        st.replace_by_pure_dict(
            CheckpointWeightLoaderKeepMissing(str(a.ckpt_root / step / "params")).load(st.to_pure_dict())
        )
        model = nnx.merge(gd, st)

        @nnx.jit(static_argnums=(3,))
        def draw(model, obs, key, k):
            return model.sample_n_actions_batched(key, obs, num_samples=k, num_steps=a.ode_steps)

        key = jax.random.key(0)
        spreads = []
        for b, batch in enumerate(loader):
            obs = _model.Observation.from_dict({k: _np(v) for k, v in batch.items() if k != "actions"})
            key, k1 = jax.random.split(key)
            ch = np.asarray(draw(model, obs, k1, a.samples))[..., :ad]  # [B, k, H, ad]
            spreads.extend(float(np.sqrt(np.mean(c.var(0)))) for c in ch)
            print(f"  [{step}] {len(spreads)}/{len(rows)} states", flush=True)
            if b == 0:
                print(f"    first state: per-dim std {spreads[0]:.4f}", flush=True)
        sp = np.array(spreads)
        res["policy"][step] = {
            "per_dim_std_median": float(np.median(sp)),
            "p10": float(np.percentile(sp, 10)),
            "p90": float(np.percentile(sp, 90)),
            "n_states": len(sp),
            "ratio_human_over_policy": float(res["human_conditional_spread"]["per_dim_std_median"] / np.median(sp)),
        }
        d = res["policy"][step]
        print(
            f"policy @ {step}: per-dim std {d['per_dim_std_median']:.4f} "
            f"-> demonstrators are {d['ratio_human_over_policy']:.1f}x wider",
            flush=True,
        )

    if len(a.steps) > 1:
        first, last = res["policy"][a.steps[0]], res["policy"][a.steps[-1]]
        res["collapse_ratio_first_to_last"] = first["per_dim_std_median"] / last["per_dim_std_median"]
        print(
            f"\nover-training check: {a.steps[0]} -> {a.steps[-1]} the spread changes by "
            f"{res['collapse_ratio_first_to_last']:.2f}x "
            f"({'TIGHTER, consistent with progressive collapse' if res['collapse_ratio_first_to_last'] > 1.1 else 'no meaningful narrowing'})"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
