"""Sample K action chunks per state from the FROZEN BC policy, keyed by patch-cache row.

Why this exists. Every OOD-defence we are considering needs the one thing our critic has never
been shown: the action distribution the POLICY actually emits. Our patch critic was trained by
IQL expectile on 347 human demonstrations, and IQL by construction never queries an action
outside the dataset (train_patch_critic_cached.py:276 weights only the demonstrated chunk), so
the Q-value of a sampled chunk is unconstrained at serving time. This pass materialises that
distribution once, offline, so it can be used as
  - CQL/Cal-QL negatives            (train_patch_critic_cached.py --cql-negatives bank)
  - the candidate set for the gradient diagnostic (scripts/diag_critic_gradient.py)
without re-running a 3.35B VLA inside either.

Index contract. src/openpi/extraction/data.py:3-7 -- torch-dataset index == LeRobot global frame
index == patch-cache row index (cache N == dataset total_frames == 937,993). The saved `idx` array
is therefore directly usable to address /data1/jellyho/pc_cache/yam_s347 rows.

Action space. `sample_actions` returns MODEL-space (pi05-normalized) chunks, which per the critic's
input_spec ("pi05-normalized JOINT DELTA -- identical to the sampler's raw output",
src/openpi/extraction/critic_q.py:6-8) is exactly the critic's input space. No conversion; we keep
the first `robot_ad` dims, matching scripts/train_flowdpg.py's `a_hat_sg[..., :robot_ad]`.

Cost. One prefix pass per STATE, K suffix Euler chains sharing its KV cache -- the batched form of
`Pi0.sample_n_actions` (src/openpi/models/pi0.py:300-345), whose docstring gives the rationale:
sampling by calling `sample_actions` K times would re-run the VLM prefix (images, prompt, state) K
times over one unchanged frame. Here the extra candidates cost only their own suffix forwards.
"""

# ruff: noqa: PLC0415  (heavy imports after argparse for fast --help)

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init-ckpt",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/200000"
        ),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--num-samples", "-k", type=int, default=8, help="chunks drawn per state")
    ap.add_argument("--num-states", type=int, default=100_000, help="states to cover (evenly strided over the set)")
    ap.add_argument("--num-steps", type=int, default=10, help="Euler steps, the pi05 serving default")
    ap.add_argument("--batch", type=int, default=8, help="STATES per prefix pass (device batch = batch*k)")
    ap.add_argument("--robot-ad", type=int, default=14, help="dims the critic scores; the rest are pi05 padding")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/policy_chunks_bc")
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import torch

    from openpi.extraction import data as exdata
    import openpi.models.model as _model
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    meta = json.loads((a.cache / "meta.json").read_text())
    dataset, cfg = exdata.make_bc_dataset(str(a.init_ckpt / "assets"))
    if len(dataset) != meta["N"]:
        raise SystemExit(f"index contract broken: dataset has {len(dataset)} rows, cache has {meta['N']}")

    # Evenly strided rather than random: strided keeps whole-episode coverage uniform, so the bank
    # is a sample of the STATE distribution and not of the long-episode distribution.
    stride = max(1, len(dataset) // a.num_states)
    idx = np.arange(0, len(dataset), stride, dtype=np.int64)[: a.num_states]
    print(f"{len(idx)} states, stride {stride}, k={a.num_samples}", flush=True)

    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(
        CheckpointWeightLoaderKeepMissing(str(a.init_ckpt / "params")).load(state.to_pure_dict())
    )
    model = nnx.merge(graphdef, state)
    H, K = cfg.model.action_horizon, a.num_samples

    @nnx.jit(static_argnums=(3,))
    def sample_k(model, obs, rng, k):
        """[B, k, H, ad] from ONE prefix pass per state -- the model's own batched sampler, so this
        pass and the served policy cannot drift apart (Pi0.sample_n_actions_batched, pi0.py)."""
        return model.sample_n_actions_batched(rng, obs, num_samples=k, num_steps=a.num_steps)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx.tolist()),
        batch_size=a.batch,
        shuffle=False,  # keeps row order == idx order, which is the whole index contract
        num_workers=a.num_workers,
        drop_last=False,
    )

    def _to_np(v):
        if isinstance(v, dict):
            return {k: _to_np(x) for k, x in v.items()}
        arr = np.asarray(v)
        return arr.astype(np.float32) if arr.dtype == np.float64 else arr

    a.out.mkdir(parents=True, exist_ok=True)
    chunks = np.lib.format.open_memmap(
        a.out / "chunks.npy", mode="w+", dtype=np.float16, shape=(len(idx), K, H, a.robot_ad)
    )
    rng = jax.random.key(a.seed)
    import time

    t0, done = time.time(), 0
    for batch in loader:
        obs = _model.Observation.from_dict({k: _to_np(v) for k, v in batch.items() if k != "actions"})
        rng, k1 = jax.random.split(rng)
        out = np.asarray(sample_k(model, obs, k1, K))[..., : a.robot_ad]
        chunks[done : done + len(out)] = out.astype(np.float16)
        done += len(out)
        if done % (a.batch * 50) < a.batch:
            r = done / (time.time() - t0)
            print(f"{done}/{len(idx)} states  ({r:.1f} states/s, eta {(len(idx) - done) / r / 60:.0f} min)", flush=True)
    chunks.flush()
    np.save(a.out / "idx.npy", idx)
    (a.out / "meta.json").write_text(
        json.dumps(
            {
                "init_ckpt": str(a.init_ckpt),
                "cache": str(a.cache),
                "num_samples": K,
                "num_steps": a.num_steps,
                "horizon": H,
                "robot_ad": a.robot_ad,
                "n_states": len(idx),
                "stride": int(stride),
                "seed": a.seed,
                "space": "pi05-normalized joint delta (sampler raw output, first robot_ad dims)",
            },
            indent=2,
        )
    )
    print(f"wrote {a.out} ({len(idx)} x {K} x {H} x {a.robot_ad})", flush=True)


if __name__ == "__main__":
    main()
