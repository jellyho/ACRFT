"""Convert a PRE-2026-09-03 extraction-arm run into an ordinary openpi checkpoint.

Trainers now write servable checkpoints directly (`<out>/<step>/params` + `assets`, see
openpi/extraction/checkpoint.py), so nothing trained after 2026-09-03 needs this. Runs from before
that date are a bare orbax tree at `<arm>_<run>/<step>` with no params/ or assets/ subdirectory,
in one of two shapes that the top-level key tells apart:

  {"expert": ...}   an expert-only run (`_run1`): only the action-expert subtree was trained, so it
                    is OVERLAID onto the BC checkpoint's full parameter tree here.
  {"params": ...}   a backbone run (`_bb`, `--train-backbone`): the WHOLE model was trained and
                    saved, so it is used as-is -- overlaying it on BC would be wrong, and reading
                    it as "expert" is a KeyError, which is how this case was first noticed.

Either way the output is a normal checkpoint served by the one serving entry point:

    uv run python scripts/export_extraction_checkpoint.py --arm qam            # expert run, _run1
    uv run python scripts/export_extraction_checkpoint.py --arm awr --run bb   # backbone run, _bb
    uv run python scripts/serve_policy.py --policy.config pi05_yam_lego_taxi \\
        --policy.dir /data1/jellyho/acrft_ckpts/extraction/exported/awr_bb_30000
"""

# ruff: noqa: PLC0415

import argparse
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["awr", "cfgrl", "flowdpg", "qam", "fqlx"])
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--run", default="run1", help="run suffix: run1 = expert-only, bb = backbone (whole model)")
    ap.add_argument("--base-config", default="pi05_yam_lego_taxi", help="the task config the arm was trained on")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    import flax.nnx as nnx
    import jax

    from openpi.extraction import serving
    import openpi.training.config as _config
    from openpi.training.weight_loaders import CheckpointWeightLoaderKeepMissing

    src = serving.CKPT_ROOT / f"{a.arm}_{a.run}"
    steps = sorted((int(p.name) for p in src.iterdir() if p.name.isdigit()), reverse=True)
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {src}")
    step = a.step or steps[0]
    expert_dir = src / str(step)
    if (expert_dir / "params").is_dir():
        raise SystemExit(f"{expert_dir} already has params/: it was written servable, serve it directly")
    out = a.out or serving.CKPT_ROOT / "exported" / (
        f"{a.arm}_{step}" if a.run == "run1" else f"{a.arm}_{a.run}_{step}"
    )

    # base task config, or its CFGRL variant (config.with_cfgrl) -- the arm never picks a
    # task-specific name itself, so exporting an arm trained on another task changes one flag
    cfg_name = f"{a.base_config}_cfgrl" if a.arm == "cfgrl" else a.base_config
    cfg = _config.get_config(cfg_name)
    model = cfg.model.create(jax.random.key(0))
    _graphdef, state = nnx.split(model)
    params = CheckpointWeightLoaderKeepMissing(str(serving.BC_CKPT / "params")).load(state.to_pure_dict())

    import numpy as np
    import orbax.checkpoint as ocp

    # Restore every leaf as a host numpy array. The run checkpoints were written on a GPU node and
    # carry `cuda:0` sharding metadata; a StandardCheckpointer restore replays that sharding and
    # fails on any machine without that device ("Device cuda:0 was not found"). Restoring by
    # metadata with an explicit numpy restore type ignores the stored sharding entirely.
    with ocp.PyTreeCheckpointer() as c:
        meta = c.metadata(expert_dir.absolute())
        saved = c.restore(
            expert_dir.absolute(),
            ocp.args.PyTreeRestore(
                item=meta, restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta)
            ),
        )
    if "params" in saved:
        # backbone run: the whole model was trained. Use it as-is; do not overlay on BC.
        params = saved["params"]
        print(f"{a.arm}: whole-model checkpoint from {expert_dir} (backbone run), used as-is")
    elif "expert" in saved:
        serving._deep_update(params, saved["expert"])
        print(f"{a.arm}: overlaid the trained subtree from {expert_dir}")
    else:
        raise KeyError(f"{expert_dir} has neither 'params' nor 'expert' at the top level: {list(saved)}")

    # the same writer the trainers use; norm stats travel WITH the checkpoint (create_trained_policy
    # reads <dir>/assets), and these arms train in the BC checkpoint's input space, so its assets apply
    from openpi.extraction.checkpoint import save_servable

    save_servable(out, params, assets_from=serving.BC_CKPT)

    print(
        f"\nwrote {out}\nserve with:\n  uv run python scripts/serve_policy.py --policy.config {cfg_name} --policy.dir {out}"
    )


if __name__ == "__main__":
    main()
