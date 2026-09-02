"""Export a weight-bearing extraction arm as an ORDINARY openpi checkpoint.

The extraction trainers save only the trained subtree (`{"expert": ...}`) because that is all
they touch. Serving must not know about that: this writes a normal `<out>/params` + `<out>/assets`
checkpoint, so the arm is served by the one serving entry point with no special path:

    uv run python scripts/export_extraction_checkpoint.py --arm qam
    uv run python scripts/serve_policy.py --policy.config pi05_yam_lego_taxi \\
        --policy.dir /data1/jellyho/acrft_ckpts/extraction/exported/qam_30000

CFGRL keeps its own train config (`pi05_yam_lego_taxi_cfgrl`) because its model carries the
optimality embedding and samples with classifier-free guidance; the export is otherwise identical.
"""

# ruff: noqa: PLC0415

import argparse
import pathlib
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["awr", "cfgrl", "flowdpg", "qam", "fqlx"])
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--run", default="run1")
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
    out = a.out or serving.CKPT_ROOT / "exported" / f"{a.arm}_{step}"

    # base task config, or its CFGRL variant (config.with_cfgrl) -- the arm never picks a
    # task-specific name itself, so exporting an arm trained on another task changes one flag
    cfg_name = f"{a.base_config}_cfgrl" if a.arm == "cfgrl" else a.base_config
    cfg = _config.get_config(cfg_name)
    model = cfg.model.create(jax.random.key(0))
    _graphdef, state = nnx.split(model)
    params = CheckpointWeightLoaderKeepMissing(str(serving.BC_CKPT / "params")).load(state.to_pure_dict())

    import orbax.checkpoint as ocp

    with ocp.StandardCheckpointer() as c:
        expert = c.restore(expert_dir.absolute())["expert"]
    serving._deep_update(params, expert)
    print(f"{a.arm}: overlaid the trained subtree from {expert_dir}")

    out.mkdir(parents=True, exist_ok=True)
    with ocp.StandardCheckpointer() as c:
        c.save((out / "params").absolute(), {"params": params}, force=True)
    # norm stats travel WITH the checkpoint (create_trained_policy reads <dir>/assets), and these
    # arms train in the BC checkpoint's input space, so its assets are the correct ones
    if (out / "assets").exists():
        shutil.rmtree(out / "assets")
    shutil.copytree(serving.BC_CKPT / "assets", out / "assets")

    print(
        f"\nwrote {out}\nserve with:\n  uv run python scripts/serve_policy.py --policy.config {cfg_name} --policy.dir {out}"
    )


if __name__ == "__main__":
    main()
