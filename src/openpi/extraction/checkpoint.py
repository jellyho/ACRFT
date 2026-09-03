"""Write an extraction-arm checkpoint in the SAME layout openpi's BC trainer writes.

    <out>/<step>/params/   orbax PyTree {"params": <whole-model pure dict>}  -- what `restore_params` reads
    <out>/<step>/assets/   norm stats, copied from the init checkpoint       -- what `load_norm_stats` reads

Every trainer saves the WHOLE model, whatever subset it trained: an expert-only run carries the frozen
backbone verbatim. That is ~12 GB per step instead of ~2 GB, and it buys the thing that matters: no
export step, no "which subtree was trained" bookkeeping, and

    scripts/serve_policy.py --policy.config pi05_yam_lego_taxi --policy.dir <out>/<step>

works the moment the directory lands, exactly as for a BC checkpoint (user decision, 2026-09-03: there
is no expert-only save pipeline). `train_state/` is not written -- these runs are never resumed, and
serving does not read it.

Checkpoints written before 2026-09-03 used a bare orbax tree ({"expert": ...} or {"params": ...}) with
no params/ or assets/ subdirectory; `scripts/export_extraction_checkpoint.py` converts those.
"""

import pathlib
import shutil

import orbax.checkpoint as ocp


def save_servable(step_dir: pathlib.Path | str, params: dict, *, assets_from: pathlib.Path | str) -> pathlib.Path:
    """Write `params` (a whole-model pure dict) as `<step_dir>/params` and copy `<assets_from>/assets` next to it."""
    step_dir = pathlib.Path(step_dir).absolute()
    assets_src = pathlib.Path(assets_from) / "assets"
    if not assets_src.is_dir():
        raise FileNotFoundError(f"{assets_src} missing: the init checkpoint must carry its norm stats")
    step_dir.mkdir(parents=True, exist_ok=True)
    with ocp.StandardCheckpointer() as c:
        c.save(step_dir / "params", {"params": params}, force=True)
    assets_dst = step_dir / "assets"
    if assets_dst.exists():
        shutil.rmtree(assets_dst)
    shutil.copytree(assets_src, assets_dst)
    return step_dir
