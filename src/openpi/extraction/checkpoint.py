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

import os
import pathlib
import shutil

from flax import traverse_util
import orbax.checkpoint as ocp


def _check_leaf_keys(params: dict) -> None:
    """`restore_params` strips a trailing "value" key only when EVERY leaf path ends with one (model.py); a tree
    that mixes nnx-wrapped and pure leaves would restore with a partly-shifted structure and fail at serve time."""
    paths = list(traverse_util.flatten_dict(params))
    if not paths:
        raise ValueError("refusing to save an empty parameter tree")
    ends = [kp[-1] == "value" for kp in paths]
    if any(ends) and not all(ends):
        raise ValueError("parameter tree mixes nnx-wrapped ('value') and pure leaves; pass a pure dict")


def save_servable(step_dir: pathlib.Path | str, params: dict, *, assets_from: pathlib.Path | str) -> pathlib.Path:
    """Write `params` (a whole-model pure dict) as `<step_dir>/params` and copy `<assets_from>/assets` next to it.

    The step is built in `<step_dir>.tmp` and renamed into place, so a run killed mid-write (12 GB takes a while)
    never leaves a half-written `<step>` that looks servable; a stale `.tmp` from such a kill is replaced.
    """
    step_dir = pathlib.Path(step_dir).absolute()
    assets_src = pathlib.Path(assets_from) / "assets"
    if not assets_src.is_dir():
        raise FileNotFoundError(f"{assets_src} missing: the init checkpoint must carry its norm stats")
    _check_leaf_keys(params)
    tmp = step_dir.with_name(step_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with ocp.StandardCheckpointer() as c:
        c.save(tmp / "params", {"params": params})
    shutil.copytree(assets_src, tmp / "assets")
    if step_dir.exists():
        shutil.rmtree(step_dir)
    os.rename(tmp, step_dir)
    return step_dir
