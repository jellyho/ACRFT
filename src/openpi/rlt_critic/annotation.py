"""Derived annotations, built on demand by whoever needs them.

The discount is not a training flag on its own: it sets `mc_return` and the value support as well as
the bootstrap. Training at gamma=0.999 against returns accumulated at 0.99 silently mixes two
definitions of what a step costs - and it is worst exactly where mc_return is used as a target floor,
because then the wrong number is not a bystander but a bound the target is clamped to.

So the discount selects a *dataset*, and this module makes that dataset appear. `ensure_discount`
returns the annotation directory that actually matches the requested gamma, building it if it does
not exist yet. Nothing has to be prepared by hand first: running the trainer is sufficient.

Everything here is safe to call from many SLURM array tasks at once. Variants are built into a
private temporary directory and moved into place with a single rename, so a concurrent reader either
sees no directory or sees a complete one - never a half-written one - and a task that loses the race
simply uses the winner's copy.
"""

import json
import logging
import os
import pathlib
import shutil

import numpy as np

logger = logging.getLogger(__name__)

# Identical across discounts and never rewritten in place -> shared by hardlink rather than copied.
# These are ~99.9% of the bytes: rl_token alone is 2.3 GB.
SHARED = (
    "rl_token",
    "action_chunk",
    "base_action",
    "base_action_heldout",
    "progress",
    "frame_index",
    "proprio",
)
# Read by the critic and rewritten by any re-labelling -> must be independent bytes, so that writing
# to a variant can never reach back into the source through a shared inode.
PRIVATE = ("reward", "mc_return", "done", "episode_index")

_TOL = 1e-12


def discount_tag(discount: float) -> str:
    """0.999 -> 'g999'. The suffix a variant directory carries."""
    return "g" + format(discount, ".10g").split(".")[1]


def mc_return_at(reward, done, episode, discount: float) -> np.ndarray:
    """Backward return-to-go per episode, reset at every terminal.

    This is how annotate_rlt.py built the column in the first place, so re-running it from the stored
    per-frame reward reproduces the annotation exactly rather than approximating it - and it needs
    only the arrays already on disk, not the source LeRobot dataset.
    """
    T = len(reward)
    mc = np.zeros(T, np.float64)
    if not T:
        return mc.astype(np.float32)
    bounds = np.flatnonzero(np.diff(episode, prepend=episode[0] - 1))
    bounds = np.append(bounds, T)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        acc = 0.0
        for t in range(b - 1, a - 1, -1):
            if done[t]:
                acc = 0.0  # nothing accumulates across the end of the decision problem
            acc = float(reward[t]) + discount * acc
            mc[t] = acc
    return mc


def _link(src: pathlib.Path, dst: pathlib.Path) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        # Different filesystem, or a backend without link support. A symlink reads the same and is
        # still zero-copy; it is only weaker against the source being moved.
        dst.symlink_to(src.resolve())
        return "symlink"


def _matches(path: pathlib.Path, discount: float) -> bool:
    meta = path / "meta.json"
    if not meta.exists():
        return False
    try:
        return abs(json.loads(meta.read_text()).get("discount", 0.99) - discount) <= _TOL
    except (json.JSONDecodeError, OSError):
        return False


def build_discount_variant(data: pathlib.Path, discount: float, out: pathlib.Path) -> pathlib.Path:
    """Write the gamma-`discount` copy of `data` to `out`. Assumes `out` does not exist."""
    meta = json.loads((data / "meta.json").read_text())
    T = meta["num_frames"]
    scheme = meta.get("reward_scheme", "sparse")
    if scheme not in ("sparse", "v3"):
        raise ValueError(f"unknown reward_scheme {scheme!r}; only sparse and v3 are handled")

    rd = lambda n, dt: np.asarray(np.memmap(data / f"{n}.dat", dt, "r", shape=(T,)))  # noqa: E731
    reward, done, episode = rd("reward", np.float32), rd("done", np.int8), rd("episode_index", np.int32)
    mc = mc_return_at(reward, done, episode, discount)

    reward_out = reward.astype(np.float32)
    support = meta.get("value_support", [-1.0, 0.0] if scheme == "v3" else [0.0, 1.0])
    z_scale = 1.0
    if scheme == "v3":
        # v3 divides everything by Z = |min return| so the support is exactly [-1, 0]. Z moves with
        # the discount (a longer horizon accumulates more living cost), so reward and return are
        # rescaled together; the stored reward is already in Z_old units, hence the ratio.
        z_scale = abs(float(mc.min())) or 1.0
        mc = mc / z_scale
        reward_out = (reward / z_scale).astype(np.float32)

    # Build beside the destination (same filesystem, so the rename is atomic and hardlinks work) and
    # under a pid-unique name, so two array tasks racing here cannot write into each other's copy.
    tmp = out.parent / f".{out.name}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for name in SHARED:
            src = data / f"{name}.dat"
            if src.exists():
                _link(src, tmp / f"{name}.dat")
        for name in PRIVATE:
            src = data / f"{name}.dat"
            if src.exists():
                shutil.copy2(src, tmp / f"{name}.dat")
        np.memmap(tmp / "reward.dat", np.float32, "r+", shape=(T,))[:] = reward_out
        np.memmap(tmp / "mc_return.dat", np.float32, "r+", shape=(T,))[:] = mc.astype(np.float32)

        meta_out = meta | {"discount": discount, "reward_scheme": scheme, "value_support": support}
        if scheme == "v3":
            meta_out["z"] = meta.get("z", 1.0) * z_scale
        # meta.json is written last and is what `_matches` reads, so a directory that appears
        # complete really is: the rename below only ever publishes a finished tree.
        (tmp / "meta.json").write_text(json.dumps(meta_out, indent=2))
        os.rename(tmp, out)
    except OSError:
        if not _matches(out, discount):
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        shutil.rmtree(tmp, ignore_errors=True)  # lost the race; the winner's copy is equivalent
    return out


def ensure_discount(data: pathlib.Path, discount: float | None) -> pathlib.Path:
    """The annotation directory whose returns are accumulated at `discount`, building it if needed.

    `discount=None` (or a discount the source already carries) returns `data` untouched, so the
    common path costs one meta.json read.
    """
    data = pathlib.Path(data)
    meta = json.loads((data / "meta.json").read_text())
    have = meta.get("discount", 0.99)
    if discount is None or abs(have - discount) <= _TOL:
        return data

    out = data.parent / f"{data.name}_{discount_tag(discount)}"
    if _matches(out, discount):
        logger.info(f"discount {discount}: using existing variant {out}")
        return out
    if out.exists():
        # A directory under the right name with the wrong (or unreadable) discount is not something
        # to silently overwrite - a run may be reading it. Name the conflict instead of guessing.
        raise ValueError(f"{out} exists but its meta.json does not record discount={discount}; remove it or rename")

    logger.info(f"discount {discount} != annotation's {have}: building {out} (mc_return re-accumulated)")
    build_discount_variant(data, discount, out)
    logger.info(f"discount {discount}: built {out}")
    return out
