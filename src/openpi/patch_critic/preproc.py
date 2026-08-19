"""Put the patch-critic's inputs in the SAME space the base VLA works in.

The critic used to eat raw dataset units (absolute joint targets in radians, a 42-d state whose
channels differ in scale by 30x). The base pi05 policy does not: its data pipeline turns each action
into a joint DELTA against the current joint position and then quantile-normalizes both state and
actions into roughly [-1, 1].

Training the critic in raw units while serving it a policy that speaks normalized deltas is what made
the serving path silently score the wrong action space. Sharing one preprocessing removes that whole
failure class -- the sampler's output IS the critic's input, with no conversion in between -- and it
also fixes the critic's conditioning, since raw state channels span std 0.096 to 3.1.

The pipeline reproduced here is exactly, in order:
  1. ``JointDeltaActions(ref)``  actions[..., i] -= state[..., ref[i]]  for ref[i] >= 0 (grippers stay
     absolute). One state -- the chunk's base frame -- is used for the whole horizon, as in training.
  2. ``Normalize(use_quantiles=True)``  (x - q01) / (q99 - q01) * 2 - 1, applied to state and actions.

Padding to the model action dim is deliberately NOT reproduced: the critic has its own action_dim and
a 42-d state that pi05's 32-d padding would truncate.
"""

import dataclasses
import json
import pathlib

import numpy as np

MODES = ("raw", "pi05")

# Which proprio channels the critic is allowed to see.
#
# YAM logs 42 dims per frame: each arm is pos(7), vel(7), eff(7). Nearly every VLA -- openpi's own
# ALOHA (14), Libero (8) and DROID (joint_position + gripper) included -- feeds POSITIONS ONLY, so
# taking all 42 quietly hands our model two sensor streams the baselines never get, and any headline
# number stops being a method-only difference. Torque is the sharp case: contact effort says almost
# directly whether the gripper has hold of something, which is much of what separates success from
# failure, so a critic reading it can score well without understanding the task at all.
PROPRIO_SETS = {
    "all": None,  # every channel (legacy)
    "pos": [*range(7), *range(21, 28)],  # left pos(6)+gripper, right pos(6)+gripper -- 14 dims
}


def compare(a: dict, b: dict, *, tol: float = 1e-5) -> list[str]:
    """How two sets of norm stats disagree (empty list = same numbers).

    Only the keys and statistics present in BOTH are compared: the critic stores what it uses, the
    served policy may carry more, and a key one side simply does not have is not a disagreement.
    """
    out = []
    for key in sorted(set(a) & set(b)):
        for stat in sorted(set(a[key]) & set(b[key])):
            x, y = np.asarray(a[key][stat], np.float64), np.asarray(b[key][stat], np.float64)
            n = min(x.shape[-1], y.shape[-1])  # the policy pads; compare the shared prefix
            d = float(np.abs(x[..., :n] - y[..., :n]).max())
            if d > tol:
                out.append(f"{key}.{stat}: max abs difference {d:.3g} over {n} dims")
    return out


def load_norm_stats(path) -> dict:
    """openpi norm_stats.json -> {'state': {...}, 'actions': {...}} of float64 arrays."""
    d = json.loads(pathlib.Path(path).read_text())
    d = d.get("norm_stats", d)
    return {k: {kk: np.asarray(vv, np.float64) for kk, vv in v.items() if vv is not None} for k, v in d.items()}


@dataclasses.dataclass(frozen=True)
class Pi05Preproc:
    """State/action preprocessing shared with the base VLA. Stateless; safe to use per batch."""

    ref: np.ndarray  # action dim -> state index to subtract, -1 = leave absolute
    stats: dict
    use_quantiles: bool = True
    delta: bool = True

    @classmethod
    def build(cls, norm_stats_path, ref, *, use_quantiles: bool = True, delta: bool = True) -> "Pi05Preproc":
        return cls(np.asarray(ref, np.int64), load_norm_stats(norm_stats_path), use_quantiles, delta)

    def _norm(self, x, key: str):
        s = self.stats[key]
        if self.use_quantiles:
            q01, q99 = s["q01"][: x.shape[-1]], s["q99"][: x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return (x - s["mean"][: x.shape[-1]]) / (s["std"][: x.shape[-1]] + 1e-6)

    def state(self, s):
        """[..., S] raw state -> normalized."""
        return self._norm(np.asarray(s, np.float64), "state").astype(np.float32)

    def actions(self, chunk, base_state):
        """[B, H, A] raw ABSOLUTE actions + [B, S] raw state at the chunk's base frame -> normalized delta.

        Copies before mutating: openpi's JointDeltaActions edits its input in place, which silently
        corrupts a caller that still needs the absolute actions.
        """
        a = np.array(chunk, np.float64, copy=True)
        if self.delta:
            s = np.asarray(base_state, np.float64)
            for i, r in enumerate(self.ref):
                if r >= 0:
                    a[..., i] -= s[..., None, r]
        return self._norm(a, "actions").astype(np.float32)

    def digest(self) -> str:
        """Short content hash of the stats, so a server can tell "same numbers" from "same path"."""
        import hashlib

        h = hashlib.sha256()
        for key in sorted(self.stats):
            for stat in sorted(self.stats[key]):
                h.update(key.encode())
                h.update(stat.encode())
                h.update(np.asarray(self.stats[key][stat], np.float64).tobytes())
        return h.hexdigest()[:16]

    def embedded(self) -> dict:
        """The stats themselves, for writing INTO the checkpoint.

        The spec also records the path they came from, but a path is not portable: move the checkpoint
        or run from another cwd and it either fails or -- worse -- resolves to a different pi05
        checkpoint's stats. The embedded copy is what serving actually uses.
        """
        return {k: {kk: np.asarray(vv, np.float64).tolist() for kk, vv in v.items()} for k, v in self.stats.items()}

    def spec(self, norm_stats_path) -> dict:
        """The part of the input contract this preprocessing determines."""
        return {
            "normalization": "pi05",
            "norm_stats": str(norm_stats_path),  # provenance only -- serving reads the embedded copy
            "norm_stats_file": "pi05_norm_stats.json",
            "norm_stats_digest": self.digest(),
            "use_quantiles": bool(self.use_quantiles),
            "delta_mode": "joint" if self.delta else "none",
            "joint_delta_reference": self.ref.tolist(),
            "state_units": "pi05-normalized state (quantile) -- same space the base VLA sees",
            "action_units": "pi05-normalized JOINT DELTA -- identical to the sampler's raw output",
        }
