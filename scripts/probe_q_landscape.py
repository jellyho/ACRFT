"""Does the patch critic overestimate Q away from the behaviour policy's support?

Two observations motivated this, and neither is a measurement yet:

  bon      exploiting the critic even slightly makes the SELECTED value optimistic -- the classic
           max-over-N bias, which is only a bias if the critic's error grows where it is picked
  qpilots  steering follows grad_a Q, and a gradient that points off-support is a gradient that
           points somewhere the critic was never told the answer

Both are the same question: what does Q do as an action leaves the region the demonstrations
cover? This probe answers it on the critic's OWN training set, so "off-support" means what it
meant during fitting.

WHAT IS MEASURED, and against what
----------------------------------
There is no ground truth for Q off-support -- that is the whole problem. So the demonstrator's own
continuation is the anchor: on a SUCCESSFUL episode it is an action known to reach the goal, and
the critic is cost-to-goal, so Q(data) is a value the critic should not be able to beat by much.
Everything is reported relative to it.

  data     the demonstrator's next H actions                    in-distribution, known-good
  bc       N draws from the base BC policy                      the policy's own support
  ray      BC mean + t * grad_a Q direction, t swept outward     the direction an exploiter moves
  grid     a 2D slice spanned by grad_a Q and one orthogonal    the landscape itself

Distance is in units of the BC cloud's own spread (sigma), because "far" only means anything
relative to how wide the policy already is -- the same reasoning as the drift readout.

The signature of exploitable overestimation is Q RISING as t grows past the BC cloud. An honest
critic falls off: actions the demonstrations never took should not look better than the ones they
did. Reported alongside it is the ensemble's disagreement, because pessimism (min, or mean - rho *
std) can only defend against error it can SEE -- if std stays flat while Q rises, no amount of rho
saves best-of-N.

Everything the figures need is written to JSON; the figures regenerate from it.
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "misc"))


#: yam_policy.py fills pi0's three image slots from these cameras.
CAMERAS = {
    "observation/image": "agentview",
    "observation/wrist_image": "wrist_left",
    "observation/image_right": "wrist_right",
}


def _obs_dict(imgs, state, task: str):
    """Client-format observation -- the same keys serve_policy receives from the robot."""
    if state is None or not imgs:
        return None
    out = {}
    for k, cam in CAMERAS.items():
        if cam not in imgs or imgs[cam] is None:
            return None
        out[k] = imgs[cam]
    out["observation/state"] = np.asarray(state, np.float32)
    out["prompt"] = task
    return out


@dataclasses.dataclass
class Probe:
    """The critic and the policy, wired exactly as the serving wrapper wires them.

    Built ON the wrapper rather than beside it: the normalisations, the proprio slice and the
    policy->critic action map are the ones a rollout uses, so a landscape measured here is the
    landscape steering actually climbs. A second implementation would be a second answer.
    """

    wrapper: object
    critic: object

    def features(self, obs):
        return np.asarray(self.wrapper._patches_of(obs), np.float32)

    def proprio(self, raw_state):
        return np.asarray(self.wrapper._critic_proprio(raw_state), np.float32)

    def to_critic_space(self, policy_norm_chunks, norm_state, raw_state):
        """Policy-normalized [N, H, Ad] -> the critic's action space, via the wrapper's own map."""
        k, c = self.wrapper._critic_space_affine(norm_state, raw_state)
        h, ad = k.shape
        return np.asarray(policy_norm_chunks[:, :h, :ad] * k + c, np.float32)

    def dataset_to_critic_space(self, abs_chunk, raw_state):
        """Absolute joint targets [H, A] -> the critic's space, the selection path's own call."""
        pre = self.wrapper._pre
        h, ad = self.wrapper._critic_horizon, self.wrapper._critic_action_dim
        if pre is None:
            return np.asarray(abs_chunk[None, :h, :ad], np.float32)
        return np.asarray(pre.actions(abs_chunk[None], raw_state)[:, :h, :ad], np.float32)

    def q_members(self, feats, proprio, chunks, *, batch: int = 64):
        """[K, N] full-chunk Q per ensemble member -- NOT reduced.

        The reduction is the thing under test: `min` is what bon selects on, `mean - rho*std` is
        what qpilots steers on, and whether either can SEE off-support error is the question. So
        the members are kept and reduced in the analysis.

        Batched because the 2D slice is grid_n^2 chunks (441 at the default) and the critic is a
        transformer over 192 patches: one call for all of them allocates a gigabyte of attention
        weights, which is where this ran the GPU out of memory with nine critics resident.
        """
        cq = self.critic
        f = jnp.asarray(feats)
        p = jnp.asarray(proprio)
        outs = []
        for i in range(0, len(chunks), batch):
            c = jnp.asarray(chunks[i : i + batch])
            n = c.shape[0]
            logits = cq.net.apply({"params": cq.params}, f[None].repeat(n, 0), c, p[None].repeat(n, 0))
            outs.append(np.asarray(cq.hl.from_logits(logits)[..., -1], np.float32))  # [K, n]
        return np.concatenate(outs, axis=1)

    def grad_q(self, feats, proprio, chunk):
        """d(mean Q)/d(action), UNCLIPPED.

        critic_q.grad_q_chunk clips inside the call, which makes the gradient exactly zero outside
        the normalized box -- correct for the trainers that use it as a guard, and useless here,
        where the whole point is to look outside the box.
        """
        cq = self.critic
        f = jnp.asarray(feats)[None]
        p = jnp.asarray(proprio)[None]

        def q_of(a):
            logits = cq.net.apply({"params": cq.params}, f, a[None], p)
            return cq.hl.from_logits(logits).mean(axis=0)[..., -1].sum()

        return np.asarray(jax.grad(q_of)(jnp.asarray(chunk)), np.float32)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def bc_draws(wrapper, obs, *, n_bc, flow_steps, rng):
    """N draws from the base policy, in the POLICY's normalized space, plus the normalized state
    every critic needs to interpret them.

    Drawn ONCE per frame and shared across critics: the 3B sampler is the expensive part, and more
    importantly a comparison BETWEEN critics is only that if the actions they score are the same
    actions. Re-drawing per critic would mix sampling variance into every difference.
    """
    import openpi.models.model as _model

    inputs = wrapper._pol._input_transform(dict(obs))
    norm_state = np.asarray(inputs["state"], np.float32).reshape(-1)
    tree = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(tree)
    _tok, draws = wrapper._extract(rng, observation, num_samples=n_bc, num_steps=flow_steps)
    return np.asarray(draws[0], np.float32), norm_state


def probe_frame(pb, obs, bc_norm, norm_state, data_chunk, *, ray_sigmas, abs_ts, grid_abs, grid_n):
    """Everything measured for ONE critic at one frame. Plain lists, so it serialises."""
    raw_state = np.asarray(obs[pb.wrapper._state_key], np.float32).reshape(-1)
    feats = pb.features(obs)
    pro = pb.proprio(raw_state)
    bc = pb.to_critic_space(bc_norm, norm_state, raw_state)  # [N, h, ad]

    centre = bc.mean(axis=0)
    sigma = float(bc.std(axis=0).mean())  # the policy's own spread, one number
    if sigma < 1e-9:
        return None

    # The exploit direction is the CRITIC's own: each is asked where IT thinks value increases,
    # which is what an arm consuming it would follow.
    g = pb.grad_q(feats, pro, centre)
    e1 = _unit(g)
    r = np.random.default_rng(0).normal(size=e1.shape).astype(np.float32)
    e2 = _unit(r - (r * e1).sum() * e1)  # orthogonal to the exploit direction

    # TWO parameterisations, because two methods reach different distances.
    #
    # In BC-sigma units: best-of-N can only pick among draws, so it never leaves the cloud -- a few
    # sigma is the whole of what it can exploit. But sigma is small (~0.009 measured), so 8 sigma is
    # still 0.07 of a box half-width and a sweep in sigma alone never goes off-support at all.
    #
    # In ABSOLUTE normalized units: steering moves by alpha * ||v|| per step, which has nothing to
    # do with the policy's spread, so it reaches the box edge and past it. That is where the
    # question "is off-support overestimated" actually lives.
    ray = np.stack([centre + t * sigma * e1 for t in ray_sigmas])
    absray = np.stack([centre + t * e1 for t in abs_ts])
    ts = np.linspace(-grid_abs, grid_abs, grid_n)
    grid = np.stack([centre + x * e1 + y * e2 for x in ts for y in ts])

    out = {
        "sigma": sigma,
        "ray_sigmas": [float(t) for t in ray_sigmas],
        "abs_ts": [float(t) for t in abs_ts],
        "grid_ts": [float(t) for t in ts],
        "grad_norm": float(np.linalg.norm(g)),
    }
    for name, arr in (("bc", bc), ("ray", ray), ("absray", absray), ("grid", grid)):
        q = pb.q_members(feats, pro, arr)  # [K, N]
        out[f"q_{name}"] = q.tolist()
        out[f"outbox_{name}"] = float(((arr < -1) | (arr > 1)).mean())
    d = pb.dataset_to_critic_space(data_chunk, raw_state)
    out["q_data"] = pb.q_members(feats, pro, d).tolist()
    return out


def main(a):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from dataset_reader import DatasetReader
    from dataset_reader import SequentialImages

    from openpi.extraction import critic_q
    from openpi.policies import patch_critic_policy as pcp
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config

    cfg = _config.get_config(a.policy_config)
    policy = _pc.create_trained_policy(cfg, pathlib.Path(a.policy_dir))
    # One wrapper per critic, all over the SAME loaded policy. Each carries its own normalisation,
    # proprio slice and policy->critic action map, so they can disagree about everything except
    # which actions were drawn.
    probes = {}
    for cdir in a.critic:
        name = pathlib.Path(cdir).name
        w = pcp.PatchCriticSelectPolicy(policy, cdir, mode="bon", default_samples=a.n_bc)
        probes[name] = Probe(wrapper=w, critic=critic_q.load(cdir))
        logging.info("critic %s ready", name)
    first = next(iter(probes.values())).wrapper

    ds = pathlib.Path(a.dataset)
    reader = DatasetReader(ds.name, str(ds.parent))
    reader.load()  # metadata: without it episode_length is 0 and every frame is silently skipped
    outcomes = {}
    op = ds / "outcomes.jsonl"
    if op.exists():
        for line in op.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                outcomes[int(d["episode"])] = d
    # Successful episodes only: the anchor has to be an action that actually reached the goal, or
    # "the critic beats the demonstrator" says nothing.
    eps = sorted(e for e, d in outcomes.items() if d.get("outcome") == "success")
    if not eps:
        raise SystemExit(f"no successful episodes in {op}; the anchor would be meaningless")
    logging.info("%d successful episodes of %d", len(eps), len(outcomes))

    rng = jax.random.key(a.seed)
    pick = np.random.default_rng(a.seed)
    rows, skipped = [], {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    # Episode at a time, frames in increasing order. SequentialImages walks each camera's file once
    # (~1 ms/frame) where random access re-seeks and re-decodes per frame (221 ms measured in that
    # class's own docstring), and it is forward-only -- so the SAMPLING is sorted rather than the
    # reader made random-access. DatasetReader.get_images would have done the latter, and on this
    # machine it fails outright: its LeRobotDataset path needs torchcodec, whose FFmpeg libraries
    # are missing here, and it swallows that into an empty dict.
    order = pick.permutation(eps)
    per_ep = max(1, a.per_episode)
    for ep_idx in order:
        if len(rows) >= a.frames:
            break
        ep = int(ep_idx)
        n = reader.episode_length(ep)
        if not n or n < a.horizon + 2:
            skip("episode too short / unreadable")
            continue
        acts = reader.column(ep, "action")
        states = reader.column(ep, "observation.state")
        if acts is None or states is None:
            skip("episode has no action/state columns")
            continue
        want = min(per_ep, a.frames - len(rows))
        hi = min(int(n), acts.shape[0]) - a.horizon - 1
        if hi <= 0:
            skip("episode too short / unreadable")
            continue
        frames = sorted({int(x) for x in pick.integers(0, hi, size=want * 3)})[:want]
        try:
            seq = SequentialImages(str(ds), ep, cameras=list(CAMERAS.values()))
        except Exception as e:
            skip(f"video open failed ({type(e).__name__})")
            continue
        try:
            for fr in frames:
                obs = _obs_dict(seq.frame(fr), states[fr], outcomes[ep].get("task", ""))
                if obs is None:
                    skip("no images or state at that frame")
                    continue
                data_chunk = np.asarray(acts[fr : fr + a.horizon], np.float32)
                rng, sub = jax.random.split(rng)
                # ONE draw, every critic scores it.
                bc_norm, norm_state = bc_draws(first, obs, n_bc=a.n_bc, flow_steps=a.flow_steps, rng=sub)
                per_critic = {}
                for name, pb in probes.items():
                    r = probe_frame(
                        pb,
                        obs,
                        bc_norm,
                        norm_state,
                        data_chunk,
                        ray_sigmas=np.arange(0, a.ray_max + 1e-9, a.ray_step),
                        abs_ts=np.arange(0, a.abs_max + 1e-9, a.abs_step),
                        grid_abs=a.grid_max,
                        grid_n=a.grid_n,
                    )
                    if r is not None:
                        per_critic[name] = r
                if not per_critic:
                    skip("BC spread collapsed (sigma ~ 0)")
                    continue
                rows.append({"episode": ep, "frame": int(fr), "frames_in_episode": int(n), "critics": per_critic})
                if len(rows) % 5 == 0:
                    logging.info("  %d/%d frames", len(rows), a.frames)
        finally:
            seq.close()

    if skipped:
        # A probe that returns nothing must say why. The first run of this wrote 0 frames and
        # looked like a successful run -- the reader needed .load() and said so nowhere.
        logging.info("skipped: %s", ", ".join(f"{v}x {k}" for k, v in sorted(skipped.items())))
    if not rows:
        raise SystemExit("no frames probed -- see the skip reasons above")
    out = {
        "rows": rows,
        "critics": [pathlib.Path(c).name for c in a.critic],
        "policy": str(a.policy_dir),
        "dataset": str(a.dataset),
        "horizon": a.horizon,
        "n_bc": a.n_bc,
    }
    pathlib.Path(a.out).write_text(json.dumps(out))
    logging.info("wrote %s (%d frames)", a.out, len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(pathlib.Path.home() / "lerobot_data/yam_lego_taxi"))
    ap.add_argument("--critic", required=True, nargs="+", help="one or more critic dirs, scored on the same frames")
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--policy-config", default="pi05_yam_lego_taxi")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--per-episode", type=int, default=2, help="frames sampled per episode visited")
    ap.add_argument("--n-bc", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--ray-max", type=float, default=8.0, help="how many BC sigmas to sweep out to")
    ap.add_argument("--ray-step", type=float, default=0.5)
    ap.add_argument(
        "--abs-max", type=float, default=3.0, help="sweep in absolute normalized units; the box half-width is 1"
    )
    ap.add_argument("--abs-step", type=float, default=0.1)
    ap.add_argument("--grid-max", type=float, default=2.0, help="2D slice half-width, absolute normalized units")
    ap.add_argument("--grid-n", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="q_landscape.json")
    main(ap.parse_args())
