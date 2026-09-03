"""One noise draw, one frame, and what QPILOTS steering does to it as alpha grows.

Every other measurement of steering in this repo is an aggregate: displacement averaged over a
rollout, Q averaged over frames, drift summarised as a scalar. This one is the opposite. It fixes
everything that can be fixed -- the observation, the flow's initial noise, the critic -- so the
ONLY thing that varies is the guidance strength, and then shows the chunk itself.

That is possible because ``Pi0Steered.sample_steered`` takes the noise as an argument and
``alpha=0`` reproduces the unsteered draw THROUGH THE SAME PATH. So the alpha=0 curve here is not
"a BC sample" in the loose sense; it is THIS noise's BC sample, and every other curve is what
steering did to that exact draw. The difference between two curves is the steering term and
nothing else -- no sampling variance, no re-normalisation, no second model.

The four panels answer four different questions.

  1  joint trajectories      what the arm was actually told to do, per joint, over the 30-step
                             chunk. One line per alpha. This is the picture of "how far does it
                             bend".
  2  displacement            ||a(alpha) - a(0)|| per step of the chunk, in BC-sigma units. Says
                             WHERE in the chunk steering bites: uniformly, or only at the end.
  3  the policy's own plane  every alpha projected onto PC1/PC2 of the BC draw cloud at this
                             frame, with the cloud drawn to scale. This is where "steering leaves
                             the support" stops being an abstraction: the cloud is the region the
                             demonstrations cover, and the alpha path either stays in it or does
                             not.
  4  what it bought          the critic's own pessimistic Q at each alpha, against the value of
                             the demonstrator's actual continuation. Steering climbs Q by
                             construction; whether that climb is real is the question the
                             q-landscape probe answers, and this panel is where the two meet.

The chunk is NOT clipped (this branch removed the output clip), so saturation past the normalized
box edge is visible rather than hidden. Panel 1 marks that edge.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))

CAMERAS = {
    "observation/image": "agentview",
    "observation/wrist_image": "wrist_left",
    "observation/image_right": "wrist_right",
}
#: The 14 action dimensions of the YAM bimanual chunk, in order.
JOINT_NAMES = [f"L{i}" for i in range(1, 7)] + ["Lgrip"] + [f"R{i}" for i in range(1, 7)] + ["Rgrip"]


def _obs_dict(imgs, state, task):
    out = {}
    for k, cam in CAMERAS.items():
        if cam not in imgs or imgs[cam] is None:
            return None
        out[k] = imgs[cam]
    out["observation/state"] = np.asarray(state, np.float32)
    out["prompt"] = task
    return out


@dataclasses.dataclass
class Sweep:
    """Everything measured at one frame, for one critic, across the alpha ladder."""

    alphas: list
    chunks: np.ndarray  # [S, A, H, AD] policy-normalized, S noise draws x A alphas
    bc: np.ndarray  # [N, H, AD] the BC cloud at this frame, same normalization
    data: np.ndarray  # [H, AD] the demonstrator's continuation, policy-normalized
    q: np.ndarray  # [A] pessimistic Q of each steered chunk, critic's own reduction
    q_data: float
    sigma: float  # mean PER-COORDINATE std of the BC cloud
    pc_sigma: float  # std along the draws' first principal direction, for reference
    episode: int
    frame: int


def measure(wrapper, sampler, obs, data_chunk, *, alphas, n_bc, flow_steps, seed, n_noise=4):
    """Run the ladder. One rng for the flow noise, reused at every alpha."""
    import openpi.models.model as _model

    # The wrapper must have run once so the policy->critic affine map is set; the sampler refuses
    # otherwise, and it refuses for a good reason (see serving.py).
    wrapper.infer(dict(obs))

    inputs = wrapper._pol._input_transform(dict(obs))
    tree = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(tree)

    # The BC cloud at this frame: the scale everything else is read against.
    rng = jax.random.key(seed)
    rng, sub = jax.random.split(rng)
    _tok, draws = wrapper._extract(sub, observation, num_samples=n_bc, num_steps=flow_steps)
    bc = np.asarray(draws[0], np.float32)

    obs_pre = _model.preprocess_observation(None, observation, train=False)
    feats = np.asarray(wrapper._patches_of(dict(obs)), np.float32)
    proprio = np.asarray(wrapper._critic_proprio(obs["observation/state"]), np.float32)
    feats_j = jnp.asarray(feats)[None] if feats.ndim == 2 else jnp.asarray(feats)
    prop_j = jnp.asarray(proprio)[None] if proprio.ndim == 1 else jnp.asarray(proprio)
    ad = sampler.critic.config["action_dim"]
    k, c = sampler.to_critic_space

    # One rng per NOISE, reused across the whole alpha ladder: sample_steered draws x0 from it, so
    # a ladder shares its noise and the only thing varying along it is alpha. alpha is traced in
    # that jit, so this is one compilation for the whole grid.
    #
    # Several noises, not one, because a single draw can land in the tail of the policy's own cloud
    # (the first version of this figure had its alpha=0 point 3 sigma out along PC1, which reads as
    # "steering starts off-support" and is just one unlucky sample).
    paired = False  # the alpha=0 twin is a rung of this ladder already, so no pairing is needed
    chunks = []
    for si in range(n_noise):
        steer_rng = jax.random.key(seed + 1 + si)
        row = [
            np.asarray(
                sampler._steer_jit(sampler.params, steer_rng, obs_pre, feats_j, prop_j, ad, float(al), paired, k, c)[0],
                np.float32,
            )
            for al in alphas
        ]
        chunks.append(np.stack(row))
    chunks = np.stack(chunks)  # [S, A, H, AD]

    def q_of(chunk_norm):
        a = jnp.asarray(chunk_norm)[None, : sampler.critic_horizon, :ad] * k + c
        return float(np.asarray(sampler._q(feats_j, a, prop_j, reduce="pess"))[0])

    q = np.array([[q_of(ch) for ch in row] for row in chunks], np.float32)  # [S, A]

    # The demonstrator's continuation, put through the POLICY's normalization so it lands in the
    # same space as the draws. Its Q is the anchor: an action known to have reached the goal.
    pre = wrapper._pol._input_transform
    dn = np.asarray(pre({**obs, "actions": data_chunk})["actions"], np.float32)
    return Sweep(
        alphas=list(map(float, alphas)),
        chunks=chunks,
        bc=bc,
        data=dn,
        q=q,
        q_data=q_of(dn),
        sigma=float(bc.std(axis=0).mean()),
        pc_sigma=float(
            np.linalg.svd(bc.reshape(len(bc), -1) - bc.reshape(len(bc), -1).mean(0), compute_uv=False)[0]
            / max(len(bc) - 1, 1) ** 0.5
        ),
        episode=-1,
        frame=-1,
    )


def rms_sigma(delta: np.ndarray, sigma: float) -> float:
    """Displacement in units of the BC cloud's PER-COORDINATE spread.

    A raw ``norm(delta)`` over a 30x14 chunk divided by a per-coordinate sigma inflates the number
    by sqrt(420) ~ 20.5, which reads as hundreds of sigma when each coordinate moved a few. Root
    MEAN square, so one unit here means "every coordinate moved by one BC sigma".
    """
    return float(np.sqrt(np.mean(np.square(delta))) / max(sigma, 1e-9))


def figure(sw: Sweep, out: pathlib.Path, *, critic_name: str, joints: int = 6) -> None:
    from matplotlib import cm
    from matplotlib import colors as mcolors
    import matplotlib.pyplot as plt
    from plot_style import GRAY
    from plot_style import apply

    apply()
    a = np.asarray(sw.alphas)
    ch = sw.chunks  # [S, A, H, AD]
    ns, na, h, _ = ch.shape
    ref = ch[0]  # the noise shown joint-by-joint
    norm = mcolors.Normalize(a.min(), a.max())
    cmap = cm.viridis
    col = [cmap(norm(x)) for x in a]

    fig = plt.figure(figsize=(19.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, 1.0, 1.0, 1.0], wspace=0.32)

    # --- 1. the chunk itself, per joint -------------------------------------------------------
    sub = gs[0].subgridspec(2, joints // 2, wspace=0.38, hspace=0.55)
    moved = sorted(np.argsort(-np.abs(ref[-1] - ref[0]).max(axis=0))[:joints])
    for i, j in enumerate(moved):
        ax = fig.add_subplot(sub[i // (joints // 2), i % (joints // 2)])
        for ai, cc in enumerate(col):
            ax.plot(np.arange(h), ref[ai, :, j], color=cc, lw=1.4)
        ax.plot(np.arange(h), sw.data[:, j], color="k", lw=1.2, ls="--", zorder=5)
        ax.axhline(1, color=GRAY, ls=":", lw=0.8)
        ax.axhline(-1, color=GRAY, ls=":", lw=0.8)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.tick_params(labelsize=7)
        if i >= joints // 2:
            ax.set_xlabel("step", fontsize=8)
    fig.text(0.10, 1.015, "the chunk, joint by joint (one noise)", fontsize=11)
    fig.text(0.10, 0.955, "dashed = the demonstrator   ·   dotted = the normalized box", fontsize=8, color=GRAY)

    # --- 2. where in the chunk it bends -------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    for ai, cc in enumerate(col):
        per = np.stack([[rms_sigma(ch[si, ai, t] - ch[si, 0, t], sw.sigma) for t in range(h)] for si in range(ns)])
        m = per.mean(0)
        ax.plot(np.arange(h), m, color=cc, lw=1.7)
        if ns > 1:
            ax.fill_between(np.arange(h), per.min(0), per.max(0), color=cc, alpha=0.16, lw=0)
    ax.set_xlabel("step in the chunk")
    ax.set_ylabel("displacement from this noise's\n$\\alpha{=}0$ draw (BC $\\sigma$ per coordinate)")
    ax.axhline(1, color=GRAY, ls="--", lw=1.0)
    ax.text(h - 1, 1, "the policy's own spread ($1\\sigma$)  ", ha="right", va="bottom", fontsize=8, color=GRAY)
    ax.set_title("where steering bites", fontsize=11)

    # --- 3. the policy's own plane ------------------------------------------------------------
    ax = fig.add_subplot(gs[2])
    flat_bc = sw.bc.reshape(len(sw.bc), -1)
    mu = flat_bc.mean(0)
    _u, _s, vt = np.linalg.svd(flat_bc - mu, full_matrices=False)
    basis = np.stack([vt[0], vt[1]])

    def proj(x):
        return (x.reshape(len(x), -1) - mu) @ basis.T

    p_bc = proj(sw.bc)
    ax.scatter(p_bc[:, 0], p_bc[:, 1], s=16, color=GRAY, alpha=0.5, lw=0, label="BC draws")
    for si in range(ns):
        p = proj(ch[si])
        ax.plot(p[:, 0], p[:, 1], color="k", lw=0.9, alpha=0.35 if si else 1.0, zorder=3)
        ax.scatter(
            p[:, 0],
            p[:, 1],
            c=a,
            cmap=cmap,
            norm=norm,
            s=40 if si == 0 else 20,
            zorder=4,
            edgecolor="k",
            linewidth=0.35,
            alpha=1.0 if si == 0 else 0.6,
        )
    p_d = proj(sw.data[None])
    ax.scatter(p_d[:, 0], p_d[:, 1], marker="*", s=170, color="crimson", zorder=5, label="demonstrator")
    ax.set_xlabel("PC1 of the BC draws")
    ax.set_ylabel("PC2")
    ax.set_title("does it leave the policy's support?", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="best")

    # --- 4. what it bought --------------------------------------------------------------------
    ax = fig.add_subplot(gs[3])
    qm = sw.q.mean(0)
    ax.plot(a, qm, color="k", lw=1.5, zorder=2)
    if ns > 1:
        ax.fill_between(a, sw.q.min(0), sw.q.max(0), color=GRAY, alpha=0.25, lw=0)
    ax.scatter(a, qm, c=a, cmap=cmap, norm=norm, s=44, zorder=3, edgecolor="k", linewidth=0.4)
    ax.axhline(sw.q_data, color="crimson", ls="--", lw=1.3)
    ax.text(a.max(), sw.q_data, " demonstrator ", color="crimson", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("guidance strength $\\alpha$")
    ax.set_ylabel("critic's pessimistic $Q$")
    ax.set_title("what the climb bought", fontsize=11)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.011, pad=0.012)
    cb.set_label("$\\alpha$", fontsize=9)
    fig.suptitle(
        f"one noise draw, steered harder and harder   ·   episode {sw.episode} frame {sw.frame}   ·   "
        f"{critic_name}   ·   {ns} noises",
        fontsize=12,
        y=1.08,
    )
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print("wrote", out)


def animation(sw: Sweep, out: pathlib.Path, *, critic_name: str, joints: int = 6, fps: int = 6) -> None:
    """The same chunk, one frame per alpha, so the bend is watched rather than read."""
    from matplotlib import cm
    from matplotlib import colors as mcolors
    import matplotlib.animation as manim
    import matplotlib.pyplot as plt
    from plot_style import GRAY
    from plot_style import apply

    apply()
    a = np.asarray(sw.alphas)
    ch = sw.chunks[0]  # the animation follows one noise; the ladder is what moves
    h = ch.shape[1]
    norm = mcolors.Normalize(a.min(), a.max())
    cmap = cm.viridis
    moved = sorted(np.argsort(-np.abs(ch[-1] - ch[0]).max(axis=0))[:joints])

    fig, axes = plt.subplots(2, joints // 2, figsize=(13, 5.0))
    axes = axes.ravel()
    lines, ghosts = [], []
    for ax, j in zip(axes, moved, strict=True):
        lo = min(ch[:, :, j].min(), sw.data[:, j].min())
        hi = max(ch[:, :, j].max(), sw.data[:, j].max())
        pad = 0.12 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, h - 1)
        ax.plot(np.arange(h), sw.data[:, j], color="k", lw=1.2, ls="--")
        (g,) = ax.plot(np.arange(h), ch[0, :, j], color=GRAY, lw=1.2)  # the alpha=0 draw, always shown
        (ln,) = ax.plot([], [], lw=2.2)
        ax.axhline(1, color=GRAY, ls=":", lw=0.8)
        ax.axhline(-1, color=GRAY, ls=":", lw=0.8)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.tick_params(labelsize=7)
        lines.append(ln)
        ghosts.append(g)
    title = fig.suptitle("", fontsize=13)

    def draw(i):
        for ln, j in zip(lines, moved, strict=True):
            ln.set_data(np.arange(h), ch[i, :, j])
            ln.set_color(cmap(norm(a[i])))
        d = rms_sigma(ch[i] - ch[0], sw.sigma)
        title.set_text(
            f"$\\alpha$ = {a[i]:.2f}     every coordinate moved {d:.1f} BC $\\sigma$ from the same noise's draw"
            f"     Q {sw.q[0, i]:+.1f} (demonstrator {sw.q_data:+.1f})"
        )
        return [*lines, title]

    order = list(range(len(a))) + list(range(len(a) - 2, 0, -1))  # up the ladder and back down
    anim = manim.FuncAnimation(fig, draw, frames=order, interval=1000 // fps, blit=False)
    anim.save(out, writer=manim.PillowWriter(fps=fps))
    print("wrote", out)


def main(a):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from dataset_reader import DatasetReader
    from dataset_reader import SequentialImages

    from openpi.policies import patch_critic_policy as pcp
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config
    from openpi.training import outcomes as _outcomes

    cfg = _config.get_config(a.policy_config)
    policy = _pc.create_trained_policy(cfg, pathlib.Path(a.policy_dir))
    wrapper = pcp.PatchCriticSelectPolicy(policy, a.critic, mode="qpilots", steer_alpha=a.alphas[0])
    sampler = wrapper._arm_sampler
    if sampler is None:
        raise SystemExit("no arm sampler: the wrapper was not built in a steering mode")

    ds = pathlib.Path(a.dataset)
    reader = DatasetReader(ds.name, str(ds.parent))
    reader.load()
    verdicts = _outcomes.episode_outcomes(ds)
    if verdicts is None:
        raise SystemExit(f"{ds} has no next.success / next.done; migrate it first")
    ok = sorted(e for e, v in verdicts.items() if v == "success")

    ep = a.episode if a.episode is not None else ok[len(ok) // 2]
    if ep not in ok:
        raise SystemExit(f"episode {ep} is not a success episode; the demonstrator anchor needs one")
    n = reader.episode_length(ep)
    acts = reader.column(ep, "action")
    states = reader.column(ep, "observation.state")
    fr = a.frame if a.frame is not None else int(0.45 * (min(int(n), acts.shape[0]) - a.horizon - 1))
    seq = SequentialImages(str(ds), ep, cameras=list(CAMERAS.values()))
    obs = _obs_dict(seq.frame(fr), states[fr], reader.task_of(ep) if hasattr(reader, "task_of") else "")
    if obs is None:
        raise SystemExit(f"no images/state at episode {ep} frame {fr}")
    logging.info("episode %d frame %d of %d", ep, fr, n)

    sw = measure(
        wrapper,
        sampler,
        obs,
        np.asarray(acts[fr : fr + a.horizon], np.float32),
        alphas=a.alphas,
        n_bc=a.n_bc,
        flow_steps=a.flow_steps,
        seed=a.seed,
        n_noise=a.n_noise,
    )
    sw.episode, sw.frame = ep, fr
    for ai, al in enumerate(sw.alphas):
        d = np.array([rms_sigma(sw.chunks[si, ai] - sw.chunks[si, 0], sw.sigma) for si in range(len(sw.chunks))])
        oob = np.array(
            [float(((sw.chunks[si, ai] < -1) | (sw.chunks[si, ai] > 1)).mean()) for si in range(len(sw.chunks))]
        )
        logging.info(
            "alpha %5.2f   displaced %5.2f-%5.2f sigma/coord   Q %+8.2f   %4.1f%% of entries outside the box",
            al,
            d.min(),
            d.max(),
            sw.q[:, ai].mean(),
            100 * oob.mean(),
        )
    logging.info(
        "demonstrator Q %+8.2f   BC sigma %.4f per coordinate (%.3f along PC1)",
        sw.q_data,
        sw.sigma,
        sw.pc_sigma,
    )

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = pathlib.Path(a.critic).name
    figure(sw, out, critic_name=name, joints=a.joints)
    if a.gif:
        animation(sw, pathlib.Path(a.gif), critic_name=name, joints=a.joints, fps=a.fps)
    if a.json:
        pathlib.Path(a.json).write_text(
            json.dumps(
                {
                    "episode": ep,
                    "frame": fr,
                    "critic": name,
                    "alphas": sw.alphas,
                    "n_noise": int(len(sw.chunks)),
                    "q": sw.q.tolist(),
                    "q_data": sw.q_data,
                    "sigma_per_coordinate": sw.sigma,
                    "sigma_pc1": sw.pc_sigma,
                    "displacement_sigma_per_coordinate": [
                        [rms_sigma(sw.chunks[si, ai] - sw.chunks[si, 0], sw.sigma) for ai in range(len(sw.alphas))]
                        for si in range(len(sw.chunks))
                    ],
                    "frac_outside_box": [
                        [
                            float(((sw.chunks[si, ai] < -1) | (sw.chunks[si, ai] > 1)).mean())
                            for ai in range(len(sw.alphas))
                        ]
                        for si in range(len(sw.chunks))
                    ],
                },
                indent=2,
            )
        )
        print("wrote", a.json)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(pathlib.Path.home() / "lerobot_data/yam_lego_taxi"))
    ap.add_argument("--critic", required=True)
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--policy-config", default="pi05_yam_lego_taxi")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    ap.add_argument("--n-bc", type=int, default=32)
    ap.add_argument("--n-noise", type=int, default=4, help="independent noise draws; the ladder is run from each")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--joints", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="guidance_sweep.png")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--json", default=None)
    main(ap.parse_args())
