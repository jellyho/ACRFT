"""Visualize what CFAC does differently, on one matched PlanReach episode.

Four panels, all from the same scene (same latent directions, same seed), so the two critics are
compared on identical situations:

  (1,2) the executed path, integrating the 2-D actions into positions. Grey arrows are the target
        direction of each segment; the coloured path is what the policy actually did, and a marker
        sits at every point where the system re-queried the policy.
  (3)   the commitment timeline. Each bar is one query, its width the number of actions committed.
        The junction reveal is drawn as a vertical line: committing across it means acting before
        the information arrives.
  (4)   the critic's prefix-value profile at the two decision types. This is the mechanism: at a
        junction entry a confounded critic still ranks long commitments highest, while CFAC's
        profile peaks at k=1.

    python slurm/probes/toy_cfac_viz.py --out /scratch/jellyho/acrft/hub_figs/toy_cfac_viz.png
"""

import argparse
import pathlib
import sys

sys.path.insert(0, "slurm")
sys.path.insert(0, "slurm/probes")

import matplotlib.pyplot as plt
import numpy as np
from plot_style import PALETTE as COLORS
from plot_style import apply as apply_style
import torch
import toy_cfac_nn as T  # noqa: N812


@torch.no_grad()
def trace(seed, pi, q, n_ep=1, start=0):
    """Replay episodes, recording every decision: where it queried, how long it committed, the
    prefix values it saw, and the path it produced."""
    rng = np.random.default_rng(seed)
    env = T.PlanReach(rng)
    out = []
    for _ep in range(start + n_ep):
        o = env.reset()
        hist, pos, path, decisions = [], np.zeros(2), [np.zeros(2)], []
        while True:
            seg, step = env._seg_step()
            if step == 0:
                hist = []
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            ht = torch.as_tensor(T.hist_feat(hist), dtype=torch.float32).unsqueeze(0)
            c = pi.sample(ot)
            qv = q(ot, ht, c)[0].numpy()
            k = int(T.select_k(q(ot, ht, c))[0].item())
            decisions.append(
                {"t": env.t, "seg": T.SEGS[seg], "step": step, "k": k, "q": qv.copy(), "g": env.g[seg].copy()}
            )
            acts = c.view(T.H, T.ADIM).numpy()
            done = False
            for j in range(min(k, T.T - env.t)):
                a = np.clip(acts[j], -1, 1)
                _o2, _r, done = env.step(acts[j])
                pos = pos + a
                path.append(pos.copy())
                hist = [] if env.t % T.H == 0 else [*hist, a]
                o = _o2
                if done:
                    break
            if done:
                break
        out.append({"path": np.array(path), "decisions": decisions, "g": env.g.copy()})
    return out[start:]


@torch.no_grad()
def probe_states(seed, pi, critics, episode=0):
    """Walk one episode with a fixed rule (full commitment), stop at a corridor entry and a junction
    entry, and score one shared chunk with every critic."""
    rng = np.random.default_rng(seed)
    env = T.PlanReach(rng)
    for _ in range(episode + 1):
        o = env.reset()
    out = {}
    hist = []
    while True:
        seg, step = env._seg_step()
        if step == 0:
            hist = []
        ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
        ht = torch.as_tensor(T.hist_feat(hist), dtype=torch.float32).unsqueeze(0)
        c = pi.sample(ot)  # one chunk, shared by every critic
        if step == 0 and T.SEGS[seg] not in out:
            out[T.SEGS[seg]] = {
                "q": [q(ot, ht, c)[0].numpy() for q in critics],
                "k": [int(T.select_k(q(ot, ht, c))[0].item()) for q in critics],
            }
        acts = c.view(T.H, T.ADIM).numpy()
        done = False
        for j in range(min(T.H, T.T - env.t)):
            a = np.clip(acts[j], -1, 1)
            o, _r, done = env.step(acts[j])
            hist = [] if env.t % T.H == 0 else [*hist, a]
            if done:
                break
        if done or len(out) == 2:
            break
    return out


def draw_path(ax, tr, title, color):
    path, g = tr["path"], tr["g"]
    # ideal direction of each segment, anchored where that segment starts
    for i, seg in enumerate(T.SEGS):
        p0 = path[min(i * T.H, len(path) - 1)]
        ax.arrow(*p0, *(g[i] * 2.2), width=0.04, color="0.75", length_includes_head=True, head_width=0.22, zorder=1)
        ax.text(*(p0 + g[i] * 2.5), "corridor" if seg == "C" else "junction", fontsize=6, color="0.45", ha="center")
    ax.plot(path[:, 0], path[:, 1], "-", color=color, lw=1.8, zorder=3)
    qpts = np.array([path[min(d["t"], len(path) - 1)] for d in tr["decisions"]])
    ax.plot(qpts[:, 0], qpts[:, 1], "o", ms=5, mfc="white", mec=color, mew=1.6, zorder=4, label="re-query")
    ax.plot(path[0, 0], path[0, 1], "s", ms=6, color="0.3", zorder=5, label="start")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--episode-index", type=int, default=3)
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/hub_figs/toy_cfac_viz.png")
    )
    a = ap.parse_args()
    torch.set_num_threads(4)

    rng = np.random.default_rng(a.seed)
    data = T.gen_demos(rng, a.episodes)
    pi = T.train_bc(data, a.steps, a.seed)
    vnet = T.dataset_v(data, a.steps, a.seed)
    q_naive = T.train_naive_critic(data, vnet, a.steps, a.seed)
    q_cfac = T.train_cfac_critic(data, pi, a.steps, a.seed, use_hist=True, interventional=True)

    tr_n = trace(7_000 + a.seed, pi, q_naive, start=a.episode_index)[0]
    tr_c = trace(7_000 + a.seed, pi, q_cfac, start=a.episode_index)[0]
    # Counterfactual probe: the two critics are asked about the SAME decision, with the same chunk,
    # at a corridor entry and at a junction entry. Without this, an arm that committed across the
    # junction never queried there and would simply be missing from the comparison.
    probe = probe_states(7_000 + a.seed, pi, [q_naive, q_cfac], episode=a.episode_index)

    apply_style()
    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.3)

    ax_n = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1], sharex=ax_n, sharey=ax_n)
    draw_path(ax_n, tr_n, "naive critic: what it executed", COLORS[3])
    draw_path(ax_c, tr_c, "CFAC: what it executed", COLORS[2])
    both = np.vstack([tr_n["path"], tr_c["path"]])
    pad = 1.0
    ax_n.set_xlim(both[:, 0].min() - pad, both[:, 0].max() + pad)
    ax_n.set_ylim(both[:, 1].min() - pad, both[:, 1].max() + pad)

    # (3) commitment timeline
    ax = fig.add_subplot(gs[0, 2])
    for row, (tr, col) in enumerate([(tr_n, COLORS[3]), (tr_c, COLORS[2])]):
        for d in tr["decisions"]:
            ax.barh(row, min(d["k"], T.T - d["t"]), left=d["t"], height=0.55, color=col, edgecolor="white", lw=1.2)
            ax.text(d["t"] + 0.12, row, str(d["k"]), va="center", fontsize=6.5, color="white")
    for x in range(T.H, T.T, T.H):
        ax.axvline(x, color="0.8", lw=0.8)
    reveal = T.H + 1  # the junction's event becomes visible one step after the segment starts
    ax.axvline(reveal, color=COLORS[1], ls="--", lw=1.4, label="junction reveal")
    ax.set_yticks([0, 1], ["naive", "CFAC"])
    ax.set_xlabel("environment step")
    ax.set_title("commitments (bar width = k)")
    ax.legend(fontsize=6.5, loc="lower right")

    # (4) prefix-value profiles at the two decision types
    ks = np.arange(1, T.H + 1)
    for col, (kind, label) in enumerate(
        [("C", "corridor entry (commitment is right)"), ("J", "junction entry (reaction is right)")]
    ):
        ax = fig.add_subplot(gs[1, col])
        pr = probe.get(kind)
        if pr is None:
            ax.axis("off")
            continue
        truth = T.H if kind == "C" else 1
        ax.axvline(truth, color="0.8", lw=6, alpha=0.5, zorder=0)
        ax.text(truth, 1.04, "right answer", fontsize=6, color="0.45", ha="center")
        for idx, (name, c) in enumerate([("naive", COLORS[3]), ("CFAC", COLORS[2])]):
            v = pr["q"][idx]
            v = (v - v.min()) / (np.ptp(v) + 1e-9)  # shapes are what matter, not levels
            ax.plot(ks, v, "-o", color=c, ms=5, label=f"{name} picks k={pr['k'][idx]}")
        ax.set_xticks(ks)
        ax.set_xlabel("commitment length k")
        ax.set_ylabel("prefix value (rescaled)")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "How to read this\n\n"
        "Corridor: the plan is visible only when the segment starts, so\n"
        "the actions that follow carry information the observation no\n"
        "longer shows. Committing keeps that plan; re-querying throws\n"
        "it away and the Markov policy guesses.\n\n"
        "Junction: the event is revealed one step in. Committing across\n"
        "the dashed line acts before the information arrives.\n\n"
        "The bottom-left profile is where the two critics agree, and the\n"
        "bottom-middle one is where they do not: the confounded critic\n"
        "still ranks long commitments highest at the junction, because\n"
        "in the demonstrations the person already knew the event when\n"
        "choosing those actions.",
        va="top",
        ha="left",
        fontsize=7.2,
        linespacing=1.5,
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
