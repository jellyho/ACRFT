"""What the critic thinks of the actions the policy would actually take, and of the ones near them.

The other landscape figure is spanned by grad_a Q and something orthogonal to it. That answers
"what does moving along the exploit direction cost", but it draws the BC cloud as a single dot,
because the cloud is small and the plane is two units wide. This figure is the same field seen
from the policy's side:

  the plane      the top two principal components OF THE DRAWS -- the two directions this policy
                 is actually uncertain about at this state, so the cloud fills the frame
  the dots       the 16 chunks the BC policy actually proposed
  the star       what the demonstrator did (a successful episode, so it reached the goal)
  the arrow      grad_a Q at the cloud's centre -- the direction steering pushes
  the colour     Q, relative to the demonstrator's own action

Two zooms, and the PAIR is the argument:

  left   +/- 4 sigma of the cloud       everything best-of-N can reach. It picks a DRAW, so it
                                        never leaves this frame however large N is.
  right  the full normalized box        everything steering can reach. The same cloud is the
                                        small circle near the middle.

If the colour keeps getting hotter as you leave the cloud, the critic prefers actions the data
never took -- and the arrow points that way by construction.
"""

import argparse
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "slurm"))


def _field(rows, key, n):
    """Frame-averaged Q on a grid, each frame centred on its own demonstrator value first.

    Averaging raw Q would be meaningless -- frames differ by hundreds of cost-to-goal steps
    depending on how far along the episode they are. Relative to each frame's own anchor they are
    the same quantity: how much better than the demonstrator this action looks.
    """
    q = np.asarray([r[f"q_{key}"] for r in rows], np.float32).mean(axis=1)  # [F, n*n] ensemble mean
    anchor = np.asarray([r["q_data"] for r in rows], np.float32)[..., 0].mean(axis=1)[:, None]
    return (q - anchor).mean(axis=0).reshape(n, n).T


def main(a):
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
    from plot_style import apply

    apply()
    pp = pathlib.Path(a.probe)
    d = json.loads(gzip.decompress(pp.read_bytes()) if pp.suffix == ".gz" else pp.read_text())
    name = a.critic_name or d["critics"][0]
    rows = [r["critics"][name] for r in d["rows"] if name in r["critics"]]
    n = int(round(len(rows[0]["q_near"][0]) ** 0.5))
    print(f"critic: {name}  ({len(rows)} frames, {n}x{n} grid)")

    sig = np.asarray([r["pc_sigma"] for r in rows], np.float32).mean(axis=0)
    varf = np.asarray([r["pc_var_frac"] for r in rows], np.float32).mean(axis=0)
    near, far = _field(rows, "near", n), _field(rows, "far", n)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lim = max(np.abs(near).max(), 1e-6)

    # ---- left: the cloud's own scale ----------------------------------------------------------
    ax = axes[0]
    ts = np.linspace(-4, 4, n)  # in units of each component's own sigma
    im = ax.pcolormesh(ts, ts, near, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    ax.contour(ts, ts, near, levels=7, colors="k", linewidths=0.4, alpha=0.35)
    # the draws, in the same units the axes are in
    pb = np.concatenate([np.asarray(r["proj_bc"], np.float32) / np.asarray(r["pc_sigma"], np.float32) for r in rows])
    ax.scatter(pb[:, 0], pb[:, 1], s=9, c="k", alpha=0.35, lw=0, label="BC draws (what N picks among)")
    pd = np.stack([np.asarray(r["proj_data"], np.float32) / np.asarray(r["pc_sigma"], np.float32) for r in rows])
    pd = pd[(np.abs(pd) < 4).all(axis=1)]
    ax.scatter(pd[:, 0], pd[:, 1], s=70, marker="*", c="w", edgecolors="k", lw=0.8, label="demonstrator", zorder=5)
    ax.set_xlabel(f"PC1 of the draws  ({100 * varf[0]:.0f}% of their variance, σ={sig[0]:.2f})")
    ax.set_ylabel(f"PC2  ({100 * varf[1]:.0f}%, σ={sig[1]:.2f})")
    ax.set_title("what best-of-N can reach")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax).set_label("$Q - Q(\\mathrm{demonstrator})$", fontsize=9)

    # ---- right: the whole box, same plane -----------------------------------------------------
    ax = axes[1]
    ts = np.linspace(-1, 1, n)  # normalized action units
    lim2 = max(np.abs(far).max(), 1e-6)
    im = ax.pcolormesh(ts, ts, far, cmap="RdBu_r", vmin=-lim2, vmax=lim2, shading="nearest")
    ax.contour(ts, ts, far, levels=9, colors="k", linewidths=0.4, alpha=0.35)
    # the same cloud, to scale: 2 sigma along each component
    ax.add_patch(Circle((0, 0), 2 * float(sig.mean()), fill=False, color="k", lw=1.4))
    ax.annotate(
        "the BC cloud,\nto scale",
        (2 * float(sig.mean()), 0),
        (0.42, 0.30),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    g = np.stack([np.asarray(r["proj_grad"], np.float32) for r in rows]).mean(axis=0)
    g = g / max(np.linalg.norm(g), 1e-9) * 0.8
    ax.annotate("", (g[0], g[1]), (0, 0), arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "k"})
    ax.text(g[0] * 1.02, g[1] * 1.02, "  $\\nabla_a Q$\n  (where steering goes)", fontsize=8, va="center")
    ax.set_xlabel("PC1 of the draws  (normalized action units)")
    ax.set_ylabel("PC2")
    ax.set_title("what steering can reach")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax).set_label("$Q - Q(\\mathrm{demonstrator})$", fontsize=9)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)

    pathlib.Path(a.summary).write_text(
        json.dumps(
            {
                "critic": name,
                "frames": len(rows),
                "grid": n,
                "pc_sigma": sig.tolist(),
                "pc_var_frac": varf.tolist(),
                "near_range": [float(near.min()), float(near.max())],
                "far_range": [float(far.min()), float(far.max())],
                "cloud_sigma_mean": float(sig.mean()),
            },
            indent=1,
        )
    )
    print("wrote", a.summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="slurm/probes/q_landscape.json.gz")
    ap.add_argument("--critic-name", default=None)
    ap.add_argument("--out", default="hub_figs/q_cloud.png")
    ap.add_argument("--summary", default="q_cloud_summary.json")
    main(ap.parse_args())
