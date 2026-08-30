"""Figure for the acrft_ogbench experiment-audit synthesis (house style, report_style).

One data figure: on the SAME task (cube-double-task5, offline 1M), the success rate swings from 0.01
to 0.77 depending on a single flag (BoN vs DDPG, target critic on/off, alpha). The point is fragility:
most single-config conclusions in the thread rest on numbers this sensitive. Values are read from the
acrft_ogbench exps notes (0803_bon_vs_ddpg, 0803_prior_sweeps, 0807_prefix_bias, 0807_target_critic).

    uv run python scripts/ogbench_audit_figs.py --out space_v2/figs
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import report_style as rs

# cube-double-play-singletask-task5-v0, offline 1M, 3 seeds (median / point estimates from the notes).
BARS = [
    ("BoN N=4\n(no target)", 0.767, rs.BLUE),
    ("BoN N=4\ntarget-on", 0.320, rs.TEAL),
    ("DDPG (aqc)\nα=300", 0.307, rs.ORANGE),
    ("DDPG\nα=100", 0.013, rs.RED),
    ("DDPG\nα=900", 0.233, rs.PURPLE),
]
QC_MLP_REF = 0.40  # QC-FQL fixed-chunk, MLP critic (0731_inference_viz reference)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("space_v2/figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rs.use()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(BARS))
    ax.bar(x, [b[1] for b in BARS], width=0.62, color=[b[2] for b in BARS], alpha=0.9)
    for i, (_, v, _c) in enumerate(BARS):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    rs.baseline(ax, QC_MLP_REF, "QC-FQL (MLP critic) ref 0.40")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in BARS], fontsize=8.5)
    ax.set_ylabel("offline success rate")
    ax.set_ylim(0, 0.85)
    ax.set_title("cube-double task5 — one flag swings the same family 0.01 → 0.77", loc="left")
    rs.save(fig, a.out / "ogbench_fragility.png")
    print("wrote", a.out / "ogbench_fragility.png")


if __name__ == "__main__":
    main()
