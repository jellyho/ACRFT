"""Shared plot style for experiment reports on the acrft-reports Space.

The Space's master_report.html (worker-B) sets the house style — light background,
despined axes, muted categorical palette, sentence-case English titles, statistical
annotations inside the axis label ("30 paired trials; filled dot = McNemar p < 0.05").
Import and call `use()` before plotting so every report's figures match.

Conventions beyond rcParams:
- Significance on dot/lollipop plots: open circle = n.s., filled = significant.
- Small multiples share the y axis; thin lines = individual runs, bold = group median.
- Reference levels are dashed gray lines, explained by gray text INSIDE the axes
  (e.g. "dashed = own vla median"), not in a legend.
- Titles and labels are English, sentence case, and say what the reader should
  conclude, not just what is plotted.
- Export: fig.savefig(..., dpi=110, bbox_inches="tight", facecolor="white"), embed
  as jpeg quality ~80 to keep the single-file HTML small.
"""

import matplotlib as mpl

# seaborn "deep"-adjacent muted palette, in the order worker-B uses them:
# red for the incumbent/failing family, green for IQL, blue for high-gamma, purple for dueling.
RED = "#c44e52"
GREEN = "#2e8b6d"
BLUE = "#4c72b0"
PURPLE = "#8172b3"
ORANGE = "#dd8452"
GRAY = "#555555"
LIGHTGRAY = "#aaaaaa"
PALETTE = [RED, GREEN, BLUE, PURPLE, ORANGE, GRAY]


def use():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1c1917",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "xtick.color": "#1c1917",
        "ytick.color": "#1c1917",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "legend.frameon": False,
        "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })
