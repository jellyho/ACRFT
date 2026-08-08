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

# seaborn "deep" — the palette Seohong Park's papers (HILP/METRA/OGBench) draw from.
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"
BROWN = "#937860"
GRAY = "#555555"
LIGHTGRAY = "#aaaaaa"
PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN]


def use():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1c1917",
        # Titles are labels, not sentences: keep them one or two words; every explanation,
        # reading instruction, and statistical caveat belongs in the surrounding prose.
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "xtick.color": "#1c1917",
        "ytick.color": "#1c1917",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        # Computer Modern text — the LaTeX look of the reference papers. cmr10 has no unicode
        # minus, so unicode_minus must be off or negative ticks render as boxes.
        "font.family": "serif",
        "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })
