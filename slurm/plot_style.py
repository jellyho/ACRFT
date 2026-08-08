"""Paper-grade matplotlib style, following the visual conventions of Seohong Park's papers
(OGBench / HIQL / horizon-reduction figures): white background, y-grid only, no top/right
spines, seaborn-deep solid palette with shaded CI bands, frameless legends, bold sans titles.

    from plot_style import apply, PALETTE
    apply()
"""

import matplotlib as mpl

# seaborn "deep" — the solid, print-safe palette his figures use
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]
GRAY = "#444444"


def apply():
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 11,
            # 타이틀은 간결하게(정보는 본문 산문으로) — regular weight, 본문보다 약간 큰 정도
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.titlepad": 10,
            "axes.labelsize": 11.5,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": "#1a1a1a",
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "errorbar.capsize": 3,
        }
    )


def band(ax, x, mean, lo, hi, color, label=None):
    """Mean line + shaded CI band, the standard series style of these papers."""
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
