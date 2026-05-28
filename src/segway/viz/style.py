"""A unified, publication-grade matplotlib theme shared across every figure in the suite.

Applying one consistent style — refined typography (Computer-Modern mathtext for symbols),
light dotted grids, no top/right spines, a cohesive colourblind-aware palette, and high DPI —
makes the plots read like figures from a paper rather than ad-hoc script output. The theme is
applied once when :mod:`segway.viz` is imported; call :func:`apply_paper_style` again to
re-assert it (e.g. after a UI switches backends).
"""

from __future__ import annotations

import matplotlib as mpl

# Cohesive, colourblind-aware palette used throughout the suite.
PALETTE = {
    "planned": "#3b6fb6",     # planned global path
    "driven": "#2a9d8f",      # driven trajectory (success)
    "driven_bad": "#e63946",  # driven trajectory (failure)
    "start": "#111111",
    "goal": "#e8862e",
    "obstacle": "#5a5a66",
    "accent": "#00838f",
    "muted": "#8a8a8a",
    "grid": "#b9b9b9",
}

# Categorical cycle for multi-series comparison plots (planners, followers, controllers).
CYCLE = [
    "#1f77b4", "#d1495b", "#2a9d8f", "#e8862e", "#7b6cae",
    "#8c6d31", "#e377c2", "#17becf", "#9aa033", "#555555",
]

# A perceptually-uniform sequential map for heightfields / heatmaps.
SEQ_CMAP = "viridis"

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.facecolor": "#fbfbfd",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
    "grid.color": PALETTE["grid"],
    "grid.linestyle": ":",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.6,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "cm",   # Computer-Modern look for θ, τ, ψ … (ships with matplotlib)
}


def apply_paper_style() -> None:
    """Apply the publication theme to the global matplotlib rcParams."""
    mpl.rcParams.update(_RC)


apply_paper_style()
