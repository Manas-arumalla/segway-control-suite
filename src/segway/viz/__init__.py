"""Visualization: a unified publication theme, trajectory/navigation/terrain/benchmark plots,
and rendered 3D rollouts (GIF/MP4)."""

from __future__ import annotations

from .live import live_view, live_view_navigation
from .nav_plots import (
    plot_benchmark_summary,
    plot_nav_analysis,
    plot_navigation,
    plot_rl_analysis,
    plot_terrain,
    plot_terrain_run,
)
from .plots import plot_comparison, plot_roa, plot_trajectory
from .render import render_navigation, render_rollout
from .style import CYCLE, PALETTE, apply_paper_style

__all__ = [
    "plot_trajectory",
    "plot_comparison",
    "plot_roa",
    "render_rollout",
    "render_navigation",
    "live_view",
    "live_view_navigation",
    "apply_paper_style",
    "PALETTE",
    "CYCLE",
    "plot_navigation",
    "plot_nav_analysis",
    "plot_terrain",
    "plot_terrain_run",
    "plot_benchmark_summary",
    "plot_rl_analysis",
]
