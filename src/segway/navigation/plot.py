"""Backward-compatible re-export of the top-down navigation plot.

The navigation/terrain/benchmark figures now live in :mod:`segway.viz.nav_plots` (on the shared
publication theme). This module keeps ``segway.navigation.plot.plot_navigation`` working.
"""

from __future__ import annotations

from ..viz.nav_plots import plot_navigation

__all__ = ["plot_navigation"]
