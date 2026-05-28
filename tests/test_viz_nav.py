"""Research-grade navigation/terrain/benchmark/RL figures (headless, Agg)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import pytest  # noqa: E402

from segway.config import SimConfig, TWIPParams  # noqa: E402
from segway.navigation import build_scenario, build_terrain, navigate  # noqa: E402
from segway.viz import (  # noqa: E402
    plot_benchmark_summary,
    plot_nav_analysis,
    plot_navigation,
    plot_rl_analysis,
    plot_terrain,
)

_SIM = SimConfig(dt=0.005, duration=25.0, fall_angle=1.2)


def _result():
    sc = build_scenario("corridor")
    return navigate(TWIPParams(), sc.world, sc.start, sc.goal, sim=_SIM)


def test_publication_style_is_applied():
    assert mpl.rcParams["mathtext.fontset"] == "cm"
    assert mpl.rcParams["axes.spines.top"] is False


def test_plot_navigation_returns_axes():
    ax = plot_navigation(_result())
    assert ax.has_data()
    assert ax.get_aspect() == 1.0   # equal aspect (round obstacles)


def test_plot_nav_analysis_multipanel(tmp_path):
    fig = plot_nav_analysis(_result(), path=tmp_path / "a.png")
    assert len(fig.axes) >= 4          # route + speed + pitch + yaw/torque
    assert (tmp_path / "a.png").exists()


def test_plot_terrain(tmp_path):
    plot_terrain(build_terrain("rough"), path=tmp_path / "t.png")
    assert (tmp_path / "t.png").exists()


def test_plot_benchmark_summary(tmp_path):
    rows = [
        {"sweep": "planner", "map": "corridor", "planner": "a_star",
         "success": True, "time_s": 8.2, "path_len_m": 6.2, "min_clearance_m": 0.0},
        {"sweep": "planner", "map": "corridor", "planner": "potential_field",
         "success": False, "time_s": float("nan"), "path_len_m": 0.0, "min_clearance_m": float("inf")},
    ]
    plot_benchmark_summary(rows, path=tmp_path / "b.png")
    assert (tmp_path / "b.png").exists()


def test_plot_rl_analysis(tmp_path):
    rows = [{"method": m, "goal": i, "success": True, "fell": False,
             "time_s": 6.0 + i, "final_dist_m": 0.33 + 0.01 * i}
            for m in ("rl", "classical") for i in range(6)]
    plot_rl_analysis(rows, path=tmp_path / "r.png")
    assert (tmp_path / "r.png").exists()


@pytest.mark.parametrize("name", ["corridor"])
def test_speed_colored_route_has_collection(name):
    fig = plot_nav_analysis(_result())
    # the route panel adds a LineCollection (speed-coloured) beyond plain Line2D
    from matplotlib.collections import LineCollection
    assert any(isinstance(c, LineCollection) for ax in fig.axes for c in ax.collections)
