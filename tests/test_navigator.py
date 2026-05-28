"""Composable navigator (NAV-5): plan + balance-and-drive to a goal around obstacles."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import SimConfig, TWIPParams
from segway.navigation import Navigator, NavResult, Obstacle, World, navigate


def _corridor_world():
    """An 8x4 corridor with one obstacle squarely between start and goal."""
    return World(width=8.0, height=4.0, resolution=0.2, robot_radius=0.25,
                 obstacles=[Obstacle(4.0, 2.0, 0.6)])


_START = (1.0, 2.0)
_GOAL = (7.0, 2.0)
# Modest rollout so the test suite stays quick; early-stop ends it once the goal is hit.
_SIM = SimConfig(dt=0.005, duration=30.0, fall_angle=1.2)


def test_navigate_reaches_goal_around_obstacle():
    res = navigate(TWIPParams(), _corridor_world(), _START, _GOAL,
                   balance="lqr", planner="a_star", follower="pure_pursuit", sim=_SIM)
    assert isinstance(res, NavResult)
    assert res.planned and res.path is not None
    assert res.success, f"did not reach (reached={res.reached}, fell={res.fell})"
    assert res.final_goal_distance < 0.35
    assert not res.trajectory.fell


def test_navigation_keeps_clearance():
    res = navigate(TWIPParams(), _corridor_world(), _START, _GOAL, sim=_SIM)
    # The driven path may clip a corner slightly, but must not drive into an obstacle.
    assert res.min_clearance > -0.2, f"min clearance {res.min_clearance:.3f} m"


def test_planning_failure_when_goal_blocked():
    world = World(width=8.0, height=4.0, resolution=0.2, robot_radius=0.25,
                  obstacles=[Obstacle(7.0, 2.0, 0.8)])   # goal sits inside an obstacle
    res = navigate(TWIPParams(), world, _START, _GOAL, sim=_SIM)
    assert not res.planned
    assert res.path is None and res.trajectory is None
    assert not res.success


@pytest.mark.parametrize(
    ("planner", "follower"),
    [("a_star", "pure_pursuit"), ("dijkstra", "stanley"),
     ("rrt_star", "vector_field"), ("a_star", "mpc")],
)
def test_planner_follower_combinations_reach_goal(planner, follower):
    res = navigate(TWIPParams(), _corridor_world(), _START, _GOAL,
                   balance="lqr", planner=planner, follower=follower, sim=_SIM)
    assert res.planned
    assert res.success, f"{planner}+{follower} failed (fell={res.fell})"


def test_navigator_reuse_across_goals():
    nav = Navigator(TWIPParams(), balance="lqr", planner="a_star", follower="pure_pursuit")
    world = _corridor_world()
    r1 = nav.run(world, _START, _GOAL, sim=_SIM)
    r2 = nav.run(world, _GOAL, _START, sim=_SIM)   # drive back the other way
    assert r1.success and r2.success
    assert r1.path_length > np.hypot(_GOAL[0] - _START[0], _GOAL[1] - _START[1])


def test_metrics_are_populated():
    res = navigate(TWIPParams(), _corridor_world(), _START, _GOAL, sim=_SIM)
    assert res.path_length > 5.0           # has to detour around the obstacle
    assert np.isfinite(res.time_to_goal) and res.time_to_goal > 0.0
    assert res.driven_path.shape[1] == 2


def test_plot_navigation_renders_headless():
    pytest.importorskip("matplotlib")  # the viz extra
    import matplotlib
    matplotlib.use("Agg")
    from segway.navigation.plot import plot_navigation

    res = navigate(TWIPParams(), _corridor_world(), _START, _GOAL, sim=_SIM)
    ax = plot_navigation(res)
    assert ax.get_title() != ""
    assert len(ax.lines) >= 2   # planned + driven at least
