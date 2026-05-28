"""Path-planner suite + occupancy-grid world (NAV-3)."""

from __future__ import annotations

import numpy as np
import pytest

from segway.navigation import Obstacle, World, build_planner, list_planners


def _path_is_valid(world, path, start, goal, tol=0.4):
    assert path is not None and len(path) >= 2
    assert np.hypot(*(path[0] - np.array(start))) < tol
    assert np.hypot(*(path[-1] - np.array(goal))) < tol
    for a, b in zip(path[:-1], path[1:], strict=False):
        assert world.is_segment_free(a, b), "path segment passes through an obstacle"


@pytest.fixture
def world() -> World:
    obstacles = [Obstacle(4.0, 3.0, 1.0), Obstacle(5.5, 6.5, 1.2), Obstacle(7.0, 4.0, 0.8)]
    return World(width=10, height=10, resolution=0.2, robot_radius=0.25, obstacles=obstacles)


def test_world_occupancy_and_free(world):
    assert world.occupancy.shape == (world.ny, world.nx)
    assert world.is_free(1.0, 1.0)            # open corner
    assert not world.is_free(4.0, 3.0)        # inside an obstacle
    assert not world.is_free(-1.0, 5.0)       # out of bounds


def test_registry_has_all_planners():
    assert set(list_planners()) >= {"a_star", "dijkstra", "rrt", "rrt_star", "prm", "potential_field"}


@pytest.mark.parametrize("name", ["a_star", "dijkstra", "rrt", "rrt_star", "prm"])
def test_planner_finds_valid_path(world, name):
    planner = build_planner(name)
    path = planner.plan(world, (1.0, 1.0), (9.0, 9.0))
    _path_is_valid(world, path, (1.0, 1.0), (9.0, 9.0))


def test_potential_field_reaches_goal():
    # Obstacle offset from the straight line so the field flows around it (no trap).
    world = World(width=10, height=10, resolution=0.2, robot_radius=0.2,
                  obstacles=[Obstacle(5.0, 4.2, 0.8)])
    path = build_planner("potential_field").plan(world, (1.0, 1.0), (9.0, 9.0))
    _path_is_valid(world, path, (1.0, 1.0), (9.0, 9.0))


def test_grid_planners_report_no_path_when_blocked():
    # A wall of obstacles fully separating start from goal.
    wall = [Obstacle(5.0, y, 0.6) for y in np.arange(0.0, 10.5, 0.5)]
    world = World(width=10, height=10, resolution=0.2, robot_radius=0.25, obstacles=wall)
    assert build_planner("a_star").plan(world, (1.0, 5.0), (9.0, 5.0)) is None
    assert build_planner("dijkstra").plan(world, (1.0, 5.0), (9.0, 5.0)) is None


def test_unknown_planner_raises():
    with pytest.raises(KeyError):
        build_planner("teleport")
