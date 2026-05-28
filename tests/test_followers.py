"""Path-follower suite (NAV-4), tested closed-loop on a kinematic unicycle."""

from __future__ import annotations

import numpy as np
import pytest

from segway.navigation import World, build_follower, list_followers


def _drive_unicycle(follower, path, world=None, dt=0.1, max_time=40.0):
    """Integrate a kinematic unicycle under the follower; return (final_xy, reached)."""
    x, y, psi = path[0, 0], path[0, 1], np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
    follower.reset()
    for _ in range(int(max_time / dt)):
        v, w, done = follower.command((x, y, psi), path, world)
        if done:
            return np.array([x, y]), True
        x += v * np.cos(psi) * dt
        y += v * np.sin(psi) * dt
        psi += w * dt
    return np.array([x, y]), np.hypot(x - path[-1, 0], y - path[-1, 1]) < 0.3


def test_registry_has_all_followers():
    assert set(list_followers()) >= {"pure_pursuit", "stanley", "dwa", "mpc", "vector_field"}


@pytest.mark.parametrize("name", ["pure_pursuit", "stanley", "dwa", "mpc", "vector_field"])
def test_follower_tracks_path_to_goal(name):
    # An L-shaped path: forward, then a left turn.
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [5.0, 3.0]])
    world = World(width=12, height=12, resolution=0.3, robot_radius=0.2, obstacles=[])
    final, reached = _drive_unicycle(build_follower(name), path, world)
    assert reached, f"{name} did not reach the goal (ended at {final})"
    assert np.hypot(*(final - path[-1])) < 0.4


def test_follower_outputs_forward_command_at_start():
    path = np.array([[0.0, 0.0], [5.0, 0.0]])
    for name in ["pure_pursuit", "stanley", "vector_field"]:
        v, w, done = build_follower(name).command((0.0, 0.0, 0.0), path)
        assert v > 0 and not done and abs(w) < 1.0   # drive forward, nearly straight


def test_unknown_follower_raises():
    with pytest.raises(KeyError):
        build_follower("autopilot")
