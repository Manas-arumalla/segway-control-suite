"""Preset navigation scenarios (world + start + goal) with a name-based registry.

These give the CLI, the benchmark, and the UIs a common set of maps to plan and drive
through, ranging from a single obstacle to a cluttered random forest. Each builder returns a
fresh :class:`NavScenario` so callers can mutate worlds without side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .world import Obstacle, World


@dataclass
class NavScenario:
    """A named navigation problem: a world plus a start and goal pose."""

    name: str
    world: World
    start: tuple[float, float]
    goal: tuple[float, float]
    description: str = ""


def _corridor() -> NavScenario:
    world = World(width=8.0, height=4.0, resolution=0.1, robot_radius=0.25,
                  obstacles=[Obstacle(4.0, 2.0, 0.7)])
    return NavScenario("corridor", world, (1.0, 2.0), (7.0, 2.0),
                       "One obstacle squarely between start and goal.")


def _slalom() -> NavScenario:
    obs = [Obstacle(3.0, 1.2, 0.6), Obstacle(6.0, 2.8, 0.6), Obstacle(9.0, 1.2, 0.6)]
    world = World(width=12.0, height=4.0, resolution=0.1, robot_radius=0.25, obstacles=obs)
    return NavScenario("slalom", world, (0.8, 2.0), (11.2, 2.0),
                       "Staggered obstacles forcing a weave.")


def _rooms() -> NavScenario:
    # Two wall-like barriers with offset gaps the robot must thread.
    obs = [Obstacle(4.0, y, 0.45) for y in (0.5, 1.4, 2.3, 5.0, 6.0, 7.0, 8.0, 9.0)]
    obs += [Obstacle(7.0, y, 0.45) for y in (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 9.5)]
    world = World(width=10.0, height=10.0, resolution=0.1, robot_radius=0.25, obstacles=obs)
    return NavScenario("rooms", world, (1.0, 5.0), (9.0, 5.0),
                       "Two barriers with offset doorways.")


def _forest(seed: int = 0, n: int = 16) -> NavScenario:
    rng = np.random.default_rng(seed)
    start, goal = (1.0, 1.0), (11.0, 11.0)
    obs: list[Obstacle] = []
    while len(obs) < n:
        x, y = rng.uniform(2.0, 10.0), rng.uniform(2.0, 10.0)
        r = rng.uniform(0.3, 0.7)
        if np.hypot(x - start[0], y - start[1]) < r + 1.0:
            continue
        if np.hypot(x - goal[0], y - goal[1]) < r + 1.0:
            continue
        obs.append(Obstacle(x, y, r))
    world = World(width=12.0, height=12.0, resolution=0.1, robot_radius=0.25, obstacles=obs)
    return NavScenario("forest", world, start, goal,
                       "Randomly scattered circular obstacles (seeded).")


_BUILDERS: dict[str, Callable[[], NavScenario]] = {
    "corridor": _corridor,
    "slalom": _slalom,
    "rooms": _rooms,
    "forest": _forest,
}


def build_scenario(name: str) -> NavScenario:
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[key]()


def list_scenarios() -> list[str]:
    return sorted(_BUILDERS)
