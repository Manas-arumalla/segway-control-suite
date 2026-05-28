"""The navigation world: a 2-D map with circular obstacles + an inflated occupancy grid.

Obstacles are inflated by the robot radius so any collision-free path keeps clearance. The
world provides continuous checks (``is_free``, ``is_segment_free``) for sampling planners and
a boolean ``occupancy`` grid for grid planners.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    """A circular obstacle."""

    x: float
    y: float
    r: float


@dataclass
class World:
    width: float = 10.0
    height: float = 10.0
    resolution: float = 0.1          # grid cell size [m]
    robot_radius: float = 0.25       # obstacles are inflated by this
    obstacles: list[Obstacle] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.nx = max(1, int(round(self.width / self.resolution)))
        self.ny = max(1, int(round(self.height / self.resolution)))
        self._build_occupancy()

    def _build_occupancy(self) -> None:
        xs = (np.arange(self.nx) + 0.5) * self.resolution
        ys = (np.arange(self.ny) + 0.5) * self.resolution
        gx, gy = np.meshgrid(xs, ys)              # (ny, nx)
        occ = np.zeros((self.ny, self.nx), dtype=bool)
        for o in self.obstacles:
            occ |= np.hypot(gx - o.x, gy - o.y) <= (o.r + self.robot_radius)
        self.occupancy = occ

    # --- continuous queries (for sampling planners) ----------------------
    def in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def is_free(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return False
        for o in self.obstacles:
            if np.hypot(x - o.x, y - o.y) <= o.r + self.robot_radius:
                return False
        return True

    def is_segment_free(self, a, b, step: float | None = None) -> bool:
        step = step or self.resolution * 0.5
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        length = float(np.hypot(*(b - a)))
        n = max(1, int(length / step))
        for k in range(n + 1):
            p = a + (b - a) * (k / n)
            if not self.is_free(float(p[0]), float(p[1])):
                return False
        return True

    # --- grid helpers (for grid planners) --------------------------------
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        i = min(self.nx - 1, max(0, int(x / self.resolution)))
        j = min(self.ny - 1, max(0, int(y / self.resolution)))
        return i, j

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return (i + 0.5) * self.resolution, (j + 0.5) * self.resolution

    def cell_free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.occupancy[j, i]
