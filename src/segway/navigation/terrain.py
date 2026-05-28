"""Uneven-terrain generation for navigation — smooth height maps + a name-based registry.

A :class:`Terrain` is a height map over the navigation world (``x in [0, width]``,
``y in [0, height]``) plus continuous height/slope queries. Terrains are built from a few
random low-frequency waves (smooth and deterministic) and an optional constant grade, so the
balancing robot has to reject slope disturbances as it drives. The same height map feeds the
MuJoCo heightfield (see ``sim/mujoco_terrain.py``), so the analytic queries and the simulated
ground agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Terrain:
    """A smooth height map over ``[0, width] x [0, height]`` (heights in metres)."""

    width: float
    height: float
    heights: np.ndarray          # (nrow, ncol), heights[j, i] at (x_i, y_j) [m]
    name: str = "terrain"

    def __post_init__(self) -> None:
        self.nrow, self.ncol = self.heights.shape   # rows = y, cols = x

    @property
    def amplitude(self) -> float:
        return float(self.heights.max() - self.heights.min())

    def height_at(self, x: float, y: float) -> float:
        """Bilinear-interpolated ground height at ``(x, y)`` [m]."""
        fx = np.clip(x / self.width, 0.0, 1.0) * (self.ncol - 1)
        fy = np.clip(y / self.height, 0.0, 1.0) * (self.nrow - 1)
        i0, j0 = int(np.floor(fx)), int(np.floor(fy))
        i1, j1 = min(i0 + 1, self.ncol - 1), min(j0 + 1, self.nrow - 1)
        tx, ty = fx - i0, fy - j0
        h = self.heights
        top = h[j0, i0] * (1 - tx) + h[j0, i1] * tx
        bot = h[j1, i0] * (1 - tx) + h[j1, i1] * tx
        return float(top * (1 - ty) + bot * ty)

    def slope_at(self, x: float, y: float, eps: float = 0.05) -> float:
        """Local ground slope magnitude [rad] from a finite-difference gradient."""
        dzdx = (self.height_at(x + eps, y) - self.height_at(x - eps, y)) / (2 * eps)
        dzdy = (self.height_at(x, y + eps) - self.height_at(x, y - eps)) / (2 * eps)
        return float(np.arctan(np.hypot(dzdx, dzdy)))

    def max_slope(self) -> float:
        """Largest slope anywhere on the grid [rad]."""
        gy, gx = np.gradient(self.heights, self.height / (self.nrow - 1),
                             self.width / (self.ncol - 1))
        return float(np.arctan(np.hypot(gx, gy).max()))


def generate_terrain(
    width: float = 8.0,
    height: float = 8.0,
    amplitude: float = 0.05,
    n_waves: int = 6,
    wavelength: float = 3.0,
    slope: float = 0.0,
    seed: int = 0,
    nrow: int = 80,
    ncol: int = 80,
    name: str = "terrain",
) -> Terrain:
    """Build a smooth random terrain plus an optional constant grade along +x.

    ``amplitude`` is the peak bump height [m]; ``wavelength`` the typical bump spacing [m];
    ``slope`` a constant grade [rad] tilting the whole map along +x.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, width, ncol)
    ys = np.linspace(0.0, height, nrow)
    gx, gy = np.meshgrid(xs, ys)                      # (nrow, ncol)

    field = np.zeros_like(gx)
    for _ in range(n_waves):
        theta = rng.uniform(0, np.pi)
        freq = 2 * np.pi / (wavelength * rng.uniform(0.6, 1.6))
        phase = rng.uniform(0, 2 * np.pi)
        field += np.sin(freq * (gx * np.cos(theta) + gy * np.sin(theta)) + phase)

    if np.ptp(field) > 1e-9:
        field = (field - field.min()) / np.ptp(field)  # normalize to [0, 1]
    heights = amplitude * field + np.tan(slope) * gx
    heights -= heights.min()                          # keep the lowest point at 0
    return Terrain(width=width, height=height, heights=heights, name=name)


_BUILDERS: dict[str, Callable[[], Terrain]] = {
    "flat": lambda: generate_terrain(amplitude=0.0, name="flat"),
    "gentle": lambda: generate_terrain(amplitude=0.03, wavelength=3.5, seed=1, name="gentle"),
    "moderate": lambda: generate_terrain(amplitude=0.06, wavelength=3.0, seed=2, name="moderate"),
    "rough": lambda: generate_terrain(amplitude=0.10, wavelength=2.2, seed=3, name="rough"),
    "ramp": lambda: generate_terrain(amplitude=0.02, slope=np.radians(6.0), seed=4, name="ramp"),
}


def build_terrain(name: str) -> Terrain:
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise KeyError(f"unknown terrain {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[key]()


def list_terrains() -> list[str]:
    return sorted(_BUILDERS)
