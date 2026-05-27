"""Region of Attraction (ROA) estimation.

Sweeps a grid of initial conditions ``(theta0, theta_dot0)`` and marks which ones a
controller can recover from. The *area* of the recoverable region is a single-number
robustness score that lets controllers be compared on the same axes — a much fairer test
than a single trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RobotParams, SimConfig
from ..controllers.base import Controller
from ..sim.runner import simulate
from ..sim.scenarios import Scenario


@dataclass
class ROAResult:
    """Grid of recovery outcomes over initial ``(theta, theta_dot)``."""

    theta_vals: np.ndarray
    thetadot_vals: np.ndarray
    grid: np.ndarray  # bool, shape (n_theta, n_thetadot); True = recovered
    controller_name: str = "unknown"

    @property
    def area_fraction(self) -> float:
        """Fraction of the sampled grid from which the controller recovered (0..1)."""
        return float(self.grid.mean())


def compute_roa(
    params: RobotParams,
    controller: Controller,
    theta_range: tuple[float, float] = (-1.2, 1.2),
    thetadot_range: tuple[float, float] = (-6.0, 6.0),
    n_theta: int = 21,
    n_thetadot: int = 21,
    duration: float = 4.0,
    settle_angle: float = 0.2,
    dt: float = 0.005,
) -> ROAResult:
    """Estimate the ROA by simulating from each grid initial condition.

    A cell counts as *recovered* if the run does not fall and ends with ``|theta|`` below
    ``settle_angle``. Uses a coarser ``dt`` than the default sim for speed; raise ``n_*`` for
    a finer boundary.
    """
    theta_vals = np.linspace(*theta_range, n_theta)
    thetadot_vals = np.linspace(*thetadot_range, n_thetadot)
    grid = np.zeros((n_theta, n_thetadot), dtype=bool)
    sim = SimConfig(dt=dt, duration=duration, record_every=max(1, int(0.05 / dt)))

    for i, th0 in enumerate(theta_vals):
        for j, thd0 in enumerate(thetadot_vals):
            scen = Scenario(initial_tilt=float(th0), initial_tilt_rate=float(thd0))
            traj = simulate(params, controller, scen, sim)
            grid[i, j] = (not traj.fell) and abs(traj.theta[-1]) < settle_angle

    return ROAResult(theta_vals, thetadot_vals, grid, getattr(controller, "name", "unknown"))
