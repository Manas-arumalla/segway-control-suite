"""Smooth reference trajectories.

Minimum-jerk position profiles give a base setpoint that ramps smoothly from one position
to another (zero velocity/acceleration at the endpoints) instead of a step — much gentler on
a balancing robot, which has to lean to accelerate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def min_jerk(x0: float, xf: float, T: float, t: float) -> float:
    """Minimum-jerk position at time ``t`` for a move ``x0 -> xf`` over duration ``T``."""
    if t <= 0:
        return float(x0)
    if t >= T:
        return float(xf)
    s = t / T
    return float(x0 + (xf - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5))


def min_jerk_reference(x0: float, xf: float, T: float, start: float = 0.0) -> Callable[[float], np.ndarray]:
    """Return a reference function ``t -> [x_ref, 0, 0, 0]`` following a min-jerk profile.

    The robot should stay upright (``theta_ref = 0``) while its base position tracks the
    smooth profile. Plug into ``Scenario(reference_fn=...)``.
    """
    def ref(t: float) -> np.ndarray:
        return np.array([min_jerk(x0, xf, T, t - start), 0.0, 0.0, 0.0])

    return ref
