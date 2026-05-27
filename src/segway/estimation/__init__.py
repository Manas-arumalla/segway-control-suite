"""State estimation: realistic sensors + Kalman / Extended Kalman filters.

In the real world a balancing robot does not measure its full state — it reads a noisy
wheel encoder (position) and a noisy IMU (tilt), and must *estimate* the velocities. This
package provides that realism so controllers can be evaluated on estimates, not ground
truth.
"""

from __future__ import annotations

from .ekf import ExtendedKalmanFilter
from .kalman import Estimator, LinearKalmanFilter
from .sensors import SensorModel, SensorSpec

__all__ = [
    "Estimator",
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "SensorModel",
    "SensorSpec",
]
