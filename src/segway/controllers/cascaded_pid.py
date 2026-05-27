"""Cascaded PID: an outer position loop commanding an inner tilt loop.

This fixes the central weakness of the plain tilt-only :class:`PIDController` (it balances
but drifts in position). The outer loop turns a position error into a small *desired lean*;
the inner loop drives the body to that lean. The result holds position while balancing —
a strong, fully classical baseline.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotParams
from .base import Controller


class CascadedPIDController(Controller):
    """Two nested PID loops (outer: position -> tilt command; inner: tilt -> torque)."""

    name = "cascaded_pid"

    def __init__(
        self,
        params: RobotParams,
        kp: float = 140.0,
        kd: float = 45.0,
        ki: float = 0.0,
        kx: float = -0.2,
        kv: float = -0.45,
        theta_cmd_max: float = 0.25,
        i_max: float = 4.0,
    ):
        super().__init__(params)
        self.kp, self.kd, self.ki = kp, kd, ki
        self.kx, self.kv = kx, kv
        self.theta_cmd_max = theta_cmd_max
        self.i_max = i_max
        self._integral = 0.0
        self._last_t: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_t = None

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        x, x_dot, theta, theta_dot = state

        # Outer loop: a position/velocity error commands a small lean toward the target.
        theta_cmd = float(np.clip(self.kx * x + self.kv * x_dot, -self.theta_cmd_max, self.theta_cmd_max))
        err = theta - theta_cmd

        if self._last_t is not None and t > self._last_t:
            self._integral += err * (t - self._last_t)
            self._integral = float(np.clip(self._integral, -self.i_max, self.i_max))
        self._last_t = t

        # Inner loop: PD(+I) on the tilt error.
        return self.kp * err + self.kd * theta_dot + self.ki * self._integral
