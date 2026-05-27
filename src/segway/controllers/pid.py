"""PID balancing controller (classical baseline).

A deliberately simple baseline: PID on the body tilt (with optional position feedback).
With the default position gains of zero it is a *tilt-only* PID — it keeps the robot
upright but lets the base drift, which is a well-known and instructive limitation that
the benchmark makes visible.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotParams
from .base import Controller


class PIDController(Controller):
    """PID on tilt ``theta`` plus optional proportional position/velocity feedback.

    Control law::

        u = kp*theta + ki*∫theta dt + kd*theta_dot + kx*x + kv*x_dot

    The integral term uses a clamp for anti-windup. Time steps for integration are derived
    from successive ``t`` values passed to :meth:`compute`.
    """

    name = "pid"

    def __init__(
        self,
        params: RobotParams,
        kp: float = 120.0,
        ki: float = 10.0,
        kd: float = 20.0,
        kx: float = 0.0,
        kv: float = 0.0,
        i_max: float = 5.0,
    ):
        super().__init__(params)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.kx, self.kv = kx, kv
        self.i_max = i_max
        self._integral = 0.0
        self._last_t: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_t = None

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        x, x_dot, theta, theta_dot = state

        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            self._integral += theta * dt
            self._integral = float(np.clip(self._integral, -self.i_max, self.i_max))
        self._last_t = t

        return float(
            self.kp * theta
            + self.ki * self._integral
            + self.kd * theta_dot
            + self.kx * x
            + self.kv * x_dot
        )
