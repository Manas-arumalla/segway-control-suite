"""Extended Kalman filter on the full nonlinear dynamics.

Unlike the linear Kalman filter, the EKF propagates the state through the true nonlinear
model and relinearizes about the current estimate each step, so it stays accurate at larger
tilt angles where the small-angle linearization degrades.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotParams
from ..models import DEFAULT_C, linearize_numeric, nonlinear_dynamics
from .kalman import Estimator


def _rk4(state: np.ndarray, u: float, p: RobotParams, dt: float) -> np.ndarray:
    k1 = nonlinear_dynamics(state, u, p)
    k2 = nonlinear_dynamics(state + 0.5 * dt * k1, u, p)
    k3 = nonlinear_dynamics(state + 0.5 * dt * k2, u, p)
    k4 = nonlinear_dynamics(state + dt * k3, u, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class ExtendedKalmanFilter(Estimator):
    """EKF with nonlinear prediction and a (linear) position+tilt measurement model."""

    def __init__(
        self,
        params: RobotParams,
        dt: float = 0.01,
        C: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ):
        self.p = params
        self.dt = dt
        self.C = DEFAULT_C if C is None else np.asarray(C, dtype=float)
        self.Q = np.diag([1e-6, 1e-4, 1e-6, 1e-4]) if Q is None else np.asarray(Q, dtype=float)
        self.R = np.diag([0.005**2, 0.01**2]) if R is None else np.asarray(R, dtype=float)
        self.reset()

    def reset(self, x0: np.ndarray | None = None) -> None:
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, dtype=float).copy()
        self.P = np.eye(4)

    def predict(self, u: float) -> None:
        # Discrete state transition Jacobian via Euler discretization of the continuous one.
        A, _ = linearize_numeric(self.p, x_eq=self.x, u_eq=float(u))
        F = np.eye(4) + A * self.dt
        self.x = _rk4(self.x, float(u), self.p, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.C @ self.x)
        self.P = (np.eye(4) - K @ self.C) @ self.P

    def estimate(self, u: float, y: np.ndarray) -> np.ndarray:
        self.predict(u)
        self.update(y)
        return self.x.copy()
