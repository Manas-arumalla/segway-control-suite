"""Linear Kalman filter for the self-balancing robot.

Estimates the full state ``[x, x_dot, theta, theta_dot]`` from position+tilt measurements
using the discretized linearized plant. The discretization step is fixed at construction
and should match the rate at which :meth:`estimate` is called (the control period).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import cont2discrete

from ..config import RobotParams
from ..models import DEFAULT_C, linearize


class Estimator(ABC):
    """Common interface for state estimators."""

    @abstractmethod
    def reset(self, x0: np.ndarray | None = None) -> None: ...

    @abstractmethod
    def estimate(self, u: float, y: np.ndarray) -> np.ndarray:
        """One predict (with last input ``u``) + update (with measurement ``y``) cycle."""
        ...


class LinearKalmanFilter(Estimator):
    """Standard discrete-time Kalman filter on the linearized model."""

    def __init__(
        self,
        params: RobotParams,
        dt: float = 0.01,
        C: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ):
        A, B = linearize(params)
        self.C = DEFAULT_C if C is None else np.asarray(C, dtype=float)
        Ad, Bd, *_ = cont2discrete((A, B, self.C, np.zeros((self.C.shape[0], 1))), dt)
        self.Ad = Ad
        self.Bd = Bd
        self.dt = dt
        # Process noise: small on positions, larger on velocities (model uncertainty).
        self.Q = np.diag([1e-6, 1e-4, 1e-6, 1e-4]) if Q is None else np.asarray(Q, dtype=float)
        self.R = np.diag([0.005**2, 0.01**2]) if R is None else np.asarray(R, dtype=float)
        self.reset()

    def reset(self, x0: np.ndarray | None = None) -> None:
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, dtype=float).copy()
        self.P = np.eye(4)

    def predict(self, u: float) -> None:
        self.x = self.Ad @ self.x + self.Bd.flatten() * float(u)
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

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
