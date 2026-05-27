"""Linear Quadratic Regulator (optimal full-state feedback)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from ..config import RobotParams
from ..models import linearize
from .base import Controller

# Default state weights emphasize keeping the body upright (theta) over cart position.
DEFAULT_Q = (10.0, 1.0, 100.0, 1.0)
DEFAULT_R = 0.1


class LQRController(Controller):
    """LQR designed on the analytic linearization of the plant.

    Minimizes the infinite-horizon cost ``J = int (x^T Q x + u^T R u) dt``. The gain is
    ``K = R^{-1} B^T P`` where ``P`` solves the continuous-time algebraic Riccati equation,
    and the control law is ``u = -K x``.
    """

    name = "lqr"

    def __init__(
        self,
        params: RobotParams,
        Q: np.ndarray | tuple | None = None,
        R: np.ndarray | float | None = None,
    ):
        super().__init__(params)
        self.A, self.B = linearize(params)

        Q = DEFAULT_Q if Q is None else Q
        Q = np.diag(np.asarray(Q, dtype=float)) if np.ndim(Q) == 1 else np.asarray(Q, dtype=float)
        R = DEFAULT_R if R is None else R
        R = np.atleast_2d(np.asarray(R, dtype=float))

        P = solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.solve(R, self.B.T @ P)  # (1, 4)
        self.Q, self.R, self.P = Q, R, P

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        return float(-(self.K @ np.asarray(state, dtype=float)).item())
