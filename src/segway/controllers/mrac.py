"""Model-Reference Adaptive Control (direct MRAC).

Augments a stabilizing LQR baseline with an online-adapted term that cancels a *matched*
uncertainty (the kind introduced by wrong mass / length / inertia). The closed loop is
driven to follow a reference model ``x_m_dot = A_m x_m`` (the nominal LQR closed loop), and
the adaptive weights are updated by a Lyapunov-stable law with sigma-modification for
robustness.

Error dynamics with ``u = -K x - theta_hat^T x`` and true matched uncertainty ``theta*``::

    e = x - x_m,   e_dot = A_m e + B (theta* - theta_hat)^T x
    theta_hat_dot = gamma * x * (B^T P e) - sigma * theta_hat     (P solves A_m^T P + P A_m = -Q)

which gives ``V_dot = -e^T Q e <= 0``. With ``theta_hat = 0`` the controller is exactly the
LQR baseline, so adaptation can only help.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..config import RobotParams
from ..models import linearize
from .base import Controller
from .lqr import LQRController


class MRACController(Controller):
    """Direct MRAC augmenting an LQR baseline."""

    name = "mrac"

    def __init__(
        self,
        params: RobotParams,
        Q: np.ndarray | tuple | None = None,
        R: np.ndarray | float | None = None,
        gamma: float = 6.0,
        sigma: float = 0.4,
        Q_lyap: np.ndarray | None = None,
    ):
        super().__init__(params)
        self.A, self.B = linearize(params)
        base = LQRController(params, Q=Q, R=R)
        self.K = base.K  # (1, 4) stabilizing baseline
        self.Am = self.A - self.B @ self.K
        Ql = np.eye(4) if Q_lyap is None else np.asarray(Q_lyap, float)
        self.P = sla.solve_lyapunov(self.Am.T, -Ql)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.reset()

    def reset(self) -> None:
        self.theta = np.zeros(4)      # adaptive weights
        self.xm: np.ndarray | None = None  # reference-model state
        self._last_t: float | None = None

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        x = np.asarray(state, dtype=float)
        if self.xm is None:
            self.xm = x.copy()

        u = float(-(self.K @ x).item() - self.theta @ x)

        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            e = x - self.xm
            bPe = float(self.B.T @ self.P @ e)
            self.theta = self.theta + dt * (self.gamma * x * bPe - self.sigma * self.theta)
            self.xm = self.xm + dt * (self.Am @ self.xm)
        self._last_t = t
        return u
