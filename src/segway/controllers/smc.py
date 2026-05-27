"""Sliding Mode Control (robust / nonlinear)."""

from __future__ import annotations

import numpy as np

from ..config import RobotParams
from ..models import linearize
from .base import Controller

# Sliding-surface coefficients [x, x_dot, theta, theta_dot]; the x_dot coefficient is
# conventionally 1.0. Defaults give a stable sliding manifold for the default robot.
DEFAULT_LAMBDA = (2.0, 1.0, 6.0, 1.0)


class SMCController(Controller):
    """Sliding mode controller with a linear sliding surface and boundary layer.

    Surface ``s = c^T x`` (``c`` = ``lam``). The control combines equivalent control with
    a switching term smoothed by a boundary layer ``phi`` to suppress chattering::

        u = -(c^T B)^{-1} ( c^T A x + K * sat(s / phi) )
    """

    name = "smc"

    def __init__(
        self,
        params: RobotParams,
        lam: np.ndarray | tuple | None = None,
        K: float = 25.0,
        phi: float = 0.1,
    ):
        super().__init__(params)
        self.A, self.B = linearize(params)
        self.coeffs = np.asarray(DEFAULT_LAMBDA if lam is None else lam, dtype=float)
        self.K_smc = float(K)
        self.phi = float(phi)

        self.CB = float(self.coeffs @ self.B)  # scalar c^T B
        if abs(self.CB) < 1e-12:
            raise ValueError("sliding surface is not well-posed: c^T B is ~0; choose other lambda")
        self.CA = self.coeffs @ self.A  # (4,)

    def _sat(self, s: float) -> float:
        return float(np.clip(s / self.phi, -1.0, 1.0))

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        state = np.asarray(state, dtype=float)
        s = float(self.coeffs @ state)
        term = float(self.CA @ state) + self.K_smc * self._sat(s)
        return -term / self.CB
