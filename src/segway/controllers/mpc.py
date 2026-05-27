"""Model Predictive Control (constrained, receding-horizon).

Requires the ``mpc`` extra (``pip install -e ".[mpc]"``) for CVXPY/OSQP. The optimization
problem is built once with a CVXPY ``Parameter`` for the initial state and warm-started on
every solve, so re-solving each control tick is cheap.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete

from ..config import RobotParams
from ..models import linearize
from .base import Controller

DEFAULT_Q = (10.0, 1.0, 100.0, 1.0)
DEFAULT_R = 0.1


class MPCController(Controller):
    """Linear MPC with input limits, solved with CVXPY/OSQP."""

    name = "mpc"

    def __init__(
        self,
        params: RobotParams,
        Q: np.ndarray | tuple | None = None,
        R: np.ndarray | float | None = None,
        horizon: int = 20,
        control_dt: float = 0.05,
        u_max: float = 100.0,
    ):
        super().__init__(params)
        import cvxpy as cp  # local import: only needed when MPC is actually used

        self._cp = cp
        A, B = linearize(params)
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), control_dt)
        self.Ad, self.Bd = Ad, Bd
        self.N = int(horizon)
        self.control_dt = float(control_dt)
        self.u_max = float(u_max)

        Q = DEFAULT_Q if Q is None else Q
        Qm = np.diag(np.asarray(Q, dtype=float)) if np.ndim(Q) == 1 else np.asarray(Q, dtype=float)
        Rm = np.atleast_2d(np.asarray(DEFAULT_R if R is None else R, dtype=float))

        # Build the parametric QP once.
        self._x0 = cp.Parameter(4)
        u = cp.Variable((1, self.N))
        x = cp.Variable((4, self.N + 1))
        cost = 0
        cons = [x[:, 0] == self._x0]
        for k in range(self.N):
            cost += cp.quad_form(x[:, k], Qm) + cp.quad_form(u[:, k], Rm)
            cons += [x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k]]
            cons += [u[:, k] <= self.u_max, u[:, k] >= -self.u_max]
        cost += cp.quad_form(x[:, self.N], Qm)  # terminal cost
        self._u = u
        self._prob = cp.Problem(cp.Minimize(cost), cons)

        self._last_t = -np.inf
        self._last_u = 0.0

    def reset(self) -> None:
        self._last_t = -np.inf
        self._last_u = 0.0

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        # Zero-order hold between control updates.
        if np.isfinite(self._last_t) and (t - self._last_t) < self.control_dt:
            return self._last_u

        self._x0.value = np.asarray(state, dtype=float)
        try:
            self._prob.solve(solver=self._cp.OSQP, warm_start=True)
            if self._u.value is not None and self._prob.status in (
                self._cp.OPTIMAL,
                self._cp.OPTIMAL_INACCURATE,
            ):
                self._last_u = float(self._u.value[0, 0])
        except self._cp.SolverError:
            pass  # keep previous command on solver failure (fail-safe hold)

        self._last_t = t
        return self._last_u
