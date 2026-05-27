"""H-infinity (state-feedback) robust control.

Designs ``u = -K x`` that minimizes the worst-case (H-infinity) gain from disturbances to a
performance output, i.e. it optimizes for the *worst* disturbance rather than the average
one (as LQR does). Implemented from first principles via the Hamiltonian solution of the
H-infinity algebraic Riccati equation with a gamma-iteration — no ``slycot`` needed.

System:  x_dot = A x + B1 w + B2 u,  performance z = [Q^{1/2} x; R^{1/2} u].
The suboptimal gamma controller solves
    A^T X + X A + X (gamma^-2 B1 B1^T - B2 R^-1 B2^T) X + Q = 0
for the stabilizing X >= 0, giving K = R^-1 B2^T X. Gamma is reduced by bisection toward the
smallest feasible value (most robust), then backed off slightly for margin.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..config import RobotParams
from ..models import linearize
from .base import Controller

DEFAULT_Q = (10.0, 1.0, 100.0, 1.0)
DEFAULT_R = 0.1
# Disturbances enter the velocity channels (force on the base, torque on the body).
DEFAULT_B1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


def _solve_hinf_are(A, B1, B2, Q, R, gamma):
    """Stabilizing solution X of the H-infinity ARE, or None if infeasible for this gamma."""
    S = B2 @ np.linalg.solve(R, B2.T) - (gamma**-2) * (B1 @ B1.T)
    n = A.shape[0]
    H = np.block([[A, -S], [-Q, -A.T]])
    eigvals, eigvecs = sla.eig(H)

    # Feasible only if the Hamiltonian has no eigenvalues on the imaginary axis.
    if np.any(np.abs(eigvals.real) < 1e-7):
        return None

    stable = np.argsort(eigvals.real)[:n]  # n eigenvalues with most-negative real part
    U = eigvecs[:, stable]
    U1, U2 = U[:n, :], U[n:, :]
    if np.linalg.cond(U1) > 1e12:
        return None
    X = np.real(U2 @ np.linalg.inv(U1))
    X = 0.5 * (X + X.T)  # symmetrize

    if np.min(np.linalg.eigvalsh(X)) < -1e-6:  # require X >= 0
        return None
    K = np.linalg.solve(R, B2.T @ X)
    if np.max(np.linalg.eigvals(A - B2 @ K).real) >= 0:  # closed loop must be stable
        return None
    return X


class HInfController(Controller):
    """State-feedback H-infinity controller (gamma-suboptimal)."""

    name = "hinf"

    def __init__(
        self,
        params: RobotParams,
        Q: np.ndarray | tuple | None = None,
        R: np.ndarray | float | None = None,
        B1: np.ndarray | None = None,
        gamma: float | None = None,
        gamma_margin: float = 2.0,
    ):
        super().__init__(params)
        self.A, self.B = linearize(params)
        Qd = DEFAULT_Q if Q is None else Q
        self.Q = np.diag(np.asarray(Qd, float)) if np.ndim(Qd) == 1 else np.asarray(Qd, float)
        self.R = np.atleast_2d(np.asarray(DEFAULT_R if R is None else R, float))
        self.B1 = DEFAULT_B1 if B1 is None else np.asarray(B1, float)

        if gamma is None:
            gamma = self._gamma_search()
            gamma *= gamma_margin  # back off from the optimum for robustness margin
        self.gamma = float(gamma)

        X = _solve_hinf_are(self.A, self.B1, self.B, self.Q, self.R, self.gamma)
        if X is None:
            raise ValueError(f"H-infinity ARE infeasible at gamma={self.gamma:.3g}")
        self.X = X
        self.K = np.linalg.solve(self.R, self.B.T @ X)

    def _gamma_search(self, lo: float = 0.05, hi: float = 1e4, iters: int = 40) -> float:
        """Bisection for the smallest gamma with a feasible stabilizing solution."""
        # hi is assumed feasible (approaches the LQR problem); shrink lo until infeasible.
        for _ in range(iters):
            mid = np.sqrt(lo * hi)  # geometric bisection
            if _solve_hinf_are(self.A, self.B1, self.B, self.Q, self.R, mid) is not None:
                hi = mid
            else:
                lo = mid
        return hi

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        return float(-(self.K @ np.asarray(state, dtype=float)).item())
