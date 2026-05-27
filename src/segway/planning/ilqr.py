"""iLQR — iterative Linear Quadratic Regulator (trajectory optimization).

Computes a locally optimal control sequence (and time-varying feedback gains) for the full
*nonlinear* plant by repeatedly: rolling out, linearizing along the trajectory, solving a
backward Riccati-like recursion for control updates, and forward line-searching. This is the
modern workhorse behind nonlinear optimal control / MPC and yields, e.g., optimal swing-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RobotParams
from ..models import nonlinear_dynamics


@dataclass
class iLQRResult:
    xs: np.ndarray          # (N+1, 4) optimal state trajectory
    us: np.ndarray          # (N,) optimal control sequence
    gains: np.ndarray       # (N, 4) time-varying feedback gains K_t
    cost: float
    converged: bool
    iterations: int


class iLQR:
    """iLQR optimizer for the wheeled-inverted-pendulum dynamics."""

    def __init__(
        self,
        params: RobotParams,
        dt: float = 0.02,
        Q: np.ndarray | tuple = (1.0, 0.1, 10.0, 0.1),
        R: float = 0.01,
        Qf: np.ndarray | tuple = (50.0, 5.0, 500.0, 5.0),
        x_goal: np.ndarray | None = None,
        u_max: float = 80.0,
    ):
        self.p = params
        self.dt = float(dt)
        self.Q = np.diag(np.asarray(Q, float))
        self.R = float(R)
        self.Qf = np.diag(np.asarray(Qf, float))
        self.x_goal = np.zeros(4) if x_goal is None else np.asarray(x_goal, float)
        self.u_max = float(u_max)

    def _f(self, x: np.ndarray, u: float) -> np.ndarray:
        """One RK4 step of the nonlinear dynamics (discrete transition)."""
        dt = self.dt
        k1 = nonlinear_dynamics(x, u, self.p)
        k2 = nonlinear_dynamics(x + 0.5 * dt * k1, u, self.p)
        k3 = nonlinear_dynamics(x + 0.5 * dt * k2, u, self.p)
        k4 = nonlinear_dynamics(x + dt * k3, u, self.p)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _linearize(self, x: np.ndarray, u: float, eps: float = 1e-6):
        fx = np.zeros((4, 4))
        for j in range(4):
            dx = np.zeros(4)
            dx[j] = eps
            fx[:, j] = (self._f(x + dx, u) - self._f(x - dx, u)) / (2 * eps)
        fu = (self._f(x, u + eps) - self._f(x, u - eps)) / (2 * eps)
        return fx, fu

    def _rollout(self, x0, us):
        xs = np.zeros((len(us) + 1, 4))
        xs[0] = x0
        for t, u in enumerate(us):
            xs[t + 1] = self._f(xs[t], float(u))
        return xs

    def _cost(self, xs, us):
        c = 0.0
        for t in range(len(us)):
            dx = xs[t] - self.x_goal
            c += 0.5 * dx @ self.Q @ dx + 0.5 * self.R * us[t] ** 2
        dxf = xs[-1] - self.x_goal
        return c + 0.5 * dxf @ self.Qf @ dxf

    def fit(self, x0, N=100, us_init=None, iters=100, tol=1e-4) -> iLQRResult:
        """Optimize an ``N``-step control sequence from ``x0``."""
        x0 = np.asarray(x0, float)
        us = np.zeros(N) if us_init is None else np.asarray(us_init, float).copy()
        xs = self._rollout(x0, us)
        cost = self._cost(xs, us)
        mu = 1e-6  # regularization
        converged = False
        it = 0

        for it in range(iters):  # noqa: B007  (it is used after the loop for the report)
            # ---- Backward pass ----
            Vx = self.Qf @ (xs[-1] - self.x_goal)
            Vxx = self.Qf.copy()
            ks = np.zeros(N)
            Ks = np.zeros((N, 4))
            ok = True
            for t in range(N - 1, -1, -1):
                fx, fu = self._linearize(xs[t], float(us[t]))
                lx = self.Q @ (xs[t] - self.x_goal)
                lu = self.R * us[t]
                Qx = lx + fx.T @ Vx
                Qu = lu + fu @ Vx
                Qxx = self.Q + fx.T @ Vxx @ fx
                Quu = self.R + fu @ Vxx @ fu + mu
                Qux = fu @ Vxx @ fx  # (4,)
                if Quu <= 0:
                    ok = False
                    break
                k = -Qu / Quu
                K = -Qux / Quu
                ks[t] = k
                Ks[t] = K
                Vx = Qx + K * (Quu * k) + K * Qu + Qux * k
                Vxx = Qxx + Quu * np.outer(K, K) + np.outer(K, Qux) + np.outer(Qux, K)
                Vxx = 0.5 * (Vxx + Vxx.T)
            if not ok:
                mu *= 10
                continue

            # ---- Forward pass with line search ----
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
                xnew = np.zeros_like(xs)
                xnew[0] = x0
                unew = np.zeros(N)
                for t in range(N):
                    unew[t] = np.clip(us[t] + alpha * ks[t] + Ks[t] @ (xnew[t] - xs[t]),
                                      -self.u_max, self.u_max)
                    xnew[t + 1] = self._f(xnew[t], float(unew[t]))
                cnew = self._cost(xnew, unew)
                if cnew < cost:
                    improved = True
                    break
            if not improved:
                mu *= 10
                if mu > 1e6:
                    converged = True
                    break
                continue

            mu = max(mu / 10, 1e-8)
            if abs(cost - cnew) < tol:
                xs, us, cost = xnew, unew, cnew
                converged = True
                break
            xs, us, cost = xnew, unew, cnew

        return iLQRResult(xs=xs, us=us, gains=Ks, cost=float(cost),
                          converged=converged, iterations=it + 1)
