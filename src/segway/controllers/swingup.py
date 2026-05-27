"""Energy-based swing-up with a hybrid switch to balancing.

Starting from hanging-down, an energy-shaping law pumps the body up to the upright
homoclinic orbit; once the body is near vertical and slow enough, control switches to the
LQR balancer to catch and hold it. This is the project's most visually striking demo — the
robot stands itself up.

State angle ``theta`` is measured from upright, so ``theta = pi`` is hanging straight down.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotParams
from .base import Controller
from .lqr import LQRController


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class SwingUpController(Controller):
    """Hybrid energy swing-up + LQR catch/balance."""

    name = "swingup"

    def __init__(
        self,
        params: RobotParams,
        k_accel: float = 4.0,
        a_max: float = 20.0,
        k_cart: float = 2.5,
        k_pos: float = 0.6,
        switch_angle: float = 0.5,
        lqr_Q: np.ndarray | tuple | None = None,
        lqr_R: np.ndarray | float | None = None,
    ):
        super().__init__(params)
        self.lqr = LQRController(params, Q=lqr_Q, R=lqr_R)
        self.k_accel = float(k_accel)   # energy-pumping gain (on cart acceleration)
        self.a_max = float(a_max)       # cart-acceleration limit [m/s^2]
        self.k_cart = float(k_cart)     # cart-velocity damping (anti-runaway)
        self.k_pos = float(k_pos)       # cart-position centering (anti-runaway)
        self.switch_angle = float(switch_angle)
        self._Ieff = params.I_pend + params.m_pend * params.l**2
        self._E_top = params.m_pend * params.g * params.l  # energy at upright
        self._Mtot = params.M + params.m_pend
        self.mode = "swing"
        self._x_ref: float | None = None  # position to hold once balancing

    def reset(self) -> None:
        self.lqr.reset()
        self.mode = "swing"
        self._x_ref = None

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        x, x_dot, theta, theta_dot = state
        theta_w = _wrap(theta)

        # Near upright: hand over to the balancer, holding position where it was caught.
        if abs(theta_w) < self.switch_angle:
            if self.mode != "balance":
                self.mode = "balance"
                self._x_ref = x
            return self.lqr.compute(np.array([x - self._x_ref, x_dot, theta_w, theta_dot]), t)

        self.mode = "swing"
        self._x_ref = None
        # Energy shaping (Astrom-Furuta). With cart acceleration `a` as the effective input,
        # dE/dt = -m l a cos(theta) theta_dot, so a = k (E - E_top) theta_dot cos(theta) pumps
        # energy toward the upright value. Velocity/position terms keep the cart contained.
        E = 0.5 * self._Ieff * theta_dot**2 + self._E_top * np.cos(theta)
        a_des = (
            self.k_accel * (E - self._E_top) * theta_dot * np.cos(theta)
            - self.k_cart * x_dot
            - self.k_pos * x
        )
        a_des = float(np.clip(a_des, -self.a_max, self.a_max))
        # Torque that produces this cart acceleration: F = (M+m) a, tau = F r.
        return a_des * self._Mtot * self.params.r
