"""Inner balance + yaw loop for the TWIP.

Tracks a forward-speed command ``v_des`` and a yaw-rate command ``yaw_rate_des`` while
keeping the body upright. It **reuses any existing balancing controller** for the
longitudinal subsystem: the controller is fed the longitudinal error
``[0, v - v_des, theta, theta_dot]`` (position error is intentionally zero — we regulate
*speed*, not position) and returns the wheel-torque sum; a feed-forward + proportional law on
the yaw rate gives the wheel-torque difference. The two are then split into left/right wheel
torques.

(Full-state controllers — lqr, mpc, pole_placement, smc, hinf, mrac, cascaded_pid — track the
speed command; the tilt-only ``pid`` balances but does not drive, by design.)
"""

from __future__ import annotations

import numpy as np

from ..config import TWIPParams
from ..controllers import build_controller


class TWIPController:
    """Balance + yaw inner loop producing ``(tau_L, tau_R)``."""

    def __init__(
        self,
        params: TWIPParams,
        balance: str = "lqr",
        balance_kwargs: dict | None = None,
        k_yaw: float = 8.0,
    ):
        self.p = params
        self.balance_name = balance
        self.balance = build_controller(balance, params.base, **(balance_kwargs or {}))
        self.k_yaw = float(k_yaw)
        self._k_t = params.track / (2.0 * params.base.r)

    def reset(self) -> None:
        self.balance.reset()

    def compute(self, state: np.ndarray, v_des: float, yaw_rate_des: float, t: float = 0.0):
        """Return ``(tau_L, tau_R)`` for the given planar state and commands."""
        _, _, _, theta, v, psi_dot, theta_dot = state

        # Longitudinal: regulate speed + balance (position error held at 0).
        err = np.array([0.0, v - v_des, theta, theta_dot])
        tau_sum = self.balance.compute(err, t)

        # Yaw: feed-forward against damping + proportional on yaw-rate error.
        tau_ff = self.p.b_yaw * yaw_rate_des / self._k_t
        tau_diff = tau_ff + self.k_yaw * (yaw_rate_des - psi_dot)

        tau_L = 0.5 * (tau_sum - tau_diff)
        tau_R = 0.5 * (tau_sum + tau_diff)
        return tau_L, tau_R
