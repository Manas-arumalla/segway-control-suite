"""Planar two-wheeled inverted pendulum (TWIP) dynamics — the model used for navigation.

This lifts the 1-D wheeled inverted pendulum into the plane so the robot can drive to goals
and turn while balancing. State (7):

    [x, y, psi, theta, v, psi_dot, theta_dot]
      x, y       : position of the wheel-axle midpoint [m]
      psi        : yaw / heading [rad]
      theta      : body pitch from upright [rad] (0 = balanced)
      v          : forward speed along the heading [m/s]
      psi_dot    : yaw rate [rad/s]
      theta_dot  : pitch rate [rad/s]

Input: ``(tau_L, tau_R)`` — left/right wheel torques [N*m].

The dynamics approximately decouple:
  * **longitudinal / balance** (``v, theta``) is driven by the wheel-torque **sum**
    ``tau_L + tau_R`` and is *identical* to the 1-D model — so every existing balancing
    controller is reused unchanged via :func:`segway.models.dynamics.nonlinear_dynamics`;
  * **yaw** (``psi``) is driven by the wheel-torque **difference** ``tau_R - tau_L`` acting
    at the wheel track, a simple damped second-order system.

This decoupling is the standard TWIP approximation (valid for moderate speed and small
pitch) and keeps the model honest, reusable, and easy to validate.
"""

from __future__ import annotations

import numpy as np

from ..config import TWIPParams
from .dynamics import linearize, nonlinear_dynamics

STATE_NAMES = ("x", "y", "psi", "theta", "v", "psi_dot", "theta_dot")


def twip_dynamics(state: np.ndarray, torques: tuple[float, float], p: TWIPParams) -> np.ndarray:
    """Continuous-time state derivative for the planar TWIP."""
    _, _, psi, theta, v, psi_dot, theta_dot = state
    tau_L, tau_R = torques
    tau_sum = tau_L + tau_R
    tau_diff = tau_R - tau_L

    # Longitudinal/balance: reuse the validated 1-D dynamics with state [s=0, v, theta, theta_dot].
    longitudinal = nonlinear_dynamics(np.array([0.0, v, theta, theta_dot]), tau_sum, p.base)
    v_dot = longitudinal[1]
    theta_ddot = longitudinal[3]

    # Yaw: differential wheel forces (tau/r) at +/- track/2 give a yaw moment.
    k_t = p.track / (2.0 * p.base.r)
    psi_ddot = (k_t * tau_diff - p.b_yaw * psi_dot) / p.I_yaw

    return np.array([
        v * np.cos(psi),   # x_dot
        v * np.sin(psi),   # y_dot
        psi_dot,           # psi_dot
        theta_dot,         # theta_dot
        v_dot,             # v_dot
        psi_ddot,          # psi_ddot
        theta_ddot,        # theta_ddot
    ])


def longitudinal_linearization(p: TWIPParams) -> tuple[np.ndarray, np.ndarray]:
    """Linearized balance subsystem ``[s, v, theta, theta_dot]`` vs ``tau_sum`` (the 1-D model)."""
    return linearize(p.base)


def yaw_linearization(p: TWIPParams) -> tuple[np.ndarray, np.ndarray]:
    """Linearized yaw subsystem ``[psi, psi_dot]`` vs ``tau_diff``."""
    k_t = p.track / (2.0 * p.base.r)
    A = np.array([[0.0, 1.0], [0.0, -p.b_yaw / p.I_yaw]])
    B = np.array([[0.0], [k_t / p.I_yaw]])
    return A, B
