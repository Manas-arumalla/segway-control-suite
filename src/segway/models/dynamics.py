"""Canonical dynamics of the wheeled inverted pendulum.

State ``q = [x, x_dot, theta, theta_dot]`` (``theta`` from the upward vertical).
Input ``u = tau`` (motor torque). A single motor torque produces a traction force
``F = tau/r`` on the base and a reaction torque ``-tau`` on the body.

Full nonlinear equations of motion (``c = cos theta``, ``s = sin theta``)::

    [ M+m     m*l*c ] [ x_ddot     ]   [ tau/r - b_x*x_dot + m*l*s*theta_dot^2 ]
    [ m*l*c  m*l^2+I] [ theta_ddot ] = [ -tau   + m*g*l*s   - b_theta*theta_dot ]

See ``docs/implementation-notes.md`` (ND-1) for the derivation and the equivalent
SymPy version in ``segway.models.symbolic``.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotParams

STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")

# Default measurement matrix: we observe cart position and body tilt.
DEFAULT_C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def state_names() -> tuple[str, str, str, str]:
    """Names of the four state components, in order."""
    return STATE_NAMES


def nonlinear_dynamics(state: np.ndarray, u: float, p: RobotParams) -> np.ndarray:
    """Continuous-time state derivative ``q_dot = f(q, u)`` for the full nonlinear plant.

    Args:
        state: ``[x, x_dot, theta, theta_dot]``.
        u: motor torque ``tau`` [N*m].
        p: robot physical parameters.

    Returns:
        ``[x_dot, x_ddot, theta_dot, theta_ddot]``.
    """
    _, x_dot, theta, theta_dot = state
    M, m, l, I, g, r = p.M, p.m_pend, p.l, p.I_pend, p.g, p.r
    b_x, b_theta = p.b_x, p.b_theta

    c, s = np.cos(theta), np.sin(theta)
    F = u / r

    m11 = M + m
    m12 = m * l * c
    m22 = m * l**2 + I
    det = m11 * m22 - m12**2

    rhs1 = F - b_x * x_dot + m * l * s * theta_dot**2
    rhs2 = -u + m * g * l * s - b_theta * theta_dot

    x_ddot = (m22 * rhs1 - m12 * rhs2) / det
    theta_ddot = (-m12 * rhs1 + m11 * rhs2) / det
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def linearize(p: RobotParams) -> tuple[np.ndarray, np.ndarray]:
    """Analytic linearization about the upright equilibrium (``theta = 0``).

    Returns the continuous-time ``(A, B)`` of ``q_dot = A q + B u``.
    """
    M, m, l, I, g, r = p.M, p.m_pend, p.l, p.I_pend, p.g, p.r
    b_x, b_theta = p.b_x, p.b_theta
    D = (M + m) * (I + m * l**2) - (m * l) ** 2

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -(I + m * l**2) * b_x / D, -((m * l) ** 2) * g / D, m * l * b_theta / D],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, m * l * b_x / D, (M + m) * m * g * l / D, -(M + m) * b_theta / D],
        ]
    )
    B = np.array(
        [
            [0.0],
            [((I + m * l**2) / r + m * l) / D],
            [0.0],
            [-(m * l / r + (M + m)) / D],
        ]
    )
    return A, B


def linearize_numeric(
    p: RobotParams,
    x_eq: np.ndarray | None = None,
    u_eq: float = 0.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Numeric linearization via central finite differences of ``nonlinear_dynamics``.

    Used to *verify* :func:`linearize`. At the default upright equilibrium the two must
    agree to within a small tolerance (asserted in the test suite).
    """
    if x_eq is None:
        x_eq = np.zeros(4)
    n = x_eq.size

    A = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        f_plus = nonlinear_dynamics(x_eq + dx, u_eq, p)
        f_minus = nonlinear_dynamics(x_eq - dx, u_eq, p)
        A[:, j] = (f_plus - f_minus) / (2 * eps)

    f_plus = nonlinear_dynamics(x_eq, u_eq + eps, p)
    f_minus = nonlinear_dynamics(x_eq, u_eq - eps, p)
    B = ((f_plus - f_minus) / (2 * eps)).reshape(n, 1)
    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The controllability matrix ``[B, AB, A^2 B, ..., A^(n-1) B]``."""
    n = A.shape[0]
    cols = [B]
    for _ in range(1, n):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """The observability matrix stacked ``[C; CA; CA^2; ...; CA^(n-1)]``."""
    n = A.shape[0]
    rows = [C]
    for _ in range(1, n):
        rows.append(rows[-1] @ A)
    return np.vstack(rows)


def is_controllable(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``(A, B)`` is fully controllable (controllability matrix has full rank)."""
    return int(np.linalg.matrix_rank(controllability_matrix(A, B), tol=tol)) == A.shape[0]


def is_observable(A: np.ndarray, C: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``(A, C)`` is fully observable."""
    return int(np.linalg.matrix_rank(observability_matrix(A, C), tol=tol)) == A.shape[0]


def open_loop_poles(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``A`` (open-loop poles). The upright equilibrium is unstable."""
    return np.linalg.eigvals(A)
