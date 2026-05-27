"""Symbolic derivation of the plant dynamics (SymPy).

This module derives the linearized ``(A, B)`` by symbolically differentiating the full
nonlinear equations of motion and substituting the upright equilibrium. It is an
*independent* derivation from the hand-written formulas in :func:`segway.models.dynamics.linearize`
(those write the closed-form result; this one differentiates the raw EOM), so agreement
between the two — checked in ``tests/test_dynamics.py`` — is a genuine cross-validation.

Run ``python -m segway.models.symbolic`` to pretty-print the symbolic matrices.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..config import RobotParams


def _nonlinear_state_dynamics():
    """Build the symbolic state-derivative vector f(state, tau) and related symbols."""
    x, x_dot, theta, theta_dot = sp.symbols("x x_dot theta theta_dot", real=True)
    tau = sp.symbols("tau", real=True)
    M, m, l, I, g, r = sp.symbols("M m l I g r", positive=True)
    b_x, b_theta = sp.symbols("b_x b_theta", nonnegative=True)

    c, s = sp.cos(theta), sp.sin(theta)
    F = tau / r

    m11 = M + m
    m12 = m * l * c
    m22 = m * l**2 + I
    det = m11 * m22 - m12**2

    rhs1 = F - b_x * x_dot + m * l * s * theta_dot**2
    rhs2 = -tau + m * g * l * s - b_theta * theta_dot

    x_ddot = (m22 * rhs1 - m12 * rhs2) / det
    theta_ddot = (-m12 * rhs1 + m11 * rhs2) / det

    f = sp.Matrix([x_dot, x_ddot, theta_dot, theta_ddot])
    states = sp.Matrix([x, x_dot, theta, theta_dot])
    syms = (M, m, l, I, g, r, b_x, b_theta)
    return f, states, tau, syms


def symbolic_linearization() -> tuple[sp.Matrix, sp.Matrix, tuple]:
    """Return symbolic ``(A, B, syms)`` linearized about the upright equilibrium."""
    f, states, tau, syms = _nonlinear_state_dynamics()
    A = f.jacobian(states)
    B = f.jacobian(sp.Matrix([tau]))

    # Equilibrium: x_dot = 0, theta = 0, theta_dot = 0, tau = 0 (x itself does not appear).
    eq = {states[1]: 0, states[2]: 0, states[3]: 0, tau: 0}
    A0 = sp.simplify(A.subs(eq))
    B0 = sp.simplify(B.subs(eq))
    return A0, B0, syms


def linearize_symbolic_numeric(p: RobotParams) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the symbolic linearization at concrete parameter values -> NumPy arrays."""
    A0, B0, syms = symbolic_linearization()
    M, m, l, I, g, r, b_x, b_theta = syms
    subs = {
        M: p.M, m: p.m_pend, l: p.l, I: p.I_pend,
        g: p.g, r: p.r, b_x: p.b_x, b_theta: p.b_theta,
    }
    A = np.array(A0.subs(subs).evalf(), dtype=float)
    B = np.array(B0.subs(subs).evalf(), dtype=float)
    return A, B


def main() -> None:
    A0, B0, _ = symbolic_linearization()
    sp.init_printing(use_unicode=True)
    print("Symbolic A matrix (linearized about upright):")
    sp.pprint(A0)
    print("\nSymbolic B matrix:")
    sp.pprint(B0)


if __name__ == "__main__":
    main()
