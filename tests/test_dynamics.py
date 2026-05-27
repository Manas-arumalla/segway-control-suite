"""Validate the canonical dynamics: equilibrium, linearization, and structure.

The headline test cross-checks the hand-written analytic linearization against (a) a
finite-difference Jacobian of the nonlinear EOM and (b) an independent SymPy derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams
from segway.models import (
    DEFAULT_C,
    is_controllable,
    is_observable,
    linearize,
    linearize_numeric,
    nonlinear_dynamics,
    open_loop_poles,
)


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_upright_is_equilibrium(params):
    """f(0, 0) = 0: the upright state with zero torque is an equilibrium."""
    qdot = nonlinear_dynamics(np.zeros(4), 0.0, params)
    assert np.allclose(qdot, 0.0, atol=1e-12)


def test_analytic_matches_finite_difference(params):
    """Analytic (A, B) must equal the numeric linearization of the nonlinear plant."""
    A, B = linearize(params)
    A_num, B_num = linearize_numeric(params)
    assert np.allclose(A, A_num, atol=1e-5)
    assert np.allclose(B, B_num, atol=1e-5)


def test_analytic_matches_symbolic(params):
    """Analytic (A, B) must equal the independent SymPy derivation."""
    sym = pytest.importorskip("segway.models.symbolic")
    A, B = linearize(params)
    A_sym, B_sym = sym.linearize_symbolic_numeric(params)
    assert np.allclose(A, A_sym, atol=1e-9)
    assert np.allclose(B, B_sym, atol=1e-9)


def test_open_loop_is_unstable(params):
    """The upright equilibrium has at least one pole in the right-half plane."""
    A, _ = linearize(params)
    poles = open_loop_poles(A)
    assert np.max(poles.real) > 0.0


def test_controllable_and_observable(params):
    A, B = linearize(params)
    assert is_controllable(A, B)
    assert is_observable(A, DEFAULT_C)


def test_determinant_property_matches(params):
    """RobotParams.D matches the determinant used in the linearization."""
    m, ll, I = params.m_pend, params.l, params.I_pend
    expected = (params.M + m) * (I + m * ll**2) - (m * ll) ** 2
    assert params.D == pytest.approx(expected)
