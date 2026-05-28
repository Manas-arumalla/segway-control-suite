"""Planar TWIP dynamics + balance/yaw navigation inner loop (NAV-1)."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import SimConfig, TWIPParams
from segway.models.dynamics import nonlinear_dynamics
from segway.models.twip import twip_dynamics, yaw_linearization
from segway.navigation import TWIPController, simulate_twip


@pytest.fixture
def tp() -> TWIPParams:
    return TWIPParams()


def test_upright_rest_is_equilibrium(tp):
    qdot = twip_dynamics(np.zeros(7), (0.0, 0.0), tp)
    assert np.allclose(qdot, 0.0, atol=1e-12)


def test_longitudinal_matches_1d_model(tp):
    """The TWIP balance subsystem must equal the validated 1-D dynamics (driven by tau_sum)."""
    v, theta, theta_dot, tau_sum = 0.3, 0.15, -0.2, 4.0
    state = np.array([1.0, 2.0, 0.5, theta, v, 0.0, theta_dot])  # x,y,psi arbitrary
    d = twip_dynamics(state, (tau_sum / 2, tau_sum / 2), tp)      # tau_L+tau_R = tau_sum
    ref = nonlinear_dynamics(np.array([0.0, v, theta, theta_dot]), tau_sum, tp.base)
    assert d[4] == pytest.approx(ref[1])   # v_dot
    assert d[6] == pytest.approx(ref[3])   # theta_ddot


def test_kinematics(tp):
    state = np.array([0.0, 0.0, np.pi / 2, 0.0, 2.0, 0.0, 0.0])  # heading +y, v=2
    d = twip_dynamics(state, (0.0, 0.0), tp)
    assert d[0] == pytest.approx(0.0, abs=1e-9)   # x_dot ~ 0
    assert d[1] == pytest.approx(2.0)             # y_dot = v


def test_yaw_linearization_shape_and_sign(tp):
    A, B = yaw_linearization(tp)
    assert A.shape == (2, 2) and B.shape == (2, 1)
    assert B[1, 0] > 0     # positive differential torque accelerates yaw


@pytest.mark.parametrize("balance", ["lqr", "mpc"])
def test_drive_forward_while_balancing(tp, balance):
    ctrl = TWIPController(tp, balance=balance)
    traj = simulate_twip(tp, ctrl, lambda t, s: (0.5, 0.0), SimConfig(duration=8.0))
    assert not traj.fell
    assert abs(traj.theta[-1]) < 0.1                 # stayed upright
    assert abs(traj.v[-1] - 0.5) < 0.15              # tracked the speed command
    assert traj.x[-1] > 1.5                          # actually moved forward


def test_turn_in_place_while_balancing(tp):
    ctrl = TWIPController(tp, balance="lqr")
    traj = simulate_twip(tp, ctrl, lambda t, s: (0.0, 0.5), SimConfig(duration=8.0))
    assert not traj.fell
    assert abs(traj.theta[-1]) < 0.1
    assert traj.psi[-1] > 2.0                        # turned (yaw-rate tracking)
    assert abs(traj.v[-1]) < 0.15                    # stayed roughly in place


def test_drive_in_an_arc(tp):
    ctrl = TWIPController(tp, balance="lqr")
    traj = simulate_twip(tp, ctrl, lambda t, s: (0.4, 0.3), SimConfig(duration=8.0))
    assert not traj.fell
    assert traj.psi[-1] > 1.0
    assert np.hypot(traj.x[-1], traj.y[-1]) > 1.0    # traversed a curved path
