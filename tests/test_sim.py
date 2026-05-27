"""Simulator behavior: shapes, fall detection, and reference tracking."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.controllers.base import Controller
from segway.sim import Scenario, simulate


class ZeroController(Controller):
    """Applies no torque — used to confirm the open-loop plant falls over."""

    name = "zero"

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        return 0.0


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_trajectory_shapes(params):
    ctrl = build_controller("lqr", params)
    traj = simulate(params, ctrl, Scenario.balance(0.1), SimConfig(duration=2.0))
    assert traj.t.ndim == 1
    assert traj.states.shape == (traj.t.size, 4)
    assert traj.controls.shape == (traj.t.size,)
    assert np.all(np.diff(traj.t) > 0)  # monotonic time


def test_open_loop_falls(params):
    traj = simulate(params, ZeroController(params), Scenario.balance(0.2), SimConfig(duration=5.0))
    assert traj.fell


def test_lqr_setpoint_tracking(params):
    """LQR should drive the base toward a 1 m position setpoint."""
    ctrl = build_controller("lqr", params)
    traj = simulate(params, ctrl, Scenario.setpoint(x_ref=1.0), SimConfig(duration=12.0))
    assert not traj.fell
    assert abs(traj.x[-1] - 1.0) < 0.1
    assert abs(traj.theta[-1]) < 0.05


def test_disturbance_recovery(params):
    """A kick should be survived and recovered from."""
    ctrl = build_controller("lqr", params)
    traj = simulate(params, ctrl, Scenario.kick(impulse=0.6, time=2.0), SimConfig(duration=8.0))
    assert not traj.fell
    assert abs(traj.theta[-1]) < 0.05
