"""Controllers build correctly and actually stabilize the nonlinear plant."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller, list_controllers
from segway.models import linearize
from segway.sim import Scenario, simulate


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_registry_nonempty():
    names = list_controllers()
    assert {"pid", "lqr", "pole_placement", "smc"}.issubset(set(names))


@pytest.mark.parametrize("name", list_controllers())
def test_controller_balances_from_small_tilt(params, name):
    """Every registered controller recovers the body to near-upright from a 0.1 rad tilt."""
    ctrl = build_controller(name, params)
    traj = simulate(params, ctrl, Scenario.balance(tilt=0.1), SimConfig(duration=8.0))
    assert not traj.fell, f"{name} fell over"
    assert abs(traj.theta[-1]) < 0.05, f"{name} did not settle the tilt (final={traj.theta[-1]:.3f})"


def test_lqr_closed_loop_is_stable(params):
    ctrl = build_controller("lqr", params)
    A, B = linearize(params)
    eig = np.linalg.eigvals(A - B @ ctrl.K)
    assert np.max(eig.real) < 0.0


def test_pole_placement_achieves_requested_poles(params):
    poles = [-2.0, -3.0, -4.0, -5.0]
    ctrl = build_controller("pole_placement", params, poles=poles)
    A, B = linearize(params)
    achieved = np.sort(np.linalg.eigvals(A - B @ ctrl.K).real)
    assert np.allclose(achieved, np.sort(poles), atol=1e-6)


def test_pole_placement_rejects_repeated_poles(params):
    with pytest.raises(ValueError):
        build_controller("pole_placement", params, poles=[-3.0, -3.0, -4.0, -5.0])


def test_unknown_controller_raises(params):
    with pytest.raises(KeyError):
        build_controller("does_not_exist", params)
