"""Min-jerk reference tracking and iLQR trajectory optimization."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.planning import iLQR, min_jerk, min_jerk_reference
from segway.sim import Scenario, simulate


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_min_jerk_profile():
    assert min_jerk(0.0, 1.0, 2.0, 0.0) == 0.0
    assert min_jerk(0.0, 1.0, 2.0, 2.0) == 1.0
    mid = min_jerk(0.0, 1.0, 2.0, 1.0)
    assert 0.4 < mid < 0.6  # symmetric, ~0.5 at midpoint


def test_lqr_tracks_min_jerk_reference(params):
    scen = Scenario(name="track", reference_fn=min_jerk_reference(0.0, 1.0, 4.0))
    traj = simulate(params, build_controller("lqr", params), scen, SimConfig(duration=8.0))
    assert not traj.fell
    assert abs(traj.x[-1] - 1.0) < 0.1  # reached the target position


def test_ilqr_reaches_upright(params):
    opt = iLQR(params, dt=0.02)
    res = opt.fit(x0=np.array([0.0, 0.0, 0.5, 0.0]), N=100, iters=60)
    assert res.converged
    assert abs(res.xs[-1][2]) < 0.05      # final tilt ~ upright
    assert abs(res.xs[-1][0]) < 0.5       # base near origin
