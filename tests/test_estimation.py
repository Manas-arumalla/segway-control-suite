"""Sensor + estimator behavior, and closed-loop control on estimates."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.estimation import ExtendedKalmanFilter, LinearKalmanFilter, SensorModel, SensorSpec
from segway.sim import Scenario, simulate


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_sensor_measures_position_and_tilt(params):
    sensors = SensorModel(SensorSpec(pos_noise_std=0.0, tilt_noise_std=0.0), seed=1)
    state = np.array([0.3, 1.0, 0.12, -0.5])
    y = sensors.measure(state)
    assert y.shape == (2,)
    assert np.allclose(y, [0.3, 0.12])  # noiseless => exact position & tilt


@pytest.mark.parametrize("filter_cls", [LinearKalmanFilter, ExtendedKalmanFilter])
def test_estimator_recovers_velocities(params, filter_cls):
    """With a stabilizing controller, the estimator's state error stays small."""
    dt = 0.01
    sensors = SensorModel(SensorSpec(), seed=0)
    est = filter_cls(params, dt=dt, R=sensors.R)
    ctrl = build_controller("lqr", params)
    sim = SimConfig(duration=8.0, control_dt=dt)

    traj = simulate(params, ctrl, Scenario.balance(0.1), sim, sensors=sensors, estimator=est)

    assert not traj.fell
    err = traj.estimation_error
    assert err is not None
    # Use the back half of the run (after the filter has converged).
    tail = err[len(err) // 2 :]
    rms = np.sqrt(np.mean(tail**2, axis=0))
    assert rms[2] < 0.02, f"tilt estimate RMS error too large: {rms[2]:.4f} rad"
    assert rms[3] < 0.25, f"tilt-rate estimate RMS error too large: {rms[3]:.4f} rad/s"


def test_control_on_estimates_still_balances(params):
    """LQR running on noisy estimates (not ground truth) still stabilizes the robot."""
    dt = 0.01
    sensors = SensorModel(SensorSpec(), seed=3)
    est = LinearKalmanFilter(params, dt=dt, R=sensors.R)
    ctrl = build_controller("lqr", params)
    sim = SimConfig(duration=8.0, control_dt=dt)
    traj = simulate(params, ctrl, Scenario.kick(impulse=0.5, time=3.0), sim,
                    sensors=sensors, estimator=est)
    assert not traj.fell
    assert abs(traj.theta[-1]) < 0.06
