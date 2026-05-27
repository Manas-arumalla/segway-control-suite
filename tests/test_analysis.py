"""Region of attraction and Monte-Carlo robustness analysis."""

from __future__ import annotations

import pytest

from segway.analysis import compute_roa, monte_carlo
from segway.config import RobotParams
from segway.controllers import build_controller


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_roa_recovers_center_and_has_area(params):
    ctrl = build_controller("lqr", params)
    res = compute_roa(params, ctrl, n_theta=7, n_thetadot=7, duration=3.0)
    assert 0.0 <= res.area_fraction <= 1.0
    ci, cj = res.theta_vals.size // 2, res.thetadot_vals.size // 2
    assert res.grid[ci, cj]  # at-rest upright is always recoverable
    assert res.area_fraction > 0.2  # LQR recovers a meaningful region


def test_monte_carlo_lqr_is_robust(params):
    res = monte_carlo("lqr", n=20, seed=0)
    assert res.fell.shape == (20,)
    assert 0.0 <= res.success_rate <= 1.0
    assert res.success_rate > 0.5  # LQR tolerates moderate parameter error
