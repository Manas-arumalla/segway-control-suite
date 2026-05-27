"""Cascaded PID (position holding) and energy swing-up (gets up from hanging)."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.controllers.swingup import _wrap
from segway.sim import Scenario, simulate


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_cascaded_pid_holds_position_better_than_plain_pid(params):
    """Both balance, but the cascaded loop should not drift away in position."""
    scen = Scenario.kick(tilt=0.1, time=3.0, impulse=0.6)
    sim = SimConfig(duration=12.0)
    casc = simulate(params, build_controller("cascaded_pid", params), scen, sim)
    plain = simulate(params, build_controller("pid", params), scen, sim)
    assert not casc.fell and abs(casc.theta[-1]) < 0.05
    assert abs(casc.x[-1]) < 0.5                       # cascaded holds position
    assert abs(casc.x[-1]) < abs(plain.x[-1])          # and beats the plain tilt-only PID


def test_hinf_stabilizes_and_is_optimal(params):
    """H-infinity yields a stable closed loop and a lower worst-case disturbance gain than
    LQR (its design objective)."""
    import numpy as np

    from segway.controllers.hinf import DEFAULT_B1
    from segway.models import linearize

    A, B = linearize(params)
    hinf = build_controller("hinf", params)
    lqr = build_controller("lqr", params, Q=[10, 1, 100, 1], R=0.1)
    assert np.max(np.linalg.eigvals(A - B @ hinf.K).real) < 0  # stable

    Q = np.diag([10.0, 1.0, 100.0, 1.0])
    R = np.array([[0.1]])

    def hinf_norm(K):
        Acl = A - B @ K
        Cz = np.vstack([np.sqrt(Q), -np.sqrt(R) @ K])
        return max(
            np.linalg.svd(Cz @ np.linalg.solve(1j * w * np.eye(4) - Acl, DEFAULT_B1),
                          compute_uv=False)[0]
            for w in np.logspace(-2, 3, 400)
        )

    assert hinf_norm(hinf.K) < hinf_norm(lqr.K)  # H-infinity wins on worst-case gain

    traj = simulate(params, hinf, Scenario.balance(0.2), SimConfig(duration=8.0))
    assert not traj.fell and abs(traj.theta[-1]) < 0.05


def test_mrac_balances_nominal_and_mismatched(params):
    """MRAC (designed on nominal) stays stable on a significantly mismatched plant."""
    from dataclasses import replace

    ctrl = build_controller("mrac", params)
    nom = simulate(params, ctrl, Scenario.balance(0.3), SimConfig(duration=8.0))
    assert not nom.fell and abs(nom.theta[-1]) < 0.05

    mismatched = replace(params, m_pend=params.m_pend * 1.6, l=params.l * 1.3,
                         I_pend=params.I_pend * 1.6)
    ctrl2 = build_controller("mrac", params)  # still designed on nominal
    mis = simulate(mismatched, ctrl2, Scenario.balance(0.3), SimConfig(duration=8.0))
    assert not mis.fell and abs(mis.theta[-1]) < 0.05


def test_swingup_from_hanging_reaches_and_catches(params):
    """From hanging down, the energy controller swings up and the LQR catches it."""
    ctrl = build_controller("swingup", params)
    scen = Scenario(name="swingup", initial_tilt=np.pi, initial_tilt_rate=0.5)
    sim = SimConfig(duration=12.0, fall_angle=50.0)  # disable fall detection during swing-up
    traj = simulate(params, ctrl, scen, sim)
    theta_w = np.array([_wrap(a) for a in traj.theta])
    assert np.min(np.abs(theta_w)) < 0.05    # reached upright
    assert abs(theta_w[-1]) < 0.1            # and was caught / held upright
