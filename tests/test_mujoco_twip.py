"""MuJoCo TWIP backend (NAV-2): a real rolling-contact robot, cross-checked vs the analytic model.

These confirm that the same analytic-model LQR that balances ``twip_dynamics`` also stabilizes
and steers the full free-floating, contact-driven MuJoCo TWIP — and that the planar state is
read back correctly. Skipped when MuJoCo is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from segway.config import SimConfig, TWIPParams  # noqa: E402
from segway.navigation import Obstacle, World, navigate  # noqa: E402
from segway.navigation.control import TWIPController  # noqa: E402
from segway.navigation.sim import simulate_twip  # noqa: E402
from segway.sim.mujoco_twip import MuJoCoTWIP, simulate_twip_mujoco, twip_world_xml  # noqa: E402

_SIM = SimConfig(dt=0.002, duration=5.0, fall_angle=1.2, record_every=50)


def _run(command, x0=None, duration=5.0):
    p = TWIPParams()
    ctrl = TWIPController(p, balance="lqr")
    sim = SimConfig(dt=0.002, duration=duration, fall_angle=1.2, record_every=50)
    return simulate_twip_mujoco(p, ctrl, command, sim=sim, x0=x0)


def test_model_loads_with_expected_dofs():
    plant = MuJoCoTWIP(TWIPParams())
    assert plant.model.nq == 9 and plant.model.nv == 8 and plant.model.nu == 2  # free+2 wheels


def test_state_roundtrip():
    plant = MuJoCoTWIP(TWIPParams())
    x0 = np.array([1.5, -0.7, 0.6, 0.05, 0.0, 0.0, 0.0])
    plant.reset(x0)
    s = plant.state
    assert np.allclose(s[:4], x0[:4], atol=1e-3), f"pose mismatch: {np.round(s, 4)}"


def test_balances_from_small_tilt():
    x0 = np.zeros(7)
    x0[3] = 0.05
    traj = _run(lambda t, s: (0.0, 0.0), x0=x0, duration=4.0)
    assert not traj.fell
    assert np.max(np.abs(traj.theta)) < 0.2


def test_tracks_forward_speed():
    traj = _run(lambda t, s: (0.5, 0.0), duration=6.0)
    assert not traj.fell
    assert traj.x[-1] > 1.5                          # drove forward
    assert 0.3 < float(np.mean(traj.v[-20:])) < 0.8  # near the commanded 0.5 m/s


def test_turns_in_place():
    traj = _run(lambda t, s: (0.0, 0.6), duration=5.0)
    assert not traj.fell
    assert traj.psi[-1] > 1.0                                   # heading advanced
    assert float(np.mean(traj.states[-20:, 5])) > 0.3          # positive yaw rate


def test_crosscheck_analytic_vs_mujoco_forward():
    """Both backends, same LQR + forward command: both balance and drive forward."""
    p = TWIPParams()
    cmd = lambda t, s: (0.5, 0.0)  # noqa: E731
    sim = SimConfig(dt=0.002, duration=6.0, fall_angle=1.2, record_every=50)

    a = simulate_twip(p, TWIPController(p, balance="lqr"), cmd, sim=sim)
    m = simulate_twip_mujoco(p, TWIPController(p, balance="lqr"), cmd, sim=sim)

    assert not a.fell and not m.fell
    assert a.x[-1] > 1.0 and m.x[-1] > 1.0                # both make real forward progress
    assert np.max(np.abs(a.theta)) < 0.5 and np.max(np.abs(m.theta)) < 0.5
    # Same order of magnitude of travel (contact/ wheel inertia differ, so allow a wide band).
    assert 0.4 < m.x[-1] / a.x[-1] < 2.5


def test_world_scene_xml_loads_with_obstacles():
    world = World(width=8.0, height=4.0, resolution=0.2, robot_radius=0.25,
                  obstacles=[Obstacle(4.0, 2.0, 0.6)])
    xml = twip_world_xml(world, start=(1.0, 2.0), goal=(7.0, 2.0))
    assert xml.count("cylinder") >= 2          # one obstacle + the goal disc
    plant = MuJoCoTWIP(TWIPParams(), xml_string=xml)   # must compile and apply params
    assert plant.model.nu == 2


def test_navigate_mujoco_backend_reaches_goal():
    world = World(width=8.0, height=4.0, resolution=0.2, robot_radius=0.25,
                  obstacles=[Obstacle(4.0, 2.0, 0.6)])
    res = navigate(TWIPParams(), world, (1.0, 2.0), (7.0, 2.0), backend="mujoco",
                   sim=SimConfig(dt=0.002, duration=30.0, fall_angle=1.2))
    assert res.planned and res.success and not res.fell
