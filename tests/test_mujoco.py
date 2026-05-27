"""MuJoCo backend: independent physics cross-check and rendering smoke test.

These are skipped automatically where the optional MuJoCo / imageio extras are absent
(e.g. the lean CI environment).
"""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.sim import Scenario

mj = pytest.importorskip("mujoco")
from segway.sim.mujoco_backend import MuJoCoPlant, simulate_mujoco  # noqa: E402


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


def test_mujoco_plant_loads_and_steps(params):
    plant = MuJoCoPlant(params)
    plant.reset(np.array([0.0, 0.0, 0.1, 0.0]))
    assert plant.model.nq == 2 and plant.model.nv == 2
    assert np.allclose(plant.state, [0.0, 0.0, 0.1, 0.0])
    plant.apply(0.0)
    plant.step()


def test_lqr_designed_on_analytic_stabilizes_mujoco(params):
    """The cross-check: a controller designed on the analytic model also balances the
    independent MuJoCo plant — strong evidence the two models agree."""
    ctrl = build_controller("lqr", params)
    traj = simulate_mujoco(params, ctrl, Scenario.balance(0.1), SimConfig(duration=8.0))
    assert not traj.fell
    assert abs(traj.theta[-1]) < 0.05


def test_mujoco_open_loop_falls(params):
    class Zero(build_controller("lqr", params).__class__.__bases__[0]):  # Controller
        name = "zero"

        def compute(self, state, t=0.0):
            return 0.0

    traj = simulate_mujoco(params, Zero(params), Scenario.balance(0.2), SimConfig(duration=5.0))
    assert traj.fell


def test_cartpole_swingup_model_clears_floor(params):
    """The swing-up cart-pole model elevates the pivot so a hanging body clears the floor,
    and a controller designed on the analytic model still works on this geometry."""
    from segway.sim.mujoco_backend import CARTPOLE_PATH, MuJoCoPlant, simulate_mujoco

    plant = MuJoCoPlant(params, xml_path=CARTPOLE_PATH)
    assert plant.model.nq == 2 and plant.model.nv == 2
    base = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_BODY, "base")
    # Pivot must sit higher than the pendulum length so pointing down doesn't clip z=0.
    assert plant.model.body_pos[base][2] >= params.l + 0.1

    traj = simulate_mujoco(params, build_controller("lqr", params), Scenario.balance(0.1),
                           SimConfig(duration=4.0), xml_path=CARTPOLE_PATH)
    assert not traj.fell and abs(traj.theta[-1]) < 0.05


def test_render_smoke(params, tmp_path):
    pytest.importorskip("imageio")
    from segway.viz import render_rollout

    out = tmp_path / "rollout.gif"
    try:
        render_rollout(params, build_controller("lqr", params), Scenario.balance(0.1),
                       SimConfig(duration=1.0), path=out, width=160, height=120, fps=10)
    except Exception as exc:  # no GL context available (e.g. headless CI without EGL)
        pytest.skip(f"rendering unavailable: {exc}")
    assert out.exists() and out.stat().st_size > 0
