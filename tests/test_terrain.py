"""Uneven-terrain generation (pure) + heightfield traversal (NAV-6, MuJoCo)."""

from __future__ import annotations

import numpy as np
import pytest

from segway.navigation.terrain import build_terrain, generate_terrain, list_terrains


# ===== pure terrain (always run) =========================================
def test_terrain_registry():
    assert set(list_terrains()) == {"flat", "gentle", "moderate", "rough", "ramp"}


def test_flat_terrain_is_flat():
    t = build_terrain("flat")
    assert t.amplitude < 1e-6
    assert abs(t.height_at(1.0, 2.0) - t.height_at(7.0, 6.0)) < 1e-6


def test_roughness_orders_by_amplitude():
    amps = {n: build_terrain(n).amplitude for n in ("gentle", "moderate", "rough")}
    assert amps["gentle"] < amps["moderate"] < amps["rough"]


def test_height_query_matches_grid_and_is_bounded():
    t = generate_terrain(amplitude=0.1, seed=5)
    assert np.isclose(t.height_at(0.0, 0.0), t.heights[0, 0], atol=1e-9)
    lo, hi = float(t.heights.min()), float(t.heights.max())
    vals = [t.height_at(x, y) for x in np.linspace(0, t.width, 7) for y in np.linspace(0, t.height, 7)]
    assert lo - 1e-9 <= min(vals) and max(vals) <= hi + 1e-9


def test_ramp_has_grade_along_x():
    t = build_terrain("ramp")
    assert t.height_at(7.0, 4.0) > t.height_at(1.0, 4.0)     # uphill toward +x
    assert t.max_slope() > build_terrain("gentle").max_slope()


def test_generation_is_deterministic():
    a = generate_terrain(amplitude=0.08, seed=7).heights
    b = generate_terrain(amplitude=0.08, seed=7).heights
    assert np.array_equal(a, b)


# ===== heightfield traversal (MuJoCo) ====================================
# These modules import cleanly without MuJoCo (it loads lazily only when a plant is built), so
# the pure terrain tests above always run; the heightfield tests below skip without the sim extra.
import importlib.util  # noqa: E402

from segway.config import SimConfig, TWIPParams  # noqa: E402
from segway.navigation import Obstacle, World, navigate  # noqa: E402
from segway.navigation.control import TWIPController  # noqa: E402
from segway.sim.mujoco_terrain import MuJoCoTWIPTerrain, simulate_twip_terrain  # noqa: E402

requires_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None, reason="needs the sim extra (mujoco)")


def _drive(terrain_name, v=0.4, duration=5.0, x0=(2.0, 4.0)):
    p = TWIPParams()
    t = build_terrain(terrain_name)
    ctrl = TWIPController(p, balance="lqr")
    start = np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0, 0.0])
    sim = SimConfig(dt=0.002, duration=duration, fall_angle=1.2, record_every=50)
    return simulate_twip_terrain(p, ctrl, lambda tt, s: (v, 0.0), t, sim=sim, x0=start)


@requires_mujoco
def test_spawn_sits_on_surface():
    p = TWIPParams()
    t = build_terrain("moderate")
    plant = MuJoCoTWIPTerrain(p, t)
    plant.reset(np.array([4.0, 4.0, 0, 0, 0, 0, 0]))
    expected = t.height_at(4.0, 4.0) + p.base.r
    assert abs(float(plant.data.qpos[2]) - expected) < 1e-6


@requires_mujoco
@pytest.mark.parametrize("name", ["gentle", "moderate", "rough"])
def test_balances_and_drives_over_terrain(name):
    tr = _drive(name)
    assert not tr.fell, f"{name}: fell over"
    assert tr.x[-1] - 2.0 > 0.6, f"{name}: did not make forward progress"
    assert np.max(np.abs(tr.theta)) < 0.4


@requires_mujoco
def test_climbs_a_ramp_without_falling():
    tr = _drive("ramp", v=0.4, duration=5.0)
    assert not tr.fell
    assert tr.x[-1] - 2.0 > 0.3        # still climbs the grade, just slower


@requires_mujoco
def test_navigate_to_goal_over_terrain_with_obstacle():
    world = World(width=8.0, height=8.0, resolution=0.2, robot_radius=0.25,
                  obstacles=[Obstacle(4.0, 4.0, 0.6)])
    res = navigate(TWIPParams(), world, (1.0, 4.0), (7.0, 4.0), backend="mujoco",
                   terrain=build_terrain("moderate"),
                   sim=SimConfig(dt=0.002, duration=40.0, fall_angle=1.2))
    assert res.planned and res.success and not res.fell


def test_terrain_navigation_requires_mujoco_backend():
    world = World(width=8.0, height=8.0, obstacles=[])
    with pytest.raises(ValueError):
        navigate(TWIPParams(), world, (1.0, 4.0), (7.0, 4.0),
                 backend="analytic", terrain=build_terrain("gentle"))
