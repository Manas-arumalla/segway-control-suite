"""MuJoCo TWIP on uneven terrain — a heightfield ground the balancing robot must traverse.

Builds the rolling-contact TWIP (see ``mujoco_twip.py``) on a MuJoCo **heightfield** generated
from a :class:`~segway.navigation.terrain.Terrain`. The robot has no terrain sensing — the
slopes act as disturbances its balancing controller must reject — so this is where the
robustness story meets navigation. The same height map drives both the analytic queries and
the simulated ground, and the robot is spawned on the local surface.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from ..config import SimConfig, TWIPParams
from ..navigation.control import TWIPController
from ..navigation.sim import TWIPTrajectory
from ..navigation.terrain import Terrain
from .mujoco_twip import TWIP_PATH, MuJoCoTWIP

CommandFn = Callable[[float, np.ndarray], "tuple[float, float]"]
StopFn = Callable[[np.ndarray], bool]

_FLOOR_RE = re.compile(r'<geom name="floor".*?/>', re.DOTALL)


def twip_terrain_xml(terrain: Terrain, world=None, start=None, goal=None) -> str:
    """TWIP model XML with the flat floor replaced by ``terrain``'s heightfield."""
    base = TWIP_PATH.read_text()
    elev = max(float(terrain.heights.max()), 1e-3)
    rx, ry = terrain.width / 2.0, terrain.height / 2.0
    hfield = (f'<hfield name="terrain" nrow="{terrain.nrow}" ncol="{terrain.ncol}" '
              f'size="{rx} {ry} {elev} 0.5"/>')
    base = base.replace("</asset>", f"  {hfield}\n  </asset>", 1)
    floor = (f'<geom name="floor" type="hfield" hfield="terrain" '
             f'pos="{rx} {ry} 0" material="grid" contype="1" conaffinity="1" '
             f'friction="1.0 0.01 0.001"/>')
    base = _FLOOR_RE.sub(floor, base, count=1)

    extras: list[str] = []
    if world is not None:
        for o in world.obstacles:
            zc = terrain.height_at(o.x, o.y)
            extras.append(f'<geom type="cylinder" pos="{o.x} {o.y} {zc + 0.5}" '
                          f'size="{o.r} 0.5" rgba="0.85 0.3 0.3 0.55" '
                          f'contype="0" conaffinity="0"/>')
    if goal is not None:
        zc = terrain.height_at(goal[0], goal[1])
        extras.append(f'<geom type="cylinder" pos="{goal[0]} {goal[1]} {zc + 0.02}" '
                      f'size="0.18 0.02" rgba="0.95 0.55 0.1 0.8" '
                      f'contype="0" conaffinity="0"/>')
    if extras:
        block = "\n    " + "\n    ".join(extras) + "\n  </worldbody>"
        base = base.replace("  </worldbody>", block, 1)
    return base


class MuJoCoTWIPTerrain(MuJoCoTWIP):
    """Rolling-contact TWIP standing on a generated heightfield."""

    def __init__(self, params: TWIPParams, terrain: Terrain, world=None,
                 start=None, goal=None):
        self.terrain = terrain
        super().__init__(params, xml_string=twip_terrain_xml(terrain, world, start, goal))
        self._set_heightfield(terrain)

    def _set_heightfield(self, terrain: Terrain) -> None:
        mj, m = self._mj, self.model
        elev = max(float(terrain.heights.max()), 1e-3)
        data = (terrain.heights / elev).astype(np.float64).ravel()
        m.hfield_data[:] = np.clip(data, 0.0, 1.0)   # single hfield ("terrain")
        mj.mj_forward(m, self.data)

    def reset(self, x0: np.ndarray | None = None) -> None:
        s = np.zeros(7) if x0 is None else np.asarray(x0, dtype=float).copy()
        # spawn on the local surface so the wheels start in contact, not buried or floating.
        super().reset(s)
        z = self.terrain.height_at(float(s[0]), float(s[1])) + self.params.base.r
        self.data.qpos[2] = z
        self._mj.mj_forward(self.model, self.data)


def simulate_twip_terrain(
    params: TWIPParams,
    controller: TWIPController,
    command_fn: CommandFn,
    terrain: Terrain,
    sim: SimConfig | None = None,
    x0: np.ndarray | None = None,
    command_dt: float = 0.1,
    stop_fn: StopFn | None = None,
    world=None,
    plant: MuJoCoTWIPTerrain | None = None,
    step_callback: Callable[[MuJoCoTWIPTerrain, float], None] | None = None,
) -> TWIPTrajectory:
    """Closed-loop TWIP rollout over ``terrain`` (heightfield contact), returning a
    :class:`TWIPTrajectory`. Mirrors :func:`segway.sim.mujoco_twip.simulate_twip_mujoco`."""
    sim = sim or SimConfig(dt=0.002, duration=40.0, fall_angle=1.2)
    controller.reset()
    if plant is None:
        plant = MuJoCoTWIPTerrain(params, terrain, world=world)
    plant.model.opt.timestep = sim.dt
    plant.reset(x0)

    dt = sim.dt
    n_steps = int(round(sim.duration / dt))
    control_dt = sim.control_dt

    ts: list[float] = []
    xs: list[np.ndarray] = []
    cmds: list[tuple[float, float]] = []
    us: list[tuple[float, float]] = []
    u = (0.0, 0.0)
    cmd = (0.0, 0.0)
    last_ctrl_t = -np.inf
    last_cmd_t = -np.inf
    fell = False
    reached = False

    for i in range(n_steps + 1):
        t = i * dt
        state = plant.state
        if abs(state[3]) > sim.fall_angle:
            fell = True
            break
        if stop_fn is not None and stop_fn(state):
            reached = True
            break
        if (t - last_cmd_t) >= command_dt - 1e-12:
            cmd = tuple(command_fn(t, state))
            last_cmd_t = t
        if control_dt is None or (t - last_ctrl_t) >= control_dt - 1e-12:
            u = controller.compute(state, cmd[0], cmd[1], t)
            last_ctrl_t = t
        plant.apply(u)
        if i % sim.record_every == 0:
            ts.append(t)
            xs.append(state.copy())
            cmds.append(cmd)
            us.append(u)
        if step_callback is not None:
            step_callback(plant, t)
        if i < n_steps:
            plant.step()

    if not ts:
        ts.append(0.0)
        xs.append(plant.state.copy())
        cmds.append(cmd)
        us.append(u)

    return TWIPTrajectory(
        t=np.asarray(ts),
        states=np.asarray(xs),
        commands=np.asarray(cmds),
        torques=np.asarray(us),
        fell=fell,
        params=params,
        controller_name=getattr(controller, "balance_name", "unknown"),
        reached=reached,
    )
