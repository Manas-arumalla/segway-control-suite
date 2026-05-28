"""Interactive live 3D view of a controller running in MuJoCo (the "View Model" feature).

Opens a passive MuJoCo viewer and steps the plant in real time under the given controller,
applying disturbances and reference tracking just like the headless runner. Blocks until the
window is closed, so callers (e.g. the GUI) typically run it on a background thread.

Requires the ``sim`` extra (MuJoCo >= 2.3, which provides ``mujoco.viewer``).
"""

from __future__ import annotations

import time

from ..config import RobotParams, SimConfig
from ..controllers.base import Controller
from ..sim.mujoco_backend import MuJoCoPlant
from ..sim.scenarios import Scenario


def live_view(
    params: RobotParams,
    controller: Controller,
    scenario: Scenario | None = None,
    sim: SimConfig | None = None,
    realtime: bool = True,
    xml_path=None,
) -> None:
    """Launch an interactive MuJoCo viewer and drive it with ``controller``.

    ``xml_path`` selects the model geometry (e.g. the cart-pole used for swing-up).
    """
    import mujoco
    import mujoco.viewer

    scenario = scenario or Scenario()
    sim = sim or SimConfig()
    plant = MuJoCoPlant(params, xml_path=xml_path)
    plant.model.opt.timestep = sim.dt
    plant.reset(scenario.initial_state())
    controller.reset()

    dt = sim.dt
    n_steps = int(sim.duration / dt)
    fired: set[int] = set()

    with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
        start = time.time()
        for i in range(n_steps + 1):
            if not viewer.is_running():
                break
            t = i * dt
            state = plant.state
            for j, d in enumerate(scenario.disturbances):
                if j not in fired and abs(t - d.time) < dt / 2:
                    plant.kick(d.impulse)
                    fired.add(j)
                    state = plant.state
            u = controller.compute(state - scenario.reference_at(t), t)
            plant.apply(u)
            plant.step()
            viewer.sync()
            if realtime:
                lag = t - (time.time() - start)
                if lag > 0:
                    time.sleep(lag)


def live_view_navigation(
    params,
    world,
    start,
    goal,
    *,
    balance: str = "lqr",
    planner: str = "a_star",
    follower: str = "pure_pursuit",
    terrain=None,
    sim: SimConfig | None = None,
    command_dt: float = 0.1,
    goal_tol: float = 0.3,
    realtime: bool = True,
):
    """Plan a route, then drive the balancing TWIP along it in a live MuJoCo viewer.

    Obstacles, start, and goal are drawn in the scene; with ``terrain`` given the robot drives
    over the heightfield. Blocks until the goal is reached or the window is closed, then returns
    a :class:`~segway.navigation.navigator.NavResult` for the post-run plots. Requires MuJoCo.
    """
    import time

    import mujoco
    import mujoco.viewer
    import numpy as np

    from ..navigation.navigator import Navigator, NavResult
    from ..navigation.sim import TWIPTrajectory
    from ..sim.mujoco_twip import MuJoCoTWIP, twip_world_xml

    sim = sim or SimConfig(dt=0.002, duration=40.0, fall_angle=1.2)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    nav = Navigator(params, balance=balance, planner=planner, follower=follower, backend="mujoco")
    route = nav.plan(world, start, goal)
    if route is None or len(route) == 0:
        raise ValueError("planner found no path from start to goal")

    psi0 = float(np.arctan2(route[1, 1] - route[0, 1], route[1, 0] - route[0, 0])) \
        if len(route) >= 2 else float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    x0 = np.zeros(7)
    x0[0], x0[1], x0[2] = start[0], start[1], psi0

    if terrain is not None:
        from ..sim.mujoco_terrain import MuJoCoTWIPTerrain
        plant = MuJoCoTWIPTerrain(params, terrain, world=world, start=start, goal=goal)
    else:
        plant = MuJoCoTWIP(params, xml_string=twip_world_xml(world, start, goal))
    plant.model.opt.timestep = sim.dt
    plant.reset(x0)
    nav.controller.reset()
    nav.follower.reset()

    dt = sim.dt
    n_steps = int(sim.duration / dt)
    ts, xs, cmds, us = [], [], [], []
    cmd, u = (0.0, 0.0), (0.0, 0.0)
    last_cmd_t = -np.inf
    fell = reached = False

    with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
        wall0 = time.time()
        for i in range(n_steps + 1):
            if not viewer.is_running():
                break
            t = i * dt
            state = plant.state
            if abs(state[3]) > sim.fall_angle:
                fell = True
                break
            if np.hypot(state[0] - goal[0], state[1] - goal[1]) < goal_tol:
                reached = True
                break
            if (t - last_cmd_t) >= command_dt - 1e-12:
                v, w, _done = nav.follower.command(
                    (float(state[0]), float(state[1]), float(state[2])), route, world)
                cmd = (v, w)
                last_cmd_t = t
            u = nav.controller.compute(state, cmd[0], cmd[1], t)
            plant.apply(u)
            plant.step()
            viewer.sync()
            if i % sim.record_every == 0:
                ts.append(t)
                xs.append(state.copy())
                cmds.append(cmd)
                us.append(u)
            if realtime:
                lag = t - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)

    if not ts:
        ts.append(0.0)
        xs.append(plant.state.copy())
        cmds.append(cmd)
        us.append(u)
    traj = TWIPTrajectory(t=np.asarray(ts), states=np.asarray(xs), commands=np.asarray(cmds),
                          torques=np.asarray(us), fell=fell, params=params,
                          controller_name=balance, reached=reached)
    final = float(np.hypot(traj.x[-1] - goal[0], traj.y[-1] - goal[1]))
    return NavResult(path=route, trajectory=traj,
                     reached=bool(reached or final < goal_tol), fell=bool(fell),
                     world=world, start=start, goal=goal,
                     balance_name=balance, planner_name=planner, follower_name=follower)
