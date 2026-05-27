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
