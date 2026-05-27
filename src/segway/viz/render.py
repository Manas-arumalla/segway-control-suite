"""Render a closed-loop rollout to an animated GIF or MP4 using MuJoCo offscreen rendering.

Requires the ``sim`` (MuJoCo) and ``viz`` (imageio) extras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import RobotParams, SimConfig
from ..controllers.base import Controller
from ..sim.mujoco_backend import MuJoCoPlant, simulate_mujoco
from ..sim.scenarios import Scenario


def render_rollout(
    params: RobotParams,
    controller: Controller,
    scenario: Scenario | None = None,
    sim: SimConfig | None = None,
    path: str | Path = "rollout.gif",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    distance: float = 3.5,
    elevation: float = -12.0,
    azimuth: float = 90.0,
    xml_path: str | Path | None = None,
) -> str:
    """Simulate ``controller`` on ``scenario`` in MuJoCo and write an animation to ``path``.

    Returns the output path. The format (``.gif`` or ``.mp4``) is inferred from the suffix.
    ``xml_path`` selects the model geometry (e.g. the cart-pole used for swing-up).
    """
    import imageio.v2 as imageio
    import mujoco as mj

    sim = sim or SimConfig()
    scenario = scenario or Scenario()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # One plant instance, shared by the simulator and the renderer (same MjModel).
    plant = MuJoCoPlant(params, xml_path=xml_path)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_BODY, "base")
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth

    renderer = mj.Renderer(plant.model, height=height, width=width)
    frames: list[np.ndarray] = []
    frame_every = max(1, int(round(1.0 / (fps * sim.dt))))
    counter = {"i": 0}

    def grab(plant: MuJoCoPlant, t: float) -> None:
        if counter["i"] % frame_every == 0:
            renderer.update_scene(plant.data, camera=cam)
            frames.append(renderer.render().copy())
        counter["i"] += 1

    simulate_mujoco(params, controller, scenario, sim, step_callback=grab, plant=plant, xml_path=xml_path)
    renderer.close()

    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)
    return str(path)
