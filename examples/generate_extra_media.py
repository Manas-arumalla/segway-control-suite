"""Generate the advanced-controller showcase GIFs (headless MuJoCo render).

  - assets/ilqr_swingup.gif   : iLQR-optimized swing-up from hanging, tracked in MuJoCo.
  - assets/mrac_mismatch.gif  : MRAC (designed on the nominal robot) balancing and recovering
                                from a kick on a heavier/longer MISMATCHED robot.

Run from the repo root:  python examples/generate_extra_media.py
"""

from __future__ import annotations

import sys as _sys
from dataclasses import replace
from pathlib import Path as _Path

_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import numpy as np

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.controllers.base import Controller
from segway.planning import iLQR
from segway.sim import Scenario
from segway.sim.mujoco_backend import CARTPOLE_PATH
from segway.sim.scenarios import Disturbance
from segway.viz import render_rollout

ASSETS = _Path("assets")


class iLQRReplay(Controller):
    """Tracks a precomputed iLQR trajectory with its time-varying feedback gains."""

    name = "ilqr"

    def __init__(self, params, result, dt):
        super().__init__(params)
        self.us = result.us
        self.K = result.gains
        self.xs = result.xs
        self.dt = dt
        self.N = len(result.us)

    def compute(self, state, t: float = 0.0) -> float:
        k = min(int(t / self.dt), self.N - 1)
        return float(self.us[k] + self.K[k] @ (np.asarray(state, dtype=float) - self.xs[k]))


def make_ilqr_swingup():
    p = RobotParams()
    dt = 0.02
    opt = iLQR(p, dt=dt, u_max=80.0, Qf=(20, 2, 800, 5))
    res = opt.fit(x0=np.array([0.0, 0.0, np.pi, 0.0]), N=220, iters=150)
    thw = float(np.arctan2(np.sin(res.xs[-1][2]), np.cos(res.xs[-1][2])))
    print(f"iLQR: converged={res.converged} final|theta|={abs(thw):.4f}")
    ctrl = iLQRReplay(p, res, dt)
    scen = Scenario(name="ilqr_swingup", initial_tilt=float(np.pi), initial_tilt_rate=0.0)
    out = render_rollout(p, ctrl, scen,
                         SimConfig(dt=0.005, duration=res.us.shape[0] * dt + 2.0, fall_angle=50.0),
                         path=ASSETS / "ilqr_swingup.gif", width=560, height=420, fps=30,
                         distance=4.0, elevation=-12.0, xml_path=CARTPOLE_PATH)
    print("wrote", out)


def make_mrac_mismatch():
    nominal = RobotParams()
    mismatched = replace(nominal, m_pend=nominal.m_pend * 1.6, l=nominal.l * 1.3,
                         I_pend=nominal.I_pend * 1.6)
    ctrl = build_controller("mrac", nominal)  # designed on the NOMINAL robot
    scen = Scenario(name="mrac_mismatch", initial_tilt=0.25,
                    disturbances=[Disturbance(time=4.0, impulse=0.7)])
    out = render_rollout(mismatched, ctrl, scen, SimConfig(duration=8.0),
                         path=ASSETS / "mrac_mismatch.gif", width=560, height=420, fps=30)
    print("wrote", out)


if __name__ == "__main__":
    make_ilqr_swingup()
    make_mrac_mismatch()
