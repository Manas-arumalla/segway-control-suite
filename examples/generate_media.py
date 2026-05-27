"""Generate the showcase media used in the README and docs.

Run from the repo root:  python examples/generate_media.py
Outputs land in assets/.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running directly from a source checkout (before `pip install -e .`).
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from segway.config import RobotParams, SimConfig  # noqa: E402
from segway.controllers import REGULATORS, build_controller  # noqa: E402
from segway.estimation import LinearKalmanFilter, SensorModel, SensorSpec  # noqa: E402
from segway.sim import Scenario, simulate  # noqa: E402
from segway.viz import plot_comparison  # noqa: E402

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)


def make_comparison() -> None:
    p = RobotParams()
    scen = Scenario.kick(tilt=0.1, time=3.0, impulse=0.6)
    sim = SimConfig(duration=8.0)
    trajs = {name: simulate(p, build_controller(name, p), scen, sim) for name in REGULATORS}
    plot_comparison(trajs, ASSETS / "comparison_kick.png",
                    title="Controller comparison: 0.1 rad start + 0.6 rad/s kick at 3 s")
    print("wrote", ASSETS / "comparison_kick.png")


def make_estimation_plot() -> None:
    p = RobotParams()
    dt = 0.01
    sensors = SensorModel(SensorSpec(), seed=0)
    est = LinearKalmanFilter(p, dt=dt, R=sensors.R)
    traj = simulate(p, build_controller("lqr", p), Scenario.balance(0.2),
                    SimConfig(duration=6.0, control_dt=dt), sensors=sensors, estimator=est)
    t = traj.t
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, np.degrees(traj.states[:, 2]), color="#e63946", lw=2, label=r"true $\theta$")
    ax[0].plot(t, np.degrees(traj.estimates[:, 2]), color="#1d75d8", lw=1.4, ls="--", label="KF estimate")
    ax[0].set_ylabel(r"Tilt $\theta$ (deg)")
    ax[1].plot(t, traj.states[:, 3], color="#e63946", lw=2, label=r"true $\dot{\theta}$ (unmeasured)")
    ax[1].plot(t, traj.estimates[:, 3], color="#1d75d8", lw=1.4, ls="--", label="KF estimate")
    ax[1].set_ylabel(r"Tilt rate $\dot{\theta}$ (rad/s)")
    ax[1].set_xlabel("Time (s)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Kalman filter reconstructs the unmeasured tilt rate from noisy position+tilt",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "estimation_kf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", ASSETS / "estimation_kf.png")


def make_gif() -> None:
    try:
        from segway.viz import render_rollout
    except Exception as exc:  # pragma: no cover
        print("skip GIF (viz import failed):", exc)
        return
    p = RobotParams()
    try:
        out = render_rollout(
            p, build_controller("lqr", p),
            Scenario.kick(tilt=0.15, time=3.0, impulse=0.8),
            SimConfig(duration=6.0),
            path=ASSETS / "lqr_balance.gif", width=560, height=420, fps=30,
        )
        print("wrote", out)
    except Exception as exc:  # pragma: no cover
        print("skip GIF (rendering unavailable):", exc)


def make_swingup_gif() -> None:
    import numpy as np

    try:
        from segway.viz import render_rollout
    except Exception as exc:  # pragma: no cover
        print("skip swing-up GIF (viz import failed):", exc)
        return
    p = RobotParams()
    scen = Scenario(name="swingup", initial_tilt=float(np.pi), initial_tilt_rate=0.5)
    try:
        from segway.sim.mujoco_backend import CARTPOLE_PATH
        out = render_rollout(
            p, build_controller("swingup", p), scen,
            SimConfig(duration=9.0, fall_angle=50.0),  # disable fall detection while swinging up
            path=ASSETS / "swingup.gif", width=560, height=420, fps=30,
            distance=4.0, elevation=-12.0, xml_path=CARTPOLE_PATH,
        )
        print("wrote", out)
    except Exception as exc:  # pragma: no cover
        print("skip swing-up GIF (rendering unavailable):", exc)


if __name__ == "__main__":
    make_comparison()
    make_estimation_plot()
    make_gif()
    make_swingup_gif()
