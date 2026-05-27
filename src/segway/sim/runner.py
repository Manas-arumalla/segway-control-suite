"""Headless nonlinear simulator (fixed-step RK4).

This is the default backend: pure NumPy/SciPy, deterministic, no display. It integrates the
full nonlinear plant while a controller acts through a zero-order hold, and returns a
:class:`Trajectory` that knows how to score itself. The MuJoCo backend shares the same
controller and metric code for high-fidelity visualization and cross-checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import RobotParams, SimConfig
from ..controllers.base import Controller
from ..models import nonlinear_dynamics
from .scenarios import Scenario

if TYPE_CHECKING:  # avoid importing the estimation package at runtime
    from ..estimation.kalman import Estimator
    from ..estimation.sensors import SensorModel


@dataclass
class Trajectory:
    """The result of a simulation run."""

    t: np.ndarray              # (N,)
    states: np.ndarray         # (N, 4) = [x, x_dot, theta, theta_dot]
    controls: np.ndarray       # (N,) applied torque
    fell: bool
    params: RobotParams
    scenario: Scenario
    controller_name: str = "unknown"
    estimates: np.ndarray | None = None  # (N, 4) state estimates, if an estimator was used

    @property
    def x(self) -> np.ndarray:
        return self.states[:, 0]

    @property
    def estimation_error(self) -> np.ndarray | None:
        """Per-step estimate minus truth, ``(N, 4)``; ``None`` if no estimator was used."""
        if self.estimates is None:
            return None
        return self.estimates - self.states

    @property
    def theta(self) -> np.ndarray:
        return self.states[:, 2]

    def metrics(self, **kwargs) -> dict[str, float | bool]:
        """Performance metrics for this run (see :func:`segway.analysis.compute_metrics`)."""
        from ..analysis.metrics import compute_metrics  # lazy: avoids an import cycle

        return compute_metrics(self.t, self.states, self.controls, fell=self.fell, **kwargs)


def _rk4_step(state: np.ndarray, u: float, p: RobotParams, dt: float) -> np.ndarray:
    """One fixed-step RK4 integration of the nonlinear dynamics with a held input ``u``."""
    k1 = nonlinear_dynamics(state, u, p)
    k2 = nonlinear_dynamics(state + 0.5 * dt * k1, u, p)
    k3 = nonlinear_dynamics(state + 0.5 * dt * k2, u, p)
    k4 = nonlinear_dynamics(state + dt * k3, u, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    params: RobotParams,
    controller: Controller,
    scenario: Scenario | None = None,
    sim: SimConfig | None = None,
    sensors: SensorModel | None = None,
    estimator: Estimator | None = None,
) -> Trajectory:
    """Run a closed-loop simulation and return the recorded :class:`Trajectory`.

    The controller regulates ``state - reference`` (so reference tracking is uniform), is
    updated at most every ``sim.control_dt`` seconds (zero-order hold), and the run stops
    early if ``|theta|`` exceeds ``sim.fall_angle``.

    If both ``sensors`` and ``estimator`` are provided, the controller acts on the
    estimator's reconstruction of the state from noisy measurements (instead of ground
    truth). The estimator's discretization step should match ``sim.control_dt``.
    """
    scenario = scenario or Scenario()
    sim = sim or SimConfig()
    controller.reset()

    use_estimator = sensors is not None and estimator is not None

    state = scenario.initial_state()
    dt = sim.dt
    n_steps = int(round(sim.duration / dt))
    control_dt = sim.control_dt

    ts: list[float] = []
    xs: list[np.ndarray] = []
    us: list[float] = []
    ests: list[np.ndarray] = []

    # Seed the estimator from the first measurement (velocities unknown -> 0).
    x_hat = state.copy()
    if use_estimator:
        y0 = sensors.measure(state, 0.0)
        x_hat = np.array([y0[0], 0.0, y0[1], 0.0])
        estimator.reset(x_hat)

    u = 0.0
    last_ctrl_t = -np.inf
    fell = False

    for i in range(n_steps + 1):
        t = i * dt

        # Apply any disturbance kick scheduled for this step.
        for d in scenario.disturbances:
            if abs(t - d.time) < dt / 2:
                state = state.copy()
                state[3] += d.impulse

        # Fall check (stop early).
        if abs(state[2]) > sim.fall_angle:
            fell = True
            break

        # Control update with zero-order hold.
        if control_dt is None or (t - last_ctrl_t) >= control_dt - 1e-12:
            if use_estimator:
                y = sensors.measure(state, t)
                x_hat = estimator.estimate(u, y)
                control_state = x_hat
            else:
                control_state = state
            u = controller.compute(control_state - scenario.reference_at(t), t)
            last_ctrl_t = t

        # Record.
        if i % sim.record_every == 0:
            ts.append(t)
            xs.append(state.copy())
            us.append(u)
            if use_estimator:
                ests.append(x_hat.copy())

        # Integrate forward (except past the final step).
        if i < n_steps:
            state = _rk4_step(state, u, params, dt)

    if not ts:  # extremely short / immediate fall — record at least the initial sample
        ts.append(0.0)
        xs.append(scenario.initial_state())
        us.append(0.0)
        if use_estimator:
            ests.append(x_hat.copy())

    return Trajectory(
        t=np.asarray(ts),
        states=np.asarray(xs),
        controls=np.asarray(us),
        fell=fell,
        params=params,
        scenario=scenario,
        controller_name=getattr(controller, "name", "unknown"),
        estimates=np.asarray(ests) if use_estimator else None,
    )
