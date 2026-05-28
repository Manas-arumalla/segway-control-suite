"""Headless planar TWIP simulator (fixed-step RK4).

Rolls out the planar TWIP under a `TWIPController` while a *command function* supplies the
(forward-speed, yaw-rate) targets over time. The loop runs at two rates: the inner balance +
yaw controller fires at ``sim.control_dt`` (every step by default, ~hundreds of Hz) while the
command function — a path follower, during navigation — fires at the slower ``command_dt``
(~10 Hz). An optional ``stop_fn(state)`` ends the rollout early (e.g. once the goal is
reached). NAV-1 uses constant commands (drive / turn); the navigator (NAV-5) supplies commands
from a planner + follower.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import SimConfig, TWIPParams
from ..models.twip import twip_dynamics
from .control import TWIPController

# command_fn(t, state) -> (v_des, yaw_rate_des)
CommandFn = Callable[[float, np.ndarray], "tuple[float, float]"]
# stop_fn(state) -> bool : True ends the rollout early (goal reached)
StopFn = Callable[[np.ndarray], bool]


@dataclass
class TWIPTrajectory:
    """Result of a planar TWIP rollout."""

    t: np.ndarray            # (N,)
    states: np.ndarray       # (N, 7) = [x, y, psi, theta, v, psi_dot, theta_dot]
    commands: np.ndarray     # (N, 2) = [v_des, yaw_rate_des]
    torques: np.ndarray      # (N, 2) = [tau_L, tau_R]
    fell: bool
    params: TWIPParams
    controller_name: str = "unknown"
    reached: bool = False    # True if a stop_fn ended the rollout (e.g. goal reached)

    @property
    def x(self) -> np.ndarray:
        return self.states[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.states[:, 1]

    @property
    def psi(self) -> np.ndarray:
        return self.states[:, 2]

    @property
    def theta(self) -> np.ndarray:
        return self.states[:, 3]

    @property
    def v(self) -> np.ndarray:
        return self.states[:, 4]


def _rk4(state, torques, p, dt):
    k1 = twip_dynamics(state, torques, p)
    k2 = twip_dynamics(state + 0.5 * dt * k1, torques, p)
    k3 = twip_dynamics(state + 0.5 * dt * k2, torques, p)
    k4 = twip_dynamics(state + dt * k3, torques, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_twip(
    params: TWIPParams,
    controller: TWIPController,
    command_fn: CommandFn,
    sim: SimConfig | None = None,
    x0: np.ndarray | None = None,
    command_dt: float = 0.1,
    stop_fn: StopFn | None = None,
) -> TWIPTrajectory:
    """Simulate the planar TWIP with a fast inner loop and a slow command loop.

    The inner balance + yaw controller fires at ``sim.control_dt`` (every integrator step when
    ``None``); ``command_fn`` fires at the slower ``command_dt``. The rollout stops early if
    ``|theta|`` exceeds ``sim.fall_angle`` (``fell=True``) or if ``stop_fn(state)`` is truthy
    (``reached=True``).
    """
    sim = sim or SimConfig()
    controller.reset()
    state = np.zeros(7) if x0 is None else np.asarray(x0, dtype=float).copy()

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
        if abs(state[3]) > sim.fall_angle:   # theta is index 3
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
        if i % sim.record_every == 0:
            ts.append(t)
            xs.append(state.copy())
            cmds.append(cmd)
            us.append(u)
        if i < n_steps:
            state = _rk4(state, u, params, dt)

    if not ts:
        ts.append(0.0)
        xs.append(state.copy())
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
