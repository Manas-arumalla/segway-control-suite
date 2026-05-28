"""Learned navigation: drive the balancing TWIP to a goal with a trained policy (NAV-7).

Wraps a Stable-Baselines3 policy (trained on :class:`~segway.envs.twip_nav_env.TWIPNavEnv`) as
a drop-in navigator that produces the same :class:`~segway.navigation.navigator.NavResult` as
the classical stack — so the two can be compared head-to-head. The policy is end-to-end: it
maps the goal-in-body-frame + balance state straight to wheel torques, with no planner or
follower. The observation must match the training env exactly, so it is built here identically.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig, TWIPParams
from .navigator import NavResult
from .sim import simulate_twip
from .world import World


class RLNavController:
    """Adapts a trained policy + a goal into the ``controller.compute`` interface."""

    balance_name = "rl"

    def __init__(self, model, goal, u_max: float = 8.0):
        self.model = model
        self.goal = np.asarray(goal, dtype=float)
        self.u_max = float(u_max)

    def reset(self) -> None:  # the MLP policy is stateless
        pass

    def _obs(self, state: np.ndarray) -> np.ndarray:
        x, y, psi, theta, v, psi_dot, theta_dot = state
        dx, dy = self.goal[0] - x, self.goal[1] - y
        c, s = np.cos(-psi), np.sin(-psi)
        gx_b = c * dx - s * dy
        gy_b = s * dx + c * dy
        dist = float(np.hypot(dx, dy))
        return np.array([gx_b, gy_b, dist, v, theta, theta_dot, psi_dot], dtype=np.float32)

    def compute(self, state, v_des=0.0, yaw_rate_des=0.0, t: float = 0.0):
        action, _ = self.model.predict(self._obs(np.asarray(state)), deterministic=True)
        a = np.clip(np.asarray(action).reshape(-1), -self.u_max, self.u_max)
        return float(a[0]), float(a[1])


def rl_navigate(
    model,
    params: TWIPParams,
    start,
    goal,
    *,
    world: World | None = None,
    sim: SimConfig | None = None,
    goal_tol: float = 0.45,
    u_max: float = 8.0,
) -> NavResult:
    """Drive the balancing TWIP from ``start`` to ``goal`` with a trained policy.

    Returns a :class:`NavResult` with ``planner_name='—'`` and ``follower_name='rl'``; there is
    no global path, so ``path`` is the straight start→goal segment (for plotting/metrics). The
    default 0.45 m arrival tolerance reflects how precisely a balancing robot can decelerate onto
    a point goal — the policy reliably arrives within ~0.4 m and balances there (the exact
    standoff depends on the trained policy; a velocity-tracking pendulum cannot pinpoint-stop).
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    world = world or World(width=20.0, height=20.0, obstacles=[])
    # Decision rate matches the training env (50 Hz) over a finer integrator step.
    sim = sim or SimConfig(dt=0.005, control_dt=0.02, duration=20.0, fall_angle=0.8)

    psi0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    x0 = np.zeros(7)
    x0[0], x0[1], x0[2] = start[0], start[1], psi0

    controller = RLNavController(model, goal, u_max=u_max)

    def stop_fn(state):
        return np.hypot(state[0] - goal[0], state[1] - goal[1]) < goal_tol

    traj = simulate_twip(params, controller, lambda t, s: (0.0, 0.0),
                         sim=sim, x0=x0, stop_fn=stop_fn)
    final_dist = float(np.hypot(traj.x[-1] - goal[0], traj.y[-1] - goal[1]))
    reached = (traj.reached or final_dist < goal_tol) and not traj.fell
    return NavResult(
        path=np.array([start, goal]), trajectory=traj,
        reached=bool(reached), fell=bool(traj.fell),
        world=world, start=start, goal=goal,
        balance_name="rl", planner_name="—", follower_name="rl",
    )
