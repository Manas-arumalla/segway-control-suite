"""Goal-conditioned navigation environment for the planar TWIP (learned navigation, NAV-7).

A single end-to-end policy must do what the classical stack splits across a controller and a
follower: drive the balancing robot to an arbitrary goal point while staying upright. The
observation is the goal expressed in the robot's own frame plus the balance state; the action
is the two wheel torques. Optional **domain randomization** perturbs the physics each episode
for transfer. Reuses the same planar dynamics as every other backend.

Requires the ``rl`` extra (Gymnasium / Stable-Baselines3).
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import TWIPParams
from ..models.twip import twip_dynamics

# Per-episode multiplicative domain-randomization ranges (on the base RobotParams).
DR_RANGES = {"m_pend": 0.3, "l": 0.2, "I_pend": 0.3, "b_x": 0.5, "b_theta": 0.5}


def _rk4(state, torques, p, dt):
    k1 = twip_dynamics(state, torques, p)
    k2 = twip_dynamics(state + 0.5 * dt * k1, torques, p)
    k3 = twip_dynamics(state + 0.5 * dt * k2, torques, p)
    k4 = twip_dynamics(state + dt * k3, torques, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class TWIPNavEnv(gym.Env):
    """Drive the balancing TWIP to a random goal point. Obs = goal-in-body + balance state."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: TWIPParams | None = None,
        randomize: bool = False,
        u_max: float = 8.0,
        dt: float = 0.02,
        substeps: int = 4,
        max_seconds: float = 12.0,
        goal_range: tuple[float, float] = (1.0, 4.0),
        goal_tol: float = 0.45,
        fall_angle: float = 0.8,
    ):
        super().__init__()
        self.base_params = params or TWIPParams()
        self.randomize = randomize
        self.u_max = float(u_max)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.max_steps = int(max_seconds / dt)
        self.goal_range = goal_range
        self.goal_tol = float(goal_tol)
        self.fall_angle = float(fall_angle)

        self.action_space = spaces.Box(-self.u_max, self.u_max, shape=(2,), dtype=np.float32)
        # obs = [gx_body, gy_body, dist, v, theta, theta_dot, psi_dot]
        high = np.array([10.0, 10.0, 14.0, 20.0, np.pi, 20.0, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.params = self.base_params
        self.state = np.zeros(7)
        self.goal = np.zeros(2)
        self._step = 0
        self._prev_dist = 0.0

    def _sample_params(self) -> TWIPParams:
        if not self.randomize:
            return self.base_params
        base = self.base_params.base
        changes = {k: getattr(base, k) * (1.0 + self.np_random.uniform(-f, f))
                   for k, f in DR_RANGES.items()}
        return replace(self.base_params, base=replace(base, **changes))

    def _obs(self) -> np.ndarray:
        x, y, psi, theta, v, psi_dot, theta_dot = self.state
        dx, dy = self.goal[0] - x, self.goal[1] - y
        c, s = np.cos(-psi), np.sin(-psi)
        gx_b = c * dx - s * dy          # goal in the robot's body frame
        gy_b = s * dx + c * dy
        dist = float(np.hypot(dx, dy))
        return np.array([gx_b, gy_b, dist, v, theta, theta_dot, psi_dot], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.params = self._sample_params()
        # Robot at the origin facing a random heading; goal at a random range/bearing.
        psi0 = self.np_random.uniform(-np.pi, np.pi)
        self.state = np.array([0.0, 0.0, psi0, 0.0, 0.0, 0.0, 0.0])
        r = self.np_random.uniform(*self.goal_range)
        ang = self.np_random.uniform(-np.pi, np.pi)
        self.goal = np.array([r * np.cos(ang), r * np.sin(ang)])
        self._step = 0
        self._prev_dist = float(np.hypot(*self.goal))
        return self._obs(), {}

    def step(self, action):
        tau = np.clip(action, -self.u_max, self.u_max)
        torques = (float(tau[0]), float(tau[1]))
        sdt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.state = _rk4(self.state, torques, self.params, sdt)
        self._step += 1

        _, _, _, theta, v, psi_dot, theta_dot = self.state
        dist = float(np.hypot(self.goal[0] - self.state[0], self.goal[1] - self.state[1]))
        reached = dist < self.goal_tol
        fell = abs(theta) > self.fall_angle

        # Reward: progress + a sharp proximity attractor (gradient stays strong right up to the
        # goal, so the robot closes the last few cm instead of parking) + upright, minus effort.
        reward = (4.0 * (self._prev_dist - dist)         # progress toward the goal
                  + 2.0 * np.exp(-3.0 * dist)             # sharp proximity attractor (peaks at goal)
                  + 0.3 * np.cos(theta)                   # upright
                  - 0.01 * (theta_dot**2 + psi_dot**2)
                  - 0.001 * (torques[0] ** 2 + torques[1] ** 2))
        if reached:
            reward += 50.0
        if fell:
            reward -= 30.0
        self._prev_dist = dist

        terminated = bool(reached or fell)
        truncated = bool(self._step >= self.max_steps)
        info = {"reached": reached, "fell": fell, "dist": dist}
        return self._obs(), float(reward), terminated, truncated, info


def make_nav_env(randomize: bool = False, **kwargs) -> TWIPNavEnv:
    """Factory for use with Stable-Baselines3 vectorized envs."""
    return TWIPNavEnv(randomize=randomize, **kwargs)
