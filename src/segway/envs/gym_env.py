"""Gymnasium environment for the self-balancing robot.

Reuses the *same* nonlinear plant as every other backend, so a learned policy is compared
on identical physics. Optional **domain randomization** perturbs the physical parameters
each episode, producing policies that transfer across model error (the RL analogue of the
robustness benchmark). Requires the ``rl`` extra (Gymnasium / Stable-Baselines3).
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import RobotParams
from ..models import nonlinear_dynamics

# Domain-randomization ranges (multiplicative ±fraction) applied per episode.
DR_RANGES = {"m_pend": 0.4, "l": 0.25, "I_pend": 0.4, "b_x": 0.6, "b_theta": 0.6}


def _rk4(state, u, p, dt):
    k1 = nonlinear_dynamics(state, u, p)
    k2 = nonlinear_dynamics(state + 0.5 * dt * k1, u, p)
    k3 = nonlinear_dynamics(state + 0.5 * dt * k2, u, p)
    k4 = nonlinear_dynamics(state + dt * k3, u, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class SegwayEnv(gym.Env):
    """Balance-from-tilt task. Observation = full state; action = motor torque."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: RobotParams | None = None,
        randomize: bool = False,
        u_max: float = 50.0,
        dt: float = 0.02,
        substeps: int = 4,
        max_seconds: float = 10.0,
        init_tilt: float = 0.3,
        fall_angle: float = 0.8,
        x_limit: float = 4.0,
    ):
        super().__init__()
        self.base_params = params or RobotParams()
        self.randomize = randomize
        self.u_max = float(u_max)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.max_steps = int(max_seconds / dt)
        self.init_tilt = float(init_tilt)
        self.fall_angle = float(fall_angle)
        self.x_limit = float(x_limit)

        self.action_space = spaces.Box(-self.u_max, self.u_max, shape=(1,), dtype=np.float32)
        high = np.array([x_limit * 2, 20.0, np.pi, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.params = self.base_params
        self.state = np.zeros(4)
        self._step = 0

    def _sample_params(self) -> RobotParams:
        if not self.randomize:
            return self.base_params
        changes = {k: getattr(self.base_params, k) * (1.0 + self.np_random.uniform(-f, f))
                   for k, f in DR_RANGES.items()}
        return replace(self.base_params, **changes)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.params = self._sample_params()
        tilt = self.np_random.uniform(-self.init_tilt, self.init_tilt)
        self.state = np.array([0.0, 0.0, tilt, 0.0])
        self._step = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        u = float(np.clip(action[0], -self.u_max, self.u_max))
        sdt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.state = _rk4(self.state, u, self.params, sdt)
        self._step += 1

        x, x_dot, theta, theta_dot = self.state
        # Reward: stay upright and centered with modest effort.
        reward = (np.cos(theta) - 0.02 * x**2 - 0.005 * x_dot**2
                  - 0.01 * theta_dot**2 - 0.0005 * u**2)
        terminated = bool(abs(theta) > self.fall_angle or abs(x) > self.x_limit)
        truncated = bool(self._step >= self.max_steps)
        return self.state.astype(np.float32), float(reward), terminated, truncated, {}


def make_env(randomize: bool = False, **kwargs) -> SegwayEnv:
    """Factory for use with Stable-Baselines3 vectorized envs."""
    return SegwayEnv(randomize=randomize, **kwargs)
