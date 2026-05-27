"""RL environment API + a short end-to-end training/control smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import RobotParams

pytest.importorskip("gymnasium")
from segway.envs import SegwayEnv, make_env  # noqa: E402


def test_env_api_and_reward_finite():
    env = SegwayEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    total = 0.0
    for _ in range(20):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        total += r
        if terminated or truncated:
            break
    assert np.isfinite(total)


def test_env_randomization_changes_params():
    env = SegwayEnv(randomize=True)
    env.reset(seed=1)
    p1 = env.params.m_pend
    env.reset(seed=2)
    p2 = env.params.m_pend
    assert p1 != p2  # different physics per episode


def test_ppo_trains_and_controls():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from segway.controllers import RLController

    model = PPO("MlpPolicy", make_env(), n_steps=256, batch_size=64, verbose=0, seed=0)
    model.learn(total_timesteps=1024)
    ctrl = RLController(RobotParams(), model=model)
    u = ctrl.compute(np.array([0.0, 0.0, 0.1, 0.0]))
    assert np.isfinite(u)
