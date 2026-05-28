"""Learned navigation (NAV-7): goal-conditioned env API + a short train/drive smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from segway.config import TWIPParams

pytest.importorskip("gymnasium")
from segway.envs import TWIPNavEnv, make_nav_env  # noqa: E402
from segway.navigation.rl_nav import RLNavController  # noqa: E402


def test_env_api_and_reward_finite():
    env = TWIPNavEnv()
    obs, _info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (2,)
    total = 0.0
    for _ in range(30):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        total += r
        if terminated or truncated:
            break
    assert np.isfinite(total) and {"reached", "fell", "dist"} <= set(info)


def test_observation_is_goal_conditioned():
    env = TWIPNavEnv()
    env.reset(seed=0)
    env.state = np.zeros(7)
    env.goal = np.array([2.0, 0.0])
    near = env._obs()
    env.goal = np.array([-3.0, 1.0])
    far = env._obs()
    assert not np.allclose(near, far)        # the goal enters the observation


def test_randomization_changes_params():
    env = TWIPNavEnv(randomize=True)
    env.reset(seed=1)
    m1 = env.params.base.m_pend
    env.reset(seed=2)
    m2 = env.params.base.m_pend
    assert m1 != m2


def test_controller_obs_matches_env():
    env = TWIPNavEnv()
    env.state = np.array([0.3, -0.2, 0.5, 0.05, 0.1, 0.0, 0.0])
    env.goal = np.array([2.0, 1.0])
    ctrl = RLNavController(model=None, goal=[2.0, 1.0])
    assert np.allclose(env._obs(), ctrl._obs(env.state))


def test_ppo_trains_and_drives():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from segway.navigation import rl_navigate

    model = PPO("MlpPolicy", make_nav_env(), n_steps=256, batch_size=64, verbose=0, seed=0)
    model.learn(total_timesteps=1024)

    res = rl_navigate(model, TWIPParams(), (0.0, 0.0), (2.0, 0.0))
    assert res.trajectory is not None
    assert np.isfinite(res.final_goal_distance)
    tau = RLNavController(model, [2.0, 0.0]).compute(np.zeros(7))
    assert len(tau) == 2 and all(np.isfinite(tau))
