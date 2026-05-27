"""Reinforcement-learning controller (loads a trained Stable-Baselines3 policy).

Not in the default registry because it needs a trained model artifact; construct it directly
with a ``model`` object or a ``model_path``. Train one with ``scripts/train_rl.py``.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotParams
from .base import Controller


class RLController(Controller):
    """Wraps a trained SB3 policy as a controller (deterministic action = torque)."""

    name = "rl"

    def __init__(self, params: RobotParams, model=None, model_path: str | None = None):
        super().__init__(params)
        if model is None:
            if model_path is None:
                raise ValueError("RLController needs a `model` or a `model_path`")
            from stable_baselines3 import PPO  # lazy import

            model = PPO.load(model_path, device="cpu")
        self.model = model

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        action, _ = self.model.predict(np.asarray(state, dtype=np.float32), deterministic=True)
        return float(np.asarray(action).reshape(-1)[0])
