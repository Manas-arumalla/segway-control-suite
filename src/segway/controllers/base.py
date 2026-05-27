"""Common controller interface.

Every controller is a *regulator to zero*: it drives the state to the origin. The
simulator handles reference tracking by calling ``compute(state - reference, t)``, so all
controllers share one uniform contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import RobotParams


class Controller(ABC):
    """Abstract base class for all control strategies."""

    #: short, lowercase identifier used by the registry / CLI / UIs
    name: str = "base"

    def __init__(self, params: RobotParams):
        self.params = params

    def reset(self) -> None:
        """Reset any internal state (integrators, warm-starts, timers). Default: no-op."""

    @abstractmethod
    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        """Return the control torque ``u = tau`` [N*m] for the given state and time."""
        raise NotImplementedError

    def __call__(self, state: np.ndarray, t: float = 0.0) -> float:
        return self.compute(state, t)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
