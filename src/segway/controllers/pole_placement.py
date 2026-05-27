"""Pole-placement (eigenvalue assignment) full-state feedback."""

from __future__ import annotations

import numpy as np
from scipy.signal import place_poles

from ..config import RobotParams
from ..models import linearize
from .base import Controller

# A well-damped dominant complex pair (sets the response) plus two fast real poles to
# stabilize the quick modes. This LQR-inspired placement is far more robust than arbitrary
# real poles — see docs/advanced-methods.md and the pole-set analysis. (Robustness ~77% vs
# ~38% for naive poles like [-3,-4,-5,-6].)
DEFAULT_POLES = (-1.2 + 0.9j, -1.2 - 0.9j, -10.0, -25.0)


class PolePlacementController(Controller):
    """Places the closed-loop poles of ``A - B K`` at user-specified locations.

    For a single-input system the requested poles must be distinct (a multiplicity
    constraint of ``scipy.signal.place_poles``); a clear error is raised otherwise.
    """

    name = "pole_placement"

    def __init__(self, params: RobotParams, poles: np.ndarray | tuple | None = None):
        super().__init__(params)
        self.A, self.B = linearize(params)
        poles = DEFAULT_POLES if poles is None else poles
        self.poles = np.asarray(poles, dtype=complex)
        try:
            result = place_poles(self.A, self.B, self.poles)
        except ValueError as exc:  # e.g. repeated poles for a single-input system
            raise ValueError(
                f"pole placement failed for poles={list(self.poles)}: {exc}. "
                "For this single-input plant the poles must be distinct."
            ) from exc
        self.K = np.asarray(result.gain_matrix)  # (1, 4)

    def compute(self, state: np.ndarray, t: float = 0.0) -> float:
        return float(-(self.K @ np.asarray(state, dtype=float)).item())
