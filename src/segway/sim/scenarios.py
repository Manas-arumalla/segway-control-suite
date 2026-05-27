"""Scenarios: reproducible test conditions for the simulator and benchmark.

A scenario fully specifies the initial state, any timed disturbances, and a reference
setpoint, so different controllers can be compared under *identical* conditions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Disturbance:
    """An instantaneous angular-velocity kick applied to the body at ``time``."""

    time: float
    impulse: float  # added to theta_dot [rad/s]


@dataclass
class Scenario:
    """A reproducible simulation condition."""

    name: str = "default"
    initial_pos: float = 0.0
    initial_vel: float = 0.0
    initial_tilt: float = 0.1            # [rad]
    initial_tilt_rate: float = 0.0       # [rad/s]
    disturbances: list[Disturbance] = field(default_factory=list)
    reference: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Optional time-varying reference t -> [x, x_dot, theta, theta_dot] (overrides `reference`).
    reference_fn: Callable[[float], np.ndarray] | None = None

    def initial_state(self) -> np.ndarray:
        return np.array(
            [self.initial_pos, self.initial_vel, self.initial_tilt, self.initial_tilt_rate],
            dtype=float,
        )

    def reference_vec(self) -> np.ndarray:
        return np.asarray(self.reference, dtype=float)

    def reference_at(self, t: float) -> np.ndarray:
        """Reference state at time ``t`` (time-varying if ``reference_fn`` is set)."""
        if self.reference_fn is not None:
            return np.asarray(self.reference_fn(t), dtype=float)
        return self.reference_vec()

    # --- convenience constructors -----------------------------------------
    @classmethod
    def balance(cls, tilt: float = 0.1) -> Scenario:
        """Recover to upright from an initial tilt."""
        return cls(name=f"balance_{tilt:g}rad", initial_tilt=tilt)

    @classmethod
    def kick(cls, tilt: float = 0.0, time: float = 3.0, impulse: float = 0.5) -> Scenario:
        """Start near upright, then receive a disturbance kick at ``time``."""
        return cls(
            name=f"kick_{impulse:g}",
            initial_tilt=tilt,
            disturbances=[Disturbance(time=time, impulse=impulse)],
        )

    @classmethod
    def setpoint(cls, x_ref: float = 1.0, tilt: float = 0.0) -> Scenario:
        """Drive to a target base position ``x_ref`` while staying upright."""
        return cls(name=f"setpoint_{x_ref:g}m", initial_tilt=tilt, reference=(x_ref, 0.0, 0.0, 0.0))


# A small standard battery used by the benchmark harness (extended in Phase 3).
STANDARD_SCENARIOS: dict[str, Scenario] = {
    "balance_small": Scenario.balance(tilt=0.1),
    "balance_large": Scenario.balance(tilt=0.3),
    "kick": Scenario.kick(tilt=0.0, time=3.0, impulse=0.6),
    "setpoint": Scenario.setpoint(x_ref=1.0),
}
