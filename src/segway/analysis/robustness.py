"""Monte-Carlo robustness analysis.

A controller is designed on the *nominal* model, then tested on many randomly perturbed
plants (different mass, length, friction, ...). The success rate answers the question that
matters for real hardware: *how much can the model be wrong before the controller fails?*
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config import RobotParams, SimConfig
from ..controllers import build_controller
from ..sim.runner import simulate
from ..sim.scenarios import Scenario

# Default multiplicative perturbation ranges (±fraction) applied to the nominal plant.
DEFAULT_RANGES: dict[str, float] = {
    "m_pend": 0.30,
    "l": 0.20,
    "I_pend": 0.30,
    "b_x": 0.50,
    "b_theta": 0.50,
}


@dataclass
class RobustnessResult:
    controller_name: str
    n: int
    success_rate: float
    perturbed_params: list[RobotParams] = field(default_factory=list)
    fell: np.ndarray | None = None
    final_angles: np.ndarray | None = None


def _perturb(base: RobotParams, ranges: dict[str, float], rng: np.random.Generator) -> RobotParams:
    changes = {}
    for name, frac in ranges.items():
        factor = 1.0 + rng.uniform(-frac, frac)
        changes[name] = getattr(base, name) * factor
    return replace(base, **changes)


def monte_carlo(
    controller_name: str,
    scenario: Scenario | None = None,
    nominal: RobotParams | None = None,
    ranges: dict[str, float] | None = None,
    n: int = 200,
    duration: float = 6.0,
    seed: int = 0,
    settle_angle: float = 0.1,
    **controller_kwargs,
) -> RobustnessResult:
    """Design ``controller_name`` on the nominal plant; test it on ``n`` perturbed plants.

    Returns the success rate (fraction that neither fell nor ended outside ``settle_angle``)
    plus the per-sample outcomes. The controller is built **once** on the nominal model — it
    does not get to see the perturbed parameters, which is the whole point.
    """
    nominal = nominal or RobotParams()
    ranges = ranges or DEFAULT_RANGES
    scenario = scenario or Scenario.balance(0.2)
    rng = np.random.default_rng(seed)
    sim = SimConfig(duration=duration)

    controller = build_controller(controller_name, nominal, **controller_kwargs)

    fell = np.zeros(n, dtype=bool)
    final_angles = np.zeros(n)
    perturbed: list[RobotParams] = []

    for k in range(n):
        p = _perturb(nominal, ranges, rng)
        perturbed.append(p)
        traj = simulate(p, controller, scenario, sim)
        fell[k] = traj.fell
        final_angles[k] = abs(traj.theta[-1])

    success = (~fell) & (final_angles < settle_angle)
    return RobustnessResult(
        controller_name=controller_name,
        n=n,
        success_rate=float(success.mean()),
        perturbed_params=perturbed,
        fell=fell,
        final_angles=final_angles,
    )
