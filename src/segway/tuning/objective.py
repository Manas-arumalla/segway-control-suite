"""Shared tuning objective and per-controller search spaces.

The cost is a single scalar combining settling time, control effort, and position drift,
with a large penalty for falling — evaluated over a battery of scenarios so the tuned gains
generalize rather than overfit one trajectory. The same objective is used by every
optimizer (Optuna TPE, CMA-ES, GA), so they can be compared fairly.
"""

from __future__ import annotations

import math

from ..config import RobotParams, SimConfig
from ..controllers import build_controller
from ..sim import Scenario

# Flat search bounds per controller: ordered list of (name, low, high, log?) tuples.
# `flat_to_kwargs` maps a flat parameter dict back to build_controller kwargs.
search_bounds: dict[str, list[tuple]] = {
    "lqr": [("q0", 0.1, 2000, True), ("q1", 0.1, 200, True),
            ("q2", 0.1, 2000, True), ("q3", 0.1, 200, True), ("R", 1e-3, 10, True)],
    "pid": [("kp", 10, 400, False), ("ki", 0, 40, False), ("kd", 1, 80, False)],
    "cascaded_pid": [("kp", 40, 320, False), ("kd", 8, 90, False),
                     ("kx", -1.2, -0.03, False), ("kv", -2.5, -0.05, False)],
    "smc": [("l0", 0.1, 10, False), ("l2", 0.1, 12, False), ("l3", 0.1, 6, False),
            ("K", 1, 120, False), ("phi", 0.01, 1.0, False)],
    "pole_placement": [("p0", -50, -0.5, False), ("p1", -50, -0.5, False),
                       ("p2", -50, -0.5, False), ("p3", -50, -0.5, False)],
}


def flat_to_kwargs(controller_name: str, flat: dict[str, float]) -> dict:
    """Convert a flat parameter dict (as produced by the optimizers) to controller kwargs."""
    n = controller_name
    if n in ("lqr", "mpc"):
        return {"Q": [flat["q0"], flat["q1"], flat["q2"], flat["q3"]], "R": flat["R"]}
    if n == "pid":
        return {"kp": flat["kp"], "ki": flat["ki"], "kd": flat["kd"]}
    if n == "cascaded_pid":
        return {"kp": flat["kp"], "kd": flat["kd"], "kx": flat["kx"], "kv": flat["kv"]}
    if n == "smc":
        return {"lam": [flat["l0"], 1.0, flat["l2"], flat["l3"]], "K": flat["K"], "phi": flat["phi"]}
    if n == "pole_placement":
        return {"poles": [flat["p0"], flat["p1"], flat["p2"], flat["p3"]]}
    raise ValueError(f"no search space defined for controller {n!r}")


def default_scenarios() -> list[Scenario]:
    """A demanding battery: a small and a large tilt plus a hard kick.

    Including the *large* tilt (0.5 rad) and a strong kick forces tuned gains to work over a
    wide operating range, which keeps the optimizer from overfitting easy near-upright cases
    at the cost of robustness.
    """
    return [
        Scenario.balance(0.2),
        Scenario.balance(0.5),
        Scenario.kick(tilt=0.0, time=2.0, impulse=0.8),
    ]


def controller_cost(
    controller_name: str,
    kwargs: dict,
    params: RobotParams | None = None,
    scenarios: list[Scenario] | None = None,
    sim: SimConfig | None = None,
    w_effort: float = 0.003,
    w_drift: float = 0.4,
    fall_penalty: float = 500.0,
) -> float:
    """Scalar cost (lower is better) for a controller with the given kwargs.

    Returns a large value if the controller cannot be constructed (e.g. infeasible gains).
    """
    params = params or RobotParams()
    scenarios = scenarios or default_scenarios()
    sim = sim or SimConfig(duration=8.0)

    try:
        ctrl = build_controller(controller_name, params, **kwargs)
    except Exception:
        return 1e6

    total = 0.0
    for scen in scenarios:
        traj = simulate_local(params, ctrl, scen, sim)
        m = traj.metrics()
        if m["fell"]:
            total += fall_penalty
            continue
        st = m["settling_time_angle"]
        st = sim.duration if math.isinf(st) else st
        total += st + w_effort * m["control_effort"] + w_drift * m["max_pos_drift_m"]
    return total


def simulate_local(params, ctrl, scen, sim):
    # Local import keeps the module import light and avoids any import-order pitfalls.
    from ..sim import simulate

    return simulate(params, ctrl, scen, sim)
