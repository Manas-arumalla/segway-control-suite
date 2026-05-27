"""Controllers and a name-based factory.

All construction goes through :func:`build_controller`, so the CLI, benchmark harness, and
both UIs share a single, consistent path for instantiating controllers.
"""

from __future__ import annotations

from ..config import RobotParams
from .base import Controller
from .cascaded_pid import CascadedPIDController
from .hinf import HInfController
from .lqr import LQRController
from .mrac import MRACController
from .pid import PIDController
from .pole_placement import PolePlacementController
from .rl_policy import RLController
from .smc import SMCController
from .swingup import SwingUpController

_REGISTRY: dict[str, type[Controller]] = {
    PIDController.name: PIDController,
    CascadedPIDController.name: CascadedPIDController,
    PolePlacementController.name: PolePlacementController,
    LQRController.name: LQRController,
    HInfController.name: HInfController,
    SMCController.name: SMCController,
    MRACController.name: MRACController,
    SwingUpController.name: SwingUpController,
}

# Controllers that regulate the upright equilibrium (used by the regulation benchmark).
# Swing-up is a different task (getting up from hanging), so it is excluded there.
REGULATORS = ("pid", "cascaded_pid", "pole_placement", "lqr", "hinf", "smc", "mpc", "mrac")

# MPC depends on the optional `mpc` extra (CVXPY). Register it only if importable so the
# rest of the package works without that dependency.
try:  # pragma: no cover - import-guard
    from .mpc import MPCController

    _REGISTRY[MPCController.name] = MPCController
except ImportError:  # pragma: no cover
    MPCController = None  # type: ignore[assignment]


def build_controller(name: str, params: RobotParams, **kwargs) -> Controller:
    """Instantiate a controller by name (case-insensitive).

    Extra keyword arguments are forwarded to the controller constructor (gains, weights,
    horizon, ...). Raises ``KeyError`` with the available names if ``name`` is unknown.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](params, **kwargs)


def list_controllers() -> list[str]:
    """Sorted list of registered controller names."""
    return sorted(_REGISTRY)


def register_controller(cls: type[Controller]) -> type[Controller]:
    """Decorator/utility to register an additional controller class by its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


__all__ = [
    "Controller",
    "PIDController",
    "CascadedPIDController",
    "PolePlacementController",
    "LQRController",
    "SMCController",
    "SwingUpController",
    "MRACController",
    "RLController",
    "MPCController",
    "REGULATORS",
    "build_controller",
    "list_controllers",
    "register_controller",
]
