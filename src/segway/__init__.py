"""Segway Control Suite — a control benchmark for a wheeled inverted pendulum.

Public API is intentionally small and stable:

    from segway.config import RobotParams, SimConfig
    from segway.controllers import build_controller, list_controllers
    from segway.sim import simulate, Scenario
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
