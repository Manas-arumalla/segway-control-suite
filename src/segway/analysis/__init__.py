"""Analysis tools: performance metrics (and, in later phases, ROA & robustness)."""

from __future__ import annotations

from .metrics import compute_metrics
from .roa import ROAResult, compute_roa
from .robustness import RobustnessResult, monte_carlo

__all__ = [
    "compute_metrics",
    "compute_roa",
    "ROAResult",
    "monte_carlo",
    "RobustnessResult",
]
