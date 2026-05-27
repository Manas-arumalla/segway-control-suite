"""Automatic controller tuning.

Optimization techniques for finding good controller gains automatically:

* :func:`optuna_tune` — Bayesian (TPE) or CMA-ES search via Optuna.
* :func:`ga_tune` — a genetic algorithm (DEAP), a modernized, physics-aware successor to
  the legacy auto-tuner.

All share one objective (:func:`controller_cost`) that scores a controller over a battery
of scenarios (settling time + control effort + position drift, with a fall penalty).
"""

from __future__ import annotations

from .objective import controller_cost, flat_to_kwargs, search_bounds

__all__ = ["controller_cost", "flat_to_kwargs", "search_bounds", "optuna_tune", "ga_tune"]


def optuna_tune(*args, **kwargs):
    """Lazy wrapper so importing the package doesn't require Optuna installed."""
    from .optuna_tuner import optuna_tune as _f

    return _f(*args, **kwargs)


def ga_tune(*args, **kwargs):
    """Lazy wrapper so importing the package doesn't require DEAP installed."""
    from .ga_tuner import ga_tune as _f

    return _f(*args, **kwargs)
