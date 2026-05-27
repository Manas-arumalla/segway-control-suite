"""Optuna-based controller tuning (TPE Bayesian optimization or CMA-ES).

Requires the ``tuning`` extra (``pip install -e ".[tuning]"`` for Optuna).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RobotParams, SimConfig
from ..sim import Scenario
from .objective import controller_cost, flat_to_kwargs, search_bounds


@dataclass
class TuneResult:
    controller_name: str
    best_kwargs: dict
    best_cost: float
    sampler: str
    n_trials: int


def _suggest(trial, controller_name: str) -> dict:
    if controller_name not in search_bounds:
        raise ValueError(f"no search space for {controller_name!r}")
    flat = {}
    for name, low, high, log in search_bounds[controller_name]:
        flat[name] = trial.suggest_float(name, low, high, log=log)
    return flat


def optuna_tune(
    controller_name: str,
    params: RobotParams | None = None,
    scenarios: list[Scenario] | None = None,
    n_trials: int = 60,
    sampler: str = "tpe",
    seed: int = 0,
    sim: SimConfig | None = None,
) -> TuneResult:
    """Tune a controller's gains with Optuna.

    ``sampler``: ``"tpe"`` (Bayesian, default) or ``"cmaes"`` (CMA-ES). Returns the best
    controller kwargs and cost.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = params or RobotParams()
    sim = sim or SimConfig(duration=8.0)

    if sampler == "cmaes":
        s = optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler == "tpe":
        s = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f"unknown sampler {sampler!r}; use 'tpe' or 'cmaes'")

    study = optuna.create_study(direction="minimize", sampler=s)

    def objective(trial):
        flat = _suggest(trial, controller_name)
        kwargs = flat_to_kwargs(controller_name, flat)
        return controller_cost(controller_name, kwargs, params, scenarios, sim)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_kwargs = flat_to_kwargs(controller_name, study.best_params)
    return TuneResult(controller_name, best_kwargs, float(study.best_value), sampler, n_trials)
