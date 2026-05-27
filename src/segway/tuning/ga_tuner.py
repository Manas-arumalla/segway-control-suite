"""Genetic-algorithm controller tuning (DEAP).

A modernized, physics-aware successor to the legacy auto-tuner: it uses the shared,
multi-scenario objective and the same search bounds as the Optuna tuner, so GA / TPE /
CMA-ES can be compared head-to-head. Requires the ``tuning`` extra (DEAP).
"""

from __future__ import annotations

import random

from ..config import RobotParams, SimConfig
from ..sim import Scenario
from .objective import controller_cost, flat_to_kwargs, search_bounds
from .optuna_tuner import TuneResult


def _clamp(lows, highs):
    def deco(func):
        def wrapper(*a, **k):
            offspring = func(*a, **k)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = min(max(child[i], lows[i]), highs[i])
            return offspring

        return wrapper

    return deco


def ga_tune(
    controller_name: str,
    params: RobotParams | None = None,
    scenarios: list[Scenario] | None = None,
    pop_size: int = 30,
    ngen: int = 15,
    seed: int = 42,
    sim: SimConfig | None = None,
) -> TuneResult:
    """Tune a controller's gains with a genetic algorithm."""
    from deap import algorithms, base, creator, tools

    params = params or RobotParams()
    sim = sim or SimConfig(duration=8.0)
    bounds = search_bounds[controller_name]
    names = [b[0] for b in bounds]
    lows = [b[1] for b in bounds]
    highs = [b[2] for b in bounds]
    ndim = len(bounds)

    if not hasattr(creator, "FitnessMinGA"):
        creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualGA"):
        creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

    toolbox = base.Toolbox()
    for i, (_, lo, hi, _log) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)
    toolbox.register("individual", tools.initCycle, creator.IndividualGA,
                     tuple(getattr(toolbox, f"attr_{i}") for i in range(ndim)), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        flat = {names[i]: ind[i] for i in range(ndim)}
        return (controller_cost(controller_name, flat_to_kwargs(controller_name, flat),
                                params, scenarios, sim),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, low=lows, up=highs, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", _clamp(lows, highs))
    toolbox.decorate("mutate", _clamp(lows, highs))

    random.seed(seed)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=ngen, halloffame=hof, verbose=False)

    best = hof[0]
    flat = {names[i]: best[i] for i in range(ndim)}
    return TuneResult(controller_name, flat_to_kwargs(controller_name, flat),
                      float(best.fitness.values[0]), "ga", pop_size * ngen)
