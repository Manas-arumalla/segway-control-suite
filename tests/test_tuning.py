"""Auto-tuning: Optuna (TPE/CMA-ES) and GA produce valid, stabilizing gains."""

from __future__ import annotations

import pytest

from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.sim import Scenario, simulate

pytest.importorskip("optuna")

FAST_SIM = SimConfig(duration=5.0)
ONE_SCENARIO = [Scenario.balance(0.2)]


@pytest.fixture
def params() -> RobotParams:
    return RobotParams()


@pytest.mark.parametrize("sampler", ["tpe", "cmaes"])
def test_optuna_tune_lqr(params, sampler):
    from segway.tuning import optuna_tune

    res = optuna_tune("lqr", scenarios=ONE_SCENARIO, sim=FAST_SIM, n_trials=10,
                      sampler=sampler, seed=0)
    assert res.best_cost < 1e5
    ctrl = build_controller("lqr", params, **res.best_kwargs)
    traj = simulate(params, ctrl, Scenario.balance(0.2), SimConfig(duration=8.0))
    assert not traj.fell and abs(traj.theta[-1]) < 0.05


def test_ga_tune_pid(params):
    pytest.importorskip("deap")
    from segway.tuning import ga_tune

    res = ga_tune("pid", scenarios=ONE_SCENARIO, sim=FAST_SIM, pop_size=8, ngen=3, seed=1)
    assert res.best_cost < 1e5
    ctrl = build_controller("pid", params, **res.best_kwargs)
    traj = simulate(params, ctrl, Scenario.balance(0.15), SimConfig(duration=8.0))
    assert not traj.fell
