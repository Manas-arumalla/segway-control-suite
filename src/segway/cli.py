"""Command-line interface: ``segway <command>``.

Commands:
    list                       list available controllers
    info                       print the linearized model + stability/controllability
    run --controller lqr ...   simulate a controller on a scenario and print metrics
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import RobotParams, SimConfig
from .controllers import build_controller, list_controllers
from .models import DEFAULT_C, is_controllable, is_observable, linearize, open_loop_poles
from .sim import STANDARD_SCENARIOS, simulate


def _cmd_list(_: argparse.Namespace) -> int:
    print("Available controllers:")
    for name in list_controllers():
        print(f"  - {name}")
    print("\nStandard scenarios:")
    for name in STANDARD_SCENARIOS:
        print(f"  - {name}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    p = RobotParams()
    A, B = linearize(p)
    np.set_printoptions(precision=4, suppress=True)
    print("Robot parameters:", p)
    print("\nA =\n", A)
    print("\nB =\n", B)
    print("\nOpen-loop poles:", np.round(open_loop_poles(A), 4))
    print("Controllable (A,B):", is_controllable(A, B))
    print("Observable  (A,C):", is_observable(A, DEFAULT_C))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    p = RobotParams()
    ctrl = build_controller(args.controller, p)
    scenario = STANDARD_SCENARIOS.get(args.scenario)
    if scenario is None:
        raise SystemExit(f"unknown scenario {args.scenario!r}; choose from {list(STANDARD_SCENARIOS)}")
    traj = simulate(p, ctrl, scenario, SimConfig(duration=args.duration))
    print(f"Controller: {ctrl.name}   Scenario: {scenario.name}")
    for k, v in traj.metrics().items():
        print(f"  {k:>22}: {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="segway", description="Segway control suite")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list controllers and scenarios").set_defaults(func=_cmd_list)
    sub.add_parser("info", help="print model analysis").set_defaults(func=_cmd_info)

    run = sub.add_parser("run", help="simulate a controller on a scenario")
    run.add_argument("--controller", default="lqr", help="controller name (see `segway list`)")
    run.add_argument("--scenario", default="balance_small", help="scenario name")
    run.add_argument("--duration", type=float, default=10.0, help="sim duration [s]")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
