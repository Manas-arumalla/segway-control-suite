"""Command-line interface: ``segway <command>``.

Commands:
    list                       list controllers, scenarios, planners, followers, maps
    info                       print the linearized model + stability/controllability
    run --controller lqr ...   simulate a controller on a scenario and print metrics
    nav --planner a_star ...   navigate a balancing TWIP to a goal through an obstacle map
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import RobotParams, SimConfig, TWIPParams
from .controllers import build_controller, list_controllers
from .models import DEFAULT_C, is_controllable, is_observable, linearize, open_loop_poles
from .navigation import (
    build_scenario,
    build_terrain,
    list_followers,
    list_planners,
    list_scenarios,
    list_terrains,
    navigate,
)
from .sim import STANDARD_SCENARIOS, simulate


def _cmd_list(_: argparse.Namespace) -> int:
    def _section(title, names):
        print(f"{title}:")
        for name in names:
            print(f"  - {name}")

    _section("Balancing controllers", list_controllers())
    print()
    _section("Standard scenarios", STANDARD_SCENARIOS)
    print()
    _section("Navigation planners", list_planners())
    print()
    _section("Navigation followers", list_followers())
    print()
    _section("Navigation maps", list_scenarios())
    print()
    _section("Navigation terrains", list_terrains())
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


def _cmd_nav(args: argparse.Namespace) -> int:
    scenario = build_scenario(args.map)
    terrain = None
    if args.terrain and args.terrain != "none":
        if args.backend != "mujoco":
            raise SystemExit("--terrain requires --backend mujoco")
        terrain = build_terrain(args.terrain)
    res = navigate(
        TWIPParams(), scenario.world, scenario.start, scenario.goal,
        balance=args.balance, planner=args.planner, follower=args.follower,
        backend=args.backend, terrain=terrain,
    )
    print(f"Map: {scenario.name}   ({scenario.description})")
    print(f"Stack: balance={args.balance}  planner={args.planner}  "
          f"follower={args.follower}  backend={args.backend}"
          f"{'  terrain=' + args.terrain if terrain is not None else ''}")
    print(f"Start: {tuple(round(s, 2) for s in scenario.start)}   "
          f"Goal: {tuple(round(g, 2) for g in scenario.goal)}")
    if not res.planned:
        print("Result: NO PATH FOUND (planner could not connect start to goal)")
        return 1
    status = "REACHED" if res.success else ("FELL" if res.fell else "DID NOT REACH")
    print(f"Result: {status}")
    print(f"  {'final goal distance':>22}: {res.final_goal_distance:.3f} m")
    print(f"  {'driven path length':>22}: {res.path_length:.3f} m")
    print(f"  {'time':>22}: {res.time_to_goal:.2f} s")
    print(f"  {'min obstacle clearance':>22}: {res.min_clearance:.3f} m")
    if args.plot:
        import pathlib

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .navigation.plot import plot_navigation
        out = pathlib.Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_navigation(res)
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  saved plot -> {out}")
    return 0 if res.success else 1


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

    nav = sub.add_parser("nav", help="navigate a balancing TWIP to a goal through a map")
    nav.add_argument("--balance", default="lqr", help="inner balancing controller")
    nav.add_argument("--planner", default="a_star", help="global planner (see `segway list`)")
    nav.add_argument("--follower", default="pure_pursuit", help="path follower")
    nav.add_argument("--map", default="corridor", help="preset map (see `segway list`)")
    nav.add_argument("--backend", default="analytic", choices=["analytic", "mujoco"],
                     help="analytic RK4 or full MuJoCo rolling-contact physics")
    nav.add_argument("--terrain", default="none",
                     help="uneven-terrain preset to drive over (requires --backend mujoco)")
    nav.add_argument("--plot", default=None, help="optional PNG path for a top-down plot")
    nav.set_defaults(func=_cmd_nav)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
