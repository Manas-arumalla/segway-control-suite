"""Composable navigator: drive a *balancing* TWIP to a goal through an obstacle map.

This is the piece that ties the stack together. Pick any combination of

  * **balance controller** — the inner upright + speed loop (lqr, mpc, smc, hinf, ...),
  * **global planner**     — A*, Dijkstra, RRT, RRT*, PRM, Potential-Field,
  * **path follower**      — Pure Pursuit, Stanley, DWA, MPC, Vector-Field,

and :class:`Navigator` plans a collision-free path, then rolls out the planar TWIP so the
follower steers it along that path while the inner loop keeps it balanced. The result bundles
the planned path, the closed-loop trajectory, and success/clearance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimConfig, TWIPParams
from .control import TWIPController
from .followers import build_follower
from .planners import build_planner
from .sim import TWIPTrajectory, simulate_twip
from .world import World


@dataclass
class NavResult:
    """Outcome of a navigation run."""

    path: np.ndarray | None             # planned global path (K, 2), or None if planning failed
    trajectory: TWIPTrajectory | None   # closed-loop TWIP rollout, or None if planning failed
    reached: bool                       # goal reached without falling
    fell: bool                          # body exceeded the fall angle
    world: World
    start: np.ndarray
    goal: np.ndarray
    balance_name: str
    planner_name: str
    follower_name: str

    @property
    def planned(self) -> bool:
        return self.path is not None

    @property
    def success(self) -> bool:
        return self.reached and not self.fell

    @property
    def driven_path(self) -> np.ndarray:
        """The (M, 2) x-y track the robot actually drove."""
        if self.trajectory is None:
            return np.empty((0, 2))
        return np.column_stack([self.trajectory.x, self.trajectory.y])

    @property
    def path_length(self) -> float:
        """Length of the *driven* trajectory [m]."""
        xy = self.driven_path
        if len(xy) < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))

    @property
    def time_to_goal(self) -> float:
        """Simulated time at the end of the run [s] (the arrival time when ``reached``)."""
        if self.trajectory is None or len(self.trajectory.t) == 0:
            return float("nan")
        return float(self.trajectory.t[-1])

    @property
    def min_clearance(self) -> float:
        """Smallest gap between the robot centre and any (un-inflated) obstacle along the run."""
        xy = self.driven_path
        if len(xy) == 0 or not self.world.obstacles:
            return float("inf")
        gaps = [
            np.hypot(xy[:, 0] - o.x, xy[:, 1] - o.y) - o.r - self.world.robot_radius
            for o in self.world.obstacles
        ]
        return float(np.min(gaps))

    @property
    def final_goal_distance(self) -> float:
        xy = self.driven_path
        if len(xy) == 0:
            return float("nan")
        return float(np.hypot(xy[-1, 0] - self.goal[0], xy[-1, 1] - self.goal[1]))


class Navigator:
    """A fixed (balance, planner, follower) triple that can be run on any world/goal."""

    def __init__(
        self,
        params: TWIPParams,
        balance: str = "lqr",
        planner: str = "a_star",
        follower: str = "pure_pursuit",
        *,
        balance_kwargs: dict | None = None,
        planner_kwargs: dict | None = None,
        follower_kwargs: dict | None = None,
        k_yaw: float = 8.0,
        goal_tol: float = 0.3,
        backend: str = "analytic",
    ):
        if backend not in ("analytic", "mujoco"):
            raise ValueError(f"backend must be 'analytic' or 'mujoco', got {backend!r}")
        self.params = params
        self.goal_tol = float(goal_tol)
        self.backend = backend
        self.balance_name = balance
        self.planner_name = planner
        self.follower_name = follower

        self.controller = TWIPController(
            params, balance=balance, balance_kwargs=balance_kwargs, k_yaw=k_yaw
        )
        self.planner = build_planner(planner, **(planner_kwargs or {}))
        # Keep the follower's own stop tolerance aligned with the navigator's, unless overridden.
        fkw = dict(follower_kwargs or {})
        fkw.setdefault("goal_tol", goal_tol)
        self.follower = build_follower(follower, **fkw)

    def plan(self, world: World, start, goal) -> np.ndarray | None:
        return self.planner.plan(world, start, goal)

    def run(
        self,
        world: World,
        start,
        goal,
        sim: SimConfig | None = None,
        command_dt: float = 0.1,
        terrain=None,
    ) -> NavResult:
        """Plan a path then drive the balancing TWIP along it to ``goal``.

        With ``backend="mujoco"`` a :class:`~segway.navigation.terrain.Terrain` may be passed to
        drive the route over a heightfield (the slopes act as disturbances on the inner loop).
        """
        if terrain is not None and self.backend != "mujoco":
            raise ValueError("terrain navigation requires backend='mujoco'")
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        def _result(path, traj, reached, fell):
            return NavResult(
                path=path, trajectory=traj, reached=reached, fell=fell,
                world=world, start=start, goal=goal,
                balance_name=self.balance_name,
                planner_name=self.planner_name,
                follower_name=self.follower_name,
            )

        path = self.plan(world, start, goal)
        if path is None or len(path) == 0:
            return _result(None, None, False, False)

        # Start facing the first leg of the path so the follower isn't fighting the heading.
        if len(path) >= 2:
            psi0 = float(np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0]))
        else:
            psi0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
        x0 = np.zeros(7)
        x0[0], x0[1], x0[2] = start[0], start[1], psi0

        self.follower.reset()

        def command_fn(t, state):
            v, w, _done = self.follower.command(
                (float(state[0]), float(state[1]), float(state[2])), path, world
            )
            return v, w

        def stop_fn(state):
            return np.hypot(state[0] - goal[0], state[1] - goal[1]) < self.goal_tol

        if self.backend == "mujoco" and terrain is not None:
            from ..sim.mujoco_terrain import simulate_twip_terrain
            sim = sim or SimConfig(dt=0.002, duration=40.0, fall_angle=1.2)
            traj = simulate_twip_terrain(
                self.params, self.controller, command_fn, terrain,
                sim=sim, x0=x0, command_dt=command_dt, stop_fn=stop_fn, world=world,
            )
        elif self.backend == "mujoco":
            from ..sim.mujoco_twip import simulate_twip_mujoco
            sim = sim or SimConfig(dt=0.002, duration=40.0, fall_angle=1.2)
            traj = simulate_twip_mujoco(
                self.params, self.controller, command_fn,
                sim=sim, x0=x0, command_dt=command_dt, stop_fn=stop_fn,
            )
        else:
            sim = sim or SimConfig(dt=0.005, duration=40.0, fall_angle=1.2)
            traj = simulate_twip(
                self.params, self.controller, command_fn,
                sim=sim, x0=x0, command_dt=command_dt, stop_fn=stop_fn,
            )
        final_dist = np.hypot(traj.x[-1] - goal[0], traj.y[-1] - goal[1])
        reached = (traj.reached or final_dist < self.goal_tol) and not traj.fell
        return _result(path, traj, bool(reached), bool(traj.fell))


def navigate(
    params: TWIPParams,
    world: World,
    start,
    goal,
    *,
    balance: str = "lqr",
    planner: str = "a_star",
    follower: str = "pure_pursuit",
    balance_kwargs: dict | None = None,
    planner_kwargs: dict | None = None,
    follower_kwargs: dict | None = None,
    k_yaw: float = 8.0,
    goal_tol: float = 0.3,
    backend: str = "analytic",
    sim: SimConfig | None = None,
    command_dt: float = 0.1,
    terrain=None,
) -> NavResult:
    """One-shot convenience wrapper around :class:`Navigator`."""
    nav = Navigator(
        params, balance=balance, planner=planner, follower=follower,
        balance_kwargs=balance_kwargs, planner_kwargs=planner_kwargs,
        follower_kwargs=follower_kwargs, k_yaw=k_yaw, goal_tol=goal_tol, backend=backend,
    )
    return nav.run(world, start, goal, sim=sim, command_dt=command_dt, terrain=terrain)
