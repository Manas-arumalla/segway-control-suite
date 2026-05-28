"""Navigation: drive a balancing TWIP to goals, around obstacles, over uneven terrain.

A composable stack — pick a balancing controller, a planner, and a path follower — built up
across the NAV phases. NAV-1 provides the planar inner loop and simulator.
"""

from __future__ import annotations

from .control import TWIPController
from .followers import Follower, build_follower, list_followers
from .maps import NavScenario, build_scenario, list_scenarios
from .navigator import Navigator, NavResult, navigate
from .planners import Planner, build_planner, list_planners
from .rl_nav import RLNavController, rl_navigate
from .sim import TWIPTrajectory, simulate_twip
from .terrain import Terrain, build_terrain, generate_terrain, list_terrains
from .world import Obstacle, World

__all__ = [
    "TWIPController",
    "TWIPTrajectory",
    "simulate_twip",
    "World",
    "Obstacle",
    "Planner",
    "build_planner",
    "list_planners",
    "Follower",
    "build_follower",
    "list_followers",
    "Navigator",
    "NavResult",
    "navigate",
    "NavScenario",
    "build_scenario",
    "list_scenarios",
    "Terrain",
    "build_terrain",
    "generate_terrain",
    "list_terrains",
    "rl_navigate",
    "RLNavController",
]
