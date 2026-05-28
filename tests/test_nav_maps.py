"""Preset navigation maps + CLI wiring (NAV-5)."""

from __future__ import annotations

import pytest

from segway.cli import main
from segway.navigation import build_planner, build_scenario, list_scenarios


def test_registry_lists_all_maps():
    assert set(list_scenarios()) == {"corridor", "slalom", "rooms", "forest"}


@pytest.mark.parametrize("name", ["corridor", "slalom", "rooms", "forest"])
def test_scenario_start_and_goal_are_free_and_solvable(name):
    sc = build_scenario(name)
    assert sc.world.is_free(*sc.start), f"{name}: start blocked"
    assert sc.world.is_free(*sc.goal), f"{name}: goal blocked"
    # A* must be able to connect them — the maps are guaranteed solvable.
    path = build_planner("a_star").plan(sc.world, sc.start, sc.goal)
    assert path is not None and len(path) >= 2, f"{name}: no path found"


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        build_scenario("nowhere")


def test_cli_list_runs():
    assert main(["list"]) == 0
