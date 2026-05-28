"""Navigation benchmark core (NAV-8) — the importable sweep runner and report writer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load benchmarks/run_nav.py as a module (it lives outside the package).
_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_nav.py"
_spec = importlib.util.spec_from_file_location("run_nav", _PATH)
run_nav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_nav)


def test_stack_builders_cover_registries():
    from segway.navigation import list_followers, list_planners

    planners = {s[1] for s in run_nav.planner_stacks()}
    followers = {s[2] for s in run_nav.follower_stacks()}
    assert planners == set(list_planners())
    assert followers == set(list_followers())


def test_run_sweep_produces_metric_rows():
    rows = run_nav.run_sweep("planner", [("lqr", "a_star", "pure_pursuit")], ["corridor"])
    assert len(rows) == 1
    r = rows[0]
    assert set(run_nav.METRIC_COLS) <= set(r)
    assert r["planned"] and r["success"]      # lqr+a_star+pure_pursuit solves the corridor
    assert r["map"] == "corridor"


def test_write_report(tmp_path):
    rows = run_nav.run_sweep("follower", [("lqr", "a_star", "pure_pursuit")], ["corridor"])
    out = tmp_path / "nav_report.md"
    run_nav.write_report(rows, [], out)
    text = out.read_text(encoding="utf-8")
    assert "Navigation Benchmark" in text and "Follower sweep" in text


def test_run_sweep_rejects_unknown_map():
    with pytest.raises(KeyError):
        run_nav.run_sweep("planner", [("lqr", "a_star", "pure_pursuit")], ["atlantis"])
