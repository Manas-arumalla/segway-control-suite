"""Navigation comparison benchmark.

Drives the composable navigator across preset maps and reports per-run metrics — success,
time, driven-path length, and obstacle clearance — for three curated sweeps that each vary one
axis of the stack:

  * **planner**  sweep (fixed balance=lqr, follower=pure_pursuit),
  * **follower** sweep (fixed balance=lqr, planner=a_star),
  * **balance**  sweep (fixed planner=a_star, follower=pure_pursuit).

It also pits the **learned** navigator against the classical stack on open-space goals (no
obstacles), if a trained policy is available. Outputs a CSV and a Markdown report under
``benchmarks/nav_results/``.

Usage:
    python benchmarks/run_nav.py                       # analytic backend (fast)
    python benchmarks/run_nav.py --quick               # fewer maps
    python benchmarks/run_nav.py --rl-model models/ppo_twip_nav.zip
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running directly from a source checkout (before `pip install -e .`).
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import argparse
import csv
from pathlib import Path

import numpy as np

from segway.config import TWIPParams
from segway.navigation import (
    build_scenario,
    list_followers,
    list_planners,
    navigate,
)

RESULTS = Path("benchmarks/nav_results")

# Full-state controllers that can track a speed command (tilt-only pid cannot drive).
DRIVE_CONTROLLERS = ["lqr", "mpc", "smc", "hinf", "pole_placement", "cascaded_pid"]
METRIC_COLS = ["sweep", "map", "balance", "planner", "follower",
               "planned", "success", "fell", "time_s", "path_len_m",
               "min_clearance_m", "final_dist_m"]


def _row(sweep, scenario, balance, planner, follower, res) -> dict:
    return {
        "sweep": sweep, "map": scenario.name,
        "balance": balance, "planner": planner, "follower": follower,
        "planned": res.planned, "success": res.success, "fell": res.fell,
        "time_s": round(res.time_to_goal, 2) if res.trajectory is not None else float("nan"),
        "path_len_m": round(res.path_length, 2),
        "min_clearance_m": round(res.min_clearance, 3),
        "final_dist_m": round(res.final_goal_distance, 3) if res.trajectory is not None else float("nan"),
    }


def run_sweep(sweep, stacks, maps, params=None, backend="analytic") -> list[dict]:
    """Run a list of (balance, planner, follower) stacks across named maps."""
    params = params or TWIPParams()
    rows = []
    for map_name in maps:
        sc = build_scenario(map_name)
        for balance, planner, follower in stacks:
            res = navigate(params, sc.world, sc.start, sc.goal, balance=balance,
                           planner=planner, follower=follower, backend=backend)
            rows.append(_row(sweep, sc, balance, planner, follower, res))
    return rows


def planner_stacks() -> list[tuple[str, str, str]]:
    return [("lqr", p, "pure_pursuit") for p in list_planners()]


def follower_stacks() -> list[tuple[str, str, str]]:
    return [("lqr", "a_star", f) for f in list_followers()]


def balance_stacks() -> list[tuple[str, str, str]]:
    return [(b, "a_star", "pure_pursuit") for b in DRIVE_CONTROLLERS]


def run_rl_vs_classical(model, params=None, n_goals=12, seed=0, radius=3.0) -> list[dict]:
    """Open-space goal reaching: learned navigator vs the classical stack."""
    from segway.navigation import World, rl_navigate

    params = params or TWIPParams()
    rng = np.random.default_rng(seed)
    world = World(width=2 * radius + 4, height=2 * radius + 4, obstacles=[])
    center = np.array([radius + 2, radius + 2])
    rows = []
    for k in range(n_goals):
        ang = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(1.5, radius)
        start = center
        goal = center + r * np.array([np.cos(ang), np.sin(ang)])

        rl = rl_navigate(model, params, start, goal, world=world, goal_tol=0.45)
        rows.append({"method": "rl", "goal": k, "success": rl.success, "fell": rl.fell,
                     "time_s": round(rl.time_to_goal, 2),
                     "final_dist_m": round(rl.final_goal_distance, 3)})

        cl = navigate(params, world, start, goal, balance="lqr", planner="a_star",
                      follower="pure_pursuit", goal_tol=0.45)
        rows.append({"method": "classical", "goal": k, "success": cl.success, "fell": cl.fell,
                     "time_s": round(cl.time_to_goal, 2),
                     "final_dist_m": round(cl.final_goal_distance, 3)})
    return rows


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _sweep_table(rows, key) -> str:
    hdr = [key, "map", "success", "fell", "time (s)", "path (m)", "clearance (m)"]
    body = [[r[key], r["map"], "yes" if r["success"] else "no", "yes" if r["fell"] else "no",
             r["time_s"], r["path_len_m"], r["min_clearance_m"]] for r in rows]
    return _md_table(hdr, body)


def write_report(all_rows, rl_rows, path: Path) -> None:
    lines = ["# Navigation Benchmark", "",
             "_Auto-generated by `benchmarks/run_nav.py`._", "",
             "![benchmark summary](nav_benchmark.png)", ""]
    titles = {"planner": "Planner sweep (balance=lqr, follower=pure_pursuit)",
              "follower": "Follower sweep (balance=lqr, planner=a_star)",
              "balance": "Balance-controller sweep (planner=a_star, follower=pure_pursuit)"}
    keys = {"planner": "planner", "follower": "follower", "balance": "balance"}
    for sweep, title in titles.items():
        rows = [r for r in all_rows if r["sweep"] == sweep]
        if not rows:
            continue
        n_ok = sum(r["success"] for r in rows)
        lines += [f"## {title}", "", f"_{n_ok}/{len(rows)} runs reached the goal._", "",
                  _sweep_table(rows, keys[sweep]), ""]

    if rl_rows:
        for method in ("classical", "rl"):
            mr = [r for r in rl_rows if r["method"] == method]
            ok = sum(r["success"] for r in mr)
            mean_dist = np.mean([r["final_dist_m"] for r in mr])
            lines += [] if method != "classical" else ["## Learned vs classical (open-space goals)", ""]
            lines += [f"- **{method}**: reached {ok}/{len(mr)} goals, "
                      f"mean final distance {mean_dist:.2f} m"]
        lines += ["", "![learned vs classical](nav_rl.png)", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the navigation benchmark")
    ap.add_argument("--quick", action="store_true", help="fewer maps for a fast run")
    ap.add_argument("--backend", default="analytic", choices=["analytic", "mujoco"])
    ap.add_argument("--rl-model", default=None, help="path to a trained nav policy (.zip)")
    args = ap.parse_args(argv)

    maps = ["corridor", "slalom"] if args.quick else ["corridor", "slalom", "rooms", "forest"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    params = TWIPParams()

    all_rows: list[dict] = []
    for sweep, stacks in (("planner", planner_stacks()),
                          ("follower", follower_stacks()),
                          ("balance", balance_stacks())):
        print(f"[{sweep}] {len(stacks)} stacks x {len(maps)} maps ...")
        all_rows += run_sweep(sweep, stacks, maps, params, backend=args.backend)

    with (RESULTS / "nav_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLS)
        w.writeheader()
        w.writerows(all_rows)

    rl_rows = []
    if args.rl_model and Path(args.rl_model).exists():
        print("[rl] learned vs classical on open-space goals ...")
        from stable_baselines3 import PPO
        model = PPO.load(args.rl_model, device="cpu")
        rl_rows = run_rl_vs_classical(model, params)

    # Research-grade summary figures alongside the report.
    try:
        from segway.viz import plot_benchmark_summary, plot_rl_analysis
        plot_benchmark_summary(all_rows, path=RESULTS / "nav_benchmark.png")
        print(f"  figure -> {RESULTS / 'nav_benchmark.png'}")
        if rl_rows:
            log = Path(args.rl_model).with_name("nav_train_log") / "progress.csv" if args.rl_model else None
            plot_rl_analysis(rl_rows, learning_csv=log, path=RESULTS / "nav_rl.png")
            print(f"  figure -> {RESULTS / 'nav_rl.png'}")
    except Exception as exc:  # plotting is optional
        print(f"  (skipped summary figures: {exc})")

    write_report(all_rows, rl_rows, RESULTS / "nav_report.md")
    ok = sum(r["success"] for r in all_rows)
    print(f"\nDone. {ok}/{len(all_rows)} classical runs reached the goal. "
          f"See {RESULTS / 'nav_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
