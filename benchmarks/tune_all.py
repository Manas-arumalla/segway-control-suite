"""Auto-tune every tunable controller and measure how much it improved.

For each controller we record the objective cost with its hand-set default gains, run Optuna
(TPE) to find better gains, then re-measure cost and Monte-Carlo robustness (designed on
nominal, tested on a hard randomized regime). Outputs tuned gains (JSON), before/after
response plots, and a Markdown report under benchmarks/results/.

Usage:
    python benchmarks/tune_all.py                 # full
    python benchmarks/tune_all.py --quick         # small budgets
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running directly from a source checkout (before `pip install -e .`).
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import argparse
import json
from pathlib import Path

from segway.analysis import monte_carlo
from segway.config import RobotParams, SimConfig
from segway.controllers import build_controller
from segway.sim import STANDARD_SCENARIOS, Scenario, simulate
from segway.tuning import optuna_tune
from segway.tuning.objective import controller_cost, default_scenarios
from segway.viz import plot_comparison

RESULTS = Path("benchmarks/results")
TUNABLE = ["pid", "cascaded_pid", "pole_placement", "lqr", "smc"]

# Same hard robustness regime as the main benchmark.
ROB_SCEN = Scenario.balance(1.0)
ROB_RANGES = {"m_pend": 0.6, "l": 0.4, "I_pend": 0.6, "b_x": 0.8, "b_theta": 0.8}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tune all controllers and compare")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--mc-n", type=int, default=None)
    args = ap.parse_args(argv)
    trials = args.trials or (15 if args.quick else 70)
    mc_n = args.mc_n or (20 if args.quick else 80)

    RESULTS.mkdir(parents=True, exist_ok=True)
    params = RobotParams()
    sim = SimConfig(duration=8.0)
    scen = default_scenarios()

    rows = []
    gains: dict[str, dict] = {}
    for name in TUNABLE:
        print(f"[tune] {name} ...", flush=True)
        default_cost = controller_cost(name, {}, params, scen, sim)
        res = optuna_tune(name, params=params, scenarios=scen, sim=sim, n_trials=trials, sampler="tpe")
        tuned_cost = res.best_cost
        gains[name] = res.best_kwargs

        rob_def = monte_carlo(name, scenario=ROB_SCEN, ranges=ROB_RANGES, n=mc_n).success_rate
        rob_tun = monte_carlo(name, scenario=ROB_SCEN, ranges=ROB_RANGES, n=mc_n,
                              **res.best_kwargs).success_rate

        impr = 100.0 * (default_cost - tuned_cost) / default_cost if default_cost > 0 else 0.0
        rows.append((name, default_cost, tuned_cost, impr, rob_def, rob_tun))
        print(f"       cost {default_cost:.2f} -> {tuned_cost:.2f}  ({impr:+.1f}%)   "
              f"robustness {rob_def:.0%} -> {rob_tun:.0%}", flush=True)

        # Before/after response on the kick scenario.
        kick = STANDARD_SCENARIOS["kick"]
        trajs = {
            "default": simulate(params, build_controller(name, params), kick, sim),
            "tuned": simulate(params, build_controller(name, params, **res.best_kwargs), kick, sim),
        }
        plot_comparison(trajs, RESULTS / f"tuned_{name}.png", title=f"{name}: default vs tuned (kick)")

    (RESULTS / "tuned_gains.json").write_text(json.dumps(gains, indent=2), encoding="utf-8")

    # Markdown report.
    lines = ["# Auto-Tuning Report", "",
             "_Optuna (TPE). Cost = settling + control effort + position drift over a "
             "scenario battery (lower is better). Robustness = success rate on randomized "
             "plants from a 1.0 rad tilt._", "",
             "| controller | default cost | tuned cost | improvement | robustness (def→tuned) |",
             "|---|---|---|---|---|"]
    for name, dc, tc, impr, rd, rt in rows:
        lines.append(f"| {name} | {dc:.2f} | {tc:.2f} | {impr:+.1f}% | {rd:.0%} → {rt:.0%} |")
    lines += ["", "## Tuned gains", "", "```json", json.dumps(gains, indent=2), "```", ""]
    for name, *_ in rows:
        lines.append(f"![{name}](tuned_{name}.png)")
    (RESULTS / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone. See {RESULTS / 'tuning_report.md'} and {RESULTS / 'tuned_gains.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
