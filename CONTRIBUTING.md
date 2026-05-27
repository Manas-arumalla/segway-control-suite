# Contributing

Thanks for your interest in the Segway Control Suite. Contributions — bug reports, new
controllers, analysis tools, documentation, and tests — are welcome.

## Development setup

```bash
git clone <your-fork-url>
cd segway-control-suite
python -m pip install -e ".[all]"      # full toolchain
python -m pip install cmaes            # CMA-ES tuner backend (optional)
pytest                                 # run the test suite
```

The package uses a `src/` layout. Entry-point scripts under `apps/`, `benchmarks/`,
`scripts/`, and `examples/` add `src/` to the path automatically, so they run from a checkout
without installing.

## Quality gates

Please make sure the following pass before opening a pull request:

```bash
ruff check src tests benchmarks scripts examples apps    # lint + import order
ruff format src tests                                    # formatting (optional but encouraged)
pytest                                                   # all tests green
```

CI runs the same checks on Python 3.10 and 3.11.

## Guidelines

- **Don't modify `Control GUI/`** — it is the preserved original prototype and is kept as-is.
- **One source of truth for physics.** `RobotParams` is shared by the dynamics, controllers,
  simulator, and analysis. Avoid introducing parallel parameter sets.
- **Controllers** subclass `segway.controllers.base.Controller`, implement
  `compute(state, t)`, and register via the factory so the CLI, benchmark, and UIs pick them
  up automatically.
- **Add a test** for new behavior. The headless RK4 backend keeps tests fast and
  deterministic (no display required).
- **Keep it documented.** Update the docstring, the relevant docs page, and the changelog.

## Adding a controller (quick recipe)

1. Create `src/segway/controllers/<name>.py` with a `Controller` subclass.
2. Register it in `src/segway/controllers/__init__.py` (and `REGULATORS` if it regulates
   the upright equilibrium).
3. If it has tunable gains, add a search space entry in `src/segway/tuning/objective.py` and
   a UI spec in `apps/_common.py`.
4. Add a test asserting it stabilizes the nonlinear plant.
