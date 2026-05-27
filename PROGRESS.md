# Development Log

A phase-by-phase record of how the suite was built, with the key engineering decisions and
verified results at each stage. For a concise, version-oriented summary see
[`CHANGELOG.md`](CHANGELOG.md); for reproducible milestone snapshots see
[`CHECKPOINTS.md`](CHECKPOINTS.md); for the forward plan see [`docs/roadmap.md`](docs/roadmap.md).

The full suite is feature-complete: **50 tests passing, lint clean**. The original prototype
is preserved untouched under `Control GUI/`.

---

## Phase 5 — Presentation: UIs, docs, and media

- **Desktop GUI** (`apps/desktop_gui.py`, CustomTkinter): a polished control center with
  tabbed configuration (Controller · Robot · Scenario), a persistent action bar, an
  indeterminate progress bar, and tabbed results (Response · Metrics cards · ROA). Runs on
  background threads with live embedded plots; includes a live 3D MuJoCo viewer, region of
  attraction, auto-tuning, GIF rendering, and a compare-all overlay.
- **Streamlit dashboard** (`apps/streamlit_app.py`): the same configuration and analysis in
  the browser — single-run, compare-all, and region-of-attraction views. Both front-ends
  share one parameter module (`apps/_common.py`) so they expose identical controls.
- **Documentation site** (MkDocs Material): home, modeling derivation (with LaTeX),
  architecture (with a module diagram), advanced-methods catalog, roadmap, and notes.
- **Media**: MuJoCo-rendered GIFs for LQR balancing, energy swing-up, iLQR swing-up, and
  MRAC on a mismatched robot, plus controller-comparison and estimation plots.
- **Swing-up visualization fix**: swing-up is a full 180° maneuver, so it is rendered on an
  elevated cart-pole model (`models/assets/cartpole.xml`) where the body swings freely above
  the floor; the dynamics are identical to the main model (verified by a dedicated test).

## Phase 4 — Advanced control, learning, and tuning

- **Controllers added**: cascaded PID (holds position where tilt-only PID drifts ~2.8 m → ~0.1 m),
  H∞ state feedback (Hamiltonian Riccati + γ-iteration, no `slycot`; lowest worst-case
  disturbance gain — H∞ norm 2.57 vs LQR 2.92), adaptive MRAC (Lyapunov-stable LQR
  augmentation; 89% ROA, 100% robustness), energy-based swing-up + LQR catch, and a learned
  PPO policy.
- **Reinforcement learning**: a Gymnasium environment on the shared plant with domain
  randomization; PPO (Stable-Baselines3) matches LQR at 100% robustness in-distribution and
  the out-of-distribution limit is documented.
- **iLQR** trajectory optimizer: optimal reach and swing-up from hanging; plus min-jerk
  reference tracking (time-varying references in the runner).
- **Auto-tuning** (`segway.tuning`): a shared multi-scenario objective optimized by Optuna
  (TPE), CMA-ES, and a genetic algorithm. Tuning lifts LQR and SMC to 100% robustness;
  `benchmarks/tune_all.py` reports default-vs-tuned for every controller.
- **Pole-placement robustness**: default poles changed to a well-damped, LQR-inspired set
  `[-1.2±0.9j, -10, -25]`, raising ROA 65%→85% and robustness 37%→78%.

## Phase 3 — Benchmark & analysis

- **Metrics-driven benchmark** (`benchmarks/run_all.py`): every regulator × every scenario →
  CSV, comparison plots, region-of-attraction maps, and a Markdown report.
- **Region of attraction** (`analysis/roa.py`) and **Monte-Carlo robustness**
  (`analysis/robustness.py`, designed on nominal, tested on randomized plants).
- Final 8-regulator benchmark: ROA 76–96%, robustness 62–100% — the metrics discriminate
  the controllers (e.g. MPC/PID/cascaded-PID/MRAC most robust; H∞ optimizes a different axis).

## Phase 2 — Realism: sensing and estimation

- **Sensor model**: encoder (position) + IMU (tilt) with noise, bias, and quantization.
- **Kalman filter and EKF**: reconstruct the unmeasured velocities; controllers can run on
  estimates instead of ground truth and still balance.
- **MuJoCo backend**: an independent articulated-body simulation used for 3D visualization
  and as a cross-check of the analytic model; offscreen rendering to GIF/MP4.

## Phase 1 — Validated core

- **One canonical plant** (`models/dynamics.py`): full nonlinear equations of motion plus an
  analytic linearization, **machine-validated three ways** — against a finite-difference
  Jacobian, an independent SymPy derivation, and (qualitatively) the MuJoCo model.
- **Controllers**: a uniform `Controller` interface and a `build_controller` registry,
  starting with PID, pole placement, LQR, SMC, and MPC — each proven to stabilize the
  *nonlinear* plant by the test suite.
- **Deterministic headless simulator** (RK4) with scenarios, disturbances, reference
  tracking, and fall detection; a metrics module; and a `segway` CLI.

## Phase 0 — Foundations

- Installable `src/` package, dependency extras, GitHub Actions CI, linting/formatting/type
  configuration, license, and documentation scaffolding. The original prototype is preserved
  untouched under `Control GUI/`.

---

## Roadmap snapshot

All planned phases are complete. Forward-looking ideas (a true rolling-contact wheel model,
hardware sim-to-real, additional learned controllers) are catalogued in
[`docs/roadmap.md`](docs/roadmap.md) and [`docs/advanced-methods.md`](docs/advanced-methods.md).
