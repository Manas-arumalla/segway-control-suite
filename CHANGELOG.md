# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Navigation extension — the balancer lifted from the line into the plane.

### Added

- **Planar TWIP model** — a two-wheeled inverted pendulum (`models/twip.py`) whose
  longitudinal/balance block is identical to the 1-D plant, so every balancing controller is
  reused unchanged for the inner loop; a yaw loop tracks heading.
- **Composable navigation stack** — `navigate(...)` / `Navigator` selecting a
  *balance controller × global planner × path follower* at run time, with a `NavResult` of
  success / time / driven-path length / obstacle clearance.
  - **Planners**: A*, Dijkstra, RRT, RRT*, PRM, potential field (registry `build_planner`).
  - **Followers**: pure pursuit, Stanley, DWA, sampling-MPC, vector field (registry
    `build_follower`).
  - **Maps**: corridor, slalom, rooms, forest (`build_scenario`), with a top-down plot.
- **Real rolling-contact robot** — a free-floating MuJoCo TWIP on two driven, ground-contacting
  wheels (`models/assets/twip.xml`, `sim/mujoco_twip.py`); the analytic-model controller
  balances and drives it, cross-checking the decoupling. Navigate it with `backend="mujoco"`.
- **Uneven terrain** — procedural heightfields (`navigation/terrain.py`, `sim/mujoco_terrain.py`)
  the robot traverses with no terrain sensing; `navigate(..., terrain=...)`.
- **Learned navigation** — a goal-conditioned PPO policy (`envs/twip_nav_env.py`) wrapped as
  `rl_navigate(...)`, with a training script and a learned-vs-classical comparison.
- **Navigation benchmark** — `benchmarks/run_nav.py` sweeps planner / follower / balance across
  maps and reports findings.
- **3-D navigation rendering** — `viz.render_navigation` animates a run with obstacles, start,
  and goal; and a live MuJoCo navigation viewer (`viz.live_view_navigation`) drives the robot
  along a planned route (or over terrain) in an interactive window. Navigation design is
  documented in `docs/navigation.md`.
- **Research-grade figures + unified theme** — `viz/style.py` (one publication theme) and
  `viz/nav_plots.py`: per-run analysis (speed-coloured route + speed/pitch/yaw/torque), terrain
  height-map/elevation/pitch-vs-slope, benchmark success matrices, and a learned-vs-classical
  figure with the PPO learning curve.
- **Redesigned front-ends** — both the desktop GUI and the web dashboard are now mode-based
  (🛴 Balancing / 🧭 Navigation) with a clean **Watch** (live/rendered 3-D) vs **Run** (headless
  results) split, auto-disabling of incompatible options, and an interactive custom-map editor.

## 0.1.0 — 2026-05-27

First public release: a complete, validated control benchmark for a self-balancing wheeled
inverted pendulum.

### Added

- **Validated plant model** — full nonlinear equations of motion and an analytic
  linearization, cross-checked against a finite-difference Jacobian and an independent
  symbolic (SymPy) derivation.
- **Ten control strategies** — PID, cascaded PID, pole placement, LQR, MPC, sliding-mode
  (SMC), H∞ (state feedback), adaptive MRAC, energy-based swing-up, and a learned PPO policy
  — plus an iLQR trajectory optimizer and min-jerk reference tracking.
- **State estimation** — sensor model (encoder + IMU with noise/bias/quantization) and
  Kalman / Extended Kalman filters; controllers can run on estimates.
- **Two simulation backends** — a fast, deterministic headless RK4 integrator and a
  high-fidelity MuJoCo backend (also used as a physics cross-check and for 3D rendering).
- **Analysis** — performance metrics, region-of-attraction estimation, and Monte-Carlo
  robustness, with an automated multi-controller benchmark harness.
- **Auto-tuning** — a shared objective optimized by Optuna (TPE), CMA-ES, or a genetic
  algorithm, with a default-vs-tuned comparison report.
- **Interactive applications** — a CustomTkinter desktop control center (live plots, 3D
  viewer, ROA, tuning, rendering) and a Streamlit web dashboard, sharing one parameter module.
- **Documentation** — a MkDocs Material site (modeling derivation, architecture, advanced
  methods, roadmap, implementation notes), a test suite (50 tests), and CI.

### Notes

- The original prototype is preserved untouched under `Control GUI/` for reference.
