# Development Log

A phase-by-phase record of how the suite was built, with the key engineering decisions and
verified results at each stage. For a concise, version-oriented summary see
[`CHANGELOG.md`](CHANGELOG.md); for reproducible milestone snapshots see
[`CHECKPOINTS.md`](CHECKPOINTS.md); for the forward plan see [`docs/roadmap.md`](docs/roadmap.md).

The core suite is feature-complete and a **planar navigation extension** is underway, taking
the balancer from the line into the plane: **130 tests passing, lint clean**. The original
prototype is preserved untouched under `Control GUI/`.

---

## Navigation extension — driving to goals while balancing

A composable navigation stack lifts the 1-D balancer into the plane so the robot can drive to
goal points, around obstacles, while staying upright. The design is deliberately modular: at
run time you pick **{balancing controller} × {global planner} × {path follower}**.

- **Planar TWIP model** (`models/twip.py`): a two-wheeled inverted pendulum with the 7-state
  `[x, y, ψ, θ, v, ψ̇, θ̇]`. It decouples into a longitudinal/balance subsystem driven by the
  wheel-torque *sum* — *identical* to the validated 1-D plant, so every existing balancing
  controller is reused unchanged — and a yaw subsystem driven by the torque *difference*.
- **Inner loop** (`navigation/control.py`): a `TWIPController` that feeds the chosen balancing
  controller the longitudinal error `[0, v−v_des, θ, θ̇]` for the torque sum, and a
  feed-forward + proportional yaw law for the torque difference.
- **Planner suite** (`navigation/planners.py`): A*, Dijkstra, RRT, RRT*, PRM, and an
  artificial Potential Field, behind a `build_planner` registry, on an inflated occupancy-grid
  `World` (`navigation/world.py`).
- **Follower suite** (`navigation/followers.py`): Pure Pursuit, Stanley, DWA, a sampling MPC
  tracker, and a Vector/Potential field, behind a `build_follower` registry, each with
  monotonic look-ahead and a robust goal-reached test.
- **Composable navigator** (`navigation/navigator.py`): `navigate(...)` plans a collision-free
  path then rolls out the planar TWIP at two rates — a fast inner balance loop and a slower
  follower command loop — driving the robot to the goal and reporting success, driven-path
  length, time, and obstacle clearance.
- **Maps and tooling**: preset scenarios (`navigation/maps.py`: corridor, slalom, rooms,
  forest), a top-down plot (`navigation/plot.py`), and a `segway nav` CLI command. The suite
  cleanly separates planners that always find a path from reactive followers that can stall in
  tight maps — exactly the kind of contrast the navigation benchmark is built to surface.
- **Real rolling-contact robot** (`models/assets/twip.xml`, `sim/mujoco_twip.py`): a
  free-floating MuJoCo chassis on two driven wheels that roll on the ground through frictional
  contact. The *same* analytic-model controller drives it — wheel torques produce traction and
  the chassis reaction emerge from contact physics rather than being imposed — and it balances,
  tracks forward speed, and turns. This independently checks that the analytic decoupling holds
  against full contact dynamics, and the navigator runs through it via `backend="mujoco"`.
- **3-D navigation visualization** (`viz.render_navigation`): renders the balancing robot
  driving a planned route past obstacles, start, and goal markers to an animated GIF/MP4.
- **Uneven terrain** (`navigation/terrain.py`, `sim/mujoco_terrain.py`): smooth procedural
  height maps (flat / gentle / moderate / rough / ramp) drive a MuJoCo **heightfield** the robot
  must traverse. It carries no terrain sensing — the slopes are pure disturbances on the inner
  loop — yet the same controller balances and drives across bumps to ~20° local grade and climbs
  ramps. Goal-to-goal navigation runs over terrain via `navigate(..., backend="mujoco",
  terrain=...)`, and the 3-D renderer shows the robot driving the undulating ground.
- **Learned navigation** (`envs/twip_nav_env.py`, `navigation/rl_nav.py`): a goal-conditioned
  PPO policy (observation = goal-in-body-frame + balance state, action = two wheel torques,
  trained with domain randomization) drives end-to-end — one network replacing the controller +
  follower. It reaches goals in any direction without falling, settling at a ~0.4 m radial
  standoff (a velocity-tracking pendulum must lean back to arrive), and is wrapped as
  `rl_navigate(...)` returning the same `NavResult` for a fair comparison.
- **Navigation benchmark** (`benchmarks/run_nav.py`): planner / follower / balance sweeps across
  the maps plus learned-vs-classical on open-space goals. Findings: every full-state controller
  drives navigation on all maps (24/24); grid/sampling planners solve all maps while the
  potential field stalls in local minima; pure-pursuit / Stanley / MPC followers solve all maps
  while reactive DWA / vector-field stall in tight doorways; and the learned policy is
  competitive with the classical stack on open-space goals (both reach all goals within 0.45 m).
- **Research-grade visualization** (`viz/style.py`, `viz/nav_plots.py`): one publication theme
  across the suite; a per-run analysis figure (speed-coloured route + speed/pitch/yaw/torque
  tracking), terrain figures (height map, elevation profile, pitch-vs-slope), benchmark
  success-matrix heatmaps, and a learned-vs-classical figure (arrival distribution, success
  rate, PPO learning curve). Both front-ends gain a navigation tab with an interactive map
  editor — click-to-place obstacles/start/goal in the desktop GUI and a custom-map builder in
  the web dashboard.

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
