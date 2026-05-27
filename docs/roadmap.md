# Roadmap

The phased plan for turning the legacy student project into an engineering-grade,
portfolio-quality control benchmark. Legacy code in `Control GUI/` is preserved untouched
throughout; all new work lives under `src/segway/` and friends.

## Guiding thesis

> A rigorous, reproducible benchmark of control strategies for a self-balancing robot —
> from classical to learned — with validated dynamics, realistic sensing, quantitative
> head-to-head metrics, and publication-quality visuals.

## Phases

### Phase 0 — Foundations ✅
Installable package, CI, linting/typing, license, tracking docs, dependency check.

### Phase 1 — Validated core 🚧
- One canonical dynamics model (nonlinear EOM + analytic linearization).
- Linearization **machine-verified** vs finite-difference Jacobian and vs SymPy.
- Controllability/observability, open-loop pole analysis.
- Config system (single source of truth for physical parameters).
- Controllers: PID, Pole Placement, LQR, SMC, MPC (clean interface + registry).
- Headless RK4 simulator with a scenario system.
- Metrics module + full pytest suite.

### Phase 2 — Realism
- Sensor models: encoder (position) + IMU/gyro (tilt rate) with noise, bias, delay,
  quantization.
- Kalman filter / EKF; controllers run on **estimates**, not ground truth.
- MuJoCo backend for high-fidelity sim + 3D, used as an independent cross-check.
- Headless rendering → GIF/MP4; matplotlib plotting helpers.

### Phase 3 — Benchmark & analysis
- Metrics-driven harness: every controller × every scenario → CSV + comparison plots +
  auto-generated report. Deterministic seeds.
- Fast, comparable Region-of-Attraction (vectorized/parallel; overlay controllers).
- Monte-Carlo robustness over randomized mass/length/friction.

### Phase 4 — Standout controllers
- H∞ / robust synthesis against modeled uncertainty.
- Energy-based swing-up + hybrid switch to balancing.
- RL: a Gymnasium env on the shared plant + PPO (Stable-Baselines3), benchmarked against
  the classical controllers ("model-based vs learned").
- Optuna-based tuning alongside the legacy-style genetic algorithm.

### Phase 5 — Presentation
- Modernized desktop GUI (CustomTkinter) — all legacy features, no self-destruct, live
  plots + embedded 3D, threaded runs, a benchmark tab.
- Streamlit web dashboard for sharing/portfolio.
- MkDocs-Material site: LaTeX theory derivations, architecture diagrams, live benchmark
  results.
- README hero GIF, results table, badges; CI builds docs.

### Future extensions
- True rolling-contact wheel model (genuine WIP, not the cart abstraction).
- Hardware reference (ESP32 + MPU6050) and sim-to-real notes.
- Docker / devcontainer; PyPI release.

## Critical path
Phase 1 unblocks everything. Phases 2–4 then run largely in parallel. Phase 5 capture
(GIFs, results) happens continuously as features land — not saved for the end.
