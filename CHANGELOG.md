# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-27

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

[0.1.0]: https://github.com/Manas-arumalla/segway-control-suite/releases/tag/v0.1.0
