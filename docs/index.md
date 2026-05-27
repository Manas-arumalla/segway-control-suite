# Segway Control Suite

A rigorous, reproducible benchmark of control strategies for a **self-balancing wheeled
inverted pendulum** — from classical to learned — with honest dynamics, realistic sensing,
quantitative head-to-head metrics, and publication-quality visuals.

<div align="center">
  <img src="../assets/lqr_balance.gif" width="360"/>
  <img src="../assets/swingup.gif" width="360"/>
</div>

## What's inside

- **One validated plant** — analytic linearization machine-checked against a finite-difference
  Jacobian *and* an independent SymPy derivation (see [Modeling](theory/modeling.md)).
- **Ten control strategies** — PID, cascaded PID, pole placement, LQR, MPC, sliding-mode,
  H∞, MRAC (adaptive), energy swing-up, and a learned PPO policy; plus an iLQR trajectory
  optimizer.
- **Realistic sensing** — encoder + IMU noise with a Kalman/EKF reconstructing the
  unmeasured velocities.
- **A real benchmark** — region-of-attraction, Monte-Carlo robustness, and an automated
  multi-controller report.
- **Auto-tuning** — Optuna (TPE / CMA-ES) and a genetic algorithm over a shared objective.
- **Two simulation backends** — a fast headless RK4 integrator and a high-fidelity MuJoCo
  model that also cross-checks the analytic dynamics.

## Quickstart

```bash
pip install -e ".[all]"
pytest                                  # validated test suite
segway info                             # model analysis
python benchmarks/run_all.py            # full benchmark report
```

See the [Architecture](architecture.md) page for how the pieces fit together, and the
[Roadmap](roadmap.md) / [Implementation notes](implementation-notes.md) for the plan and the
"why" behind each decision.
