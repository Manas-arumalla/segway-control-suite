# Checkpoints

Reproducible milestone snapshots. Each checkpoint records **what demonstrably works** at that
stage and the exact commands to reproduce it, providing a known-good reference point.

---

## CP-0 — Scaffold (2026-05-25)

**State:** Installable package skeleton; no functional code yet.

Works:
- `pip install -e .` succeeds; package metadata + extras resolve.
- Repo structure, CI, license, and tracking docs in place.
- Legacy implementation preserved untouched in `Control GUI/`.

Verification: file structure created; environment dependency check passed.

---

## CP-1 — Validated core (2026-05-25)

**State:** A working, tested control engine. Design and simulation use one shared,
machine-validated plant model.

Works:
- **Validated dynamics** — analytic `(A,B)` agrees with both a finite-difference Jacobian
  and an independent SymPy derivation. Verified by `tests/test_dynamics.py`.
- **Five controllers** (PID, Pole Placement, LQR, SMC, MPC) all stabilize the *nonlinear*
  plant from a tilt and survive a disturbance kick. Verified by the parametrized
  `tests/test_controllers.py`.
- **Deterministic headless simulator** with scenarios, disturbances, reference tracking,
  fall detection, and self-scoring `Trajectory`. Verified by `tests/test_sim.py`.
- **Metrics** module. Verified by `tests/test_metrics.py`.
- **CLI**: `segway list | info | run`.
- Full suite: **22 passed**.

Known gaps:
- No sensor model / state estimation yet (controllers see ground truth).
- No 3D / MuJoCo backend wired into the new engine yet.
- No automated multi-controller report (manual comparison only so far).

Reproduce:
```bash
pip install -e ".[mpc,symbolic,dev]"
pytest                      # 22 passed
segway info                 # model analysis
segway run --controller lqr --scenario kick
```

---

## CP-2 — Realism + visuals (2026-05-25)

**State:** Controllers can run on noisy estimates; an independent MuJoCo backend renders
the robot; the project produces publication-quality media.

Works:
- **State estimation** — Kalman filter + EKF reconstruct unmeasured velocities; LQR on
  estimates still balances. Verified by `tests/test_estimation.py`.
- **MuJoCo backend** — independent articulated-body sim; the analytic-designed LQR also
  stabilizes it (model cross-check). Verified by `tests/test_mujoco.py`.
- **Rendering** — offscreen MuJoCo → GIF/MP4 (works headlessly here); matplotlib plots.
- Showcase assets generated: `assets/lqr_balance.gif`, `comparison_kick.png`,
  `estimation_kf.png`.
- Full suite: **30 passed**.

Known gaps:
- No automated benchmark report / ROA / Monte-Carlo yet (Phase 3).
- Advanced controllers (H∞, swing-up, RL) and the UIs are not built yet (Phases 4–5).

Reproduce:
```bash
pip install -e ".[all]"
pytest                          # 30 passed
python examples/generate_media.py   # regenerates assets/
```

---

## CP-3 — Benchmark + advanced methods + auto-tuning (2026-05-25)

**State:** A real benchmark with quantitative, discriminating results; two new controllers;
and a full auto-tuning subsystem.

Works:
- **Benchmark harness** (`benchmarks/run_all.py`) → CSV + comparison plots + ROA maps +
  Markdown report. ROA areas 65–96%; robustness 37–100% (discriminates controllers).
- **Cascaded PID** holds position (~0.1 m) vs plain PID drift (~2.8 m).
- **Energy swing-up** stands the robot up from hanging, then balances (`assets/swingup.gif`).
- **Auto-tuning**: Optuna TPE + CMA-ES + DEAP GA over a shared objective.
- Full suite: **39 passed**.

Known gaps:
- H∞, Adaptive (MRAC), RL (PPO), iLQR not yet built (Phase 4 cont.).
- No UIs or docs site yet (Phase 5).

Reproduce:
```bash
pip install -e ".[all]" && python -m pip install cmaes
pytest                                  # 39 passed
python benchmarks/run_all.py            # full report under benchmarks/results/
python -c "from segway.tuning import optuna_tune; print(optuna_tune('lqr', n_trials=40).best_kwargs)"
```

---

## CP-4 — H∞, pole-placement fix, full-roster tuning (2026-05-25)

**State:** 7 regulators + swing-up, all tested; robust defaults; every controller tunable.

Works:
- **H∞ state-feedback** (Hamiltonian ARE + γ-iteration, no slycot) — best worst-case
  disturbance gain (H∞ norm 2.57 < LQR 2.92).
- **Pole placement fixed**: default poles `[-1.2±0.9j,-10,-25]` → ROA 65%→85%,
  robustness 37%→78%.
- **All controllers auto-tuned** (`benchmarks/tune_all.py`); LQR & SMC reach 100%
  robustness when tuned; pole-placement overfits (documented lesson).
- Refreshed benchmark (7 regulators) — ROA 76–96%, robustness 62–100%.
- Full suite: **41 passed**, ruff clean.

Known gaps:
- Adaptive (MRAC), RL (PPO), iLQR not built yet.
- No UIs or docs site yet (Phase 5).

Reproduce:
```bash
pytest                                       # 41 passed
python benchmarks/run_all.py                 # 7-regulator report
python benchmarks/tune_all.py                # default-vs-tuned report
```

---

## CP-5 — Phase 4 complete: full controller suite (2026-05-25)

**State:** 8 regulators + swing-up + a learned RL policy + iLQR planner + adaptive control.
The control breadth is now exceptional: classical, optimal, predictive, robust, nonlinear,
adaptive, learned, and trajectory-optimized.

Works:
- **MRAC** (adaptive) — 89% ROA, 100% robustness.
- **RL/PPO** with domain randomization — 100% robust in-distribution; documented OOD limit.
- **iLQR** — optimal reach + swing-up from hanging.
- **Min-jerk** reference tracking (time-varying references in the runner).
- Final benchmark (8 regulators): ROA 76–96%, robustness 62–100%.
- Full suite: **49 passed**, ruff clean.

Known gaps:
- No interactive UIs or docs site yet (Phase 5).

Reproduce:
```bash
pytest                                                  # 49 passed
python scripts/train_rl.py --timesteps 250000 --randomize
python benchmarks/run_all.py                            # 8-regulator report
python -c "from segway.planning import iLQR; from segway.config import RobotParams; \
import numpy as np; print(iLQR(RobotParams()).fit(np.array([0,0,np.pi,0]), N=200, iters=120).xs[-1])"
```

---

<!-- Template for future checkpoints:

## CP-N — <title> (YYYY-MM-DD)

**State:** <one line>

Works:
- <capability> — verified by <test/command/artifact>

Known gaps:
- <thing not yet done>

Reproduce: `<command>`
-->
