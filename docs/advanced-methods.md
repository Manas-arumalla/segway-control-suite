# Advanced methods — research addendum

A curated set of methods beyond the core roadmap, chosen for **impact per unit effort** and
for how well they fit a self-balancing robot. Each entry notes *why it's worth it* and a
rough effort (S/M/L). Items marked ⭐ are the recommended high-value additions.

## Control methods

| Method | Why it elevates the project | Effort |
|---|---|---|
| ⭐ **LQG** (LQR + Kalman) | The "correct" optimal output-feedback controller; pairs the LQR with the Kalman estimator I already built. Completes the optimal-control story. | S |
| ⭐ **Cascaded PID** (position outer loop → tilt inner loop) | Fixes the tilt-only PID's position drift; a fair, strong classical baseline and a great before/after story. | S |
| ⭐ **iLQR / DDP** (trajectory optimization) | Modern nonlinear optimal control; computes an optimal swing-up trajectory. Genuinely advanced and visually spectacular. | M |
| **Feedback linearization / partial FL** | Cancels nonlinearities for large-angle control; textbook nonlinear control. | M |
| **Gain-scheduled LQR** | Interpolates LQR gains across tilt — bridges linear control to wide operating range. | S |
| ⭐ **Adaptive control (MRAC / L1)** | Online adaptation to unknown mass/length — directly attacks the robustness story. | M |
| **H∞ / μ-synthesis** (already in roadmap) | Robust control against modeled uncertainty. | M |
| **Tube / robust MPC** | MPC with guaranteed constraint satisfaction under disturbance. | L |

## Optimization & tuning

| Method | Why | Effort |
|---|---|---|
| ⭐ **Optuna** (TPE/Bayesian) | Modern, sample-efficient auto-tuning; complements the legacy GA. | S |
| **CMA-ES** | Strong black-box optimizer for controller gains; nice comparison vs GA/Optuna. | S |
| **Differentiable simulation (JAX)** | Gradient-based tuning through the dynamics; cutting-edge. | L |

## Learning methods

| Method | Why | Effort |
|---|---|---|
| ⭐ **PPO** (already in roadmap) | Model-free RL baseline; "learned vs model-based" headline. | M |
| **SAC** | Off-policy, sample-efficient; a second RL point of comparison. | S (given PPO) |
| ⭐ **Domain randomization** | Train RL over randomized physics → robust, sim-to-real-ready policy. Ties RL to the robustness benchmark. | S |
| **Behavior cloning / DAgger from MPC** | Distill an expensive MPC into a fast neural policy. | M |

## Planning & trajectory (needs the 2D extension)

The current plant balances in place (1-D base motion), so classical *path planning* only
becomes meaningful once the model is extended to a planar robot (x, y, heading):

| Method | Why | Effort |
|---|---|---|
| ⭐ **Reference trajectory generator** (min-jerk / S-curve) | Smooth position setpoints for tracking — works *today* in 1-D. | S |
| **A\*** / **RRT\*** path planning | Real navigation around obstacles — requires the planar (2D) model. | L (incl. 2D model) |
| **Differential-drive WIP (2D)** | The true wheeled robot with yaw; unlocks planning + a richer demo. | L |

## Robotics frameworks (assessment)

Already in use: **MuJoCo** (sim/render), **Gymnasium** + **Stable-Baselines3** (RL),
**CVXPY/OSQP** (MPC), **python-control / SciPy** (classical), **SymPy** (derivation),
**DEAP** (GA). These are the right, well-supported choices. Considered but *not* adopting
now (cost > benefit at this stage): ROS 2 (overkill for a single sim), Drake (heavy),
acados (hard to install), Pinocchio (not needed for 2-DOF). Revisit ROS 2 only if/when I
target real hardware.

## Suggested implementation order

1. ⭐ LQG and ⭐ Cascaded PID (quick wins that enrich the benchmark immediately).
2. ⭐ Energy swing-up + hybrid switch, then ⭐ iLQR swing-up (the standout demos).
3. ⭐ H∞ and ⭐ Adaptive (MRAC) (the robustness story).
4. ⭐ RL: PPO + ⭐ domain randomization (with SAC as an option).
5. ⭐ Optuna tuning (with a CMA-ES comparison).
6. ⭐ Min-jerk reference tracking; defer A\*/RRT until the **2D extension**.
