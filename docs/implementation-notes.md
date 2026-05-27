# Implementation Notes

My technical decisions, derivations, and rationale — the "why" companion to the code.

---

## ND-1 — Canonical plant model (2026-05-25)

### The problem in the legacy code
The legacy implementation contained **two inconsistent dynamics models**:

1. `Control GUI/segway_dynamics.py` — parametric, `Ip = 0.2`, `r = 0.1`, **torque input**
   with the traction/reaction split folded into `B`.
2. `Control GUI/segway_control_{mpc,poleplacement,slidingmode}.py` — `Ip = 5·0.4² = 0.8`,
   `r = 0.0726`, **pure cart-force input**, and a different `a,b,c,d,Δ` parameterization.

These are *different plants*. A controller "validated" against one was silently run on the
other, and the two even apply control to MuJoCo differently (two actuators vs one).

### The decision
The v2 canonical model is the **torque-input wheeled inverted pendulum**:

- State `q = [x, ẋ, θ, θ̇]`, input `u = τ` (motor torque).
- A single motor torque `τ` produces a horizontal traction force `F = τ/r` on the base and,
  by Newton's third law, a reaction torque `−τ` on the body about the axle.

I chose this because (a) it is the more physically faithful description of a real
self-balancing robot driven by wheel motors, and (b) it matches the dual-actuator mapping
I already used in the better legacy runner. It also reduces cleanly to the classic cart-pole
and gives me a natural upgrade path to a true rolling-contact wheel model.

### Equations of motion (full nonlinear)
With `M = m_base + 2·m_wheel`, body mass `m`, COM distance `l`, body inertia `I`,
`c = cosθ`, `s = sinθ`, viscous damping `b_x`, `b_θ`:

```
[ M+m      m·l·c ] [ ẍ ]   [ τ/r − b_x·ẋ + m·l·s·θ̇²     ]
[ m·l·c   m·l²+I ] [ θ̈ ] = [ −τ + m·g·l·s − b_θ·θ̇       ]
```

Solve the 2×2 system for `[ẍ, θ̈]` → the nonlinear `f(state, τ)` used by the simulator.

### Linearization about the upright equilibrium (`θ=0, θ̇=0, ẋ=0, τ=0`)
With `D = (M+m)(m·l²+I) − (m·l)²`:

```
        [ 0   1                       0                  0                  ]
A =     [ 0  −(I+m·l²)·b_x/D         −(m·l)²·g/D          m·l·b_θ/D          ]
        [ 0   0                       0                  1                  ]
        [ 0   m·l·b_x/D               (M+m)·m·g·l/D      −(M+m)·b_θ/D        ]

        [ 0                       ]
B =     [ (I+m·l²)/r + m·l ) / D  ]
        [ 0                       ]
        [ −( m·l/r + (M+m) ) / D  ]
```

This reproduces `segway_dynamics.py`'s `A`/`B` exactly — confirming that file was the
correct one. I machine-verify it two ways in `tests/test_dynamics.py`:
1. against a finite-difference Jacobian of the nonlinear `f` (numeric linearization), and
2. against an independent SymPy derivation in `src/segway/models/symbolic.py`.

### Default parameters
`m_base=1.0, m_wheel=0.432, m_pend=5.0, l=0.4, r=0.1, I_pend=0.2, g=9.81,
b_x=0.05, b_theta=0.2`. The upright equilibrium is open-loop **unstable** (a positive
real eigenvalue from the `(M+m)·m·g·l/D` gravity term) and the pair `(A,B)` is
**controllable** — both asserted in the tests.

---

## ND-2 — Two simulation backends (2026-05-25)

I needed the benchmark to be fast, deterministic, and headless (CI-friendly), but I also
wanted high-fidelity 3D for demos, so I built two backends that share the *same*
controllers/metrics:

- **Analytic backend** (default): fixed-step RK4 integration of the nonlinear `f`. Pure
  NumPy/SciPy, no display, fully deterministic. This drives tests and benchmarks.
- **MuJoCo backend**: the `segway.xml` articulated model, used for visualization and as an
  independent physics cross-check of the analytic model.

---

## ND-3 — Controller interface (2026-05-25)

Every controller subclasses `segway.controllers.base.Controller` and implements
`compute(state, t) -> float` (+ optional `reset()`). The runner regulates to a reference
by calling `compute(state − reference, t)`, so all controllers are zero-regulators and
reference tracking is handled uniformly. Controllers are constructed via the
`build_controller(name, params, **kwargs)` factory backed by a registry, which the
benchmark harness and both UIs use so there is a single construction path.

---

## ND-4 — Cascaded PID (2026-05-25)

The plain tilt-only PID balances but drifts in position (its base coordinate is an
uncontrolled cyclic mode → a marginal pole at the origin). The cascaded controller adds an
**outer position loop** that turns a position/velocity error into a small *desired lean*
`theta_cmd = clip(kx·x + kv·ẋ, ±θ_cmd_max)` (with `kx, kv < 0`), and an **inner PD(+I)**
loop that drives the tilt to that command. Time-scale separation matters: a weakly damped
inner loop produces a slow inter-loop **limit cycle** (~0.15 rad). Raising the inner
derivative gain (`kd≈45`) restores clean settling. The result holds position to ~0.12 m
where the plain PID drifts ~2.8 m — which I verify in `tests/test_advanced_controllers.py`.

## ND-5 — Energy-based swing-up (2026-05-25)

A hybrid controller: an **energy-shaping** law swings the body up from hanging, then an LQR
**catches and balances** it once within `switch_angle` of upright.

Key subtlety for *this* plant: the textbook Åström–Furuta law assumes the input is **cart
acceleration** `a`, giving `dE/dt = −m l a cos θ · θ̇`, so `a = k (E − E_top) θ̇ cos θ`
pumps energy toward the upright value `E_top = m g l`. But the actual input here is **motor
torque**; applying the energy law *directly as torque* fails — the `τ→F=τ/r` force
amplification (×10) just throws the cart away. I fix this by commanding a desired cart
acceleration and converting it to torque via `τ = (M+m)·a·r`. I add two terms
(`−k_cart·ẋ − k_pos·x`) to keep the cart from running away during pumping, and I have the
balancer hold the position *where it caught* (not the origin) so it isn't fighting a large
position error at the handoff. From hanging it reaches upright (min |θ| < 0.05) and is
caught — which I verify in the test suite. Swing-up is a different
task from regulation, so it is **excluded from the regulation benchmark** (`REGULATORS`) and
showcased on its own (`assets/swingup.gif`).

## ND-6 — H∞ robust control (2026-05-25)

I implemented **state-feedback H∞** from first principles (no `slycot`): the suboptimal-γ
controller solves the H∞ algebraic Riccati equation via its **Hamiltonian matrix** (stable
invariant subspace → `X = U₂U₁⁻¹`), with a **γ-bisection** toward the smallest feasible γ
(then backed off by `gamma_margin` for margin). `K = R⁻¹B₂ᵀX`.

**Important nuance (measured, not assumed):** H∞ minimizes the worst-case *disturbance →
performance* gain, **not** parametric robustness. Measured closed-loop H∞ norm (w→z):
**H∞ = 2.57 vs LQR = 2.92** — H∞ wins on its own objective. But on the *parametric* Monte-
Carlo test (±60% mass etc. from a 1.0 rad tilt) the aggressive H∞ is **less** robust than
LQR; backing γ off (larger `gamma_margin`) recovers parametric robustness as H∞ → H₂/LQR
(margin 10 → 80%, matching LQR). Default `gamma_margin=2.0` balances the two. These are
different robustness axes — an instructive result rather than "H∞ is just better."

## ND-7 — Why pole placement looked bad, and the fix (2026-05-25)

The benchmark first showed pole placement at **37% robustness / 64% ROA** — much worse than
LQR. I investigated with a pole-set sweep, which showed this is **entirely a pole-selection
problem**, not a code bug:

| poles | robustness | ROA |
|---|---|---|
| `[-3,-4,-5,-6]` (naive default) | 38% | 64% |
| legacy `[-43.9,-0.76,-2.05,-48.8]` | 37% | 64% |
| fast `[-6,-7,-8,-9]` | 10% | 27% |
| **LQR-optimal poles** | **80%** | **85%** |
| **chosen default `[-1.2±0.9j, -10, -25]`** | **77%** | **84%** |

Lesson: pole placement is *only as good as its poles*; arbitrary placement gives no
robustness guarantee, while the LQR-optimal locations are excellent (LQR **is** the
cost-optimal pole placement). The new default is a well-damped dominant complex pair plus
two fast real poles — robust **and** distinct from LQR. Note also that **auto-tuning pole
placement on the nominal cost makes robustness *worse*** (38%→20%): the optimizer finds
aggressive poles that overfit the nominal scenarios. By contrast, tuning LQR and SMC raised
their robustness to **100%** — because those methods optimize a cost/surface with inherent
margin. This contrast is one of the project's headline control-theory takeaways.

## ND-8 — Adaptive control (MRAC) (2026-05-25)

Direct model-reference adaptive control augmenting the LQR baseline:
``u = -K x - theta_hat^T x``. The reference model is the nominal closed loop
``x_m_dot = A_m x_m`` (``A_m = A - B K``), and the weights follow the Lyapunov-stable law
``theta_hat_dot = gamma * x * (B^T P e) - sigma * theta_hat`` (``P`` solves
``A_m^T P + P A_m = -Q``; sigma-modification adds robustness). Because ``theta_hat = 0``
recovers the LQR baseline exactly, adaptation can only help. I verified it balances the
nominal plant and a +60% mass / +30% length / +60% inertia **mismatched** plant.

## ND-9 — Reinforcement learning (PPO) (2026-05-25)

A Gymnasium env (`segway/envs/gym_env.py`) wraps the *same* nonlinear plant (RK4 substeps,
20 ms control). Reward = ``cos(theta)`` minus quadratic penalties on position, velocities,
and effort; episodes end on fall or cart-limit. **Domain randomization** perturbs the
physical parameters each episode (±40% mass/inertia, ±25% length, ±60% damping), producing
a policy robust to model error — the RL analogue of the Monte-Carlo robustness benchmark.
I train it with Stable-Baselines3 PPO (`scripts/train_rl.py`) and wrap the learned policy as
a normal `Controller` (`RLController`) so it drops straight into the benchmark and the UIs —
the "model-based vs learned" comparison on identical physics.

## ND-10 — Swing-up visualization: the elevated cart-pole model (2026-05-27)

**Bug:** in the swing-up GIF the body appeared to pass *through the floor* and back out.
Cause is purely geometric, not dynamics: the ground-robot model (`segway.xml`) pivots at
wheel height (`base` at z = 0.1), so when the body rotates *below* the pivot during swing-up
(it must travel 180° from hanging to upright) the ~0.5 m body clips the floor at z = 0.

**Fix:** swing-up is the classic *cart-pole* maneuver, properly drawn with an **elevated
pivot**. I added `models/assets/cartpole.xml` — a cart on a raised horizontal rail (pivot at
z = 0.7) with the *same* body/joint/actuator names, so `MuJoCoPlant._apply_params` and the
`[τ/r, −τ]` torque mapping are unchanged. The **dynamics are identical** (gravity is uniform
and the slide is horizontal, so absolute height doesn't enter the EOM) — only the rendering
geometry differs, and the body now swings freely above the floor. `MuJoCoPlant`,
`simulate_mujoco`, `render_rollout`, and `live_view` all take an optional `xml_path`; the
swing-up renders and the GUI's swing-up View/Render pass `CARTPOLE_PATH`. I guard this with
`tests/test_mujoco.py::test_cartpole_swingup_model_clears_floor`.

## ND-11 — GUI redesign (2026-05-27)

I rebuilt `apps/desktop_gui.py` for usability while keeping every feature. I grouped the
configuration into a **tabbed panel** (🎛 Controller · 🤖 Robot · 🌍 Scenario) instead of one
long scroll; put a **persistent action bar** (Run / View-3D / ROA / Tune / Compare / Render)
with an **indeterminate progress bar** below it; and moved results into **tabbed panes**
(📈 Response · 🧮 Metrics cards · ◌ ROA). I added per-controller **descriptions**, **hover
tooltips**, a **reset-to-defaults** button, and a **Compare-all** overlay. All parameter
logic still comes from the shared `apps/_common.py`, so the desktop GUI and Streamlit
dashboard stay identical.
