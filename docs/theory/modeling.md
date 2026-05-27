# Modeling the self-balancing robot

This page derives the plant used throughout the project. The implementation lives in
[`src/segway/models/dynamics.py`](https://github.com/Manas-arumalla/segway-control-suite/blob/main/src/segway/models/dynamics.py)
and is cross-checked symbolically in
[`src/segway/models/symbolic.py`](https://github.com/Manas-arumalla/segway-control-suite/blob/main/src/segway/models/symbolic.py).

## System

A **wheeled inverted pendulum (WIP)**: a body (the "pendulum") balances on a wheeled base
that moves along a line. Generalized coordinates:

- $x$ — horizontal position of the wheel axle.
- $\theta$ — body tilt measured from the upward vertical ($\theta = 0$ is balanced).

The single control input is the **motor torque** $\tau$. The wheel of radius $r$ turns it
into a horizontal traction force $F = \tau/r$ on the base; by Newton's third law a reaction
torque $-\tau$ acts on the body about the axle.

### Parameters

| Symbol | Meaning | Default |
|---|---|---|
| $M = m_\text{base} + 2 m_w$ | total translating base mass | 1.864 kg |
| $m$ | body mass | 5.0 kg |
| $l$ | axle → body-COM distance | 0.4 m |
| $I$ | body inertia about its COM | 0.2 kg·m² |
| $r$ | wheel radius | 0.1 m |
| $b_x, b_\theta$ | viscous damping (axle, hinge) | 0.05, 0.2 |

## Lagrangian derivation

With COM position $(x + l\sin\theta,\; l\cos\theta)$, the kinetic and potential energies give
the Lagrangian $\mathcal{L} = T - V$. Applying the Euler–Lagrange equations and the
generalized forces $Q_x = \tau/r$, $Q_\theta = -\tau$ yields the coupled **nonlinear** EOM
($c=\cos\theta,\ s=\sin\theta$):

$$
\begin{bmatrix} M+m & m l c \\ m l c & m l^2 + I \end{bmatrix}
\begin{bmatrix} \ddot x \\ \ddot\theta \end{bmatrix}
=
\begin{bmatrix} \tau/r - b_x \dot x + m l s\, \dot\theta^2 \\
-\tau + m g l s - b_\theta \dot\theta \end{bmatrix}.
$$

This is integrated directly by the analytic simulator.

## Linearization about upright

Linearizing at $\theta=0,\ \dot\theta=0$ (so $c\to1,\ s\to\theta,\ \dot\theta^2\to0$) and
writing $D = (M+m)(I+m l^2) - (m l)^2$ gives the state-space model with
$q = [x, \dot x, \theta, \dot\theta]^\top$, input $\tau$:

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & -\frac{(I+m l^2) b_x}{D} & -\frac{(m l)^2 g}{D} & \frac{m l\, b_\theta}{D} \\
0 & 0 & 0 & 1 \\
0 & \frac{m l\, b_x}{D} & \frac{(M+m) m g l}{D} & -\frac{(M+m) b_\theta}{D}
\end{bmatrix},
\qquad
B = \begin{bmatrix}
0 \\ \frac{(I+m l^2)/r + m l}{D} \\ 0 \\ -\frac{m l/r + (M+m)}{D}
\end{bmatrix}.
$$

For the default parameters this evaluates to an open-loop pole at $+6.62$ (unstable, as
expected for an inverted pendulum), with $(A,B)$ controllable and $(A,C)$ observable for
$C = \begin{bmatrix}1&0&0&0\\0&0&1&0\end{bmatrix}$ (position + tilt).

## Validation

The analytic $(A,B)$ is checked three ways:

1. **Finite differences** — central-difference Jacobian of the nonlinear EOM.
2. **Symbolic** — an independent SymPy derivation differentiating the raw EOM.
3. **MuJoCo** — a controller designed on this model also stabilizes the independent
   MuJoCo articulated-body plant.

All three are enforced by the test suite (`tests/test_dynamics.py`, `tests/test_mujoco.py`).

## Assumptions & planned extensions

The wheels are currently a *rolling abstraction*: their only dynamical role is the
$\tau \to F$ conversion; wheel spin inertia and slip are neglected, and motion is planar. A
**true rolling-contact wheel model** (genuine WIP with no-slip constraints) is a planned
extension — the architecture isolates the model so this can be swapped in without touching
the controllers.
