# Navigation

The navigation extension takes the balancer from the line into the plane: the robot drives to
goal points, around obstacles, and over uneven terrain — **while staying upright**. The whole
stack is composable, so at run time you choose a **balancing controller × global planner ×
path follower**, on either an analytic or a full-physics backend.

<div align="center">
  <img src="https://raw.githubusercontent.com/Manas-arumalla/segway-control-suite/main/assets/nav_maps.png" width="640"/>
</div>

## The planar robot (TWIP)

Navigation uses a **two-wheeled inverted pendulum** (TWIP) — the 1-D balancer lifted into the
plane. Its 7-state is `[x, y, ψ, θ, v, ψ̇, θ̇]` (position, heading, pitch, forward speed, yaw
rate, pitch rate), driven by the two wheel torques `(τ_L, τ_R)`. The dynamics decouple:

- the **longitudinal / balance** subsystem (`v, θ`) is driven by the torque **sum**
  `τ_L + τ_R` and is *identical* to the validated 1-D plant — so every existing balancing
  controller is reused unchanged;
- the **yaw** subsystem (`ψ`) is driven by the torque **difference** `τ_R − τ_L`, a damped
  second-order system.

This is the standard TWIP approximation, and it is checked against full contact physics (see
[Backends](#backends-analytic-and-mujoco)).

## Inner loop: balance + yaw

`TWIPController` is the inner loop. It feeds the chosen balancing controller the longitudinal
error `[0, v − v_des, θ, θ̇]` to get the torque sum (regulating *speed*, not position, so the
robot tracks a velocity command while balancing), and a feed-forward + proportional law on the
yaw-rate error to get the torque difference; the two are split into left/right wheel torques.

Full-state controllers (LQR, MPC, SMC, H∞, MRAC, pole placement, cascaded PID) track the speed
command; the tilt-only PID balances but does not drive, by design.

## Global planners

A planner turns an obstacle map (an inflated occupancy grid, `World`) into a collision-free
path. All share one interface and a registry (`build_planner`):

| Planner | Kind | Notes |
|---|---|---|
| `a_star` | grid search | shortest path, admissible heuristic |
| `dijkstra` | grid search | uniform-cost (no heuristic) |
| `rrt` | sampling | fast in open space |
| `rrt_star` | sampling | asymptotically optimal (rewiring) |
| `prm` | sampling | roadmap, good for repeated queries |
| `potential_field` | reactive | gradient descent on an artificial field |

## Path followers

A follower turns the global path + the robot's pose into a `(v_des, ψ̇_des)` command for the
inner loop. All share one interface and a registry (`build_follower`):

| Follower | Kind | Notes |
|---|---|---|
| `pure_pursuit` | geometric | look-ahead point tracking |
| `stanley` | geometric | front-axle cross-track + heading |
| `dwa` | reactive | dynamic-window sampling, obstacle-aware |
| `mpc` | sampling | rollout-scored cross-track + heading |
| `vector_field` | reactive | attractive goal + repulsive obstacles |

Each follower densifies the (possibly sparse) path and tracks a *monotonically advancing*
look-ahead target, so it never aims at a waypoint it has already passed, and stops cleanly at
the goal.

## The composable navigator

`navigate(...)` (or the `Navigator` class) ties it together: it plans a path, points the robot
down the first leg, then rolls the TWIP forward at two rates — a fast inner balance loop and a
slower follower command loop — and reports success, driven-path length, time, and obstacle
clearance.

```python
from segway.config import TWIPParams
from segway.navigation import navigate, build_scenario

scenario = build_scenario("slalom")
result = navigate(TWIPParams(), scenario.world, scenario.start, scenario.goal,
                  balance="lqr", planner="a_star", follower="pure_pursuit")
print(result.success, result.path_length, result.min_clearance)
```

Preset maps (`build_scenario`): `corridor`, `slalom`, `rooms`, `forest`. A top-down plot is
available via `segway.viz.plot_navigation`, and a full run analysis — the speed-coloured route
beside speed/pitch/yaw tracking and wheel torques — via `segway.viz.plot_nav_analysis`:

<div align="center">
  <img src="https://raw.githubusercontent.com/Manas-arumalla/segway-control-suite/main/assets/nav_analysis.png" width="760"/>
</div>

## Backends: analytic and MuJoCo

The navigator runs on either backend:

- **`analytic`** (default) — a fast fixed-step RK4 integrator of the decoupled TWIP dynamics.
- **`mujoco`** — the **real rolling-contact robot**: a free-floating chassis on two driven
  wheels that roll on the ground through frictional contact. Nothing about the balance or
  traction is imposed — wheel torques produce traction and the chassis reaction *emerge* from
  contact physics — yet the same analytic-model controller balances, tracks speed, and turns.
  This is an independent check that the decoupling holds against full contact dynamics.

```python
navigate(TWIPParams(), world, start, goal, backend="mujoco")
```

A 3-D animation of a run (robot, obstacles, start, and goal) is produced by
`segway.viz.render_navigation`.

## Uneven terrain

Procedural height maps (`build_terrain`: `flat`, `gentle`, `moderate`, `rough`, `ramp`) drive a
MuJoCo **heightfield** the robot must traverse. It has no terrain sensing — the slopes act as
pure disturbances on the inner loop — yet the controller balances and drives across bumps and
up ramps. Terrain navigation runs on the MuJoCo backend:

```python
from segway.navigation import build_terrain
navigate(TWIPParams(), world, start, goal, backend="mujoco", terrain=build_terrain("rough"))
```

`segway.viz.plot_terrain_run` shows the route over the height map, the elevation profile along
the path, and the body pitch tracking the ground slope:

<div align="center">
  <img src="https://raw.githubusercontent.com/Manas-arumalla/segway-control-suite/main/assets/nav_terrain_analysis.png" width="760"/>
</div>

## Learned navigation (RL)

Alongside the classical stack there is an **end-to-end learned navigator**: a single PPO policy
(`TWIPNavEnv`) that maps the goal — expressed in the robot's own frame — plus the balance state
straight to the two wheel torques, with no planner or follower. It is trained with domain
randomization and wrapped as a drop-in `rl_navigate(...)` that returns the same `NavResult`, so
it is compared head-to-head with the classical stack.

```bash
python scripts/train_nav_rl.py --timesteps 1000000 --out models/ppo_twip_nav.zip
```

The policy learns to turn toward and drive to goals in any direction while balancing. It
settles at a **~0.4 m radial standoff** rather than landing exactly on the point — a
velocity-tracking inverted pendulum must lean back to decelerate as it arrives — so the learned
navigator uses a 0.45 m arrival tolerance, realistic for a 0.5 m-wide robot. The classical stack
reaches a tighter tolerance because it has explicit stop logic; the learned policy trades a
little precision for a single network that does the whole job.

## Benchmark

`benchmarks/run_nav.py` runs three curated sweeps across the preset maps — varying the planner,
the follower, and the balance controller one axis at a time — and pits the learned navigator
against the classical stack on open-space goals. It writes a CSV and a Markdown report under
`benchmarks/nav_results/`.

```bash
python benchmarks/run_nav.py --rl-model models/ppo_twip_nav.zip
```

<div align="center">
  <img src="https://raw.githubusercontent.com/Manas-arumalla/segway-control-suite/main/assets/nav_benchmark.png" width="820"/>
</div>

What the sweeps show across the four maps:

- **Balance controller (24/24):** all six full-state controllers — LQR, MPC, SMC, H∞,
  pole-placement, cascaded-PID — drive the navigation on every map. This is the payoff of the
  decoupled design: any balancing controller drops straight into the inner loop.
- **Planner (20/24):** A*, Dijkstra, PRM, RRT, and RRT* solve every map (A*/Dijkstra give the
  shortest paths; RRT the longest); the artificial **potential field gets stuck** in local
  minima and never reaches.
- **Follower (17/20):** pure-pursuit, Stanley, and the sampling-MPC follower solve every map,
  while the **reactive followers (DWA, vector-field) stall** in the tight `rooms` doorways —
  the classic global-vs-reactive trade-off, made concrete.
- **Learned vs classical (open-space goals):** both reach every goal within the 0.45 m
  tolerance; the end-to-end PPO policy is competitive with the classical planner-plus-follower
  stack and, with no global path to track, actually arrives a little faster in open space.

<div align="center">
  <img src="https://raw.githubusercontent.com/Manas-arumalla/segway-control-suite/main/assets/nav_rl.png" width="760"/>
</div>

## Command line

```bash
segway list                                              # controllers, planners, followers, maps, terrains
segway nav --planner a_star --follower pure_pursuit --map slalom
segway nav --map corridor --backend mujoco --terrain rough
segway nav --map forest --plot nav_forest.png           # save a top-down plot
```
